#!/usr/bin/env python3
"""
NDOG/NWOG Strategie v2 — Gap-Erkennung + Rücklauf + ATR-Trail + Bestätigung
"""

import sys, json, argparse, logging
from datetime import timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

C = {
    "min_gap_points": 8.0,
    "max_gap_points": 150.0,
    "min_gap_atr": 0.3,
    "retrace_zone_pct": 0.6,
    "max_bars_retrace": 720,
    "min_retrace_bars": 5,
    "confirm_body_ratio": 0.4,
    "confirm_volume_factor": 1.2,
    "trail_mult": 2.5,
    "trail_activation": 1.5,
    "initial_stop_mult": 1.5,
    "session_ny": True,
    "session_london": True,
    "session_asia": False,
    "data_path": "data/nq_1m_databento_2024_2026.parquet",
}


def calc_atr(high, low, close, period=14):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def get_session(hour):
    if hour < 7: return "asia"
    elif hour < 13: return "london"
    return "ny"


def gap_index(df, d):
    """Finde ersten Bar-Index für Datum d (tz-safe)"""
    mask = df.index.date == d
    if mask.any():
        return int(np.where(mask)[0][0])
    return 0


def detect_entries(df):
    n = len(df)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    atr = calc_atr(high, low, close)
    vol_med = float(np.median(volume))
    
    # ─── NDOG: Daily Gaps ─────────────────────────────────────────
    dates = pd.Series(df.index.date, index=df.index)
    daily_open = df.groupby(dates)["Open"].first()
    daily_close = df.groupby(dates)["Close"].last()
    yesterday_close = daily_close.shift(1)
    
    gap_top = pd.concat([daily_open, yesterday_close], axis=1).max(axis=1)
    gap_btm = pd.concat([daily_open, yesterday_close], axis=1).min(axis=1)
    gap_size = gap_top - gap_btm
    gap_ce = (gap_top + gap_btm) / 2
    is_bull_gap = daily_open > yesterday_close
    is_bear_gap = daily_open < yesterday_close
    
    ndog = {}
    for d in daily_close.index:
        sz = gap_size.loc[d]
        if sz < C["min_gap_points"] or sz > C["max_gap_points"]:
            continue
        idx = gap_index(df, d)
        atr_val = atr[idx] if 0 <= idx < n else 14.0
        if sz < atr_val * C["min_gap_atr"]:
            continue
        ndog[d] = {
            "top": float(gap_top.loc[d]),
            "bot": float(gap_btm.loc[d]),
            "ce": float(gap_ce.loc[d]),
            "size": float(sz),
            "bull": bool(is_bull_gap.loc[d]),
            "bear": bool(is_bear_gap.loc[d]),
            "atr": float(atr_val),
        }
    
    # ─── NWOG: Weekly Gaps ────────────────────────────────────────
    week_start = pd.Series(
        [d - timedelta(days=d.weekday()) for d in dates],
        index=df.index
    )
    first_of_week = df.groupby(week_start).first()
    last_of_prev_week = df.groupby(week_start).last().shift(1)
    
    w_gap_top = pd.concat([first_of_week["Open"], last_of_prev_week["Close"]], axis=1).max(axis=1)
    w_gap_btm = pd.concat([first_of_week["Open"], last_of_prev_week["Close"]], axis=1).min(axis=1)
    w_gap_size = w_gap_top - w_gap_btm
    w_gap_ce = (w_gap_top + w_gap_btm) / 2
    w_is_bull = first_of_week["Open"] > last_of_prev_week["Close"]
    
    nwog = {}
    for ws in first_of_week.index:
        sz = w_gap_size.loc[ws]
        if sz < C["min_gap_points"] or sz > C["max_gap_points"]:
            continue
        idx = gap_index(df, ws.date())
        atr_val = atr[idx] if 0 <= idx < n else 14.0
        if sz < atr_val * C["min_gap_atr"]:
            continue
        nwog[ws] = {
            "top": float(w_gap_top.loc[ws]),
            "bot": float(w_gap_btm.loc[ws]),
            "ce": float(w_gap_ce.loc[ws]),
            "size": float(sz),
            "bull": bool(w_is_bull.loc[ws]),
            "bear": not bool(w_is_bull.loc[ws]),
            "atr": float(atr_val),
        }
    
    logger.info("  Gaps: %d NDOG, %d NWOG", len(ndog), len(nwog))
    
    # ─── Entry-Erkennung ─────────────────────────────────────────
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)
    entry_prices = np.zeros(n, dtype=float)
    entry_stops = np.zeros(n, dtype=float)
    active_gaps = []
    
    for i in range(1440, n):
        ts = df.index[i]
        d = ts.date()
        h = ts.hour if hasattr(ts, 'hour') else 0
        
        sess = get_session(h)
        if sess == "asia" and not C["session_asia"]: continue
        if sess == "london" and not C["session_london"]: continue
        if sess == "ny" and not C["session_ny"]: continue
        
        hi, lo, cl, vol = high[i], low[i], close[i], volume[i]
        
        # Neue Gaps aktivieren (erste Bar des Tages)
        is_new_day = i > 0 and df.index[i-1].date() != d
        if is_new_day:
            if d in ndog:
                g = dict(ndog[d])
                g["direction"] = -1 if g["bull"] else 1
                g["entry_bar"] = i
                g["max_bars"] = i + C["max_bars_retrace"]
                g["min_bars"] = i + C["min_retrace_bars"]
                active_gaps.append(g)
            if d.weekday() == 0:
                ws = d - timedelta(days=d.weekday())
                if ws in nwog:
                    g = dict(nwog[ws])
                    g["direction"] = -1 if g["bull"] else 1
                    g["entry_bar"] = i
                    g["max_bars"] = i + C["max_bars_retrace"]
                    g["min_bars"] = i + C["min_retrace_bars"]
                    active_gaps.append(g)
        
        # Prüfe aktive Gaps
        for gi in list(active_gaps):
            if i > gi["max_bars"]:
                active_gaps.remove(gi)
                continue
            if i < gi["min_bars"]:
                continue
            
            g_top = max(gi["top"], gi["bot"])
            g_bot = min(gi["top"], gi["bot"])
            g_ce = gi["ce"]
            in_zone = lo <= g_top and hi >= g_bot
            if not in_zone:
                continue
            
            zone_range = abs(g_top - g_bot)
            if zone_range <= 0:
                continue
            
            if gi["direction"] == 1:  # LONG (Gap Down → Fill Up)
                near_ce = abs(lo - g_bot) <= zone_range * C["retrace_zone_pct"]
            else:
                near_ce = abs(hi - g_top) <= zone_range * C["retrace_zone_pct"]
            
            if not near_ce:
                continue
            
            # Rejection-Bestätigung
            candle_range = hi - lo
            if candle_range <= 0:
                continue
            
            if gi["direction"] == 1:
                body_ratio = (cl - lo) / candle_range
            else:
                body_ratio = (hi - cl) / candle_range
            
            if body_ratio < C["confirm_body_ratio"]:
                continue
            if vol < vol_med * C["confirm_volume_factor"]:
                continue
            
            # ENTRY!
            entries[i] = True
            directions[i] = gi["direction"]
            entry_prices[i] = cl
            atr_val = atr[i]
            if gi["direction"] == 1:
                entry_stops[i] = cl - atr_val * C["initial_stop_mult"]
            else:
                entry_stops[i] = cl + atr_val * C["initial_stop_mult"]
            active_gaps.remove(gi)
    
    logger.info("  Entries: %d", int(np.sum(entries)))
    return entries, directions, entry_prices, entry_stops, atr


def backtest(df, entries, directions, entry_prices, entry_stops, atr):
    trades = []
    n = len(df)
    
    for i in range(n):
        if not entries[i] or directions[i] == 0:
            continue
        
        entry_price = entry_prices[i]
        direction = directions[i]
        atr_entry = atr[i]
        trail_stop = entry_stops[i]
        trail_active = False
        best_close = entry_price
        
        max_hold = min(n, i + 1440)
        
        for j in range(i + 1, max_hold):
            cl_j = float(df.iloc[j]["Close"])
            
            if direction == 1:  # LONG
                lo_j = float(df.iloc[j]["Low"])
                best_close = max(best_close, cl_j)
                profit = best_close - entry_price
                if not trail_active and profit >= atr_entry * C["trail_activation"]:
                    trail_active = True
                    trail_stop = best_close - atr_entry * C["trail_mult"]
                if trail_active:
                    trail_stop = max(trail_stop, best_close - atr[j] * C["trail_mult"])
                if lo_j <= trail_stop:
                    trades.append({"entry_bar": i, "entry_time": str(df.index[i]),
                        "entry_price": entry_price, "dir": "LONG", "exit_bar": j,
                        "exit_time": str(df.index[j]), "exit_price": trail_stop,
                        "pnl": trail_stop - entry_price, "bars_held": j - i})
                    break
            else:  # SHORT
                hi_j = float(df.iloc[j]["High"])
                best_close = min(best_close, cl_j)
                profit = entry_price - best_close
                if not trail_active and profit >= atr_entry * C["trail_activation"]:
                    trail_active = True
                    trail_stop = best_close + atr_entry * C["trail_mult"]
                if trail_active:
                    trail_stop = min(trail_stop, best_close + atr[j] * C["trail_mult"])
                if hi_j >= trail_stop:
                    trades.append({"entry_bar": i, "entry_time": str(df.index[i]),
                        "entry_price": entry_price, "dir": "SHORT", "exit_bar": j,
                        "exit_time": str(df.index[j]), "exit_price": trail_stop,
                        "pnl": entry_price - trail_stop, "bars_held": j - i})
                    break
        else:
            # Trade hält bis Ende
            last_cl = float(df.iloc[max_hold - 1]["Close"])
            if direction == 1:
                pnl = last_cl - entry_price
            else:
                pnl = entry_price - last_cl
            trades.append({"entry_bar": i, "entry_time": str(df.index[i]),
                "entry_price": entry_price, "dir": "LONG" if direction == 1 else "SHORT",
                "exit_bar": max_hold - 1, "exit_time": str(df.index[max_hold - 1]),
                "exit_price": last_cl, "pnl": pnl, "bars_held": max_hold - 1 - i})
    
    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    print("  Lade Daten...")
    df = pd.read_parquet(C["data_path"])
    df.columns = [c.capitalize() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if args.bars and args.bars < len(df):
        df = df.iloc[-args.bars:]
    
    n = len(df)
    print(f"  {n:,} Bars ({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})")
    
    entries, dirs, prices, stops, atr = detect_entries(df)
    
    print("  Backtest...")
    trades = backtest(df, entries, dirs, prices, stops, atr)
    
    if args.json:
        winners = [t for t in trades if t["pnl"] > 0]
        gross_win = sum(t["pnl"] for t in winners) if winners else 0
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        print(json.dumps({
            "n_bars": n, "entries": int(np.sum(entries)), "n_trades": len(trades),
            "total_pnl": sum(t["pnl"] for t in trades),
            "pf": round(gross_win / max(gross_loss, 1), 2),
            "wr": round(len(winners) / max(len(trades), 1) * 100, 1),
        }, indent=2))
        return
    
    if not trades:
        print("\n  Keine Trades.")
        return
    
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    gross_win = sum(t["pnl"] for t in winners)
    gross_loss = abs(sum(t["pnl"] for t in losers))
    pf = gross_win / max(gross_loss, 1)
    wr = len(winners) / max(len(trades), 1) * 100
    avg_win = gross_win / max(len(winners), 1)
    avg_loss = gross_loss / max(len(losers), 1)
    avg_bars = np.mean([t["bars_held"] for t in trades])
    max_win = max(t["pnl"] for t in trades)
    max_loss = min(t["pnl"] for t in trades)
    
    longs = [t for t in trades if t["dir"] == "LONG"]
    shorts = [t for t in trades if t["dir"] == "SHORT"]
    
    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║     NDOG/NWOG v2 — BACKTEST                 ║")
    print(f"  ╚══════════════════════════════════════════════╝")
    print(f"  Daten:    {n:,} Bars 1min NQ")
    print(f"  Trades:   {len(trades)}")
    print(f"  Exit:     ATR-Trail ({C['trail_mult']}x, Act {C['trail_activation']}x)")
    print(f"  ─── PERFORMANCE ───")
    print(f"  PnL:      {total_pnl:>+8.0f}")
    print(f"  PF:       {pf:.2f}")
    print(f"  WR:       {wr:.1f}%")
    print(f"  Avg W/L:  {avg_win:.0f} / {avg_loss:.0f}")
    print(f"  Avg Hold: {avg_bars:.0f} Bars")
    print(f"  Max W/L:  {max_win:>+8.0f} / {max_loss:>+8.0f}")
    print(f"  ─── LONG vs SHORT ───")
    print(f"  LONG:  {len(longs):>3}  PnL {sum(t['pnl'] for t in longs):>+8.0f}")
    print(f"  SHORT: {len(shorts):>3}  PnL {sum(t['pnl'] for t in shorts):>+8.0f}")
    
    sorted_t = sorted(trades, key=lambda t: t["pnl"])
    print("\n  ─── BESTE ───")
    for t in sorted_t[-5:][::-1]:
        print(f"  +{t['pnl']:>6.0f}  {t['dir']:>5}  {t['entry_time']}")
    print("\n  ─── SCHLECHTESTE ───")
    for t in sorted_t[:5]:
        print(f"  {t['pnl']:>+7.0f}  {t['dir']:>5}  {t['entry_time']}")


if __name__ == "__main__":
    main()
