#!/usr/bin/env python3
"""
NDOG/NWOG Strategie — Gap erkennen, Rücklauf abwarten, dann Entry mit ATR-Trail.
Test auf NQ 1min Daten.

Usage:
    python3 test_ndog_strategy.py
    python3 test_ndog_strategy.py --bars 100000
"""

import sys, json, argparse, logging
from datetime import timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── PARAMETER ────────────────────────────────────────────────
MIN_GAP_PTS = 10.0       # Mindest-Gap in Punkten
RETRACE_TOUCH_PCT = 0.5  # Rücklauf bis CE (50% der Gap-Zone)
ATR_TRAIL_MULT = 2.0     # ATR-Trail Multiplikator
ATR_ACTIVATION = 1.0     # Trail-Aktivierung nach 1×ATR Gewinn
USE_NDOG = True          # Daily Gaps
USE_NWOG = True          # Weekly Gaps

DATA = str(Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet")


def detect_gaps(df):
    """Erkenne NDOG (Daily) und NWOG (Weekly) Gaps.
    Gibt Dict mit gap-Daten pro Bar zurück.
    """
    n = len(df)
    dates = pd.Series(df.index.date, index=df.index)
    
    # Daily Aggregation
    daily_open = df.groupby(dates)["Open"].first()
    daily_close = df.groupby(dates)["Close"].last()
    yesterday_close = daily_close.shift(1)
    
    # Tägliche Gaps (NDOG)
    gap_top = pd.concat([daily_open, yesterday_close], axis=1).max(axis=1)
    gap_btm = pd.concat([daily_open, yesterday_close], axis=1).min(axis=1)
    gap_size = gap_top - gap_btm
    gap_ce = (gap_top + gap_btm) / 2
    
    is_bull_gap = daily_open > yesterday_close  # Gap Up
    is_bear_gap = daily_open < yesterday_close  # Gap Down
    has_gap = gap_size >= MIN_GAP_PTS
    
    # Weekly Gaps (NWOG) — Monday Open vs Friday Close
    week_start = pd.Series([d - timedelta(days=d.weekday()) for d in daily_open.index], index=daily_open.index)
    weekly_open = daily_open.groupby(week_start).first()
    weekly_close = daily_close.groupby(week_start).last()
    weekly_close_prev = weekly_close.shift(1)
    
    w_gap_top = pd.concat([weekly_open, weekly_close_prev], axis=1).max(axis=1)
    w_gap_btm = pd.concat([weekly_open, weekly_close_prev], axis=1).min(axis=1)
    w_gap_size = w_gap_top - w_gap_btm
    w_gap_ce = (w_gap_top + w_gap_btm) / 2
    w_is_bull = weekly_open > weekly_close_prev
    w_is_bear = weekly_open < weekly_close_prev
    w_has_gap = w_gap_size >= MIN_GAP_PTS
    
    # Map zurück auf 1min Bars
    gaps_info = []
    
    for i in range(n):
        d = df.index[i].date() if hasattr(df.index[i], 'date') else df.index[i].date()
        # Nur an erster Bar des Tages prüfen
        is_first_bar = i == 0 or (hasattr(df.index[i], 'date') and 
            (df.index[i].date() != df.index[i-1].date() if i > 0 else False))
        
        info = {"ndog_active": False, "nwog_active": False, "entry_signal": False, "direction": 0}
        
        if is_first_bar and d in has_gap.index and has_gap.loc[d]:
            gt = float(gap_top.loc[d])
            gb = float(gap_btm.loc[d])
            gc = float(gap_ce.loc[d])
            bull = bool(is_bull_gap.loc[d])
            info["ndog_active"] = True
            info["ndog_top"] = gt
            info["ndog_bot"] = gb
            info["ndog_ce"] = gc
            info["ndog_bull"] = bull
            info["ndog_bear"] = not bull
        
        # Weekly Gap nur an Montag erster Bar
        if is_first_bar and d.weekday() == 0:
            wm = d - timedelta(days=d.weekday())
            if wm in w_has_gap.index and w_has_gap.loc[wm]:
                wgt = float(w_gap_top.loc[wm])
                wgb = float(w_gap_btm.loc[wm])
                wgc = float(w_gap_ce.loc[wm])
                wbull = bool(w_is_bull.loc[wm])
                info["nwog_active"] = True
                info["nwog_top"] = wgt
                info["nwog_bot"] = wgb
                info["nwog_ce"] = wgc
                info["nwog_bull"] = wbull
                info["nwog_bear"] = not wbull
        
        gaps_info.append(info)
    
    return gaps_info


def detect_retrace_entries(df, gaps_info):
    """Erkenne Rücklauf-Entries:
    - Gap existiert (NDOG/NWOG)
    - Preis toucht Gap-Zone (zwischen gap_top und gap_btm)
    - Preis kommt NAH an CE (Consequent Encroachment = 50%)
    """
    n = len(df)
    entries = np.zeros(n, dtype=bool)
    directions = np.zeros(n, dtype=int)  # 1 = LONG, -1 = SHORT
    
    active_gaps = []  # Liste aktiver Gaps [(top, bot, ce, direction), ...]
    
    for i in range(1, n):
        gi = gaps_info[i]
        hi = float(df.iloc[i]["High"])
        lo = float(df.iloc[i]["Low"])
        cl = float(df.iloc[i]["Close"])
        
        # Neuen Gap aktivieren
        if USE_NDOG and gi.get("ndog_active"):
            if gi["ndog_bull"]:
                active_gaps.append((gi["ndog_top"], gi["ndog_bot"], gi["ndog_ce"], -1))  # Gap Up → Short (Fill)
            else:
                active_gaps.append((gi["ndog_top"], gi["ndog_bot"], gi["ndog_ce"], 1))   # Gap Down → Long (Fill)
        
        if USE_NWOG and gi.get("nwog_active"):
            if gi["nwog_bull"]:
                active_gaps.append((gi["nwog_top"], gi["nwog_bot"], gi["nwog_ce"], -1))
            else:
                active_gaps.append((gi["nwog_top"], gi["nwog_bot"], gi["nwog_ce"], 1))
        
        # Prüfe ob Preis einen Gap retraced
        for j, (gt, gb, gc, direction) in enumerate(active_gaps):
            # Preis muss in Gap-Zone sein (zwischen bottom und top)
            in_zone = lo <= gt and hi >= gb
            
            if not in_zone:
                continue
            
            # Rücklauf-Kriterium: Preis nahe CE (50% der Zone)
            ce_range = abs(gt - gb) * RETRACE_TOUCH_PCT
            near_ce = abs(cl - gc) <= ce_range if ce_range > 0 else False
            
            if near_ce and not entries[i]:
                entries[i] = True
                directions[i] = direction
                # Gap nach Entry entfernen (einmal getradet)
                active_gaps.pop(j)
                break
    
    return entries, directions


def backtest(df, entries, directions):
    """Backtest mit ATR-Trail"""
    n = len(df)
    trades = []
    in_trade = False
    trade_dir = 0
    entry_price = 0.0
    entry_bar = 0
    trail_price = 0.0
    trail_active = False

    for i in range(100, n):
        if not in_trade:
            if entries[i] and directions[i] != 0:
                entry_price = float(df.iloc[i]["Close"])
                in_trade = True
                trade_dir = directions[i]
                entry_bar = i
                trail_price = entry_price - entry_price * 0.002 if trade_dir == 1 else entry_price + entry_price * 0.002
                trail_active = False
                trades.append({
                    "entry_bar": i, "entry_time": str(df.index[i]),
                    "entry_price": entry_price, "dir": "LONG" if trade_dir == 1 else "SHORT",
                    "exit_bar": None, "exit_price": None, "pnl": None, "bars_held": None,
                })
        else:
            t = trades[-1]
            hi = float(df.iloc[i]["High"])
            lo = float(df.iloc[i]["Low"])
            cl = float(df.iloc[i]["Close"])

            if trade_dir == 1:  # LONG
                if not trail_active and cl >= entry_price * 1.002:
                    trail_active = True
                if trail_active:
                    trail_price = max(trail_price, cl - cl * 0.002)
                if lo <= trail_price:
                    t["exit_bar"] = i
                    t["exit_price"] = trail_price
                    t["pnl"] = trail_price - entry_price
                    t["bars_held"] = i - entry_bar
                    in_trade = False
            else:  # SHORT
                if not trail_active and cl <= entry_price * 0.998:
                    trail_active = True
                if trail_active:
                    trail_price = min(trail_price, cl + cl * 0.002)
                if hi >= trail_price:
                    t["exit_bar"] = i
                    t["exit_price"] = trail_price
                    t["pnl"] = entry_price - trail_price
                    t["bars_held"] = i - entry_bar
                    in_trade = False

    if in_trade:
        t = trades[-1]
        t["exit_bar"] = n - 1
        t["exit_price"] = float(df.iloc[-1]["Close"])
        t["pnl"] = (t["exit_price"] - entry_price) if trade_dir == 1 else (entry_price - t["exit_price"])
        t["bars_held"] = n - 1 - entry_bar

    return trades


def main():
    parser = argparse.ArgumentParser(description="NDOG/NWOG Strategie Backtest")
    parser.add_argument("--bars", type=int, default=0)
    args = parser.parse_args()

    print("  Lade Daten...")
    df = pd.read_parquet(DATA)
    df.columns = [c.capitalize() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if args.bars:
        df = df.iloc[-args.bars:]
    
    n = len(df)
    print(f"  {n:,} Bars geladen ({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})")

    print("  Erkenne Gaps...")
    gaps = detect_gaps(df)
    
    ndog_count = sum(1 for g in gaps if g.get("ndog_active"))
    nwog_count = sum(1 for g in gaps if g.get("nwog_active"))
    print(f"  NDOG: {ndog_count}, NWOG: {nwog_count}")

    print("  Erkenne Rücklauf-Entries...")
    entries, dirs = detect_retrace_entries(df, gaps)
    entry_count = int(np.sum(entries))
    long_count = int(np.sum(dirs == 1))
    short_count = int(np.sum(dirs == -1))
    print(f"  Entries: {entry_count} ({long_count} Long, {short_count} Short)")

    print("  Backtest...")
    trades = backtest(df, entries, dirs)

    # Report
    if not trades:
        print("\n  Keine Trades.")
        return

    winners = [t for t in trades if t["pnl"] is not None and t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] is not None and t["pnl"] <= 0]
    closed = [t for t in trades if t["pnl"] is not None]
    
    total_pnl = sum(t["pnl"] for t in closed)
    gross_win = sum(t["pnl"] for t in winners)
    gross_loss = abs(sum(t["pnl"] for t in losers))
    pf = gross_win / max(gross_loss, 1)
    wr = len(winners) / max(len(closed), 1) * 100
    avg_win = gross_win / max(len(winners), 1)
    avg_loss = gross_loss / max(len(losers), 1)
    avg_bars = np.mean([t["bars_held"] for t in closed if t["bars_held"]])

    longs = [t for t in closed if t["dir"] == "LONG"]
    shorts = [t for t in closed if t["dir"] == "SHORT"]

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║   NDOG/NWOG STRATEGIE — BACKTEST        ║")
    print(f"  ╚══════════════════════════════════════════╝")
    print(f"  Daten:      {n:,} Bars 1min NQ")
    print(f"  Trades:     {len(trades)}")
    print(f"  Exit:       Trail (0.2%)")
    print()
    print(f"  ─── PERFORMANCE ───")
    print(f"  Gesamt PnL:  {total_pnl:>+8.0f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Win Rate:    {wr:.1f}%")
    print(f"  Avg Win:     {avg_win:.0f}   Avg Loss: {avg_loss:.0f}")
    print(f"  Avg Hold:    {avg_bars:.0f} Bars")
    print()
    print(f"  ─── LONG vs SHORT ───")
    print(f"  LONG:  {len(longs):>4} Trades")
    print(f"  SHORT: {len(shorts):>4} Trades")
    print()
    
    sorted_t = sorted(closed, key=lambda t: t["pnl"])
    print("  ─── BESTE TRADES ───")
    for t in sorted_t[-5:][::-1]:
        print(f"  +{t['pnl']:>6.0f}  {t['dir']:>5}  {t['entry_time']}")
    print()
    print("  ─── SCHLECHTESTE TRADES ───")
    for t in sorted_t[:5]:
        print(f"  {t['pnl']:>7.0f}  {t['dir']:>5}  {t['entry_time']}")


if __name__ == "__main__":
    main()
