#!/usr/bin/env python3
"""
EPE Strategy Backtest — Pre-Displacement Entry + Event-Ketten Filter + Trend
Testet auf NQ 1min Daten (Parquet) und gibt Performance-Report.

Usage:
    python3 test_epe_strategy.py
    python3 test_epe_strategy.py --bars 100000
    python3 test_epe_strategy.py --json
"""

import sys, os, json, argparse, logging
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── PARAMETER ────────────────────────────────────────────────
SWEEP_MIN_PTS = 1.0
REJECTION_BODY = 0.30
VOL_FACTOR = 1.2
TREND_TF = "15min"  # Trend auf 15min
TREND_EMA = 200
LIFT_MIN = 0.0  # Min Lift für Entry (0 = kein Filter)
SL_PTS = 0  # 0 = verwendet ATR-Trail
TP_PTS = 0  # 0 = verwendet ATR-Trail
ATR_TRAIL_MULT = 2.0  # Trail-Abstand = ATR × Multiplikator
ATR_TRAIL_ACTIVATION = 1.0  # Trail aktiv erst nach ATR × X Gewinn

DATA = str(Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet")

# ─── LIFT TABLE (validiert 717k Bars) ─────────────────────────
LIFT_TABLE_3EVENT = {
    "SU→SU→DD": (0.0, 26.4),
    "SD→SD→DU": (19.6, 0.0),
    "SU→SU→DU": (18.7, 0.0),
    "SD→SD→DD": (0.0, 16.6),
    "SU→DU→DU": (20.0, 0.0),
    "SD→DD→DD": (0.0, 18.6),
    "N→SD→DD": (0.0, 27.1),
    "N→SU→DD": (0.0, 19.1),
    "N→SD→DU": (17.7, 0.0),
    "N→SU→DU": (16.1, 0.0),
    "N→N→DD": (0.0, 16.9),
    "N→N→DU": (16.5, 0.0),
}
LIFT_TABLE_2EVENT = {
    "SU→DU": (20.2, 0.0), "SD→DD": (0.0, 15.4),
    "N→DD": (0.0, 16.0), "N→DU": (13.1, 0.0),
    "C→DD": (0.0, 18.5), "C→DU": (7.6, 0.0),
}
LIFT_TABLE_SINGLE = {
    "SU": (5.0, 3.0), "SD": (3.0, 5.0),
    "C": (4.0, 4.0), "DU": (11.6, 3.0), "DD": (3.0, 12.3),
}


def calc_atr(high, low, close, period=14):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def detect_raw_states(high, low, close, volume, atr_smooth, vol_median):
    """Erkenne 1min Roh-States: C, SU, SD, DU, DD, N"""
    n = len(high)
    states = np.full(n, "N", dtype="U8")
    for i in range(30, n):
        range5 = float(np.mean(high[i-5:i] - low[i-5:i]))
        if range5 < atr_smooth[i] * 0.5:
            states[i] = "C"
            continue
        max10 = float(np.max(high[i-10:i]))
        if high[i] > max10 + SWEEP_MIN_PTS and volume[i] > vol_median * VOL_FACTOR:
            cr = high[i] - low[i]
            if cr > 0 and (high[i] - close[i]) / cr > REJECTION_BODY:
                states[i] = "SU"
                continue
        min10 = float(np.min(low[i-10:i]))
        if low[i] < min10 - SWEEP_MIN_PTS and volume[i] > vol_median * VOL_FACTOR:
            cr = high[i] - low[i]
            if cr > 0 and (close[i] - low[i]) / cr > REJECTION_BODY:
                states[i] = "SD"
                continue
        if abs(close[i] - close[i-1]) > atr_smooth[i] * 2:
            states[i] = "DU" if close[i] > close[i-1] else "DD"
            continue
        if i > 15:
            max15 = float(np.max(high[i-15:i]))
            min15 = float(np.min(low[i-15:i]))
            if close[i] > max15 and volume[i] > vol_median:
                states[i] = "BU"
            elif close[i] < min15 and volume[i] > vol_median:
                states[i] = "BD"
    return states


def detect_window_states(raw_states, window_size=5, event_threshold=3):
    """Wandle Roh-States in Window-States um"""
    n = len(raw_states)
    window = np.full(n, "N", dtype="U8")
    for i in range(window_size, n):
        w = raw_states[i - window_size:i]
        c_count = int(np.sum(w == "C"))
        su_count = int(np.sum(w == "SU"))
        sd_count = int(np.sum(w == "SD"))
        du_count = int(np.sum(w == "DU"))
        dd_count = int(np.sum(w == "DD"))
        if du_count >= 2 or dd_count >= 2:
            window[i] = "DU" if du_count >= dd_count else "DD"
        elif su_count >= event_threshold:
            window[i] = "SU"
        elif sd_count >= event_threshold:
            window[i] = "SD"
        elif c_count >= event_threshold:
            window[i] = "C"
        elif du_count >= 1:
            window[i] = "DU"
        elif dd_count >= 1:
            window[i] = "DD"
        elif su_count >= 1:
            window[i] = "SU"
        elif sd_count >= 1:
            window[i] = "SD"
    return window


def get_chain(window_states, idx, lookback=3):
    if idx < lookback - 1 or idx >= len(window_states):
        return ""
    return "→".join(window_states[idx - lookback + 1:idx + 1])


def get_lift(chain, last_state):
    """Hole Lift-Werte aus Tabellen"""
    if chain in LIFT_TABLE_3EVENT:
        lb, lf = LIFT_TABLE_3EVENT[chain]
        return lb, lf, "A" if max(lb, lf) > 15 else "B"
    parts = chain.split("→")
    if len(parts) >= 2:
        short = "→".join(parts[-2:])
        if short in LIFT_TABLE_2EVENT:
            lb, lf = LIFT_TABLE_2EVENT[short]
            return lb, lf, "B" if max(lb, lf) > 10 else "C"
    if last_state in LIFT_TABLE_SINGLE:
        lb, lf = LIFT_TABLE_SINGLE[last_state]
        return lb, lf, "C"
    return 0.0, 0.0, "D"


def resample_to_tf(df, tf):
    """Resample 1min → höherer TF"""
    ohlc = df[["Open", "High", "Low", "Close"]].resample(tf).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    })
    ohlc["Volume"] = df["Volume"].resample(tf).sum()
    return ohlc


def trend_filter(df_1m):
    """Berechne 15min Trend via EMA200"""
    df_15 = resample_to_tf(df_1m, "15min")
    df_15["ema200"] = df_15["Close"].ewm(span=TREND_EMA, adjust=False).mean()
    df_15["trend_bull"] = df_15["Close"] > df_15["ema200"]
    # Forward-fill auf 1min
    trend_map = df_15["trend_bull"].reindex(df_1m.index, method="ffill")
    return trend_map.fillna(True).to_numpy()


def pre_disp_signals(high, low, close, volume, atr, vol_median):
    """Pre-Displacement Entry Signale (1min)"""
    n = len(high)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    for i in range(7, n - 1):
        # Compression
        avg_r5 = float(np.mean(high[i-5:i] - low[i-5:i]))
        atr_th = max(atr[i] * 0.5, 1.5)
        if avg_r5 >= atr_th:
            continue

        # Sweep Up
        max10 = float(np.max(high[i-10:i]))
        if high[i] > max10 + SWEEP_MIN_PTS and volume[i] > vol_median * VOL_FACTOR:
            cr = high[i] - low[i]
            if cr > 0 and (high[i] - close[i]) / cr > REJECTION_BODY:
                bull[i + 1] = True

        # Sweep Down
        min10 = float(np.min(low[i-10:i]))
        if low[i] < min10 - SWEEP_MIN_PTS and volume[i] > vol_median * VOL_FACTOR:
            cr = high[i] - low[i]
            if cr > 0 and (close[i] - low[i]) / cr > REJECTION_BODY:
                bear[i + 1] = True

    return bull, bear


def backtest(df, bull_signals, bear_signals, trend_bull, atr):
    """Führe Backtest mit ATR-Trail aus"""
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
            # Check entries
            if bull_signals[i] and trend_bull[i]:
                entry_price = float(df.iloc[i]["Close"])
                atr_val = atr[i]
                in_trade = True
                trade_dir = 1
                entry_bar = i
                trail_price = entry_price - atr_val * ATR_TRAIL_MULT
                trail_active = False
                trades.append({
                    "entry_bar": i,
                    "entry_time": str(df.index[i]),
                    "entry_price": entry_price,
                    "dir": "LONG",
                    "exit_bar": None,
                    "exit_time": None,
                    "exit_price": None,
                    "pnl": None,
                    "bars_held": None,
                })
            elif bear_signals[i] and not trend_bull[i]:
                entry_price = float(df.iloc[i]["Close"])
                atr_val = atr[i]
                in_trade = True
                trade_dir = -1
                entry_bar = i
                trail_price = entry_price + atr_val * ATR_TRAIL_MULT
                trail_active = False
                trades.append({
                    "entry_bar": i,
                    "entry_time": str(df.index[i]),
                    "entry_price": entry_price,
                    "dir": "SHORT",
                    "exit_bar": None,
                    "exit_time": None,
                    "exit_price": None,
                    "pnl": None,
                    "bars_held": None,
                })
        else:
            t = trades[-1]
            hi = float(df.iloc[i]["High"])
            lo = float(df.iloc[i]["Low"])
            cl = float(df.iloc[i]["Close"])
            atr_val = atr[i]

            if trade_dir == 1:  # LONG
                # Check activation: close must be above entry + ATR * activation
                if not trail_active and cl >= entry_price + atr_val * ATR_TRAIL_ACTIVATION:
                    trail_active = True
                    trail_price = max(trail_price, cl - atr_val * ATR_TRAIL_MULT)
                
                # Update trail
                if trail_active:
                    new_trail = cl - atr_val * ATR_TRAIL_MULT
                    if new_trail > trail_price:
                        trail_price = new_trail
                
                # Exit check
                if lo <= trail_price:
                    t["exit_bar"] = i
                    t["exit_time"] = str(df.index[i])
                    t["exit_price"] = trail_price
                    t["pnl"] = trail_price - entry_price
                    t["bars_held"] = i - entry_bar
                    in_trade = False
            else:  # SHORT
                if not trail_active and cl <= entry_price - atr_val * ATR_TRAIL_ACTIVATION:
                    trail_active = True
                    trail_price = min(trail_price, cl + atr_val * ATR_TRAIL_MULT)
                
                if trail_active:
                    new_trail = cl + atr_val * ATR_TRAIL_MULT
                    if new_trail < trail_price:
                        trail_price = new_trail
                
                if hi >= trail_price:
                    t["exit_bar"] = i
                    t["exit_time"] = str(df.index[i])
                    t["exit_price"] = trail_price
                    t["pnl"] = entry_price - trail_price
                    t["bars_held"] = i - entry_bar
                    in_trade = False

    # Close open trades at end
    if in_trade:
        t = trades[-1]
        t["exit_bar"] = n - 1
        t["exit_time"] = str(df.index[-1])
        t["exit_price"] = float(df.iloc[-1]["Close"])
        if trade_dir == 1:
            t["pnl"] = t["exit_price"] - entry_price
        else:
            t["pnl"] = entry_price - t["exit_price"]
        t["bars_held"] = n - 1 - entry_bar

    return trades


def print_report(trades, df):
    """Gib Performance-Report aus"""
    n_trades = len(trades)
    if n_trades == 0:
        print("\n  ╔══════════════════════════════════════════╗")
        print("  ║   EPE STRATEGY — KEINE TRADES           ║")
        print("  ╚══════════════════════════════════════════╝")
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
    avg_bars = np.mean([t["bars_held"] for t in closed if t["bars_held"] is not None])

    longs = [t for t in closed if t["dir"] == "LONG"]
    shorts = [t for t in closed if t["dir"] == "SHORT"]
    long_wr = len([t for t in longs if t["pnl"] > 0]) / max(len(longs), 1) * 100
    short_wr = len([t for t in shorts if t["pnl"] > 0]) / max(len(shorts), 1) * 100

    # Sessions
    sessions = {"asia": [], "london": [], "ny": []}
    for t in closed:
        if t["entry_time"]:
            h = int(t["entry_time"].split(" ")[1].split(":")[0]) if " " in t["entry_time"] else 0
            if h < 7:
                sessions["asia"].append(t)
            elif h < 13:
                sessions["london"].append(t)
            else:
                sessions["ny"].append(t)

    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   EPE STRATEGY — BACKTEST REPORT        ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"  Daten:      {len(df):,} Bars 1min NQ")
    print(f"  Zeitraum:   {str(df.index[0])[:10]} → {str(df.index[-1])[:10]}")
    print(f"  Trades:     {n_trades}")
    print(f"  Exit:       ATR-Trail ({ATR_TRAIL_MULT}x, Activation {ATR_TRAIL_ACTIVATION}x)")
    print()
    print(f"  ─── PERFORMANCE ───")
    print(f"  Gesamt PnL: {total_pnl:>+8.1f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Win Rate:   {wr:.1f}%")
    print(f"  Avg Win:    {avg_win:.1f}   Avg Loss: {avg_loss:.1f}")
    print(f"  Avg Hold:   {avg_bars:.0f} Bars")
    print()
    print(f"  ─── LONG vs SHORT ───")
    print(f"  LONG:  {len(longs):>4} Trades, WR {long_wr:.0f}%")
    print(f"  SHORT: {len(shorts):>4} Trades, WR {short_wr:.0f}%")
    print()
    print(f"  ─── SESSION ───")
    for sname, strades in sessions.items():
        if strades:
            s_wr = len([t for t in strades if t["pnl"] > 0]) / len(strades) * 100
            s_pnl = sum(t["pnl"] for t in strades)
            print(f"  {sname.upper():>6}: {len(strades):>3} Trades, WR {s_wr:.0f}%, PnL {s_pnl:>+8.1f}")
    print()

    # Top/Bottom 5
    sorted_trades = sorted(closed, key=lambda t: t["pnl"])
    print("  ─── BESTE TRADES ───")
    for t in sorted_trades[-5:][::-1]:
        print(f"  +{t['pnl']:>6.1f}  {t['dir']:>5}  {t['entry_time']}")
    print()
    print("  ─── SCHLECHTESTE TRADES ───")
    for t in sorted_trades[:5]:
        print(f"  {t['pnl']:>+7.1f}  {t['dir']:>5}  {t['entry_time']}")


def main():
    parser = argparse.ArgumentParser(description="EPE Strategy Backtest")
    parser.add_argument("--bars", type=int, default=0, help="N Bars (0 = alle)")
    parser.add_argument("--json", action="store_true", help="JSON Output")
    args = parser.parse_args()

    print("  Lade Daten...")
    df = pd.read_parquet(DATA)
    df.columns = [c.capitalize() for c in df.columns]
    df.index = pd.to_datetime(df.index)

    if args.bars and args.bars < len(df):
        df = df.iloc[-args.bars:]

    n = len(df)
    print(f"  {n:,} Bars geladen ({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})")

    # 1. ATR
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    atr = calc_atr(high, low, close)
    atr_smooth = pd.Series(atr).rolling(50, min_periods=1).mean().to_numpy()
    vol_median = float(np.median(volume))

    # 2. Event-Ketten States
    print("  Erkenne Event-Ketten States...")
    raw = detect_raw_states(high, low, close, volume, atr_smooth, vol_median)
    window = detect_window_states(raw)

    # 3. Pre-Disp Signale
    print("  Berechne Pre-Disp Signale...")
    bull_sig, bear_sig = pre_disp_signals(high, low, close, volume, atr, vol_median)
    print(f"  Pre-Disp: {int(np.sum(bull_sig))} Bull, {int(np.sum(bear_sig))} Bear")

    # 4. Event-Ketten Filter + Trend
    print("  Wende Event-Ketten + Trend Filter an...")
    trend = trend_filter(df)
    filtered_bull = np.zeros(n, dtype=bool)
    filtered_bear = np.zeros(n, dtype=bool)
    chain_counter = {}

    for i in range(5, n):
        if bull_sig[i]:
            chain = get_chain(window, i - 1, 3)
            lift_bull, lift_bear, quality = get_lift(chain, window[i-1] if i-1 < len(window) else "N")
            chain_counter[chain] = chain_counter.get(chain, 0) + 1
            if lift_bull >= LIFT_MIN and trend[i]:
                filtered_bull[i] = True
        if bear_sig[i]:
            chain = get_chain(window, i - 1, 3)
            lift_bull, lift_bear, quality = get_lift(chain, window[i-1] if i-1 < len(window) else "N")
            chain_counter[chain] = chain_counter.get(chain, 0) + 1
            if lift_bear >= LIFT_MIN and not trend[i]:
                filtered_bear[i] = True

    print(f"  Nach Filter: {int(np.sum(filtered_bull))} Bull, {int(np.sum(filtered_bear))} Bear")

    # 5. Backtest
    print("  Führe Backtest aus...")
    trades = backtest(df, filtered_bull, filtered_bear, trend, atr)

    if args.json:
        report = {
            "n_bars": n,
            "pre_disp_bull": int(np.sum(bull_sig)),
            "pre_disp_bear": int(np.sum(bear_sig)),
            "filtered_bull": int(np.sum(filtered_bull)),
            "filtered_bear": int(np.sum(filtered_bear)),
            "n_trades": len(trades),
            "top_chains": sorted(chain_counter.items(), key=lambda x: -x[1])[:10],
        }
        print(json.dumps(report, indent=2))
    else:
        print_report(trades, df)

        if chain_counter:
            print()
            print("  ─── TOP CHAINS (vor Filter) ───")
            for chain, cnt in sorted(chain_counter.items(), key=lambda x: -x[1])[:10]:
                lb, lf, q = get_lift(chain, chain.split("→")[-1] if "→" in chain else chain)
                print(f"  {chain:>20}: ×{cnt:>3}  Lift(B={lb:.0f},S={lf:.0f}) [{q}]")


if __name__ == "__main__":
    main()
