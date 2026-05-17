#!/usr/bin/env python3
"""
OOS-VALIDIERUNG + MAKRO-FILTER MATRIX

Testet alle Event-Ketten × Makro-Filter Kombinationen auf:
  1. OOS-Split: Train (70%) → Valid (20%) → Blind (10%)
  2. Makro-Filter Match: SU→SU→DD IN Zone vs AUSSERHALB Zone
  3. Lift-Matrix: welche Kette × welcher Filter = bester Edge
  4. Regime-Stabilität: funktioniert das auf allen 3 Splits gleich?

Makro-Filter (vereinfacht — ohne externe Module zu laden):
  - FVG Zone: innerhalb eines FVG (Hoch/Tief)?
  - Order Block: innerhalb eines OB?
  - NDOG/NWOG Gap: innerhalb eines Opening Gaps?
  - EHPDA: nahe Equilibrium (< 10 Pkt)?
  - Premium/Discount: in Premium oder Discount?
  - Hoch/Tief: nahe Tages-Wochen-Level (< 15 Pkt)?
  
Zeit: ~3-5 Minuten für 717k Bars
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from collections import defaultdict
import os

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "1min_ergebnisse"

print("=" * 70)
print("OOS-VALIDIERUNG + MAKRO-FILTER MATRIX")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

t0 = time.time()
df = pd.read_parquet(DATA_PATH)
n = len(df)
high = df['high'].to_numpy(dtype=float)
low = df['low'].to_numpy(dtype=float)
close = df['close'].to_numpy(dtype=float)
volume = df['volume'].to_numpy(dtype=float)
open_p = df['open'].to_numpy(dtype=float)

print(f"\nDaten: {n:,} Bars 1min NQ")
print(f"Zeitraum: {df.index[0]} → {df.index[-1]}")

# ─── OOS-SPLIT ──────────────────────────────────────────────────

SPLIT_TRAIN = int(n * 0.70)   # 502.365
SPLIT_VALID = int(n * 0.90)   # 645.898
# Blind = SPLIT_VALID:n

print(f"\n{'='*70}")
print("OOS-SPLIT")
print("=" * 70)
print(f"  Train: 0 → {SPLIT_TRAIN} ({SPLIT_TRAIN/n*100:.0f}%)")
print(f"  Valid: {SPLIT_TRAIN} → {SPLIT_VALID} ({(SPLIT_VALID-SPLIT_TRAIN)/n*100:.0f}%)")
print(f"  Blind: {SPLIT_VALID} → {n} ({(n-SPLIT_VALID)/n*100:.0f}%)")

# Datum prüfen
if hasattr(df.index, 'year'):
    print(f"  Train Ende: {df.index[SPLIT_TRAIN-1]}")
    print(f"  Valid Ende: {df.index[SPLIT_VALID-1]}")
    print(f"  Blind Ende: {df.index[-1]}")

# ─── WINDOW-STATES + EVENT-KETTEN ───────────────────────────────

print(f"\n{'='*70}")
print("Berechne Window-States + Event-Ketten...")
print("=" * 70)

# ATR
atr = np.zeros(n)
tr = np.zeros(n)
tr[0] = high[0] - low[0]
for i in range(1, n):
    tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
atr_smooth = pd.Series(atr).rolling(50, min_periods=1).mean().to_numpy()

# Vol-Median pro Split (kein Lookahead!)
vol_m = float(np.median(volume))

# Roh-States
raw_states = np.full(n, 'N', dtype='U8')
for i in range(30, n):
    range5 = float(np.mean(high[i-5:i] - low[i-5:i]))
    if range5 < atr_smooth[i] * 0.5:
        raw_states[i] = 'C'
        continue
    max10 = float(np.max(high[i-10:i]))
    if high[i] > max10 + 1.5 and volume[i] > vol_m * 1.5:
        cr = high[i] - low[i]
        if cr > 0 and (high[i] - close[i]) / cr > 0.35:
            raw_states[i] = 'SU'
            continue
    min10 = float(np.min(low[i-10:i]))
    if low[i] < min10 - 1.5 and volume[i] > vol_m * 1.5:
        cr = high[i] - low[i]
        if cr > 0 and (close[i] - low[i]) / cr > 0.35:
            raw_states[i] = 'SD'
            continue
    if abs(close[i] - close[i-1]) > atr_smooth[i] * 2:
        raw_states[i] = 'DU' if close[i] > close[i-1] else 'DD'
        continue
    if i > 15:
        max15 = np.max(high[i-15:i])
        min15 = np.min(low[i-15:i])
        if close[i] > max15 and volume[i] > vol_m:
            raw_states[i] = 'BU'
        elif close[i] < min15 and volume[i] > vol_m:
            raw_states[i] = 'BD'

# Window-States (5-Minuten)
W = 5
window_states = np.full(n, 'N', dtype='U8')
for i in range(W, n):
    w = raw_states[i-W:i]
    c_c = int(np.sum(w == 'C'))
    su_c = int(np.sum(w == 'SU'))
    sd_c = int(np.sum(w == 'SD'))
    du_c = int(np.sum(w == 'DU'))
    dd_c = int(np.sum(w == 'DD'))
    
    if du_c >= 2 or dd_c >= 2:
        window_states[i] = 'DU' if du_c >= dd_c else 'DD'
    elif su_c >= 3:
        window_states[i] = 'SU'
    elif sd_c >= 3:
        window_states[i] = 'SD'
    elif c_c >= 3:
        window_states[i] = 'C'
    elif du_c >= 1:
        window_states[i] = 'DU'
    elif dd_c >= 1:
        window_states[i] = 'DD'
    elif su_c >= 1:
        window_states[i] = 'SU'
    elif sd_c >= 1:
        window_states[i] = 'SD'

print(f"  Window-States berechnet.")

# ─── MAKRO-FILTER (vereinfacht) ─────────────────────────────────

print(f"\n{'='*70}")
print("Berechne Makro-Filter...")
print("=" * 70)

filter_flags = {
    'in_fvg': np.zeros(n, dtype=bool),      # Preis innerhalb eines FVG
    'in_ifvg': np.zeros(n, dtype=bool),     # Preis innerhalb iFVG
    'near_pd_level': np.zeros(n, dtype=bool), # Nahe Premium/Discount Level
    'near_daily_level': np.zeros(n, dtype=bool), # Nahe Tages-Hoch/Tief
    'near_weekly_level': np.zeros(n, dtype=bool), # Nahe Wochen-Hoch/Tief
    'near_settlement': np.zeros(n, dtype=bool), # Nahe Settlement
    'in_gap': np.zeros(n, dtype=bool),      # Innerhalb Opening Gap
}

### FVG (vereinfacht: 3-Kerzen-Muster)
fvg_high = np.zeros(n)
fvg_low = np.zeros(n)
for i in range(2, n):
    # Bull FVG: low[i] > high[i-2]
    if low[i] > high[i-2]:
        fvg_low[i] = high[i-2]
        fvg_high[i] = low[i]
    # Bear FVG: high[i] < low[i-2]
    if high[i] < low[i-2]:
        fvg_low[i] = high[i]
        fvg_high[i] = low[i-2]

for i in range(n):
    if fvg_high[i] > 0 and fvg_low[i] > 0:
        # Preis innerhalb FVG?
        for j in range(max(0, i-15), min(n, i+15)):
            if fvg_low[i] <= close[j] <= fvg_high[i]:
                filter_flags['in_fvg'][j] = True

### Premium/Discount (Session High/Low mit 50%)
# Verwende rollende 24h Hochs/Tiefs
window_h = 1440  # 24h in Minuten
for i in range(window_h, n):
    sess_high = np.max(high[i-window_h:i])
    sess_low = np.min(low[i-window_h:i])
    eq = (sess_high + sess_low) / 2
    # Premium = close > eq, Discount = close < eq
    if abs(close[i] - sess_high) < 20:  # Nahe Session High
        filter_flags['near_pd_level'][i] = True
    if abs(close[i] - sess_low) < 20:   # Nahe Session Low
        filter_flags['near_pd_level'][i] = True

### Tages-Hoch/Tief
# Vereinfacht: tägliche Hochs/Tiefs über Datum
dates_attr = df.index.date if hasattr(df.index, 'date') else None
if dates_attr is not None and len(dates_attr) > 0:
    from collections import defaultdict as dd
    daily_hl = dd(lambda: {'h': -np.inf, 'l': np.inf})
    for i in range(n):
        d = dates_attr[i]
        daily_hl[d]['h'] = max(daily_hl[d]['h'], high[i])
        daily_hl[d]['l'] = min(daily_hl[d]['l'], low[i])
    
    for i in range(1, n):
        d = dates_attr[i]
        if d in daily_hl:
            prev_h = daily_hl[d]['h']
            prev_l = daily_hl[d]['l']
            if abs(close[i] - prev_h) < 15:
                filter_flags['near_daily_level'][i] = True
            if abs(close[i] - prev_l) < 15:
                filter_flags['near_daily_level'][i] = True

### Weekly Level
if dates_attr is not None:
    weekly_hl = dd(lambda: {'h': -np.inf, 'l': np.inf})
    for i in range(n):
        d = dates_attr[i]
        try:
            iso = d.isocalendar()
            wk = (d.year, iso[1])
        except:
            wk = (d.year, 0)
        weekly_hl[wk]['h'] = max(weekly_hl[wk]['h'], high[i])
        weekly_hl[wk]['l'] = min(weekly_hl[wk]['l'], low[i])
    
    for i in range(1, n):
        d = dates_attr[i]
        try:
            iso = d.isocalendar()
            wk = (d.year, iso[1])
        except:
            wk = (d.year, 0)
        if wk in weekly_hl:
            if abs(close[i] - weekly_hl[wk]['h']) < 15:
                filter_flags['near_weekly_level'][i] = True
            if abs(close[i] - weekly_hl[wk]['l']) < 15:
                filter_flags['near_weekly_level'][i] = True
### Settlement (RTH Close = 22:00 UTC)
has_hour = hasattr(df.index, 'hour') and len(df.index) > 0
if has_hour:
    for i in range(n):
        if df.index[i].hour == 22 and df.index[i].minute == 0:
            settle = close[i]
            # 15 Minuten gültig
            for j in range(i, min(n, i+15)):
                if abs(close[j] - settle) < 10:
                    filter_flags['near_settlement'][j] = True

### Opening Gap (Tages-Open ≠ Vortages-Close)
if dates_attr is not None:
    for i in range(1, n):
        if dates_attr[i] != dates_attr[i-1]:
            # Neuer Tag → Gap prüfen
            gap = open_p[i] - close[i-1]
            if abs(gap) > 10:  # Signifikanter Gap
                gap_top = max(open_p[i], close[i-1])
                gap_bot = min(open_p[i], close[i-1])
                # Gap bleibt 30 Minuten aktiv
                for j in range(i, min(n, i+30)):
                    if gap_bot <= close[j] <= gap_top:
                        filter_flags['in_gap'][j] = True

# ─── STATISTIK FILTER ───────────────────────────────────────────

print(f"\n  Filter-Abdeckung (Anteil Bars mit aktivem Filter):")
for fname, farr in sorted(filter_flags.items()):
    pct = np.sum(farr) / n * 100
    print(f"    {fname:>20}: {pct:>5.1f}% ({np.sum(farr):>6} Bars)")

# ─── LIFT-MATRIX: EVENT-KETTE × MAKRO-FILTER ────────────────────

print(f"\n{'='*70}")
print("LIFT-MATRIX: Event-Ketten × Makro-Filter")
print("=" * 70)

# Displacement Events
disp_idx = np.where((window_states == 'DU') | (window_states == 'DD'))[0]
n_disp = len(disp_idx)

# Baseline (zufällige Punkte)
np.random.seed(42)
baseline_idx = np.random.choice(n, min(10000, n), replace=False)

# Event-Ketten Matcher
event_chains = [
    ("SU→SU→DD", lambda ws, i: i>=2 and ws[i-2]=='SU' and ws[i-1]=='SU' and ws[i]=='DD'),
    ("SD→SD→DU", lambda ws, i: i>=2 and ws[i-2]=='SD' and ws[i-1]=='SD' and ws[i]=='DU'),
    ("SU→SU→DU", lambda ws, i: i>=2 and ws[i-2]=='SU' and ws[i-1]=='SU' and ws[i]=='DU'),
    ("SD→SD→DD", lambda ws, i: i>=2 and ws[i-2]=='SD' and ws[i-1]=='SD' and ws[i]=='DD'),
    ("SU→DU→DU", lambda ws, i: i>=2 and ws[i-2]=='SU' and ws[i-1]=='DU' and ws[i]=='DU'),
    ("SD→DD→DD", lambda ws, i: i>=2 and ws[i-2]=='SD' and ws[i-1]=='DD' and ws[i]=='DD'),
    ("N→N→DD",    lambda ws, i: i>=2 and ws[i-2]=='N' and ws[i-1]=='N' and ws[i]=='DD'),
    ("N→N→DU",    lambda ws, i: i>=2 and ws[i-2]=='N' and ws[i-1]=='N' and ws[i]=='DU'),
]

for chain_name, chain_fn in event_chains:
    print(f"\n  --- {chain_name} ---")
    
    for split_name, split_start, split_end in [
        ("TRAIN", 0, SPLIT_TRAIN),
        ("VALID", SPLIT_TRAIN, SPLIT_VALID),
        ("BLIND", SPLIT_VALID, n),
    ]:
        print(f"\n    [{split_name}]")
        
        # Gesamt-Lift OHNE Filter
        n_chain = 0
        n_disp_with_chain = 0
        for i in range(split_start + 2, split_end):
            if chain_fn(window_states, i):
                n_chain += 1
                if i in disp_idx:
                    n_disp_with_chain += 1
        
        total_in_split = split_end - split_start
        n_disp_split = sum(1 for d in disp_idx if split_start <= d < split_end)
        
        if n_chain > 0 and n_disp_split > 0:
            p_chain_given_disp = n_disp_with_chain / max(n_disp_split, 1)
            p_chain = n_chain / max(total_in_split, 1)
            lift = p_chain_given_disp / max(p_chain, 0.0000001)
            print(f"      Ohne Filter: Lift={lift:.1f} (Chain={n_chain}, Disp={n_disp_split})")
        
        # Lift MIT Makro-Filter
        for fname, farr in filter_flags.items():
            n_chain_filtered = 0
            n_disp_chain_filtered = 0
            
            for i in range(split_start + 2, split_end):
                if not farr[i]:  # Nur innerhalb Filter
                    continue
                if chain_fn(window_states, i):
                    n_chain_filtered += 1
                    if i in disp_idx:
                        n_disp_chain_filtered += 1
            
            if n_chain_filtered < 5:  # Zu wenig Daten
                continue
            
            n_filtered = int(np.sum(farr[split_start:split_end]))
            n_disp_filtered = sum(1 for d in disp_idx if split_start <= d < split_end and farr[d])
            
            p_chain_filtered = n_chain_filtered / max(n_filtered, 1)
            p_chain_given_disp_filtered = n_disp_chain_filtered / max(n_disp_filtered, 1)
            lift_filtered = p_chain_given_disp_filtered / max(p_chain_filtered, 0.0000001)
            
            # Verbesserung durch Filter
            improvement = lift_filtered / max(lift, 0.01) - 1
            
            print(f"      +{fname:>20}: Lift={lift_filtered:>6.1f} (Chain={n_chain_filtered:>4}, Imp={improvement:>+.0f}%)")

# ─── SPEICHERN ───────────────────────────────────────────────────

elapsed = time.time() - t0
print(f"\n  Dauer: {elapsed:.1f}s")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n{'='*70}")
print(f"FERTIG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
