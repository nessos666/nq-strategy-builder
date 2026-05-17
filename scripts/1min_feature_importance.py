#!/usr/bin/env python3
"""
PHASE 1 + 2 — FEATURE IMPORTANCE + OVERFITTING-HÄRTETEST

Isoliert jede Komponente des Pre-Displacement + Event-Ketten Systems:
  1. Compression allein (Kein Sweep, Keine Rejection, Keine Kette)
  2. Sweep allein (Keine Compression, Keine Rejection, Keine Kette)
  3. Rejection allein (Keine Compression, Kein Sweep, Keine Kette)
  4. Compression + Sweep (Keine Rejection)
  5. Compression + Sweep + Rejection (VOLL — heutiger Pre-Disp)
  6. SU→SU→DD Window-State Kette (Reine Kette, Kein Pre-Disp)
  7. Compression + Sweep + SU→SU→DD (Beide kombiniert)

Overfitting-Härtetest:
  - Walk-Forward: 12 monatliche Fenster
  - Regime-Split: ADX-Trend vs ADX-Range
  - Noise-Injection: +0.05%, +0.1%, +0.2%
  - Session-Split: Asia/London/NY

Jede Kombination wird gemessen an:
  - Lift (vs Baseline)
  - Precision (WR bei verschiedenen Targets)
  - Signal-Dichte (Signale/Tag)
  - MFE/MAE Ratio
  - OOS-Stabilität (StdDev Lift über Walk-Forward)

Zeit: ~5-8 Minuten
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from collections import defaultdict
import os
import sys

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "1min_ergebnisse"

print("=" * 70)
print("PHASE 1+2: FEATURE IMPORTANCE + OVERFITTING-HÄRTETEST")
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

# ─── BASISMETRIKEN ───────────────────────────────────────────────

def calc_atr(h, l, c, period=14):
    n = len(h)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()

atr = calc_atr(high, low, close, 14)
atr_s = pd.Series(atr).rolling(50, min_periods=1).mean().to_numpy()
vol_median = float(np.median(volume))

# Median ATR für Displacement-Target
median_atr = float(np.median(atr[atr > 0]))
print(f"  ATR Median: {median_atr:.2f}, Vol Median: {vol_median:.0f}")

# ─── KOMPONENTEN-DETEKTOREN ─────────────────────────────────────

print(f"\n{'='*70}")
print("Berechne Komponenten...")
print("=" * 70)

n = len(close)
# Jede Komponente als bool-Array (Signal auf NEXT bar)
sig_compression = np.zeros(n, dtype=bool)  # 1: Nur Compression
sig_sweep = np.zeros(n, dtype=bool)        # 2: Nur Sweep
sig_rejection = np.zeros(n, dtype=bool)    # 3: Nur Rejection
sig_comp_sweep = np.zeros(n, dtype=bool)   # 4: Compression + Sweep
sig_full = np.zeros(n, dtype=bool)         # 5: Compression + Sweep + Rejection

# Window-States für Event-Ketten
raw_states = np.full(n, 'N', dtype='U8')

for i in range(30, n):
    # Gemeinsame Berechnungen
    range5 = float(np.mean(high[i-5:i] - low[i-5:i]))
    atr_th = max(atr_s[i] * 0.5, 1.5)
    is_compression = range5 < atr_th
    
    max10 = float(np.max(high[i-10:i]) if i >= 10 else high[max(0,i-10):i].max())
    min10 = float(np.min(low[i-10:i]) if i >= 10 else low[max(0,i-10):i].min())
    
    sweep_up = high[i] > max10 + 1.5 and volume[i] > vol_median * 1.5
    sweep_down = low[i] < min10 - 1.5 and volume[i] > vol_median * 1.5
    is_sweep = sweep_up or sweep_down
    
    candle_range = high[i] - low[i]
    if candle_range > 0:
        if sweep_up:
            rejection_up = (high[i] - close[i]) / candle_range > 0.35
        else:
            rejection_up = False
        if sweep_down:
            rejection_down = (close[i] - low[i]) / candle_range > 0.35
        else:
            rejection_down = False
        is_rejection = rejection_up or rejection_down
    else:
        is_rejection = False
    
    # Komponente 1: Nur Compression (KEIN Sweep)
    if is_compression and not is_sweep:
        sig_compression[i + 1] = True
    
    # Komponente 2: Nur Sweep (KEINE Compression, KEINE Rejection)
    if is_sweep and not is_compression and not is_rejection:
        sig_sweep[i + 1] = True
    
    # Komponente 3: Rejection (KEIN Sweep, KEINE Compression)
    if is_rejection and not is_sweep and not is_compression:
        sig_rejection[i + 1] = True
    
    # Komponente 4: Compression + Sweep (OHNE Rejection)
    if is_compression and is_sweep and not is_rejection:
        sig_comp_sweep[i + 1] = True
    
    # Komponente 5: Compression + Sweep + Rejection (VOLL — aktueller Pre-Disp)
    if is_compression and is_sweep and is_rejection:
        sig_full[i + 1] = True
    
    # Roh-States für Window-Ketten
    if is_compression:
        raw_states[i] = 'C'
    elif sweep_up and is_rejection:
        raw_states[i] = 'SU'
    elif sweep_down and is_rejection:
        raw_states[i] = 'SD'
    elif abs(close[i] - close[i-1]) > atr_s[i] * 2:
        raw_states[i] = 'DU' if close[i] > close[i-1] else 'DD'
    elif i > 15:
        max15 = np.max(high[i-15:i])
        min15 = np.min(low[i-15:i])
        if close[i] > max15 and volume[i] > vol_median:
            raw_states[i] = 'BU'
        elif close[i] < min15 and volume[i] > vol_median:
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

# Komponente 6: Reine SU→SU→DD Kette (KEIN Pre-Disp Signal)
sig_chain_only = np.zeros(n, dtype=bool)
for i in range(3, n):
    if window_states[i-2] == 'SU' and window_states[i-1] == 'SU' and window_states[i] == 'DD':
        sig_chain_only[i] = True

# Komponente 7: Pre-Disp + Chain
sig_combined = np.zeros(n, dtype=bool)
for i in range(3, n):
    if sig_full[i] and window_states[i-2] == 'SU' and window_states[i-1] == 'SU' and window_states[i] == 'DD':
        sig_combined[i] = True

# Stats
components = {
    '1_compression_only': sig_compression,
    '2_sweep_only': sig_sweep,
    '3_rejection_only': sig_rejection,
    '4_comp_sweep': sig_comp_sweep,
    '5_full_pre_disp': sig_full,
    '6_chain_only': sig_chain_only,
    '7_combined': sig_combined,
}

print(f"\n  Signal-Dichte der Komponenten:")
for name, arr in components.items():
    cnt = int(np.sum(arr))
    per_day = cnt / max(n / 1440, 1)
    print(f"    {name:>25}: {cnt:>6} Signale ({per_day:.2f}/Tag)")

# ─── TEST-FUNKTION ───────────────────────────────────────────────

def test_component(signals, name, n, close, high, low, atr, median_atr, lookahead=15):
    """Teste eine Komponente auf Displacement-Prädiktivität.
    
    Returns dict mit Metriken.
    """
    sig_idx = np.where(signals)[0]
    total = len(sig_idx)
    if total == 0:
        return {'name': name, 'total': 0}
    
    per_day = total / max(n / 1440, 1)
    
    # Lift: P(Signal|Disp) / P(Signal)
    disp_indices = set()
    for i in range(n):
        if i + lookahead < n:
            move = abs(close[i+lookahead] - close[i])
            if move > median_atr * 2:
                disp_indices.add(i)
    
    n_disp = len(disp_indices)
    n_sig_disp = sum(1 for si in sig_idx if si in disp_indices)
    
    p_sig = total / n
    p_sig_disp = n_sig_disp / max(n_disp, 1)
    lift = p_sig_disp / max(p_sig, 0.0000001)
    
    # WR bei verschiedenen Targets
    targets = {'1xATR': median_atr, '1.5xATR': median_atr*1.5, 
               '2xATR': median_atr*2, '3xATR': median_atr*3}
    wr_results = {}
    for tname, tval in targets.items():
        wins = 0
        for si in sig_idx:
            if si + lookahead >= n:
                continue
            move = abs(close[si+lookahead] - close[si])
            if move > tval:
                wins += 1
        wr = wins / max(total, 1) * 100
        wr_results[tname] = wr
    
    # MFE/MAE
    mfe_vals = []
    mae_vals = []
    for si in sig_idx[:1000]:  # Max 1000 für Geschwindigkeit
        if si + lookahead >= n:
            continue
        mfe = max(high[si:si+lookahead]) - min(low[si:si+lookahead])
        mae = abs(close[si] - min(low[si:si+lookahead]) if close[si] > close[si+lookahead-1] 
                  else abs(close[si] - max(high[si:si+lookahead])))
        mfe_vals.append(mfe)
        mae_vals.append(mae)
    
    mfe_mean = float(np.mean(mfe_vals)) if mfe_vals else 0
    mae_mean = float(np.mean(mae_vals)) if mae_vals else 0
    mfe_mae_ratio = mfe_mean / max(mae_mean, 0.01)
    
    return {
        'name': name,
        'total': total,
        'per_day': per_day,
        'lift': lift,
        'n_disp': n_disp,
        'n_sig_disp': n_sig_disp,
        'wr_1xatr': wr_results.get('1xATR', 0),
        'wr_1_5xatr': wr_results.get('1.5xATR', 0),
        'wr_2xatr': wr_results.get('2xATR', 0),
        'wr_3xatr': wr_results.get('3xATR', 0),
        'mfe_mean': mfe_mean,
        'mae_mean': mae_mean,
        'mfe_mae_ratio': mfe_mae_ratio,
    }

# ─── PHASE 1: FEATURE IMPORTANCE ────────────────────────────────

print(f"\n{'='*70}")
print("PHASE 1: FEATURE IMPORTANCE")
print("=" * 70)

lookahead = 15  # 15 Minuten
results_f1 = {}

for name, arr in components.items():
    print(f"\n  --- {name} ---")
    r = test_component(arr, name, n, close, high, low, atr, median_atr, lookahead)
    results_f1[name] = r
    
    if r['total'] > 0:
        print(f"      Signale: {r['total']} ({r['per_day']:.2f}/Tag)")
        print(f"      Lift: {r['lift']:.1f} (Baseline: P(Signal)={r['total']/n*100:.3f}%)")
        print(f"      WR 1×ATR: {r['wr_1xatr']:.1f}%")
        print(f"      WR 1.5×ATR: {r['wr_1_5xatr']:.1f}%")
        print(f"      WR 2×ATR: {r['wr_2xatr']:.1f}%")
        print(f"      MFE/MAE: {r['mfe_mae_ratio']:.2f}")
    else:
        print(f"      KEINE SIGNALE")

# Rangliste erstellen
print(f"\n  === RANGLISTE (nach WR 2×ATR) ===")
ranked = sorted(
    [v for v in results_f1.values() if v['total'] > 0],
    key=lambda x: x['wr_2xatr'], reverse=True
)
print(f"  {'Rang':>4}  {'Name':>25}  {'Sig/Tag':>8}  {'Lift':>6}  {'WR2x':>6}  {'MFE/MAE':>8}")
print(f"  {'-'*65}")
for rank, r in enumerate(ranked, 1):
    print(f"  {rank:>4}  {r['name']:>25}  {r['per_day']:>8.2f}  {r['lift']:>6.1f}  {r['wr_2xatr']:>5.1f}%  {r['mfe_mae_ratio']:>8.2f}")

# ─── PHASE 2a: WALK-FORWARD TEST ────────────────────────────────

print(f"\n{'='*70}")
print("PHASE 2a: WALK-FORWARD TEST (12 monatliche Fenster)")
print("=" * 70)

# Nur die 3 besten Komponenten testen
top_components = sorted(
    [v for v in results_f1.values() if v['total'] > 50],
    key=lambda x: x['wr_2xatr'], reverse=True
)[:3]

print(f"\n  Teste Top-3: {[c['name'] for c in top_components]}")

# 12 monatliche Fenster (ca. 30 Tage = 43.200 Minuten)
window_size = 43200  # 30 Tage
step_size = 14400    # 10 Tage Schritt

wf_results = defaultdict(list)
for comp in top_components:
    name = comp['name']
    arr = components[name]
    
    for start in range(0, n - window_size, step_size):
        end = start + window_size
        if end >= n:
            break
        
        # Split in Train (60%) und Test (40%)
        split = start + int(window_size * 0.6)
        
        # Train auf ersten 60%, Test auf letzten 40%
        train_arr = np.zeros(n, dtype=bool)
        test_arr = np.zeros(n, dtype=bool)
        for i in range(start, split):
            train_arr[i] = arr[i]
        for i in range(split, end):
            test_arr[i] = arr[i]
        
        r_train = test_component(train_arr, f"{name}_train", n, close, high, low, atr, median_atr, lookahead)
        r_test = test_component(test_arr, f"{name}_test", n, close, high, low, atr, median_atr, lookahead)
        
        if r_train['total'] > 0 and r_test['total'] > 0:
            wf_results[name].append({
                'start': start,
                'train_lift': r_train['lift'],
                'test_lift': r_test['lift'],
                'train_wr': r_train['wr_2xatr'],
                'test_wr': r_test['wr_2xatr'],
            })
    
    lifts = [w['test_lift'] for w in wf_results[name]]
    wrs = [w['test_wr'] for w in wf_results[name]]
    
    if lifts:
        print(f"\n  Walk-Forward [{name}]:")
        print(f"    Fenster: {len(wf_results[name])}")
        print(f"    Test Lift: Mean={np.mean(lifts):.1f}, Std={np.std(lifts):.1f}, Min={min(lifts):.1f}, Max={max(lifts):.1f}")
        print(f"    Test WR:  Mean={np.mean(wrs):.1f}%, Std={np.std(wrs):.1f}%")
        print(f"    OOS-Stabilität: {'STABIL' if np.std(lifts) < 5 else 'INSTABIL'}")

# ─── PHASE 2b: REGIME-SPLIT ─────────────────────────────────────

print(f"\n{'='*70}")
print("PHASE 2b: REGIME-SPLIT (ADX-Trend vs ADX-Range)")
print("=" * 70)

# ADX auf 1min? Zu volatil. Verwende 60min ADX.
# Vereinfacht: 60-Bar Range vs Momentum
# High Regime = 60-Bar Range > Median
# Low Regime = 60-Bar Range < Median

window_regime = 60
regime_high_vol = np.zeros(n, dtype=bool)
regime_trend = np.zeros(n, dtype=bool)

for i in range(window_regime, n):
    range60 = float(np.std(close[i-window_regime:i]))
    # High Vol = oberes Drittel
    if range60 > np.percentile(close[max(0,i-10000):i], 66) if i > 10000 else range60 > 50:
        regime_high_vol[i] = True
    
    # Trend = close ist near 60-Bar High/Low (nicht in der Mitte)
    h60 = np.max(high[i-window_regime:i])
    l60 = np.min(low[i-window_regime:i])
    pos = (close[i] - l60) / max(h60 - l60, 0.01)
    regime_trend[i] = pos > 0.8 or pos < 0.2

for comp in top_components:
    name = comp['name']
    arr = components[name]
    
    arr_high = np.zeros(n, dtype=bool)
    arr_low = np.zeros(n, dtype=bool)
    arr_trend = np.zeros(n, dtype=bool)
    arr_range = np.zeros(n, dtype=bool)
    
    for i in range(window_regime, n):
        if arr[i]:
            if regime_high_vol[i]:
                arr_high[i] = True
            else:
                arr_low[i] = True
            if regime_trend[i]:
                arr_trend[i] = True
            else:
                arr_range[i] = True
    
    r_high = test_component(arr_high, f"{name}_highvol", n, close, high, low, atr, median_atr, lookahead)
    r_low = test_component(arr_low, f"{name}_lowvol", n, close, high, low, atr, median_atr, lookahead)
    r_trend = test_component(arr_trend, f"{name}_trend", n, close, high, low, atr, median_atr, lookahead)
    r_range = test_component(arr_range, f"{name}_range", n, close, high, low, atr, median_atr, lookahead)
    
    print(f"\n  Regime [{name}]:")
    for r, label in [(r_high, 'HighVol'), (r_low, 'LowVol'), (r_trend, 'Trend'), (r_range, 'Range')]:
        if r['total'] > 0:
            print(f"    {label:>8}: WR2x={r['wr_2xatr']:.1f}% Lift={r['lift']:.1f} (n={r['total']})")

# ─── PHASE 2c: NOISE-INJECTION ──────────────────────────────────

print(f"\n{'='*70}")
print("PHASE 2c: NOISE-INJECTION (Robustheit gegen Datenrauschen)")
print("=" * 70)

np.random.seed(42)
noise_levels = [0.000, 0.0005, 0.001, 0.002, 0.005]  # 0%, 0.05%, 0.1%, 0.2%, 0.5%

for comp in top_components[:1]:  # Nur beste Komponente
    name = comp['name']
    print(f"\n  Noise-Test [{name}]:")
    print(f"  {'Noise':>8}  {'Signale':>8}  {'Lift':>6}  {'WR2×':>6}  {'MFE/MAE':>8}  {'Stabilität':>10}")
    print(f"  {'-'*55}")
    
    original_lift = None
    for nl in noise_levels:
        if nl == 0.0:
            r = results_f1[name]
        else:
            # Verrauschte Preise
            noise_h = high + high * np.random.randn(n) * nl
            noise_l = low + low * np.random.randn(n) * nl
            noise_c = close + close * np.random.randn(n) * nl
            
            # Vereinfacht: nur auf Basis der Original-Signale testen
            r = test_component(components[name], f"{name}_noise{nl}", n, noise_c, noise_h, noise_l, atr, median_atr, lookahead)
        
        if original_lift is None and r['lift'] > 0:
            original_lift = r['lift']
        
        stability = '—'
        if original_lift and r['lift'] > 0:
            pct_change = abs(r['lift'] - original_lift) / original_lift * 100
            stability = f"{'STABIL' if pct_change < 20 else 'DRIFT' if pct_change < 50 else 'BRUCH'}" if r['lift'] > 0 else '—'
        
        if r['total'] > 0:
            print(f"  {nl*100:>6.2f}%  {r['total']:>8}  {r['lift']:>6.1f}  {r['wr_2xatr']:>5.1f}%  {r['mfe_mae_ratio']:>8.2f}  {stability:>10}")

# ─── PHASE 2d: SESSION-SPLIT ────────────────────────────────────

print(f"\n{'='*70}")
print("PHASE 2d: SESSION-SPLIT (Asia/London/NY)")
print("=" * 70)

has_hour = hasattr(df.index, 'hour') and len(df.index) > 0
if has_hour:
    # UTC Sessions: ASIA=0-7, LONDON=7-13, NY=13-21, LATE=21-24
    sig_asia = np.zeros(n, dtype=bool)
    sig_london = np.zeros(n, dtype=bool)
    sig_ny = np.zeros(n, dtype=bool)
    
    for comp in top_components[:1]:
        name = comp['name']
        arr = components[name]
        
        for i in range(n):
            if not arr[i]:
                continue
            h = df.index[i].hour
            if 0 <= h < 7:
                sig_asia[i] = True
            elif 7 <= h < 13:
                sig_london[i] = True
            elif 13 <= h < 21:
                sig_ny[i] = True
        
        for sarr, slabel in [(sig_asia, 'Asia'), (sig_london, 'London'), (sig_ny, 'NY')]:
            r = test_component(sarr, f"{name}_{slabel}", n, close, high, low, atr, median_atr, lookahead)
            if r['total'] > 0:
                print(f"    {slabel:>8}: n={r['total']:>4} WR2x={r['wr_2xatr']:.1f}% Lift={r['lift']:.1f} MFE/MAE={r['mfe_mae_ratio']:.2f}")

# ─── ZUSAMMENFASSUNG ─────────────────────────────────────────────

elapsed = time.time() - t0
print(f"\n{'='*70}")
print("ENDGÜLTIGE BEWERTUNG")
print("=" * 70)

# Wer ist der Gewinner?
winner = ranked[0] if ranked else None
if winner:
    print(f"\n  🏆 BESTE KOMPONENTE: {winner['name']}")
    print(f"     WR 2×ATR: {winner['wr_2xatr']:.1f}%")
    print(f"     Lift: {winner['lift']:.1f}")
    print(f"     Signale/Tag: {winner['per_day']:.2f}")
    print(f"     MFE/MAE: {winner['mfe_mae_ratio']:.2f}")

print(f"\n  ⏱  Dauer: {elapsed:.1f}s")

# Ergebnis speichern
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vereinfachte JSON-Ausgabe
results_out = {
    'timestamp': datetime.now().isoformat(),
    'feature_importance': {k: {kk: vv for kk, vv in v.items() if kk != 'name'} for k, v in results_f1.items()},
    'walk_forward': {k: {
        'n_windows': len(v),
        'mean_test_lift': float(np.mean([w['test_lift'] for w in v])),
        'std_test_lift': float(np.std([w['test_lift'] for w in v])),
        'mean_test_wr': float(np.mean([w['test_wr'] for w in v])),
    } for k, v in wf_results.items() if v},
}

with open(f"{OUTPUT_DIR}/feature_importance_results.json", 'w') as f:
    json.dump(results_out, f, indent=2, default=str)

print(f"  💾 {OUTPUT_DIR}/feature_importance_results.json")
print(f"\n{'='*70}")
print(f"FERTIG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
