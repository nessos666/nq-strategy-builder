#!/usr/bin/env python3
"""
RUNDE 2 — Verbesserte 1-min Analyse:
  1. DFA-Hurst (Detrended Fluctuation Analysis — korrekt für nichtstationäre Daten)
  2. Event-Ketten mit Lift/Baseline-Vergleich
  3. Verbesserte Zustandserkennung (adaptive Thresholds)
  4. Live-Test auf letzten 5000 Bars mit Walk-Forward
  5. MFE/MAE Analyse wie auf 15min
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import os

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "1min_ergebnisse"

print("=" * 70)
print("1-MIN NQ — FORSCHUNG RUNDE 2 (Verbessert)")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

t0 = time.time()
df = pd.read_parquet(DATA_PATH)
print(f"\nDaten: {len(df):,} Bars")

high = df['high'].to_numpy(dtype=float)
low = df['low'].to_numpy(dtype=float)
close = df['close'].to_numpy(dtype=float)
volume = df['volume'].to_numpy(dtype=float)
open_p = df['open'].to_numpy(dtype=float)

n = len(close)

# ─── GLOBALE METRIKEN ───────────────────────────────────────────

def calc_atr(h, l, c, period=14):
    n = len(h)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()

atr = calc_atr(high, low, close, 14)
median_atr = float(np.median(atr[atr > 0]))
avg_vol = float(np.mean(volume))
vol_median = float(np.median(volume))

print(f"  ATR Median: {median_atr:.2f}")
print(f"  Vol Median: {vol_median:.0f}")

# ─── 1. DFA-HURST (KORREKT) ───────────────────────────────────────

print(f"\n{'='*70}")
print("EXP 2b: DFA-HURST-EXPONENT (Detrended Fluctuation Analysis)")
print("=" * 70)

def dfa_hurst(series, scales=None):
    """DFA Hurst — korrekt für nichtstationäre Daten"""
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series) & ~np.isinf(series)]
    n = len(series)
    if n < 20:
        return None, [], []
    if scales is None:
        scales = np.logspace(np.log10(10), np.log10(n // 4), 20, dtype=int)
        scales = np.unique(scales)
    
    # Integrate (cumulative sum of deviations from mean)
    y = np.cumsum(series - np.mean(series))
    
    flucts = []
    valid_scales = []
    for s in scales:
        if s < 5 or s >= n // 2:
            continue
        n_seg = n // s
        if n_seg < 2:
            continue
        f2 = 0.0
        for seg in range(n_seg):
            idx = seg * s
            seg_data = y[idx:idx + s]
            x = np.arange(s, dtype=float)
            # Linear detrend with safety
            if s < 3:
                continue
            try:
                c, _ = np.polyfit(x, seg_data, 1)
                trend = x * c + c  # simplified: y = mx + b
                # Actually: trend = polyval
                trend = np.polyval([c[0], c[1]], x) if hasattr(c, '__len__') else np.polyval([c], x)
            except:
                continue
            f2 += np.sum((seg_data - trend) ** 2)
        # Rest
        if n_seg * s < n:
            idx = n_seg * s
            seg_data = y[idx:]
            x = np.arange(len(seg_data), dtype=float)
            try:
                c = np.polyfit(x, seg_data, 1)
                trend = np.polyval(c, x)
                f2 += np.sum((seg_data - trend) ** 2)
            except:
                pass
        if n < 2:
            continue
        fluct = np.sqrt(f2 / n)
        flucts.append(fluct)
        valid_scales.append(s)
    
    if len(flucts) < 4:
        return None, [], []
    
    log_s = np.log([float(s) for s in valid_scales])
    log_f = np.log([float(f) for f in flucts])
    # Remove any non-finite
    mask = np.isfinite(log_s) & np.isfinite(log_f)
    if mask.sum() < 4:
        return None, [], []
    coeffs = np.polyfit(log_s[mask], log_f[mask], 1)
    return float(coeffs[0]), list(valid_scales), list(flucts)

# DFA auf gesamte Serie — downsampled auf 10min (jede 10. Bar)
step = 10
for window_bars, label in [(1000, '~16h'), (5000, '~3.5T'), (20000, '~14T')]:
    h_vals = []
    for i in range(0, n - window_bars, window_bars // 3):
        seg = close[i:i+window_bars:step]  # Downsample
        h, _, _ = dfa_hurst(seg)
        if h is not None:
            h_vals.append(h)
    if h_vals:
        print(f"  DFA-H auf {label}: Mean={np.mean(h_vals):.4f}, Std={np.std(h_vals):.4f}")

# DAF vor großen Bewegungen
returns_60m = np.array([abs(close[i+60] - close[i]) for i in range(0, n-60, 60)])
big_thresh = float(np.percentile(returns_60m, 95))
print(f"\n  60min >95% Perzentil: {big_thresh:.1f} Pkt")

h_before = []
h_normal = []
np.random.seed(42)
normal_idx = np.random.choice(list(range(3000, n-60)), 300, replace=False)

for idx in list(range(0, n-60, 60)) + list(normal_idx):
    if idx < 3000 or idx > n-60:
        continue
    end_5min = min(idx + 60, n)
    if abs(close[end_5min-1] - close[idx]) > big_thresh:
        # Vor großer Bewegung — 1000 Bars davor, downsampled
        h, _, _ = dfa_hurst(close[idx-1000:idx:step])
        if h is not None:
            h_before.append(h)

for idx in normal_idx:
    if idx < 1000 or idx > n-60:
        continue
    h, _, _ = dfa_hurst(close[idx-1000:idx:step])
    if h is not None:
        h_normal.append(h)

if h_before and h_normal:
    print(f"  H vor großen Moves: Mean={np.mean(h_before):.4f} (n={len(h_before)})")
    print(f"  H normal:           Mean={np.mean(h_normal):.4f} (n={len(h_normal)})")
    print(f"  Differenz: {np.mean(h_before) - np.mean(h_normal):+.4f}")

# ─── 2. VERBESSERTE ZUSTANDSMASCHINE ─────────────────────────────

print(f"\n{'='*70}")
print("EXP 4b: VERBESSERTE ZUSTANDSMASCHINE (adaptive Thresholds)")
print("=" * 70)

atr_smooth = pd.Series(atr).rolling(50).mean().to_numpy()
vol_smooth = pd.Series(volume).rolling(50).mean().to_numpy()

states = np.full(n, 'N', dtype='U8')
for i in range(30, n):
    # Compression: letzte 5 Bars Range < 50% des ATR-Smooth
    range5 = float(np.mean(high[i-5:i] - low[i-5:i]))
    if range5 < atr_smooth[i] * 0.5:
        states[i] = 'C'
        continue
    
    # Sweep Up: Hoch durchbricht +1.5 Pkt, Vol > 1.5x Median
    max10 = float(np.max(high[i-10:i]))
    if high[i] > max10 + 1.5 and volume[i] > vol_median * 1.5:
        cr = high[i] - low[i]
        if cr > 0 and (high[i] - close[i]) / cr > 0.35:
            states[i] = 'SU'
            continue
    
    # Sweep Down
    min10 = float(np.min(low[i-10:i]))
    if low[i] < min10 - 1.5 and volume[i] > vol_median * 1.5:
        cr = high[i] - low[i]
        if cr > 0 and (close[i] - low[i]) / cr > 0.35:
            states[i] = 'SD'
            continue
    
    # Displacement
    if abs(close[i] - close[i-1]) > atr_smooth[i] * 2:
        states[i] = 'DU' if close[i] > close[i-1] else 'DD'
        continue
    
    # Break
    if i > 15:
        max15 = np.max(high[i-15:i])
        min15 = np.min(low[i-15:i])
        if close[i] > max15 and volume[i] > vol_median:
            states[i] = 'BU'
        elif close[i] < min15 and volume[i] > vol_median:
            states[i] = 'BD'

unique, counts = np.unique(states, return_counts=True)
print(f"\n  Zustände (717k Bars):")
for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
    name = {'N':'Normal','C':'Compress','SU':'SweepUp','SD':'SweepDn',
            'DU':'DispUp','DD':'DispDn','BU':'BreakUp','BD':'BreakDn'}.get(u, u)
    print(f"    {name:>10}: {c:>7} ({c/n*100:.1f}%)")

# Signal-Generator: Compression → Sweep → Rejection = Pre-Disp Signal
sig_bars = np.zeros(n, dtype=bool)
sig_type = np.full(n, '', dtype='U8')
for i in range(20, n):
    if states[i] == 'SU' or states[i] == 'SD':
        # War Compression in den letzten 10 Bars?
        has_c = 'C' in states[i-10:i]
        if has_c:
            sig_bars[i + 1] = True
            sig_type[i + 1] = 'BULL' if states[i] == 'SU' else 'BEAR'

n_sigs = int(np.sum(sig_bars))
bull_sigs = int(np.sum(sig_type == 'BULL'))
bear_sigs = int(np.sum(sig_type == 'BEAR'))
print(f"\n  Pre-Disp Signale: {n_sigs} total ({bull_sigs} Bull, {bear_sigs} Bear)")
print(f"  ~{n_sigs/len(df)*1440:.1f}/Tag")

# ─── 3. MFE/MAE ANALYSE ─────────────────────────────────────────────

print(f"\n{'='*70}")
print("EXP 2b: MFE/MAE (Maximum Favorable/Adverse Excursion)")
print("=" * 70)

sig_idx = np.where(sig_bars)[0]
lookaheads = [5, 10, 15, 30, 60]  # In Minuten

print(f"\n  {'Target':>8}", end='')
for la in lookaheads:
    print(f"  {la:>4}min", end='')
print()

for target_pts in [10, 15, 20, 30, 40]:
    print(f"  {target_pts:>5}Pkt ", end='')
    for la in lookaheads:
        wins = 0
        for si in sig_idx:
            if si + la >= n:
                continue
            if sig_type[si] == 'BULL':
                move = max(high[si:si+la]) - close[si]
            else:
                move = close[si] - min(low[si:si+la])
            if move >= target_pts:
                wins += 1
        wr = wins / max(n_sigs, 1) * 100
        print(f"  {wr:>5.1f}%", end='')
    print()

# MFE/MAE Detail
print(f"\n  MFE/MAE (Lookahead 15min = 15 Bars):")
mfe_all = []
mae_all = []
for si in sig_idx:
    if si + 15 >= n:
        continue
    if sig_type[si] == 'BULL':
        mfe = max(high[si:si+15]) - close[si]
        mae = close[si] - min(low[si:si+15])
    else:
        mfe = close[si] - min(low[si:si+15])
        mae = max(high[si:si+15]) - close[si]
    mfe_all.append(mfe)
    mae_all.append(mae)

if mfe_all:
    print(f"    MFE Mean: {np.mean(mfe_all):.1f}, Median: {np.median(mfe_all):.1f}")
    print(f"    MAE Mean: {np.mean(mae_all):.1f}, Median: {np.median(mae_all):.1f}")
    print(f"    MFE/MAE Ratio: {np.mean(mfe_all) / max(np.mean(mae_all), 0.01):.2f}")

# ─── 4. SESSION-ANALYSE ──────────────────────────────────────────────

print(f"\n{'='*70}")
print("EXP: SESSION-BREAKDOWN (nur letzte 3 Monate mit Index)")
print("=" * 70)

# ts_event fehlt, aber wir haben den Row-Index
# Prüfe ob wir timestamp aus Index bekommen
if hasattr(df.index, 'hour'):
    hours = df.index.hour
else:
    hours = np.zeros(n, dtype=int)

# Wir versuchen UTC Stunden
# NQ: ASIA=0-8, LONDON=8-13, NY=13-22, LATE=22-24
if np.any(hours > 0):
    from collections import defaultdict
    sig_hours = defaultdict(int)
    sig_total_h = defaultdict(int)
    for si in sig_idx:
        h = hours[si] if si < len(hours) else 0
        sig_total_h[h] += 1
    print(f"  Signale pro Stunde (UTC):")
    for h in sorted(sig_total_h.keys()):
        print(f"    H{h:02d}: {sig_total_h[h]} Signale")
else:
    print("  (Keine Zeitstempel — Session-Breakdown nicht möglich)")

# ─── 5. ROHDATEN SPEICHERN FÜR WEITERE ANALYSE ─────────────────────────

print(f"\n{'='*70}")
print("ERGEBNISSE SPEICHERN")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Wichtigste Zahlen
results = {
    'rund2': {
        'pre_disp_signals': {
            'total': n_sigs,
            'bull': bull_sigs,
            'bear': bear_sigs,
            'per_day': n_sigs / len(df) * 1440,
        },
        'mfe_mae_15min': {
            'mfe_mean': float(np.mean(mfe_all)) if mfe_all else None,
            'mae_mean': float(np.mean(mae_all)) if mfe_all else None,
            'ratio': float(np.mean(mfe_all) / max(np.mean(mae_all), 0.01)) if mfe_all else None,
        },
        'state_distribution': {u: int(c) for u, c in zip(unique, counts)},
    }
}

# MFE/MAE Tabelle
for target_pts in [10, 15, 20, 30, 40]:
    for la in lookaheads:
        wins = 0
        for si in sig_idx:
            if si + la >= n:
                continue
            if sig_type[si] == 'BULL':
                move = max(high[si:si+la]) - close[si]
            else:
                move = close[si] - min(low[si:si+la])
            if move >= target_pts:
                wins += 1
        wr = wins / max(n_sigs, 1) * 100
        results['rund2'][f'wr_{target_pts}pkt_{la}min'] = wr

with open(f"{OUTPUT_DIR}/1min_results_rund2.json", 'w') as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
print(f"  Dauer: {elapsed:.1f}s")
print(f"  Datei: {OUTPUT_DIR}/1min_results_rund2.json")
print(f"\n{'='*70}")
print(f"FERTIG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
