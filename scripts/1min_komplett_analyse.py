#!/usr/bin/env python3
"""
KOMPLETTE 1-MINUTEN-NAHTO-FORSCHUNGSPIPELINE

Analysiert nq_1m_databento_2024_2026.parquet (717.665 Bars):
  1. Pre-Displacement Detektor auf 1min → 5min/15min Displacement
  2. Hurst-Exponent + Fraktale-Dimension für 1min
  3. Entropie-Analyse (Sample/Approx) vor großen Bewegungen
  4. Event-Ketten Mining: welche 1min Sequenzen gehen voraus?
  5. Multi-TF Lead-Lag: 1min→5min→15min
  6. Nächste große Bewegung Vorhersage
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta

# ─── KONFIGURATION ─────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "1min_ergebnisse"

# Pre-Disp Parameter (adaptiert für 1min)
SWEEP_MIN_PTS = 1          # 1 Punkt Sweep auf 1min
REJECTION_MIN_BODY = 0.45  # Rejection body ratio
ATR_PERIOD = 14
LOOKBACK_COMP = 5          # 5 Minuten für Compression (mehr Bars als auf 15min)
LOOKBACK_SWEEP = 10        # 10 Minuten für Sweep-Extreme
COMPRESSION_PERCENTILE = 0.30  # Top 30% niedrigste Range = Compression

# HURST
HURST_LAGS = [5, 10, 20, 50, 100, 200]

# ENTROPIE
ENTROPY_WINDOW = 60    # 1h Fenster für Entropie
ENTROPY_STEP = 10      # Alle 10 Minuten

# Multi-TF
TF_MAP = {'1min': 1, '5min': 5, '15min': 15, '1h': 60}

# ─── HELPER ────────────────────────────────────────────────────────

def calc_atr(high, low, close, period=14):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()

def calc_hurst(series, max_lag=200):
    """Hurst Exponent via Rescaled Range (R/S)"""
    n = len(series)
    if n < max_lag * 2:
        return None
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        # Split in chunks of size lag
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        chunks = series[:n_chunks * lag].reshape(n_chunks, lag)
        # R/S for each chunk
        mean = chunks.mean(axis=1, keepdims=True)
        y = np.cumsum(chunks - mean, axis=1)
        r = y.max(axis=1) - y.min(axis=1)
        s = chunks.std(axis=1, ddof=1)
        s = np.where(s == 0, 1e-10, s)
        rs = np.mean(r / s)
        tau.append(rs)
    if len(tau) < 5:
        return None
    lags = range(2, 2 + len(tau))
    coeffs = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return coeffs[0]  # H

def sample_entropy(series, m=2, r_factor=0.2):
    """Sample Entropy für Zeitreihe"""
    n = len(series)
    r = r_factor * np.std(series)
    if r == 0 or n < m + 2:
        return 0
    
    def _count_matches(template_len):
        count = 0
        templates = np.array([series[i:i+template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count
    
    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0:
        return 0
    return -np.log(A / B) if A > 0 else 0

def approximate_entropy(series, m=2, r_factor=0.2):
    """Approximate Entropy"""
    n = len(series)
    r = r_factor * np.std(series)
    if r == 0 or n < m + 2:
        return 0
    
    def _phi(template_len):
        templates = np.array([series[i:i+template_len] for i in range(n - template_len + 1)])
        counts = np.zeros(len(templates))
        for i in range(len(templates)):
            for j in range(len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    counts[i] += 1
        phi = np.mean(np.log(counts / len(templates)))
        return phi
    
    return _phi(m) - _phi(m + 1)

# ─── DATEN LADEN ───────────────────────────────────────────────────

print("=" * 70)
print("1-MINUTEN NQ-FORSCHUNGSPIPELINE")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

t0 = time.time()
print("\n--- Daten laden ---")
df = pd.read_parquet(DATA_PATH)
print(f"  {len(df):,} Bars geladen")
print(f"  Kolumnen: {list(df.columns)}")
print(f"  Zeit (aus Row-Index): {df.index[0]} bis {df.index[-1]}")

# ts_event fehlt — prüfen ob index Datetime ist
if not isinstance(df.index, pd.DatetimeIndex):
    print("  WARN: Index ist kein DatetimeIndex — versuche aus rtype/Zeilen-Nr zu rekonstruieren")
    # Ohne ts_event können wir keinen korrekten Zeitstempel setzen
    # Aber wir können trotzdem arbeiten — relativer Abstand = 1 Bar = 1 Minute

# Preis-Daten
high = df['high'].to_numpy(dtype=float)
low = df['low'].to_numpy(dtype=float)
close = df['close'].to_numpy(dtype=float)
volume = df['volume'].to_numpy(dtype=float)
open_p = df['open'].to_numpy(dtype=float)

n_total = len(close)
print(f"\n  Preisrange: {close.min():.2f} - {close.max():.2f}")
print(f"  Volume: {volume.mean():.0f} Ø, {volume.min():.0f}-{volume.max():.0f}")

# ─── 1. PRE-DISPLACEMENT AUF 1MIN ──────────────────────────────────────

print(f"\n{'='*70}")
print(f"EXP 1: PRE-DISPLACEMENT AUF 1MIN")
print(f"{'='*70}")

atr = calc_atr(high, low, close, ATR_PERIOD)
median_atr = float(np.median(atr[atr > 0])) if np.any(atr > 0) else 5
# ATR-Threshold: niedriger für 1min
atr_threshold = max(3.0, median_atr * 0.5)
avg_vol = float(np.mean(volume))

print(f"  ATR Median: {median_atr:.2f}")
print(f"  ATR Threshold: {atr_threshold:.2f}")
print(f"  Avg Volume: {avg_vol:.0f}")

# Pre-Disp für 1min
n = n_total
signal_bull = np.zeros(n, dtype=bool)
signal_bear = np.zeros(n, dtype=bool)

# Compression über 5 Minuten
sweep_count = 0
DISP_LOOKAHEAD = 15
for i in range(LOOKBACK_COMP + 2, n - DISP_LOOKAHEAD):
    # Compression
    prev_5 = high[i-LOOKBACK_COMP:i] - low[i-LOOKBACK_COMP:i]
    avg_range_5 = float(np.mean(prev_5))
    if avg_range_5 >= atr_threshold:
        continue
    
    # Sweep
    start = max(0, i - LOOKBACK_SWEEP)
    max_prev = float(np.max(high[start:i]))
    min_prev = float(np.min(low[start:i]))
    
    sweep_up = high[i] > max_prev + SWEEP_MIN_PTS
    sweep_down = low[i] < min_prev - SWEEP_MIN_PTS
    is_sweep = sweep_up or sweep_down
    
    if not is_sweep:
        continue
    
    if sweep_up:
        candle_range = high[i] - low[i]
        if candle_range <= 0:
            continue
        rejection = (high[i] - close[i]) / candle_range > REJECTION_MIN_BODY
        if rejection and volume[i] >= avg_vol * 0.3:
            signal_bull[i + 1] = True
            sweep_count += 1
    
    if sweep_down:
        candle_range = high[i] - low[i]
        if candle_range <= 0:
            continue
        rejection = (close[i] - low[i]) / candle_range > REJECTION_MIN_BODY
        if rejection and volume[i] >= avg_vol * 0.3:
            signal_bear[i + 1] = True
            sweep_count += 1

total_signals = int(np.sum(signal_bull) + np.sum(signal_bear))
print(f"  Signale gesamt: {total_signals}")
print(f"    Bull: {int(np.sum(signal_bull))}")
print(f"    Bear: {int(np.sum(signal_bear))}")

# Displacement-Check: Folgt auf Signal eine Bewegung > 1.5× ATR?
# Verschiedene Lookaheads und Ziele
lookaheads = [3, 5, 10, 15]  # 3, 5, 10, 15 Minuten
targets = [1.0, 1.5, 2.0, 3.0]  # Multiples von ATR
displacements = [5, 10, 15, 20]  # Absolute Punkte

print(f"\n  --- Displacement-Winrate (Lookahead × Target) ---")
print(f"  {'Lookahead':>10}", end="")
for t in targets:
    print(f"  {t:.1f}×ATR  ", end="")
print()
print(f"  {'-'*50}")

for la in lookaheads:
    print(f"  {la:>4}min   ", end="")
    for t in targets:
        # Prüfe ob in den nächsten la Bars eine Bewegung > t * median_atr auftritt
        wins = 0
        sig_idx = np.where(signal_bull | signal_bear)[0]
        if len(sig_idx) == 0:
            print(f"  {'N/A':>8}", end="")
            continue
        for si in sig_idx:
            if si + la >= n:
                continue
            slice_high = high[si:si+la]
            slice_low = low[si:si+la]
            movement = max(slice_high) - min(slice_low)
            if movement > t * median_atr:
                wins += 1
        wr = wins / len(sig_idx) * 100 if len(sig_idx) > 0 else 0
        print(f"  {wr:>6.1f}%  ", end="")
    print()

# Absolute Targets
print(f"\n  --- Displacement-Winrate (Lookahead × Abs-Punkte) ---")
print(f"  {'Lookahead':>10}", end="")
for d in displacements:
    print(f"  {d:>4}Pkt ", end="")
print()
print(f"  {'-'*55}")
for la in lookaheads:
    print(f"  {la:>4}min   ", end="")
    for d in displacements:
        wins = 0
        sig_idx = np.where(signal_bull | signal_bear)[0]
        if len(sig_idx) == 0:
            print(f"  {'N/A':>7}", end="")
            continue
        for si in sig_idx:
            if si + la >= n:
                continue
            movement = abs(close[si + la - 1] - close[si])
            if movement > d:
                wins += 1
        wr = wins / len(sig_idx) * 100 if len(sig_idx) > 0 else 0
        print(f"  {wr:>6.1f}%", end="")
    print()

# ─── 2. HURST-EXPONENT + FRAKTALE DIMENSION ─────────────────────────────

print(f"\n{'='*70}")
print(f"EXP 2: HURST-EXPONENT + FRAKTALE DIMENSION")
print(f"{'='*70}")

# Rolling Hurst für gesamte Serie
hurst_window = 1000  # ~16.7h Fenster
hurst_step = 200     # Alle 3.3h
hurst_values = []
hurst_times = []

for i in range(0, n - hurst_window, hurst_step):
    segment = close[i:i+hurst_window]
    h = calc_hurst(segment, max_lag=100)
    if h is not None:
        hurst_values.append(h)
        hurst_times.append(i)

print(f"  Fenster: {hurst_window} Bars, Step: {hurst_step}")
print(f"  H-Berechnungen: {len(hurst_values)}")

if hurst_values:
    h_mean = float(np.mean(hurst_values))
    h_std = float(np.std(hurst_values))
    h_gt_05 = sum(1 for h in hurst_values if h > 0.5) / len(hurst_values) * 100
    h_lt_05 = sum(1 for h in hurst_values if h < 0.5) / len(hurst_values) * 100
    
    print(f"\n  H-Exponent auf 1min:")
    print(f"    Mean H: {h_mean:.4f}")
    print(f"    Std H:  {h_std:.4f}")
    print(f"    H > 0.5 (trendend): {h_gt_05:.1f}%")
    print(f"    H < 0.5 (mean-rev): {h_lt_05:.1f}%")
    print(f"    H ≈ 0.5 (random):   {100 - h_gt_05 - h_lt_05:.1f}%")
    
    # Fraktale Dimension D = 2 - H (für Preisreihe)
    d_mean = 2 - h_mean
    print(f"\n  Fraktale Dimension D = 2 - H:")
    print(f"    Mean D: {d_mean:.4f}")
    print(f"    D=1.5=RandomWalk, D<1.5=trendend, D>1.5=mean-rev")

# Hurst vor großen Bewegungen vs. normale Phasen
# Große Bewegung = oberes 5% Perzentil der 60-Minuten-Rendite
returns_60 = np.array([abs(close[i+60] - close[i]) for i in range(0, n-60, 60)])
big_move_threshold = float(np.percentile(returns_60, 95))
print(f"\n  Große Bewegungen (>95% Perzentil, 60min):")
print(f"    Threshold: {big_move_threshold:.2f} Punkte")

# Prüfe H 1000 Bars vor jeder großen Bewegung
h_before_big = []
h_random = []
big_move_indices = [i for i in range(0, n-60, 60) if abs(close[i+60] - close[i]) > big_move_threshold]

sample_count = min(len(big_move_indices), 200)
import random
random.seed(42)
sample_random = random.sample(list(range(hurst_window, n - 60)), min(sample_count * 2, 500))

for idx in big_move_indices[:sample_count]:
    if idx < hurst_window:
        continue
    h = calc_hurst(close[idx-hurst_window:idx], max_lag=100)
    if h is not None:
        h_before_big.append(h)

for idx in sample_random:
    h = calc_hurst(close[idx-hurst_window:idx], max_lag=100)
    if h is not None:
        h_random.append(h)

if h_before_big and h_random:
    print(f"    H vor großen Moves: Mean={np.mean(h_before_big):.4f} (n={len(h_before_big)})")
    print(f"    H zufällige Phasen: Mean={np.mean(h_random):.4f} (n={len(h_random)})")
    print(f"    Differenz: {np.mean(h_before_big) - np.mean(h_random):+.4f}")

# ─── 3. ENTROPIE-ANALYSE ─────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"EXP 3: ENTROPIE-ANALYSE (Sample/Approximate Entropy)")
print(f"{'='*70}")

# Rolling ApproxEntropy auf 1h-Fenstern (60 Bars)
# Vereinfacht: berechne Entropie auf 60-Bar-Segmenten, vergleiche vor großen Moves
segment_size = 60  # 1h
seg_entropies = []
seg_returns = []

for i in range(0, n - segment_size - 60, segment_size):
    seg = close[i:i+segment_size]
    ent = approximate_entropy(seg, m=2, r_factor=0.2)
    future_move = abs(close[i+segment_size+60] - close[i+segment_size])  # nächste Stunde
    seg_entropies.append(ent)
    seg_returns.append(future_move)

seg_entropies = np.array(seg_entropies)
seg_returns = np.array(seg_returns)

print(f"  Segmente: {len(seg_entropies)}")
print(f"  Entropie: Mean={np.mean(seg_entropies):.4f}, Std={np.std(seg_entropies):.4f}")
print(f"  Return (60min): Mean={np.mean(seg_returns):.2f}, Std={np.std(seg_returns):.2f}")

# Korrelation Entropie → nächste große Bewegung?
return_percentiles = np.percentile(seg_returns, [50, 75, 90, 95, 99])
entropy_percentiles = np.percentile(seg_entropies, [10, 25, 50, 75, 90])

print(f"\n  Entropie-Perzentile:")
for p, v in zip([10, 25, 50, 75, 90], entropy_percentiles):
    print(f"    P{p}: {v:.4f}")

print(f"\n  Return (60min) nach niedriger Entropie (P10):")
low_entropy_mask = seg_entropies <= entropy_percentiles[0]
if np.any(low_entropy_mask):
    low_entropy_returns = seg_returns[low_entropy_mask]
    print(f"    Ø Return: {np.mean(low_entropy_returns):.2f}")
    print(f"    vs. Baseline: {np.mean(seg_returns):.2f}")
    print(f"    Faktor: {np.mean(low_entropy_returns) / max(np.mean(seg_returns), 0.01):.2f}×")

print(f"\n  Return (60min) nach hoher Entropie (P90):")
high_entropy_mask = seg_entropies >= entropy_percentiles[-1]
if np.any(high_entropy_mask):
    high_entropy_returns = seg_returns[high_entropy_mask]
    print(f"    Ø Return: {np.mean(high_entropy_returns):.2f}")
    print(f"    Faktor vs Baseline: {np.mean(high_entropy_returns) / max(np.mean(seg_returns), 0.01):.2f}×")

# ─── 4. EVENT-KETTEN MINING ──────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"EXP 4: EVENT-KETTEN MINING (1min Event-Sequenzen)")
print(f"{'='*70}")

# Diskrete Zustände auf 1min-Basis
n = len(high)
states = np.zeros(n, dtype='U8')
states[:] = 'NORMAL'

# Zustandserkennung
for i in range(20, n):
    # Compression (letzte 5 Bars)
    prev_5_range = np.mean(high[i-5:i] - low[i-5:i])
    if prev_5_range < atr_threshold * 0.8:
        states[i] = 'COMPRESS'
        continue
    
    # Sweep up
    max_prev = np.max(high[i-10:i])
    if high[i] > max_prev + SWEEP_MIN_PTS and volume[i] > avg_vol * 1.5:
        candle_r = high[i] - low[i]
        if candle_r > 0 and (high[i] - close[i]) / candle_r > REJECTION_MIN_BODY:
            states[i] = 'SWEEP_UP'
            continue
    
    # Sweep down
    min_prev = np.min(low[i-10:i])
    if low[i] < min_prev - SWEEP_MIN_PTS and volume[i] > avg_vol * 1.5:
        candle_r = high[i] - low[i]
        if candle_r > 0 and (close[i] - low[i]) / candle_r > REJECTION_MIN_BODY:
            states[i] = 'SWEEP_DN'
            continue
    
    # Displacement (große Bewegung)
    if abs(close[i] - close[i-1]) > median_atr * 1.5:
        if close[i] > close[i-1]:
            states[i] = 'DISP_UP'
        else:
            states[i] = 'DISP_DN'
        continue
    
    # MSS / Break
    if i > 20:
        last_10_high = np.max(high[i-10:i])
        last_10_low = np.min(low[i-10:i])
        if close[i] > last_10_high:
            states[i] = 'BREAK_UP'
        elif close[i] < last_10_low:
            states[i] = 'BREAK_DN'

# Sequenzen zählen
print(f"  Zustandsverteilung:")
unique, counts = np.unique(states, return_counts=True)
for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
    print(f"    {u:>10}: {c:>7} ({c/n*100:.1f}%)")

# 2-Event Sequenzen vor Displacement
print(f"\n  --- 2-Event Sequenzen vor DISP_UP ---")
disp_up_idx = np.where(states == 'DISP_UP')[0]
seq_counts = {}
for di in disp_up_idx:
    if di < 3:
        continue
    seq = f"{states[di-2]}→{states[di-1]}"
    seq_counts[seq] = seq_counts.get(seq, 0) + 1

top_seqs = sorted(seq_counts.items(), key=lambda x: -x[1])[:15]
for seq, cnt in top_seqs:
    print(f"    {seq:>25}: {cnt:>4}")

print(f"\n  --- 2-Event Sequenzen vor DISP_DN ---")
disp_dn_idx = np.where(states == 'DISP_DN')[0]
seq_counts_dn = {}
for di in disp_dn_idx:
    if di < 3:
        continue
    seq = f"{states[di-2]}→{states[di-1]}"
    seq_counts_dn[seq] = seq_counts_dn.get(seq, 0) + 1

top_seqs_dn = sorted(seq_counts_dn.items(), key=lambda x: -x[1])[:15]
for seq, cnt in top_seqs_dn:
    print(f"    {seq:>25}: {cnt:>4}")

# 3-Event Sequenzen
print(f"\n  --- 3-Event Sequenzen vor DISP_UP ---")
seq3_counts = {}
for di in disp_up_idx:
    if di < 4:
        continue
    seq = f"{states[di-3]}→{states[di-2]}→{states[di-1]}"
    seq3_counts[seq] = seq3_counts.get(seq, 0) + 1

top_seqs3 = sorted(seq3_counts.items(), key=lambda x: -x[1])[:10]
for seq, cnt in top_seqs3:
    pct = cnt / max(len(disp_up_idx), 1) * 100
    print(f"    {seq:>35}: {cnt:>4} ({pct:.1f}%)")

# ─── 5. MULTI-TF LEAD-LAG ────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"EXP 5: MULTI-TIMEFRAME LEAD-LAG")
print(f"{'='*70}")

# Resample zu 5min, 15min, 1h
# Da wir keine Zeitstempel haben: relativer Index
resample_5min = n // 5
resample_15min = n // 15
resample_1h = n // 60

def resample_ohlc(data, period):
    """Resample zu höherem TF"""
    n_full = (len(data) // period) * period
    reshaped = data[:n_full].reshape(-1, period)
    if len(data.shape) == 1:  # Close
        return reshaped[:, -1]
    return reshaped

# 1min Returns vs 5min Returns — Lead-Lag Cross-Correlation
returns_1m = np.diff(np.log(close[:50000]))  # Erste 50k Bars
returns_5m = []
for i in range(0, 50000 - 5, 5):
    r = np.log(close[i+5]) - np.log(close[i])
    returns_5m.append(r)
returns_5m = np.array(returns_5m)

# Cross-Correlation
min_len = min(len(returns_1m), len(returns_5m) * 5)
# Resample 1min Korrelationen
lag_corrs = {}
for lag in range(-10, 11):
    if lag < 0:
        # 1min laggt (führt — hoher TF reagiert später)
        r1 = returns_1m[:min_len + lag]
        r5 = returns_5m[:min_len//5]
        # Downsample 1min
        r1_down = r1[::5][:len(r5)]
        if len(r1_down) > len(r5):
            r1_down = r1_down[:len(r5)]
        if len(r5) > len(r1_down):
            r5 = r5[:len(r1_down)]
    else:
        r1 = returns_1m[lag:min_len+lag]
        r5 = returns_5m[:min_len//5]
        r1_down = r1[::5][:len(r5)]
        if len(r1_down) > len(r5):
            r1_down = r1_down[:len(r5)]
        if len(r5) > len(r1_down):
            r5 = r5[:len(r1_down)]
    
    if len(r1_down) > 10 and len(r5) > 10:
        corr = np.corrcoef(r1_down, r5)[0, 1]
        lag_corrs[lag] = float(corr)

print(f"  Cross-Correlation 1min → 5min:")
best_lag = max(lag_corrs, key=lag_corrs.get)
best_corr = lag_corrs[best_lag]
for lag in sorted(lag_corrs.keys()):
    marker = " <--" if lag == best_lag else ""
    print(f"    Lag {lag:+3d}: {lag_corrs[lag]:+.4f}{marker}")
print(f"  Best: Lag {best_lag:+d} mit r={best_corr:.4f}")

# ─── 6. NÄCHSTE GROSSE BEWEGUNG ──────────────────────────────────────────

print(f"\n{'='*70}")
print(f"EXP 6: VORHERSAGE NÄCHSTE GROSSE BEWEGUNG (letzte 2000 Bars)")
print(f"{'='*70}")

# Simuliere Live-Modus: trainiere auf ersten 715.665 Bars, teste auf letzten 2.000
test_start = n - 2000
train_close = close[:test_start]
test_close = close[test_start:]

# Aktuelle Zustände für Testbereich
current_state = states[test_start:]
current_close = close[test_start:]
current_high = high[test_start:]
current_low = low[test_start:]

# Signal für nächste große Bewegung (> 2× ATR in 15 Bars)
print(f"  Letzte 2.000 Bars: suche Muster für Displacement (>2×ATR in 15min)")
pred_wins = 0
pred_total = 0
pred_false = 0

for i in range(100, len(current_state)):
    # Wenn wir in den letzten 5 Bars Compression hatten, gefolgt von Sweep
    has_compression = 'COMPRESS' in current_state[i-10:i]
    has_sweep = 'SWEEP_UP' in current_state[i-10:i] or 'SWEEP_DN' in current_state[i-10:i]
    
    if has_compression and has_sweep:
        pred_total += 1
        # Prüfe nächste 15 Bars auf Bewegung > 2× median_atr
        end = min(i + 15, len(current_state))
        movement = max(current_high[i:end]) - min(current_low[i:end])
        if movement > median_atr * 2:
            pred_wins += 1
        else:
            pred_false += 1

if pred_total > 0:
    print(f"  Compression+Sweep → nächste 15min Displacement:")
    print(f"    Signale: {pred_total}")
    print(f"    Treffer: {pred_wins} ({pred_wins/pred_total*100:.1f}%)")
    print(f"    Fehlalarm: {pred_false} ({pred_false/pred_total*100:.1f}%)")

# ─── ZUSAMMENFASSUNG ─────────────────────────────────────────────────────

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"ERGEBNISSE GESAMT")
print(f"{'='*70}")
print(f"  Laufzeit: {elapsed:.1f}s")
print(f"  Daten: {n:,} Bars 1min NQ")
print(f"")

# JSON Summary speichern
summary = {
    'timestamp': datetime.now().isoformat(),
    'n_bars': n,
    'pre_disp_1min': {
        'total_signals': total_signals,
        'bull': int(np.sum(signal_bull)),
        'bear': int(np.sum(signal_bear)),
    },
    'hurst': {
        'mean': float(np.mean(hurst_values)) if hurst_values else None,
        'std': float(np.std(hurst_values)) if hurst_values else None,
        'pct_trending': h_gt_05 if hurst_values else None,
        'pct_mean_rev': h_lt_05 if hurst_values else None,
    },
    'entropy': {
        'mean': float(np.mean(seg_entropies)),
        'std': float(np.std(seg_entropies)),
        'low_entropy_return_factor': float(np.mean(low_entropy_returns) / max(np.mean(seg_returns), 0.01)) if low_entropy_mask.any() else None,
    },
    'cross_corr_1m_5m': {
        'best_lag': best_lag,
        'best_corr': best_corr,
    },
    'live_test': {
        'signals': pred_total,
        'wins': pred_wins,
        'wr': pred_wins / max(pred_total, 1) * 100,
    }
}

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/1min_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  Ergebnis JSON: {OUTPUT_DIR}/1min_summary.json")
print(f"  Fertig: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
