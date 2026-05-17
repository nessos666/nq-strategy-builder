#!/usr/bin/env python3
"""
EVENT-KETTEN MINING — Verbesserte Zustandsdiskretisierung

Problem Runde 1: 81.8% NORMAL-Zustand → keine sinnvollen Ketten
Lösung: Sliding Window Events + Lift-Analyse

Neue Methode:
  1. Statt Single-Bar Zustand: 5-Minuten Sliding Window Event-Flags
  2. Events mit Substanz: Compression (mind. 3 von 5 Bars), Sweep (mind. 2 von 5), ...
  3. Lift = P(Seq|Disp) / P(Seq|Baseline) — wieviel häufiger vor Displacement?
  4. Nur Ketten mit Lift > 2.0 und n>20 sind signifikant
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
print("EVENT-KETTEN MINING v2 — Sliding Window + Lift")
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

# ─── GLOBALE METRIKEN ───────────────────────────────────────────

def calc_atr(h, l, c, period=14):
    n = len(h)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()

atr = calc_atr(high, low, close, 14)
atr_smooth = pd.Series(atr).rolling(50).mean().to_numpy()
vol_median = float(np.median(volume))
vol_smooth = pd.Series(volume).rolling(50).mean().to_numpy()

print(f"\nDaten: {n:,} Bars 1min NQ")
print(f"ATR Median: {np.median(atr[atr>0]):.2f}, Vol Median: {vol_median:.0f}")

# ─── 1. SLIDING WINDOW EVENT-DETEKTOR ──────────────────────────

print(f"\n{'='*70}")
print("EXP 1: Sliding Window Event-Detektor (5-Minuten-Fenster)")
print("=" * 70)

# Window-Größe: 5 Minuten (5 Bars für 1min, 1 Bar für 5min)
W = 5  # Sliding Window in Bars (5 Minuten)
EVENT_THRESHOLD = 3  # Mindestens 3 von 5 Bars müssen den Zustand haben

# Roh-Zustände (pro Bar) — wie in Runde 2
raw_states = np.full(n, 'N', dtype='U8')
for i in range(30, n):
    # Compression
    range5 = float(np.mean(high[i-5:i] - low[i-5:i]))
    if range5 < atr_smooth[i] * 0.5:
        raw_states[i] = 'C'
        continue
    
    # Sweep Up
    max10 = float(np.max(high[i-10:i]))
    if high[i] > max10 + 1.5 and volume[i] > vol_median * 1.5:
        cr = high[i] - low[i]
        if cr > 0 and (high[i] - close[i]) / cr > 0.35:
            raw_states[i] = 'SU'
            continue
    
    # Sweep Down
    min10 = float(np.min(low[i-10:i]))
    if low[i] < min10 - 1.5 and volume[i] > vol_median * 1.5:
        cr = high[i] - low[i]
        if cr > 0 and (close[i] - low[i]) / cr > 0.35:
            raw_states[i] = 'SD'
            continue
    
    # Displacement
    if abs(close[i] - close[i-1]) > atr_smooth[i] * 2:
        raw_states[i] = 'DU' if close[i] > close[i-1] else 'DD'
        continue
    
    # Break
    if i > 15:
        max15 = np.max(high[i-15:i])
        min15 = np.min(low[i-15:i])
        if close[i] > max15 and volume[i] > vol_median:
            raw_states[i] = 'BU'
        elif close[i] < min15 and volume[i] > vol_median:
            raw_states[i] = 'BD'

print(f"  Roh-Zustände (pro Bar):")
unique, counts = np.unique(raw_states, return_counts=True)
for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
    print(f"    {u:>3}: {c:>7} ({c/n*100:.1f}%)")

# Sliding Window: zähle Events in den letzten W Bars
# Output: Event-Typ pro Window (z.B. 'C' wenn mind. EVENT_THRESHOLD Compression-Bars)
window_event = np.full(n - W, 'N', dtype='U8')
for i in range(W, n):
    window = raw_states[i-W:i]
    
    # Zähle jedes Event im Window
    c_count = np.sum(window == 'C')
    su_count = np.sum(window == 'SU')
    sd_count = np.sum(window == 'SD')
    du_count = np.sum(window == 'DU')
    dd_count = np.sum(window == 'DD')
    
    # Priorität: Displacement > Sweep > Compression > Break > Normal
    if du_count >= 2 or dd_count >= 2:
        window_event[i-W] = 'DU' if du_count >= dd_count else 'DD'
    elif su_count >= EVENT_THRESHOLD:
        window_event[i-W] = 'SU'
    elif sd_count >= EVENT_THRESHOLD:
        window_event[i-W] = 'SD'
    elif c_count >= EVENT_THRESHOLD:
        window_event[i-W] = 'C'
    elif du_count >= 1 or dd_count >= 1:
        window_event[i-W] = 'DU' if du_count >= dd_count else 'DD'
    elif su_count >= 1 or sd_count >= 1:
        window_event[i-W] = 'SU' if su_count >= sd_count else 'SD'
    # sonst: N (bleibt)

print(f"\n  Window-Events (W={W}, Threshold={EVENT_THRESHOLD}):")
unique_w, counts_w = np.unique(window_event, return_counts=True)
window_total = len(window_event)
for u, c in sorted(zip(unique_w, counts_w), key=lambda x: -x[1]):
    print(f"    {u:>3}: {c:>7} ({c/window_total*100:.1f}%)")

# ─── 2. LIFT-ANALYSE ──────────────────────────────────────────────

print(f"\n{'='*70}")
print("EXP 2: Lift-Analyse — welche Sequenzen sind signifikant?")
print("=" * 70)

# Displacement-Indices (aus Window-Events)
disp_idx = np.where((window_event == 'DU') | (window_event == 'DD'))[0]
n_disp = len(disp_idx)
print(f"  Displacement-Events: {n_disp} ({n_disp/window_total*100:.1f}% aller Windows)")

# Zufällige Stichprobe als Baseline (10.000 Punkte)
np.random.seed(42)
baseline_idx = np.random.choice(window_total, min(10000, window_total // 10), replace=False)

# ─── 2-EVENT SEQUENZEN ──────────────────────────────────────────

print(f"\n  --- 2-Event Sequenzen (Lift-Analyse) ---")

# Sequenzen vor Displacement: wir schauen 1-3 Windows VOR dem Event
seq_disp = defaultdict(int)
seq_baseline = defaultdict(int)

for lookback in [1, 2, 3]:
    print(f"\n  Lookback {lookback}:")
    
    # Vor Displacement
    for di in disp_idx:
        if di < lookback:
            continue
        seq = '→'.join(window_event[di-lookback:di])
        seq_disp[seq] += 1
    
    # Baseline (zufällige Punkte)
    for bi in baseline_idx:
        if bi < lookback:
            continue
        seq = '→'.join(window_event[bi-lookback:bi])
        seq_baseline[seq] += 1
    
    # Lift berechnen
    total_disp = len(disp_idx)
    total_base = len(baseline_idx)
    
    results = []
    for seq, count_d in seq_disp.items():
        if count_d < 10:  # Mindestanzahl für Signifikanz
            continue
        count_b = seq_baseline.get(seq, 0)
        p_disp = count_d / max(total_disp, 1)
        p_base = count_b / max(total_base, 1)
        lift = p_disp / max(p_base, 0.0001)
        delta = p_disp - p_base
        results.append((lift, delta, count_d, seq))
    
    results.sort(reverse=True)
    print(f"    {'Lift':>6}  {'Delta':>8}  {'n':>4}  {'Sequenz':>25}")
    print(f"    {'-'*50}")
    for lift, delta, cnt, seq in results[:20]:
        print(f"    {lift:>6.2f}  {delta:>+.4f}  {cnt:>4}  {seq:>25}")

# ─── 3-EVENT SEQUENZEN ──────────────────────────────────────────

print(f"\n\n  --- 3-Event Sequenzen (Lift-Analyse) ---")

for lookback in [2, 3]:
    print(f"\n  Lookback {lookback} (3 Events):")
    
    seq_disp_3 = defaultdict(int)
    seq_baseline_3 = defaultdict(int)
    
    for di in disp_idx:
        if di < lookback:
            continue
        # 3 Events: di-2, di-1, di
        seq = '→'.join(window_event[di-lookback:di+1])
        seq_disp_3[seq] += 1
    
    for bi in baseline_idx:
        if bi < lookback:
            continue
        seq = '→'.join(window_event[bi-lookback:bi+1])
        seq_baseline_3[seq] += 1
    
    results_3 = []
    for seq, count_d in seq_disp_3.items():
        if count_d < 10:
            continue
        count_b = seq_baseline_3.get(seq, 0)
        p_disp = count_d / max(n_disp, 1)
        p_base = count_b / max(len(baseline_idx), 1)
        lift = p_disp / max(p_base, 0.0001)
        delta = p_disp - p_base
        results_3.append((lift, delta, count_d, seq))
    
    results_3.sort(reverse=True)
    print(f"    {'Lift':>6}  {'Delta':>8}  {'n':>4}  {'Sequenz':>35}")
    print(f"    {'-'*60}")
    for lift, delta, cnt, seq in results_3[:20]:
        print(f"    {lift:>6.2f}  {delta:>+.4f}  {cnt:>4}  {seq:>35}")

# ─── 3. TRANSITION-MATRIX ─────────────────────────────────────────

print(f"\n{'='*70}")
print("EXP 3: Transition-Matrix (P(Zustand_t+1 | Zustand_t))")
print("=" * 70)

states_list = ['C', 'SU', 'SD', 'DU', 'DD', 'N']
n_states = len(states_list)
state_to_idx = {s: i for i, s in enumerate(states_list)}

# Count transitions
transitions = np.zeros((n_states, n_states), dtype=int)
for i in range(1, len(window_event)):
    s_prev = window_event[i-1]
    s_curr = window_event[i]
    if s_prev in state_to_idx and s_curr in state_to_idx:
        transitions[state_to_idx[s_prev], state_to_idx[s_curr]] += 1

# Transition Matrix (P)
trans_p = transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1)

print(f"\n  Transition-Matrix (von → zu):")
print(f"  {'':>5}", end='')
for s in states_list:
    print(f"  {s:>4}", end='')
print()
for i, s_from in enumerate(states_list):
    print(f"  {s_from:>4}", end='')
    for j, s_to in enumerate(states_list):
        if transitions[i].sum() > 0:
            print(f"  {trans_p[i,j]:.3f}", end='')
        else:
            print(f"  {' -':>4}", end='')
    print(f"  ({transitions[i].sum():>7})")

# ─── 5. WICHTIGSTE ERKENNTNISSE ───────────────────────────────────

print(f"\n{'='*70}")
print("ZUSAMMENFASSUNG: Signifikante Event-Ketten")
print("=" * 70)

# Filtere Ketten mit Lift > 2.0 und n > 20
print(f"\n  2-Event Ketten mit Lift > 2.0:")
for lookback in [1, 2]:
    seq_disp_tmp = defaultdict(int)
    seq_baseline_tmp = defaultdict(int)
    for di in disp_idx:
        if di < lookback: continue
        seq_disp_tmp['→'.join(window_event[di-lookback:di])] += 1
    for bi in baseline_idx:
        if bi < lookback: continue
        seq_baseline_tmp['→'.join(window_event[bi-lookback:bi])] += 1
    
    sig_seqs = []
    for seq, cd in seq_disp_tmp.items():
        if cd < 20: continue
        cb = seq_baseline_tmp.get(seq, 0)
        lift = (cd/n_disp) / max(cb/len(baseline_idx), 0.0001)
        if lift > 2.0:
            sig_seqs.append((lift, cd, seq))
    
    sig_seqs.sort(reverse=True)
    for lift, cnt, seq in sig_seqs[:15]:
        print(f"    Lift {lift:>5.1f}  n={cnt:>4}  {seq}")

print(f"\n  3-Event Ketten mit Lift > 2.0:")
seq_disp_3f = defaultdict(int)
seq_baseline_3f = defaultdict(int)
for di in disp_idx:
    if di < 2: continue
    seq_disp_3f['→'.join(window_event[di-2:di+1])] += 1
for bi in baseline_idx:
    if bi < 2: continue
    seq_baseline_3f['→'.join(window_event[bi-2:bi+1])] += 1

sig_seqs_3 = []
for seq, cd in seq_disp_3f.items():
    if cd < 15: continue
    cb = seq_baseline_3f.get(seq, 0)
    lift = (cd/n_disp) / max(cb/len(baseline_idx), 0.0001)
    if lift > 2.0:
        sig_seqs_3.append((lift, cd, seq))

sig_seqs_3.sort(reverse=True)
for lift, cnt, seq in sig_seqs_3[:15]:
    print(f"    Lift {lift:>5.1f}  n={cnt:>4}  {seq}")

# ─── SPEICHERN ─────────────────────────────────────────────────────

elapsed = time.time() - t0
print(f"\n  Dauer: {elapsed:.1f}s")

# JSON speichern
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {
    'timestamp': datetime.now().isoformat(),
    'n_bars': n,
    'window_size': W,
    'event_threshold': EVENT_THRESHOLD,
    'n_disp_events': n_disp,
    'state_distribution': {u: int(c) for u, c in zip(unique_w, counts_w)},
    'transition_matrix': {
        'states': states_list,
        'matrix': trans_p.tolist(),
        'counts': transitions.tolist()
    },
    'significant_2event_seqs': [
        {'lift': float(l), 'count': int(c), 'seq': s}
        for l, c, s in sig_seqs[:20]
    ],
    'significant_3event_seqs': [
        {'lift': float(l), 'count': int(c), 'seq': s}
        for l, c, s in sig_seqs_3[:20]
    ],
}

with open(f"{OUTPUT_DIR}/event_ketten_v2.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"  JSON: {OUTPUT_DIR}/event_ketten_v2.json")
print(f"\n{'='*70}")
print(f"FERTIG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
