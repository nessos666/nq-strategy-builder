#!/usr/bin/env python3
"""
Event-Chain Probability Engine — CLI-Tool

Analysiert die aktuelle Event-Kette aus 1min OHLCV Daten und gibt
Wahrscheinlichkeiten, Lift, Richtung und Qualität aus.

Usage:
    python3 chain_probability.py                    # Letzte 1000 Bars
    python3 chain_probability.py --bars 500          # Letzte 500 Bars
    python3 chain_probability.py --data pfad.csv     # Eigene CSV
    python3 chain_probability.py --watch             # Live-Monitor
    
KEINE AUTOMATISCHEN ORDERS — NUR INFORMATION
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime

import pandas as pd
import numpy as np

# Projekt-Pfad für Imports
_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "david_bibliothek", "09_Entry_Logik"
)
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from chain_engine.state_detector import (
    calc_atr, calc_atr_smooth, detect_raw_states, detect_window_states,
    get_current_chain, get_recent_states, compute_momentum
)
from chain_engine.lift_matcher import (
    load_lift_database, match_chain, score_quality, estimate_probabilities
)
from chain_engine.transition_predictor import (
    predict_next_state, top_k_paths
)

# ─── DEFAULT DATENQUELLE ─────────────────────────────────────────
_DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet")


def load_data(source: str, n_bars: int) -> pd.DataFrame:
    """Lade OHLCV Daten aus Parquet oder CSV."""
    if source.endswith('.parquet'):
        df = pd.read_parquet(source)
        if n_bars > 0 and n_bars < len(df):
            df = df.iloc[-n_bars:]
    elif source.endswith('.csv'):
        df = pd.read_csv(source, parse_dates=True, index_col=0)
        if n_bars > 0:
            df = df.iloc[-n_bars:]
    else:
        raise ValueError(f"Unbekanntes Format: {source}")
    
    # Spalten normalisieren (lowercase)
    df.columns = [c.lower() for c in df.columns]
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f"Fehlende Spalten: {required - set(df.columns)}")
    
    return df


def determine_session(df: pd.DataFrame) -> str:
    """Bestimme aktuelle Session basierend auf letztem Timestamp."""
    if not hasattr(df.index, 'hour') or len(df.index) == 0:
        return 'ny'
    h = df.index[-1].hour
    if 0 <= h < 7:
        return 'asia'
    elif 7 <= h < 13:
        return 'london'
    return 'ny'


def analyze(df: pd.DataFrame, db: dict, lookback: int = 3) -> dict:
    """Führe vollständige Analyse durch.
    
    Returns:
        dict mit allen Analyse-Ergebnissen
    """
    n = len(df)
    if n < 50:
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_bars': n,
            'last_price': 0,
            'chain': '',
            'recent_states': [],
            'momentum': 0.0,
            'session': 'ny',
            'match': {'lift_bull': 0, 'lift_bear': 0, 'direction': 'NEUTRAL',
                      'wr_2xatr': None, 'mfe_mae': None, 'n': None, 'quality': 'D', 'source': 'none'},
            'quality': 'D',
            'probabilities': {'p_1xatr': 0, 'p_1_5xatr': 0, 'p_2xatr': 0, 'p_3xatr': 0},
            'next_state': {'top_state': 'N', 'top_prob': 0.5, 'all_probs': []},
            'top_paths': [],
            'atr_median': 0,
            'error': f'Zu wenig Daten: {n} Bars (mind. 50 benötigt)',
        }
    
    # OHLCV Arrays
    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    close = df['close'].to_numpy(dtype=float)
    volume = df['volume'].to_numpy(dtype=float)
    
    # State Detection
    atr = calc_atr(high, low, close, 14)
    atr_smooth = calc_atr_smooth(atr)
    vol_median = float(np.median(volume))
    
    raw_states = detect_raw_states(high, low, close, volume, atr_smooth, vol_median)
    window_states = detect_window_states(raw_states)
    
    # Aktuelle Chain
    last_idx = len(window_states) - 1
    chain = get_current_chain(window_states, last_idx, lookback)
    recent = get_recent_states(window_states, 10)
    momentum = compute_momentum(window_states, 10)
    
    # Lift Match
    match = match_chain(chain, db)
    session = determine_session(df)
    quality = score_quality(match, session)
    probs = estimate_probabilities(match, db)
    
    # Transition Prediction
    if match['source'] != 'none':
        last_state = chain.split("→")[-1]
        next_state = predict_next_state(last_state, db, n_steps=1)
        top_paths = top_k_paths(last_state, db, k=3, depth=3)
    else:
        next_state = {'top_state': 'N', 'top_prob': 0.5, 'all_probs': []}
        top_paths = []
    
    # Letzter Preis
    last_price = close[-1] if len(close) > 0 else 0
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_bars': n,
        'last_price': round(last_price, 2),
        'chain': chain,
        'recent_states': [str(s) for s in recent],
        'momentum': round(momentum, 2),
        'session': session,
        'match': match,
        'quality': quality,
        'probabilities': {k: round(v, 3) for k, v in probs.items()},
        'next_state': next_state,
        'top_paths': top_paths,
        'atr_median': round(float(np.median(atr[atr > 0])), 2) if np.any(atr > 0) else 0,
    }


def print_report(result: dict):
    """Gib Analyse als Terminal-Report aus."""
    m = result['match']
    p = result['probabilities']
    
    # Richtungs-Pfeil
    if m['direction'] == 'LONG':
        arrow = '🔼'
        bias = 'bullish'
    elif m['direction'] == 'SHORT':
        arrow = '🔽'
        bias = 'bearish'
    else:
        arrow = '⬡'
        bias = 'neutral'
    
    # Qualitäts-Sterne
    quality_stars = {'A': '★★★★★', 'B': '★★★★', 'C': '★★★', 'D': '★★'}.get(result['quality'], '★★')
    
    # Source Label
    source_labels = {
        '3event': '✓ Validierte 3-Event Kette',
        '2event': '→ 2-Event Muster',
        'single': '● Single State',
        'none': '✗ Kein Muster erkannt',
    }
    source_label = source_labels.get(m['source'], '')
    
    print()
    print("════════════════════════════════════════════════════")
    print("     EVENT-CHAIN PROBABILITY ENGINE v1.0")
    print("════════════════════════════════════════════════════")
    print(f"  Zeit:       {result['timestamp']}")
    print(f"  Daten:      {result['n_bars']} Bars 1min NQ")
    print(f"  Letzter:    {result['last_price']}")
    print(f"  ATR Median: {result['atr_median']}")
    print(f"  Session:    {result['session'].upper()}")
    print()
    print(f"  AKTUELLE KETTE:  {result['chain']:>20}")
    print(f"  LETZTE 10:       {' '.join(result['recent_states'][:10])}")
    print(f"  MARKT-MOMENTUM:  {result['momentum']:>+.2f} ({bias})")
    print()
    
    if m['source'] != 'none':
        print("┌─────────────────────────────────────────────────┐")
        print(f"│ LIFT-ANALYSE           {arrow}                             │")
        print("├─────────────────────────────────────────────────┤")
        print(f"│ Kette:          {result['chain']:>30}  │")
        
        if m['lift_bear'] > 10:
            print(f"│ Lift Short:     {m['lift_bear']:>5.1f}×  {quality_stars}         │")
        if m['lift_bull'] > 10:
            print(f"│ Lift Long:      {m['lift_bull']:>5.1f}×  {quality_stars}         │")
        
        print(f"│ Erwartet:       {m['direction']:>30}  │")
        print(f"│ Qualität:       {result['quality']:>30}  │")
        print(f"│ Quelle:         {source_label:>30}  │")
        
        if m['n']:
            print(f"│ n:              {m['n']:>30}  │")
        
        print("│                                               │")
        print("│ WAHRSCHEINLICHKEITEN:                         │")
        print(f"│   P(1×ATR, 15min):    {p.get('p_1xatr', 0)*100:>5.1f}%{'':>17}│")
        print(f"│   P(1.5×ATR, 15min):  {p.get('p_1_5xatr', 0)*100:>5.1f}%{'':>17}│")
        print(f"│   P(2×ATR, 15min):    {p.get('p_2xatr', 0)*100:>5.1f}%{'':>17}│")
        print(f"│   P(3×ATR, 15min):    {p.get('p_3xatr', 0)*100:>5.1f}%{'':>17}│")
        print("└─────────────────────────────────────────────────┘")
        
        # Nächster Zustand
        ns = result['next_state']
        if ns and ns.get('all_probs'):
            print()
            print("  NÄCHSTER ZUSTAND (Markov):")
            for entry in ns['all_probs'][:6]:
                bar = '█' * int(entry['prob'] * 50)
                print(f"    {entry['state']:>3} → {entry['prob']*100:>4.1f}%  {bar}")
        
        # Top Pfade
        paths = result.get('top_paths', [])
        if paths:
            print()
            print("  TOP-PFADE (3 Schritte):")
            for tp in paths[:3]:
                print(f"    {tp['path']:>30}  ({tp['prob']*100:.1f}%)")
    else:
        print("  ✗ KEIN MUSTER ERKANNT")
        print("    Aktuelle States sind normal/ohne Edge-Signal.")
    
    print()
    print("════════════════════════════════════════════════════")
    print("  KEINE STRATEGIE — KEINE AUTOMATISCHEN ORDERS")
    print("  NUR PROBABILISTISCHE MARKT-ANALYSE")
    print("════════════════════════════════════════════════════")
    print()


def watch_mode(source: str, n_bars: int, interval: int = 60):
    """Watch-Mode: aktualisiert Analyse alle `interval` Sekunden."""
    db = load_lift_database()
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            df = load_data(source, n_bars)
            result = analyze(df, db)
            print_report(result)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Watch-Mode beendet.")


def main():
    parser = argparse.ArgumentParser(
        description="Event-Chain Probability Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python3 chain_probability.py
  python3 chain_probability.py --bars 500
  python3 chain_probability.py --watch
  python3 chain_probability.py --data /pfad/zu/daten.csv
        """
    )
    parser.add_argument('--data', type=str, default=_DEFAULT_DATA,
                        help=f'Datenquelle (Parquet/CSV, default: letzte 1000 Bars aus Parquet)')
    parser.add_argument('--bars', type=int, default=1000,
                        help='Anzahl Bars für Analyse (default: 1000)')
    parser.add_argument('--watch', action='store_true',
                        help='Watch-Mode: aktualisiert alle 60s')
    parser.add_argument('--json', action='store_true',
                        help='Output als JSON (für Programm-Parsing)')
    
    args = parser.parse_args()
    
    db = load_lift_database()
    
    if args.watch:
        watch_mode(args.data, args.bars)
        return
    
    df = load_data(args.data, args.bars)
    result = analyze(df, db)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
