"""Latency measurement utilities — inspired by FoxML_Trader_v2 rdtsc pattern.

Measure what matters. No guessing.

Usage:
    from sb.latency import Timer
    with Timer("walk-forward") as t:
        result = run_backtest(args)
    # prints: [LAT] walk-forward: 3.247s (1247 trades, 384 trades/s)
"""

import time
from contextlib import contextmanager


@contextmanager
def Timer(label: str = "operation", num_items: int = 0):
    """Context manager that measures and logs elapsed time."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    
    parts = [f"[LAT] {label}: {elapsed:.3f}s"]
    if num_items > 0:
        rate = num_items / elapsed if elapsed > 0 else 0
        parts.append(f"({num_items} items, {rate:.0f}/s)")
    
    print(" ".join(parts))
