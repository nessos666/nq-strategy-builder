from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


import os

REGIME_GATE_PATH = (
    Path(os.environ.get("TRADINGPROJEKT_PATH", str(Path.home() / "TRADINGPROJEKT")))
    / "nq_backtest/algo_bibliothek/v2/science/science_regime_gate_v2.py"
)


def _load_regime_gate():
    import sys

    if not REGIME_GATE_PATH.exists():
        pytest.skip(f"Externes Modul fehlt: {REGIME_GATE_PATH}")
    name = "science_regime_gate_v2"
    spec = importlib.util.spec_from_file_location(name, REGIME_GATE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so dataclass string annotations resolve
    spec.loader.exec_module(mod)
    return mod


def _make_bars(n: int = 800) -> pd.DataFrame:
    """Build OHLCV bars with alternating trend and choppy segments.

    Uses ``np.linspace`` trend segments so the Hurst exponent implementation
    (which measures std of lagged log-return differences) produces values
    clearly above 0.55 in trending windows and below it in noisy windows.
    This guarantees ``regime_passes`` is neither uniformly True nor False.
    """
    rng = np.random.default_rng(7)
    seg = n // 4
    trend_seg = np.linspace(0, 1500, seg)
    noisy_seg = rng.normal(0, 50, seg).cumsum()
    closes = np.concatenate(
        [
            18000 + noisy_seg,
            18000 + noisy_seg[-1] + trend_seg,
            18000 + noisy_seg[-1] + trend_seg[-1] + rng.normal(0, 50, seg).cumsum(),
            18000
            + noisy_seg[-1]
            + trend_seg[-1]
            + rng.normal(0, 1, n - 3 * seg).cumsum()
            + np.linspace(0, 300, n - 3 * seg),
        ]
    )
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(n, 200.0),
        },
        index=idx,
    )


def test_compute_regime_gate_df_returns_regime_passes_column():
    """compute_regime_gate_df() must return a 'regime_passes' bool column."""
    mod = _load_regime_gate()
    bars = _make_bars()  # default n=800
    result = mod.compute_regime_gate_df(bars)
    assert "regime_passes" in result.columns, f"Columns: {list(result.columns)}"
    assert (
        result["regime_passes"].dtype == bool
        or result["regime_passes"].isin([True, False]).all()
    )


def test_compute_regime_gate_df_not_all_same():
    """regime_passes must not be uniformly True or False (filter would be useless)."""
    mod = _load_regime_gate()
    bars = _make_bars()  # default n=800
    result = mod.compute_regime_gate_df(bars)
    passes = result["regime_passes"]
    assert passes.any(), "regime_passes is always False – no signal passes"
    assert not passes.all(), "regime_passes is always True – filter has no effect"
