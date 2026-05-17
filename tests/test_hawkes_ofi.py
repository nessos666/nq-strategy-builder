"""
test_hawkes_ofi.py
==================
Tests fuer:
  - science_hawkes_decay_v2.py  (Bacry/Muzy Hawkes-Prozess-Approximation)
  - science_ofi_confluence_v2.py (Cont/Kukanov/Stoikov OFI Confluence)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
import os

_TRADINGPROJEKT = Path(
    os.environ.get("TRADINGPROJEKT_PATH", str(Path.home() / "TRADINGPROJEKT"))
)
_ALGO_BIB = _TRADINGPROJEKT / "nq_backtest/algo_bibliothek/v2/science"
_HAWKES_PATH = _ALGO_BIB / "science_hawkes_decay_v2.py"
_OFI_PATH = _ALGO_BIB / "science_ofi_confluence_v2.py"


# ---------------------------------------------------------------------------
# Loader-Helpers
# ---------------------------------------------------------------------------
def _load_module(path: Path):
    if not path.exists():
        pytest.skip(f"Externes Modul fehlt: {path}")
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Kann Modul nicht laden: {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def hawkes_mod():
    return _load_module(_HAWKES_PATH)


@pytest.fixture(scope="module")
def ofi_mod():
    return _load_module(_OFI_PATH)


# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer Testdaten
# ---------------------------------------------------------------------------
def _make_bars(
    n: int = 30,
    direction: str = "bull",  # "bull" | "bear" | "mixed"
    freq: str = "1min",
) -> pd.DataFrame:
    """Erstellt kontrollierte OHLCV-Bars mit bekanntem Expected-Output.

    Bullische Bars: close > open, deutlicher Körper (body_ratio ~ 0.7)
    Bärische Bars:  close < open, deutlicher Körper
    Gemischte Bars: abwechselnd bull/bear
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq=freq, tz="UTC")
    base = 18000.0 + np.arange(n) * 0.1  # leichter Trend-Drift

    if direction == "bull":
        open_ = base
        close_ = base + 4.0  # Körper 4 Punkte bullisch
        high_ = close_ + 1.0
        low_ = open_ - 1.0
    elif direction == "bear":
        open_ = base + 4.0
        close_ = base  # Körper 4 Punkte bärisch
        high_ = open_ + 1.0
        low_ = close_ - 1.0
    else:  # mixed
        alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        open_ = base + np.where(alternating > 0, 0.0, 4.0)
        close_ = base + np.where(alternating > 0, 4.0, 0.0)
        high_ = np.maximum(open_, close_) + 1.0
        low_ = np.minimum(open_, close_) - 1.0

    volume = rng.uniform(500.0, 1500.0, n)
    return pd.DataFrame(
        {"open": open_, "high": high_, "low": low_, "close": close_, "volume": volume},
        index=idx,
    )


def _make_doji_bars(n: int = 20) -> pd.DataFrame:
    """Bars mit sehr kleinem Körper (body_ratio < 0.25 = Doji)."""
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="1min", tz="UTC")
    base = 18000.0
    open_ = np.full(n, base)
    close_ = base + 0.1  # Körper nur 0.1 Punkte
    high_ = base + 5.0  # Range 10 Punkte → body_ratio = 0.1/10 = 0.01
    low_ = base - 5.0
    volume = np.full(n, 1000.0)
    return pd.DataFrame(
        {"open": open_, "high": high_, "low": low_, "close": close_, "volume": volume},
        index=idx,
    )


# ===========================================================================
# HAWKES DECAY TESTS
# ===========================================================================

HAWKES_COLS = [
    "hawkes_bull_excitation",
    "hawkes_bear_excitation",
    "hawkes_bull_active",
    "hawkes_bear_active",
    "hawkes_decay_bull",
    "hawkes_decay_bear",
    "hawkes_bull_passes",
    "hawkes_bear_passes",
]


def test_hawkes_output_columns(hawkes_mod):
    """Alle 8 erwarteten Spalten sind im Output vorhanden."""
    df = _make_bars(30)
    out = hawkes_mod.compute_hawkes_decay(df)
    for col in HAWKES_COLS:
        assert col in out.columns, f"Spalte fehlt: {col}"


def test_hawkes_index_preserved(hawkes_mod):
    """Output-Index ist identisch mit Input-Index (kein Look-ahead-Shift)."""
    df = _make_bars(30)
    out = hawkes_mod.compute_hawkes_decay(df)
    assert len(out) == len(df)
    assert (out.index == df.index).all()


def test_hawkes_bull_active_on_trend(hawkes_mod):
    """Nach N bullischen Bars muss hawkes_bull_active=True sein."""
    df = _make_bars(n=30, direction="bull")
    cfg = hawkes_mod.HawkesDecayConfig(halflife_bars=5, min_excitation=0.10)
    out = hawkes_mod.compute_hawkes_decay(df, config=cfg)
    # Ab Bar 10 (>2x halflife) muss der Cluster aktiv sein
    assert out["hawkes_bull_active"].iloc[15:].all(), (
        "Bull-Cluster sollte nach 15 bullischen Bars aktiv sein"
    )


def test_hawkes_bear_active_on_bear_trend(hawkes_mod):
    """Nach N baerischen Bars muss hawkes_bear_active=True sein."""
    df = _make_bars(n=30, direction="bear")
    cfg = hawkes_mod.HawkesDecayConfig(halflife_bars=5, min_excitation=0.10)
    out = hawkes_mod.compute_hawkes_decay(df, config=cfg)
    assert out["hawkes_bear_active"].iloc[15:].all(), (
        "Bear-Cluster sollte nach 15 baerischen Bars aktiv sein"
    )


def test_hawkes_decay_on_counter_bars(hawkes_mod):
    """3 aufeinanderfolgende Gegenkörper loesen hawkes_decay_bull=True aus."""
    # 20 Bull-Bars, dann 3 Bear-Bars am Ende
    bull_df = _make_bars(n=20, direction="bull")
    bear_df = _make_bars(n=3, direction="bear")
    # Neuen Index fuer die Bear-Bars erstellen
    bear_df.index = pd.date_range(
        bull_df.index[-1] + pd.Timedelta("1min"),
        periods=3,
        freq="1min",
        tz="UTC",
    )
    df = pd.concat([bull_df, bear_df])

    cfg = hawkes_mod.HawkesDecayConfig(
        halflife_bars=5, min_excitation=0.10, max_counter_bars=3
    )
    out = hawkes_mod.compute_hawkes_decay(df, config=cfg)
    # Am letzten Bar (nach 3 Bear-Bars) muss Decay aktiv sein
    assert out["hawkes_decay_bull"].iloc[-1], (
        "hawkes_decay_bull sollte nach 3 Bear-Bars True sein"
    )


def test_hawkes_passes_is_active_and_not_decay(hawkes_mod):
    """hawkes_bull_passes = hawkes_bull_active AND NOT hawkes_decay_bull."""
    df = _make_bars(n=50, direction="bull")
    cfg = hawkes_mod.HawkesDecayConfig(halflife_bars=5, min_excitation=0.10)
    out = hawkes_mod.compute_hawkes_decay(df, config=cfg)
    expected = out["hawkes_bull_active"] & ~out["hawkes_decay_bull"]
    pd.testing.assert_series_equal(
        out["hawkes_bull_passes"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_hawkes_doji_ignored(hawkes_mod):
    """Doji-Bars (body_ratio < body_min_ratio) werden ignoriert."""
    df = _make_doji_bars(n=30)
    cfg = hawkes_mod.HawkesDecayConfig(
        halflife_bars=5, min_excitation=0.15, body_min_ratio=0.25
    )
    out = hawkes_mod.compute_hawkes_decay(df, config=cfg)
    # Excitation bleibt nahe 0 weil Dojis keine Events ausloesen
    assert out["hawkes_bull_excitation"].max() < 0.10, (
        "Doji-Bars sollten kaum Excitation produzieren"
    )
    assert not out["hawkes_bull_passes"].any(), (
        "Doji-Bars sollten hawkes_bull_passes=False halten"
    )


def test_hawkes_excitation_is_float(hawkes_mod):
    """Excitation-Spalten sind float (nicht bool oder int)."""
    df = _make_bars(30)
    out = hawkes_mod.compute_hawkes_decay(df)
    assert out["hawkes_bull_excitation"].dtype == "float64"
    assert out["hawkes_bear_excitation"].dtype == "float64"


def test_hawkes_bool_columns(hawkes_mod):
    """Active/Decay/Passes Spalten sind bool."""
    df = _make_bars(30)
    out = hawkes_mod.compute_hawkes_decay(df)
    for col in [
        "hawkes_bull_active",
        "hawkes_bear_active",
        "hawkes_decay_bull",
        "hawkes_decay_bear",
        "hawkes_bull_passes",
        "hawkes_bear_passes",
    ]:
        assert out[col].dtype == bool, f"{col} sollte dtype=bool haben"


# ===========================================================================
# OFI CONFLUENCE TESTS
# ===========================================================================

OFI_COLS = [
    "ofi_raw",
    "ofi_fast",
    "ofi_slow",
    "ofi_bull_passes",
    "ofi_bear_passes",
    "ofi_signal",
]


def test_ofi_output_columns(ofi_mod):
    """Alle 6 erwarteten Spalten sind im Output vorhanden."""
    df = _make_bars(30)
    out = ofi_mod.compute_ofi_confluence(df)
    for col in OFI_COLS:
        assert col in out.columns, f"Spalte fehlt: {col}"


def test_ofi_index_preserved(ofi_mod):
    """Output-Index ist identisch mit Input-Index."""
    df = _make_bars(30)
    out = ofi_mod.compute_ofi_confluence(df)
    assert len(out) == len(df)
    assert (out.index == df.index).all()


def test_ofi_bull_passes_on_bull_bars(ofi_mod):
    """Konsistente Bull-Bars -> ofi_bull_passes=True ab Bar slow_window."""
    df = _make_bars(n=50, direction="bull")
    cfg = ofi_mod.OFIConfluenceConfig(
        fast_window=3, slow_multiplier=5, min_ofi_abs=0.0, volume_normalize=False
    )
    out = ofi_mod.compute_ofi_confluence(df, config=cfg)
    slow_w = cfg.fast_window * cfg.slow_multiplier  # 15
    # Ab slow_window haben wir genug Bars fuer slow OFI
    valid = out.iloc[slow_w:]
    assert valid["ofi_bull_passes"].all(), (
        "Alle Bull-Bars sollten ofi_bull_passes=True haben (ab slow_window)"
    )


def test_ofi_bear_passes_on_bear_bars(ofi_mod):
    """Konsistente Bear-Bars -> ofi_bear_passes=True ab Bar slow_window."""
    df = _make_bars(n=50, direction="bear")
    cfg = ofi_mod.OFIConfluenceConfig(
        fast_window=3, slow_multiplier=5, min_ofi_abs=0.0, volume_normalize=False
    )
    out = ofi_mod.compute_ofi_confluence(df, config=cfg)
    slow_w = cfg.fast_window * cfg.slow_multiplier
    valid = out.iloc[slow_w:]
    assert valid["ofi_bear_passes"].all(), (
        "Alle Bear-Bars sollten ofi_bear_passes=True haben (ab slow_window)"
    )


def test_ofi_no_confluence_on_mixed_bars(ofi_mod):
    """Gemischte Bars -> kein eindeutiges Confluence-Signal."""
    df = _make_bars(n=60, direction="mixed")
    cfg = ofi_mod.OFIConfluenceConfig(
        fast_window=3, slow_multiplier=5, min_ofi_abs=0.0, volume_normalize=False
    )
    out = ofi_mod.compute_ofi_confluence(df, config=cfg)
    slow_w = cfg.fast_window * cfg.slow_multiplier
    valid = out.iloc[slow_w:]
    # Bei perfekt gemischten Bars sollte kein einheitliches Signal geben
    assert not valid["ofi_bull_passes"].all(), (
        "Gemischte Bars sollten kein durchgaengiges bull_passes liefern"
    )
    assert not valid["ofi_bear_passes"].all(), (
        "Gemischte Bars sollten kein durchgaengiges bear_passes liefern"
    )


def test_ofi_signal_consistency(ofi_mod):
    """ofi_signal ist +1 genau dann wenn ofi_bull_passes, -1 wenn ofi_bear_passes."""
    df = _make_bars(n=50, direction="bull")
    out = ofi_mod.compute_ofi_confluence(df)
    bull = out["ofi_bull_passes"]
    bear = out["ofi_bear_passes"]
    sig = out["ofi_signal"]
    assert (sig[bull] == 1).all(), "ofi_signal sollte +1 sein wenn bull_passes"
    assert (sig[bear] == -1).all(), "ofi_signal sollte -1 sein wenn bear_passes"
    assert (sig[~bull & ~bear] == 0).all(), (
        "ofi_signal sollte 0 sein wenn kein Confluence"
    )


def test_ofi_raw_positive_for_bull(ofi_mod):
    """ofi_raw ist positiv fuer bullische Bars."""
    df = _make_bars(n=20, direction="bull")
    cfg = ofi_mod.OFIConfluenceConfig(volume_normalize=False)
    out = ofi_mod.compute_ofi_confluence(df, config=cfg)
    assert (out["ofi_raw"] > 0).all(), "Bull-Bars sollten positiven ofi_raw haben"


def test_ofi_raw_negative_for_bear(ofi_mod):
    """ofi_raw ist negativ fuer baerische Bars."""
    df = _make_bars(n=20, direction="bear")
    cfg = ofi_mod.OFIConfluenceConfig(volume_normalize=False)
    out = ofi_mod.compute_ofi_confluence(df, config=cfg)
    assert (out["ofi_raw"] < 0).all(), "Bear-Bars sollten negativen ofi_raw haben"


def test_ofi_nan_before_slow_window(ofi_mod):
    """ofi_slow ist NaN fuer Bars < slow_window (kein Look-ahead durch fillna)."""
    df = _make_bars(n=30, direction="bull")
    cfg = ofi_mod.OFIConfluenceConfig(
        fast_window=3, slow_multiplier=5, volume_normalize=False
    )
    out = ofi_mod.compute_ofi_confluence(df, config=cfg)
    slow_w = cfg.fast_window * cfg.slow_multiplier  # 15
    # Erste slow_w-1 Bars sollten NaN in ofi_slow haben
    assert out["ofi_slow"].iloc[: slow_w - 1].isna().all(), (
        f"ofi_slow sollte fuer Bars < {slow_w} NaN sein"
    )


def test_ofi_volume_normalize_changes_values(ofi_mod):
    """Volume-Normierung veraendert die OFI-Werte (nicht identisch)."""
    df = _make_bars(n=30, direction="bull")
    cfg_raw = ofi_mod.OFIConfluenceConfig(volume_normalize=False)
    cfg_norm = ofi_mod.OFIConfluenceConfig(volume_normalize=True)
    out_raw = ofi_mod.compute_ofi_confluence(df, config=cfg_raw)
    out_norm = ofi_mod.compute_ofi_confluence(df, config=cfg_norm)
    # ofi_fast Werte sollten sich unterscheiden wenn Volumen variiert
    # (bei konstantem Volume waeren sie gleich – hier ist Volume random)
    assert not (out_raw["ofi_fast"].dropna() == out_norm["ofi_fast"].dropna()).all(), (
        "Volume-Normierung sollte die OFI-Fast-Werte veraendern"
    )
