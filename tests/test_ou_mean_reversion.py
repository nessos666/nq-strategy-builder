"""
test_ou_mean_reversion.py
=========================
Tests fuer science_ou_mean_reversion_v2.py
(Ornstein-Uhlenbeck Mean-Reversion Detektor, Wergieluk / Chan 1992)
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
_OU_PATH = _ALGO_BIB / "science_ou_mean_reversion_v2.py"

_QUANT_TOOLS_PATH = Path(
    os.environ.get("QUANT_TOOLS_PATH", str(Path.home() / "quant_tools"))
)


# ---------------------------------------------------------------------------
# quant_tools verfuegbar?
# ---------------------------------------------------------------------------
_quant_tools_available = False
try:
    if str(_QUANT_TOOLS_PATH) not in sys.path:
        sys.path.insert(0, str(_QUANT_TOOLS_PATH))
    import quant_tools.ou as _qt_ou_check  # noqa: F401

    _quant_tools_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _quant_tools_available,
    reason="quant_tools nicht installiert – OU-Tests uebersprungen",
)

# ---------------------------------------------------------------------------
# Loader-Helper
# ---------------------------------------------------------------------------


def _load_module(path: Path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Kann Modul nicht laden: {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def ou_mod():
    return _load_module(_OU_PATH)


# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer Testdaten
# ---------------------------------------------------------------------------


def _simulate_ou(
    n: int = 300,
    theta: float = 0.3,
    mu: float = 100.0,
    sigma: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Simuliert Ornstein-Uhlenbeck Prozess: dX = theta*(mu-X)*dt + sigma*dW."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.standard_normal()
    return x


def _random_walk(n: int = 300, seed: int = 42) -> np.ndarray:
    """Zufaelliger Preispfad (non-stationary, kein Mean-Reversion)."""
    rng = np.random.default_rng(seed)
    return 18000.0 + np.cumsum(rng.standard_normal(n) * 2.0)


def _make_ohlcv(close: np.ndarray, seed: int = 7) -> pd.DataFrame:
    """Erstellt minimalen OHLCV DataFrame aus Close-Array."""
    rng = np.random.default_rng(seed)
    n = len(close)
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="1min", tz="UTC")
    noise = rng.uniform(0.05, 0.3, n)
    high = close + noise
    low = close - noise
    open_ = close + rng.uniform(-0.15, 0.15, n)
    volume = rng.uniform(500.0, 1500.0, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


OU_COLS = [
    "ou_half_life",
    "ou_mean_level",
    "ou_mean_rev_speed",
    "ou_is_stationary",
    "ou_bull_passes",
    "ou_bear_passes",
]


# ===========================================================================
# Tests – Basisstruktur
# ===========================================================================


def test_ou_output_columns(ou_mod):
    """Alle 6 erwarteten Spalten sind im Output vorhanden."""
    df = _make_ohlcv(_simulate_ou(n=200))
    out = ou_mod.compute_ou_mean_reversion(df)
    for col in OU_COLS:
        assert col in out.columns, f"Spalte fehlt: {col}"


def test_ou_no_lookahead_bias(ou_mod):
    """Zukuenftige Bars beeinflussen vergangene Outputs nicht (Look-ahead-Bias-Test).

    Berechne OU auf 150 Bars. Haenge dann 50 voellig andere Bars an.
    Die Outputs fuer Bars 0..149 muessen in beiden Laeufen identisch sein.
    """
    close_base = _simulate_ou(n=200, theta=0.4, mu=100.0, sigma=0.4, seed=10)

    df_150 = _make_ohlcv(close_base[:150])

    # "Zukunft" manipulieren: komplett andere Werte ab Bar 150
    close_manipulated = close_base.copy()
    close_manipulated[150:] = _simulate_ou(
        n=50, theta=0.1, mu=200.0, sigma=5.0, seed=99
    )
    df_200 = _make_ohlcv(close_manipulated)
    # Gleichen Index sicherstellen (df_150 hat Index 0..149, df_200 hat 0..199)
    df_200_aligned = df_200.copy()
    df_200_aligned.index = pd.date_range(
        "2025-01-01 09:30", periods=200, freq="1min", tz="UTC"
    )
    df_150_aligned = df_150.copy()
    df_150_aligned.index = pd.date_range(
        "2025-01-01 09:30", periods=150, freq="1min", tz="UTC"
    )

    cfg = ou_mod.OUMeanReversionConfig(window=60)
    out_150 = ou_mod.compute_ou_mean_reversion(df_150_aligned, config=cfg)
    out_200 = ou_mod.compute_ou_mean_reversion(df_200_aligned, config=cfg)

    for col in ["ou_half_life", "ou_mean_level", "ou_mean_rev_speed"]:
        vals_150 = out_150[col].values
        vals_200 = out_200[col].iloc[:150].values
        # NaN an gleicher Stelle erwartet
        nan_mask = np.isnan(vals_150)
        assert np.array_equal(nan_mask, np.isnan(vals_200)), (
            f"{col}: NaN-Muster unterschiedlich – Look-ahead-Bias?"
        )
        # Nicht-NaN-Werte muessen exakt gleich sein
        np.testing.assert_array_almost_equal(
            vals_150[~nan_mask],
            vals_200[~nan_mask],
            decimal=10,
            err_msg=f"{col}: Zukunftsdaten beeinflussen Vergangenheit (Look-ahead-Bias!)",
        )

    # Boolesche Spalten ebenfalls pruefen
    for col in ["ou_is_stationary", "ou_bull_passes", "ou_bear_passes"]:
        vals_150 = out_150[col].values
        vals_200 = out_200[col].iloc[:150].values
        assert np.array_equal(vals_150, vals_200), (
            f"{col}: Zukunftsdaten beeinflussen boolesche Spalte (Look-ahead-Bias!)"
        )


def test_ou_nan_before_window(ou_mod):
    """ou_half_life ist NaN fuer die ersten window-1 Bars."""
    n = 150
    window = 60
    df = _make_ohlcv(_simulate_ou(n=n))
    cfg = ou_mod.OUMeanReversionConfig(window=window)
    out = ou_mod.compute_ou_mean_reversion(df, config=cfg)
    nan_count = out["ou_half_life"].iloc[: window - 1].isna().sum()
    assert nan_count == window - 1, f"Erwartet {window - 1} NaN-Werte, got {nan_count}"


def test_ou_float_columns(ou_mod):
    """ou_half_life, ou_mean_level, ou_mean_rev_speed sind float64."""
    df = _make_ohlcv(_simulate_ou(n=200))
    out = ou_mod.compute_ou_mean_reversion(df)
    for col in ["ou_half_life", "ou_mean_level", "ou_mean_rev_speed"]:
        assert out[col].dtype == "float64", f"{col} sollte float64 sein"


def test_ou_bool_columns(ou_mod):
    """ou_is_stationary, ou_bull_passes, ou_bear_passes sind bool."""
    df = _make_ohlcv(_simulate_ou(n=200))
    out = ou_mod.compute_ou_mean_reversion(df)
    for col in ["ou_is_stationary", "ou_bull_passes", "ou_bear_passes"]:
        assert out[col].dtype == bool, f"{col} sollte dtype=bool haben"


# ===========================================================================
# Tests – Signal-Logik
# ===========================================================================


def test_ou_stationary_series(ou_mod):
    """Stark mean-reverting OU-Serie → ou_is_stationary=True ab window."""
    close = _simulate_ou(n=300, theta=0.5, mu=100.0, sigma=0.3)
    df = _make_ohlcv(close)
    cfg = ou_mod.OUMeanReversionConfig(window=60, adf_alpha=0.05)
    out = ou_mod.compute_ou_mean_reversion(df, config=cfg)
    stationary_rate = out["ou_is_stationary"].iloc[120:].mean()
    assert stationary_rate > 0.7, (
        f"Stark mean-reverting Serie: erwartet >70% stationaer, got {stationary_rate:.1%}"
    )


def test_ou_no_signal_trending(ou_mod):
    """Random Walk (nicht stationaer) → ou_is_stationary selten True."""
    close = _random_walk(n=300)
    df = _make_ohlcv(close)
    cfg = ou_mod.OUMeanReversionConfig(window=60, adf_alpha=0.05)
    out = ou_mod.compute_ou_mean_reversion(df, config=cfg)
    stationary_rate = out["ou_is_stationary"].iloc[60:].mean()
    assert stationary_rate < 0.3, (
        f"Random Walk: erwartet <30% stationaer, got {stationary_rate:.1%}"
    )


def test_ou_bull_passes_below_mean(ou_mod):
    """Wo ou_bull_passes=True, muss close < ou_mean_level gelten.

    Verwendet theta=0.6 + 500 Bars damit Signale garantiert feuern.
    """
    close = _simulate_ou(n=500, theta=0.6, mu=100.0, sigma=0.4, seed=1)
    df = _make_ohlcv(close)
    cfg = ou_mod.OUMeanReversionConfig(window=60, max_half_life=30)
    out = ou_mod.compute_ou_mean_reversion(df, config=cfg)
    bull = out["ou_bull_passes"]
    assert bull.any(), (
        "Mit theta=0.6 und 500 Bars muessen bull_passes feuern – kein Signal gefunden!"
    )
    close_at_bull = out.loc[bull, "close"]
    mean_at_bull = out.loc[bull, "ou_mean_level"]
    assert (close_at_bull < mean_at_bull).all(), (
        "ou_bull_passes=True sollte nur wenn close < ou_mean_level"
    )


def test_ou_bear_passes_above_mean(ou_mod):
    """Wo ou_bear_passes=True, muss close > ou_mean_level gelten.

    Verwendet theta=0.6 + 500 Bars damit Signale garantiert feuern.
    """
    close = _simulate_ou(n=500, theta=0.6, mu=100.0, sigma=0.4, seed=2)
    df = _make_ohlcv(close)
    cfg = ou_mod.OUMeanReversionConfig(window=60, max_half_life=30)
    out = ou_mod.compute_ou_mean_reversion(df, config=cfg)
    bear = out["ou_bear_passes"]
    assert bear.any(), (
        "Mit theta=0.6 und 500 Bars muessen bear_passes feuern – kein Signal gefunden!"
    )
    close_at_bear = out.loc[bear, "close"]
    mean_at_bear = out.loc[bear, "ou_mean_level"]
    assert (close_at_bear > mean_at_bear).all(), (
        "ou_bear_passes=True sollte nur wenn close > ou_mean_level"
    )


def test_ou_max_half_life_filter(ou_mod):
    """Strenger max_half_life unterdrueckt mehr Signale als lockerer Wert."""
    close = _simulate_ou(n=300, theta=0.4, mu=100.0, sigma=0.4)
    df = _make_ohlcv(close)
    cfg_loose = ou_mod.OUMeanReversionConfig(window=60, max_half_life=30)
    cfg_strict = ou_mod.OUMeanReversionConfig(window=60, max_half_life=1)
    out_loose = ou_mod.compute_ou_mean_reversion(df, config=cfg_loose)
    out_strict = ou_mod.compute_ou_mean_reversion(df, config=cfg_strict)
    passes_loose = (out_loose["ou_bull_passes"] | out_loose["ou_bear_passes"]).sum()
    passes_strict = (out_strict["ou_bull_passes"] | out_strict["ou_bear_passes"]).sum()
    assert passes_loose > passes_strict, (
        f"Lockerer max_half_life=30 sollte mehr Signale liefern als max_half_life=1 "
        f"(loose={passes_loose}, strict={passes_strict})"
    )


# ===========================================================================
# Tests – Edge Cases
# ===========================================================================


def test_ou_window_larger_than_data(ou_mod):
    """window > n → alle Outputs NaN/False, kein Crash."""
    close = _simulate_ou(n=30)  # nur 30 Bars
    df = _make_ohlcv(close)
    cfg = ou_mod.OUMeanReversionConfig(window=60)  # window > n
    out = ou_mod.compute_ou_mean_reversion(df, config=cfg)
    assert out["ou_half_life"].isna().all(), "ou_half_life sollte komplett NaN sein"
    assert not out["ou_bull_passes"].any(), "ou_bull_passes sollte komplett False sein"
    assert not out["ou_bear_passes"].any(), "ou_bear_passes sollte komplett False sein"


def test_ou_empty_dataframe(ou_mod):
    """Leerer DataFrame → 6 Ausgabe-Spalten vorhanden, kein Crash."""
    idx = pd.DatetimeIndex([], tz="UTC", name=None)
    df = pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=idx,
    )
    out = ou_mod.compute_ou_mean_reversion(df)
    assert len(out) == 0, "Output sollte ebenfalls leer sein"
    for col in OU_COLS:
        assert col in out.columns, f"Spalte fehlt in leerem Output: {col}"


def test_ou_all_nan_input(ou_mod):
    """Alle NaN-Close-Werte → kein Crash, Outputs NaN/False."""
    n = 100
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(n, np.nan),
            "high": np.full(n, np.nan),
            "low": np.full(n, np.nan),
            "close": np.full(n, np.nan),
            "volume": np.ones(n),
        },
        index=idx,
    )
    out = ou_mod.compute_ou_mean_reversion(df)
    assert not out["ou_bull_passes"].any(), (
        "NaN-Input sollte keine bull_passes produzieren"
    )
    assert not out["ou_bear_passes"].any(), (
        "NaN-Input sollte keine bear_passes produzieren"
    )
