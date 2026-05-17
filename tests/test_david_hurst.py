"""
Testfeld: david_bibliothek/04_Science/hurst_exponent.py

Was getestet wird:
- Kein Look-Ahead (bfill-Bug weg): historische Werte ändern sich nicht wenn Zukunft kommt
- Mean-Reversion-Erkennung: oszillierender Preis → H < 0.45 → hurst_passes=True
- Trend-Erkennung: monoton steigender Preis → H > 0.50
- hurst_passes = hurst_passes_revert (Alias korrekt)
- Alle 5 Output-Spalten vorhanden
- Leerer Input / fehlende Close-Spalte → neutrale Werte, kein Crash
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Algo-Pfad relativ zu diesem Test-File
_ALGO = (
    Path(__file__).parent.parent
    / "david_bibliothek"
    / "04_Science"
    / "hurst_exponent.py"
)


def _load():
    """Lädt hurst_exponent.py dynamisch."""
    if not _ALGO.exists():
        pytest.skip(f"Algo nicht gefunden: {_ALGO}")
    spec = importlib.util.spec_from_file_location("hurst_exponent", _ALGO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hurst():
    return _load()


def _make_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
        },
        index=idx,
    )


# ── Output-Struktur ─────────────────────────────────────────────────────────────


def test_output_columns_present(hurst):
    """Alle 5 erwarteten Spalten im Output."""
    df = _make_df([100.0] * 200)
    result = hurst.run(df)
    expected = {
        "hurst_exp",
        "hurst_regime",
        "hurst_passes_trend",
        "hurst_passes_revert",
        "hurst_passes",
    }
    assert expected.issubset(set(result.columns)), (
        f"Fehlende Spalten: {expected - set(result.columns)}"
    )


def test_output_same_length_as_input(hurst):
    """Output hat exakt so viele Zeilen wie Input."""
    df = _make_df([100.0] * 150)
    result = hurst.run(df)
    assert len(result) == len(df)


def test_output_index_matches_input(hurst):
    """Output-Index ist identisch mit Input-Index."""
    df = _make_df([100.0] * 150)
    result = hurst.run(df)
    assert result.index.equals(df.index)


# ── Kein Look-Ahead ─────────────────────────────────────────────────────────────


def test_no_lookahead_past_values_stable(hurst):
    """Historische Hurst-Werte dürfen sich NICHT ändern wenn neue Bars hinzukommen.

    Würde bfill() Look-Ahead aktiv sein, würde das Einfügen eines Zukunfts-Bars
    die historischen Werte verändern (weil bfill rückwärts befüllt).
    """
    base_closes = [100.0 + np.sin(i * 0.3) * 5 for i in range(150)]
    df_base = _make_df(base_closes)
    df_extended = _make_df(base_closes + [999.0] * 20)  # 20 neue Bars am Ende

    result_base = hurst.run(df_base)
    result_extended = hurst.run(df_extended)

    # Die ersten 150 Werte müssen identisch sein
    base_vals = result_base["hurst_exp"].values
    ext_vals = result_extended["hurst_exp"].values[:150]
    assert np.allclose(base_vals, ext_vals, atol=1e-10), (
        "Look-Ahead-Bug: historische Hurst-Werte ändern sich durch neue Bars!"
    )


# ── Regime-Erkennung ────────────────────────────────────────────────────────────


def test_mean_reversion_detected(hurst):
    """Oszillierender Preis (±5 Punkte) → NQ-typisch mean-revertend → H < 0.45.

    hurst_passes_revert sollte auf den meisten Bars True sein.
    """
    closes = [100.0 + np.sin(i * 0.5) * 5 for i in range(300)]
    df = _make_df(closes)
    result = hurst.run(df)

    # Erst ab Bar 100 (warmup) messen
    late_bars = result.iloc[110:]
    revert_rate = late_bars["hurst_passes_revert"].mean()
    assert revert_rate > 0.5, (
        f"Erwartet >50% mean-reversion bei oszillierenden Preisen, got {revert_rate:.1%}"
    )


def test_trending_detected(hurst):
    """Monoton steigender Preis → H > 0.50 → hurst_passes_trend=True."""
    closes = [100.0 + i * 0.1 for i in range(300)]
    df = _make_df(closes)
    result = hurst.run(df)

    late_bars = result.iloc[110:]
    trend_rate = late_bars["hurst_passes_trend"].mean()
    assert trend_rate > 0.5, (
        f"Erwartet >50% trend bei monoton steigendem Preis, got {trend_rate:.1%}"
    )


def test_hurst_exp_range(hurst):
    """hurst_exp liegt immer in [0.0, 1.0]."""
    closes = [100.0 + np.random.randn() * 5 for _ in range(300)]
    df = _make_df(closes)
    result = hurst.run(df)
    assert result["hurst_exp"].min() >= 0.0
    assert result["hurst_exp"].max() <= 1.0


# ── Alias ───────────────────────────────────────────────────────────────────────


def test_hurst_passes_alias_equals_revert(hurst):
    """hurst_passes muss identisch zu hurst_passes_revert sein (Alias)."""
    closes = [100.0 + np.sin(i * 0.3) * 3 for i in range(200)]
    df = _make_df(closes)
    result = hurst.run(df)
    pd.testing.assert_series_equal(
        result["hurst_passes"],
        result["hurst_passes_revert"],
        check_names=False,
    )


# ── Regime-Spalte ───────────────────────────────────────────────────────────────


def test_regime_consistent_with_thresholds(hurst):
    """hurst_regime muss konsistent mit hurst_exp sein."""
    closes = [100.0 + np.sin(i * 0.4) * 4 for i in range(250)]
    df = _make_df(closes)
    result = hurst.run(df)

    # Regime 1 → hurst_exp > 0.50
    trend_bars = result[result["hurst_regime"] == 1]
    if len(trend_bars) > 0:
        assert (trend_bars["hurst_exp"] > 0.50).all()

    # Regime -1 → hurst_exp < 0.45
    revert_bars = result[result["hurst_regime"] == -1]
    if len(revert_bars) > 0:
        assert (revert_bars["hurst_exp"] < 0.45).all()


# ── Edge Cases ──────────────────────────────────────────────────────────────────


def test_empty_dataframe(hurst):
    """Leerer DataFrame → neutrale Werte, kein Crash."""
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df.index = pd.DatetimeIndex([], tz="UTC")
    result = hurst.run(df)
    assert len(result) == 0


def test_missing_close_column(hurst):
    """Fehlende Close-Spalte → neutrale Werte (0.5 / False), kein Crash."""
    df = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0]},
        index=pd.DatetimeIndex(["2024-01-02 09:00"], tz="UTC"),
    )
    result = hurst.run(df)
    assert result["hurst_exp"].iloc[0] == 0.5
    assert result["hurst_passes"].iloc[0] is False or not result["hurst_passes"].iloc[0]


def test_fewer_than_window_bars(hurst):
    """Weniger Bars als window=100 → erste Bars bleiben auf 0.5 (kein Crash)."""
    df = _make_df([100.0] * 50)
    result = hurst.run(df)
    assert (result["hurst_exp"] == 0.5).all()
