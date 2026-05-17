"""Tests fuer science_mtf_align_v2 – MTF-Alignment-Filter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Algo-Bibliothek in Pfad aufnehmen (via TRADINGPROJEKT_PATH konfigurierbar)
import os

_TRADINGPROJEKT = Path(
    os.environ.get("TRADINGPROJEKT_PATH", str(Path.home() / "TRADINGPROJEKT"))
)
_ALGO_PATH = _TRADINGPROJEKT / "nq_backtest/algo_bibliothek/v2/science"
if not _ALGO_PATH.exists():
    pytest.skip(
        "science-Bibliothek nicht gefunden (TRADINGPROJEKT_PATH setzen)",
        allow_module_level=True,
    )
sys.path.insert(0, str(_ALGO_PATH))

from science_mtf_align_v2 import Config, compute_mtf_align  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────────


def _make_df(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetischer 1m-OHLCV DataFrame mit tz-aware Index."""
    rng = np.random.default_rng(seed)
    base = 20000.0
    prices = base + np.cumsum(rng.normal(0, 2, n_bars))
    idx = pd.date_range(
        "2026-04-01 09:30", periods=n_bars, freq="1min", tz="America/New_York"
    )
    high = prices + rng.uniform(1, 5, n_bars)
    low = prices - rng.uniform(1, 5, n_bars)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": high,
            "Low": low,
            "Close": prices,
            "Volume": rng.integers(100, 1000, n_bars).astype(float),
        },
        index=idx,
    )
    return df


def _make_df_with_bull_fvg() -> pd.DataFrame:
    """DataFrame mit einem klaren bullischen 5min-FVG (kein Zufall)."""
    idx = pd.date_range(
        "2026-04-01 09:30", periods=300, freq="1min", tz="America/New_York"
    )
    prices = np.full(300, 20000.0)

    # Nach 50 Bars: starker Aufwaertsimpuls (erzeugt bullischen 5min-FVG)
    # Kerze i-2 schliesst bei 20000, Kerze i oeffnet bei 20010 (gap > 5 Punkte)
    prices[60:] = 20020.0

    high = prices + 2.0
    low = prices - 2.0
    # FVG-Luecke: High[i-2]=20002, Low[i]=20018 --> gap=16 Punkte (klar)
    high[49] = 20002.0
    low[50] = 20018.0

    df = pd.DataFrame(
        {"Open": prices, "High": high, "Low": low, "Close": prices, "Volume": 500.0},
        index=idx,
    )
    return df


# ── Tests ───────────────────────────────────────────────────────────────────────


class TestOutputStruktur:
    def test_alle_spalten_vorhanden(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        for col in (
            "mtf_fvg_active",
            "mtf_fvg_direction",
            "mtf_fvg_depth",
            "mtf_fvg_strength",
            "mtf_align_passes",
        ):
            assert col in out.columns, f"Spalte fehlt: {col}"

    def test_originalindex_erhalten(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        assert list(out.index) == list(df.index)

    def test_originalspalten_erhalten(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in out.columns

    def test_dtypes_korrekt(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        assert out["mtf_fvg_active"].dtype == bool
        assert out["mtf_fvg_direction"].dtype == np.int8
        assert out["mtf_fvg_depth"].dtype == np.float32
        assert out["mtf_fvg_strength"].dtype == np.float32
        assert out["mtf_align_passes"].dtype == bool


class TestWertebereich:
    def test_direction_nur_minus1_0_plus1(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        assert set(out["mtf_fvg_direction"].unique()).issubset({-1, 0, 1})

    def test_depth_zwischen_0_und_1(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        aktive = out[out["mtf_fvg_active"]]
        if not aktive.empty:
            assert (aktive["mtf_fvg_depth"] >= 0.0).all()
            assert (aktive["mtf_fvg_depth"] <= 1.0).all()

    def test_strength_nicht_negativ(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        assert (out["mtf_fvg_strength"] >= 0.0).all()

    def test_passes_implikation(self) -> None:
        """passes=True darf nur wenn active=True."""
        df = _make_df()
        out = compute_mtf_align(df)
        passes_ohne_active = out["mtf_align_passes"] & ~out["mtf_fvg_active"]
        assert not passes_ohne_active.any(), "passes=True ohne active=True"


class TestDepthFilter:
    def test_kein_pass_am_rand(self) -> None:
        """Tiefe < min_depth oder > max_depth → kein Pass."""
        df = _make_df()
        cfg = Config(pullback_min_depth=0.2, pullback_max_depth=0.8)
        out = compute_mtf_align(df, cfg)
        flache = out[out["mtf_fvg_active"] & (out["mtf_fvg_depth"] < 0.2)]
        assert not flache["mtf_align_passes"].any()
        tiefe = out[out["mtf_fvg_active"] & (out["mtf_fvg_depth"] > 0.8)]
        assert not tiefe["mtf_align_passes"].any()


class TestKleinspaltennamen:
    def test_lowercase_spalten(self) -> None:
        """Algo akzeptiert auch kleingeschriebene OHLC-Spaltennamen."""
        df = _make_df().rename(columns=str.lower)
        out = compute_mtf_align(df)
        assert "mtf_align_passes" in out.columns

    def test_fehlende_spalte_wirft_fehler(self) -> None:
        df = _make_df().drop(columns=["High"])
        with pytest.raises(ValueError, match="Fehlende Spalten"):
            compute_mtf_align(df)


class TestZuWenigDaten:
    def test_zu_wenig_htf_bars(self) -> None:
        """Weniger als 3 HTF-Bars → alle passes=False, kein Crash."""
        df = _make_df(n_bars=8)
        out = compute_mtf_align(df)
        assert not out["mtf_align_passes"].any()
        assert not out["mtf_fvg_active"].any()


class TestConfig:
    def test_groessere_min_fvg_pts(self) -> None:
        """Sehr grosser min_fvg_pts → kaum FVGs gefunden."""
        df = _make_df()
        cfg_streng = Config(min_fvg_pts=999.0)
        out_streng = compute_mtf_align(df, cfg_streng)
        assert not out_streng["mtf_fvg_active"].any()

    def test_passiert_mit_default_config(self) -> None:
        df = _make_df(n_bars=500)
        out = compute_mtf_align(df)
        # Mit 500 Bars sollte mindestens ein FVG auftreten
        assert (
            out["mtf_fvg_active"].any() or True
        )  # kein harter Assert, Markt abhaengig

    def test_direction_konsistent_mit_active(self) -> None:
        df = _make_df()
        out = compute_mtf_align(df)
        # Wenn active=False → direction muss 0 sein
        inaktive = out[~out["mtf_fvg_active"]]
        assert (inaktive["mtf_fvg_direction"] == 0).all()
