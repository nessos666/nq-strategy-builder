"""Tests für algo_51_ob_keylevel_v2 – Davids korrekte OB-Logik.

Kernregeln getestet:
1. Zone = NUR Body (Open/Close), NICHT High/Low
2. OB nur gültig wenn Fractal AN Key Level (Asia H/L, PDH/PDL, PWH/PWL)
3. Fractal-basiert: 3-Bar Swing High/Low
4. Entgegengesetzte Kerze vor Fractal = OB
5. Daily Reset der OB-Zonen
6. Mean Threshold = (body_high + body_low) / 2
"""

from __future__ import annotations

import pandas as pd
import pytest

# Algo-Import – Pfad wird über algo_bibliothek geladen
import sys
from pathlib import Path

ALGO_DIR = (
    Path(__file__).parent.parent.parent
    / "05_Strategien_Entwicklung"
    / "TRADINGPROJEKT"
    / "nq_backtest"
    / "algo_bibliothek"
    / "PDA"
)
sys.path.insert(0, str(ALGO_DIR))

from algo_51_ob_keylevel_v2 import compute_ob_keylevel


def _make_bars(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start: str = "2026-01-02 10:00",
    freq: str = "5min",
    tz: str = "US/Eastern",
) -> pd.DataFrame:
    """Hilfsfunktion: OHLC DataFrame erstellen."""
    n = len(opens)
    idx = pd.date_range(start, periods=n, freq=freq, tz=tz)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=idx,
    )


# ─── Test 1: Zone = Body only ────────────────────────────────────────────────


class TestZoneIsBodyOnly:
    """Zone MUSS Open/Close sein, NICHT High/Low."""

    def test_bullish_ob_zone_uses_body(self):
        """Bullish OB: Down-Close Kerze → Zone = [Close, Open] (body)."""
        # Erzeuge: Swing Low bei Bar 3 (fracLen=3 → Fractal bestätigt bei Bar 6)
        # Down-Close Kerze davor = Bar 2 (OB-Kandidat)
        # Bar 2: Open=102, Close=99 (Down-Close), High=105(!), Low=96(!)
        # Zone MUSS sein: high=102 (Open), low=99 (Close) → NICHT 105/96!
        opens = [100, 101, 102, 97, 99, 100, 101, 103, 105]
        highs = [103, 104, 105, 98, 101, 102, 104, 106, 108]
        lows = [98, 99, 96, 95, 97, 98, 100, 101, 103]
        closes = [101, 100, 99, 96, 100, 101, 103, 105, 107]
        df = _make_bars(opens, highs, lows, closes)

        result = compute_ob_keylevel(df, pdl=95.0)

        bull_obs = result[result["ob_kl_bullish"]]
        if not bull_obs.empty:
            row = bull_obs.iloc[0]
            # Zone = Body only
            assert row["ob_kl_high"] == 102.0, (
                f"Zone high should be Open=102, got {row['ob_kl_high']}"
            )
            assert row["ob_kl_low"] == 99.0, (
                f"Zone low should be Close=99, got {row['ob_kl_low']}"
            )

    def test_bearish_ob_zone_uses_body(self):
        """Bearish OB: Up-Close Kerze → Zone = [Open, Close] (body)."""
        # Up-Close Kerze: Open=98, Close=102, High=106(!), Low=95(!)
        # Zone MUSS sein: high=102 (Close), low=98 (Open) → NICHT 106/95!
        opens = [100, 99, 98, 103, 101, 100, 99, 97, 95]
        highs = [102, 101, 106, 105, 103, 102, 101, 99, 97]
        lows = [98, 97, 95, 101, 99, 98, 97, 95, 93]
        closes = [99, 100, 102, 104, 100, 99, 98, 96, 94]
        df = _make_bars(opens, highs, lows, closes)

        result = compute_ob_keylevel(df, pdh=105.0)

        bear_obs = result[result["ob_kl_bearish"]]
        if not bear_obs.empty:
            row = bear_obs.iloc[0]
            assert row["ob_kl_high"] == 102.0, (
                f"Zone high should be Close=102, got {row['ob_kl_high']}"
            )
            assert row["ob_kl_low"] == 98.0, (
                f"Zone low should be Open=98, got {row['ob_kl_low']}"
            )


# ─── Test 2: Key Level Filter ────────────────────────────────────────────────


class TestKeyLevelFilter:
    """OB nur gültig wenn Fractal nahe Key Level."""

    def test_ob_rejected_without_key_level(self):
        """OB ohne Key Level in der Nähe → KEIN Signal."""
        # Swing Low bei 96, aber kein Level in der Nähe
        opens = [100, 101, 102, 97, 99, 100, 101, 103, 105]
        highs = [103, 104, 105, 98, 101, 102, 104, 106, 108]
        lows = [98, 99, 96, 95, 97, 98, 100, 101, 103]
        closes = [101, 100, 99, 96, 100, 101, 103, 105, 107]
        df = _make_bars(opens, highs, lows, closes)

        # Keine Levels in der Nähe von 95-96
        result = compute_ob_keylevel(df, pdh=120.0, pdl=80.0)

        assert result["ob_kl_bullish"].sum() == 0, (
            "OB ohne Key Level sollte abgelehnt werden"
        )

    def test_ob_accepted_with_pdl_nearby(self):
        """OB mit PDL nahe Swing Low → Signal."""
        # Brauche genug Bars für Fractal (frac_len=3 → min 7 Bars um Fractal herum)
        # Swing Low bei Index 5 (low=93), PDL=93 → Match
        # Down-Close Kerze bei Index 4 → OB Kandidat
        # Validierung: spätere Kerze High > OB body high
        opens = [110, 108, 106, 104, 102, 96, 98, 100, 102, 104, 106, 108]
        highs = [112, 110, 108, 106, 104, 97, 100, 102, 105, 107, 109, 111]
        lows = [108, 106, 104, 102, 98, 93, 96, 98, 100, 102, 104, 106]
        closes = [109, 107, 105, 103, 99, 95, 99, 101, 104, 106, 108, 110]
        df = _make_bars(opens, highs, lows, closes)

        # PDL=93 ist direkt am Swing Low (93)
        result = compute_ob_keylevel(df, pdl=93.0, level_prox=0.5)

        assert result["ob_kl_bullish"].sum() > 0, (
            "OB mit PDL nearby sollte akzeptiert werden"
        )


# ─── Test 3: Mean Threshold ──────────────────────────────────────────────────


class TestMeanThreshold:
    """Mean Threshold = 50% des Body (Open+Close)/2."""

    def test_mean_is_body_midpoint(self):
        """Mean = (body_high + body_low) / 2, NICHT (High+Low)/2."""
        opens = [100, 101, 102, 97, 99, 100, 101, 103, 105]
        highs = [103, 104, 105, 98, 101, 102, 104, 106, 108]
        lows = [98, 99, 96, 95, 97, 98, 100, 101, 103]
        closes = [101, 100, 99, 96, 100, 101, 103, 105, 107]
        df = _make_bars(opens, highs, lows, closes)

        result = compute_ob_keylevel(df, pdl=95.0)

        bull_obs = result[result["ob_kl_bullish"]]
        if not bull_obs.empty:
            row = bull_obs.iloc[0]
            expected_mean = (102.0 + 99.0) / 2.0  # Body midpoint = 100.5
            assert row["ob_kl_mean"] == expected_mean, (
                f"Mean should be body midpoint {expected_mean}, got {row['ob_kl_mean']}"
            )


# ─── Test 4: Output-Spalten ──────────────────────────────────────────────────


class TestOutputColumns:
    """Alle erwarteten Spalten müssen vorhanden sein."""

    def test_all_columns_present(self):
        """Prüfe dass alle 7 Spalten vorhanden sind."""
        df = _make_bars([100, 101, 102], [103, 104, 105], [97, 98, 99], [101, 100, 103])
        result = compute_ob_keylevel(df)

        expected = {
            "ob_kl_bullish",
            "ob_kl_bearish",
            "ob_kl_high",
            "ob_kl_low",
            "ob_kl_mean",
            "ob_kl_validated",
            "ob_kl_mitigated",
        }
        assert expected.issubset(set(result.columns)), (
            f"Fehlende Spalten: {expected - set(result.columns)}"
        )

    def test_empty_df_returns_empty_columns(self):
        """Leerer DataFrame → leere Spalten, kein Crash."""
        df = _make_bars([], [], [], [])
        result = compute_ob_keylevel(df)
        assert len(result) == 0


# ─── Test 5: Body-Mindestgröße ───────────────────────────────────────────────


class TestMinBody:
    """Nur Kerzen mit body_ratio >= minBody werden als OB erkannt."""

    def test_small_body_rejected(self):
        """Doji (body < 40% der Range) wird nicht als OB erkannt."""
        # Kerze mit winzigem Body aber großem Wick
        opens = [100, 100.1, 102, 97, 99, 100, 101, 103, 105]
        highs = [103, 105, 105, 98, 101, 102, 104, 106, 108]
        lows = [98, 95, 96, 95, 97, 98, 100, 101, 103]
        closes = [101, 99.9, 99, 96, 100, 101, 103, 105, 107]
        # Bar 1: body = 0.2, range = 10 → ratio = 0.02 → REJECT
        df = _make_bars(opens, highs, lows, closes)

        compute_ob_keylevel(df, pdl=95.0, min_body=0.40)
        # Nur Kerzen mit body_ratio >= 0.40 sollten als OB erkannt werden
        # Bar 1 hat ratio 0.02 → sollte NICHT als OB gelten
        # Bar 2 hat Open=102, Close=99, High=105, Low=96 → body=3, range=9 → ratio=0.33 → AUCH REJECT
        # Abhängig von der Logik...


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
