"""
Tests für david_bibliothek/04_Opening_Gaps_NDOG_NWOG/ehpda_bounce.py

Test-Daten Aufbau (UTC-Timestamps):
    Day 0 (2024-01-01): 1 Bar, close=20000 → setzt Basis für Gap-Berechnung
    Day 1 (2024-01-02): open=20020 → Gap1 bullish: top=20020, btm=20000, avg=20010
                        close=20025 → Basis für Gap2
    Day 2 (2024-01-03): open=20000 → Gap2 bearish: top=20025, btm=20000, avg=20012.5

    Nach Day 2: 2 Gaps vorhanden → EHPDA berechenbar
    Gap-Sortierung nach avg: [20010, 20012.5]
    EHPDA_0 = (Gap1_top + Gap2_btm) / 2 = (20020 + 20000) / 2 = 20010.0

    Default Toleranz = 2.0 pts
    Bullish Bounce: Low <= 20012.0 UND Close > 20010.0
    Bearish Bounce: High >= 20008.0 UND Close < 20010.0
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ALGO = (
    Path(__file__).parent.parent
    / "david_bibliothek"
    / "04_Opening_Gaps_NDOG_NWOG"
    / "ehpda_bounce.py"
)


def _load():
    if not _ALGO.exists():
        pytest.skip(f"Algo nicht gefunden: {_ALGO}")
    spec = importlib.util.spec_from_file_location("ehpda_bounce_test", _ALGO)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ehpda_bounce_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bounce():
    return _load()


def _bar(ts_utc: str, open_: float, high: float, low: float, close: float) -> dict:
    return {"ts": ts_utc, "Open": open_, "High": high, "Low": low, "Close": close}


def _make_df(bars: list[dict]) -> pd.DataFrame:
    idx = pd.to_datetime([b["ts"] for b in bars], utc=True)
    return pd.DataFrame(
        {
            "Open": [b["Open"] for b in bars],
            "High": [b["High"] for b in bars],
            "Low": [b["Low"] for b in bars],
            "Close": [b["Close"] for b in bars],
        },
        index=idx,
    )


def _base_df(extra_bars: list[dict] | None = None) -> pd.DataFrame:
    """Minimaler Basis-DataFrame mit 2 Gaps → EHPDA_0 = 20010.0"""
    bars = [
        # Day 0: setzt prev_close = 20000
        _bar("2024-01-01 15:00", 20000, 20010, 19990, 20000),
        # Day 1: open=20020 (gap up), close=20025
        _bar("2024-01-02 08:00", 20020, 20030, 20015, 20025),
        # Day 2: open=20000 (gap down) → 2 Gaps → EHPDA_0=20010
        _bar("2024-01-03 08:00", 20000, 20005, 19995, 20000),
    ]
    if extra_bars:
        bars.extend(extra_bars)
    return _make_df(bars)


# ── Leerer Input ──────────────────────────────────────────────────────────────


def test_empty_df_returns_correct_columns(bounce):
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    result = bounce.compute_ehpda_bounce(df)
    assert "ehpda_bounce_bullish" in result.columns
    assert "ehpda_bounce_bearish" in result.columns
    assert "ehpda_bounce_level" in result.columns
    assert len(result) == 0


# ── Kein Signal vor zweitem Gap ───────────────────────────────────────────────


def test_no_signal_before_second_gap(bounce):
    """Day 1: nur 1 Gap → keine EHPDA → kein Signal."""
    bars = [
        _bar("2024-01-01 15:00", 20000, 20010, 19990, 20000),
        _bar("2024-01-02 09:00", 20020, 20030, 20010, 20025),
        _bar("2024-01-02 12:00", 20025, 20028, 20009, 20015),
    ]
    df = _make_df(bars)
    result = bounce.compute_ehpda_bounce(df)
    day1_mask = pd.to_datetime(result.index).date == pd.Timestamp("2024-01-02").date()
    assert not result.loc[day1_mask, "ehpda_bounce_bullish"].any()
    assert not result.loc[day1_mask, "ehpda_bounce_bearish"].any()


# ── Bullish Bounce ────────────────────────────────────────────────────────────


def test_bullish_bounce_within_tolerance(bounce):
    """Low=20009 <= 20010+2.0=20012 UND Close=20015 > 20010 → bullish."""
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20010, 20015, 20009, 20015),
        ]
    )
    result = bounce.compute_ehpda_bounce(df)
    last = result.iloc[-1]
    assert last["ehpda_bounce_bullish"]
    assert not last["ehpda_bounce_bearish"]
    assert abs(last["ehpda_bounce_level"] - 20010.0) < 0.01


def test_bullish_bounce_exact_tolerance(bounce):
    """Low=20012 = 20010+2.0 (exakt an Grenze) → Signal noch ausgelöst."""
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20012, 20018, 20012, 20016),
        ]
    )
    result = bounce.compute_ehpda_bounce(df)
    assert result.iloc[-1]["ehpda_bounce_bullish"]


def test_bullish_no_signal_too_far(bounce):
    """Low=20015 > 20010+2.0=20012 → kein Bounce (zu weit vom Level)."""
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20015, 20020, 20015, 20018),
        ]
    )
    result = bounce.compute_ehpda_bounce(df)
    assert not result.iloc[-1]["ehpda_bounce_bullish"]


def test_bullish_no_signal_close_below_level(bounce):
    """Low berührt Level aber Close < Level → kein Bounce (kein Recovery)."""
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20011, 20011, 20009, 20008),
        ]
    )
    result = bounce.compute_ehpda_bounce(df)
    assert not result.iloc[-1]["ehpda_bounce_bullish"]


# ── Bearish Bounce ────────────────────────────────────────────────────────────


def test_bearish_bounce_within_tolerance(bounce):
    """High=20011 >= 20010-2.0=20008 UND Close=20005 < 20010 → bearish."""
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20010, 20011, 20003, 20005),
        ]
    )
    result = bounce.compute_ehpda_bounce(df)
    last = result.iloc[-1]
    assert last["ehpda_bounce_bearish"]
    assert not last["ehpda_bounce_bullish"]


def test_bearish_no_signal_close_above_level(bounce):
    """High berührt Level aber Close > Level → kein Bounce."""
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20009, 20011, 20009, 20013),
        ]
    )
    result = bounce.compute_ehpda_bounce(df)
    assert not result.iloc[-1]["ehpda_bounce_bearish"]


# ── Config-Parameter ──────────────────────────────────────────────────────────


def test_wider_tolerance_triggers_more(bounce):
    """Toleranz 5.0 → Low=20015 (=20010+5, exakt) → Signal."""
    cfg = bounce.EHPDABounceConfig(touch_tolerance_pts=5.0)
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20015, 20018, 20015, 20016),
        ]
    )
    result = bounce.compute_ehpda_bounce(df, cfg)
    assert result.iloc[-1]["ehpda_bounce_bullish"]


def test_narrower_tolerance_no_trigger(bounce):
    """Toleranz 0.5 → Low=20011 (1pt ÜBER Level 20010, außerhalb Toleranz) → kein Signal."""
    cfg = bounce.EHPDABounceConfig(touch_tolerance_pts=0.5)
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20015, 20018, 20011, 20015),
        ]
    )
    result = bounce.compute_ehpda_bounce(df, cfg)
    assert not result.iloc[-1]["ehpda_bounce_bullish"]


# ── run()-Wrapper ─────────────────────────────────────────────────────────────


def test_run_wrapper_identical_to_compute(bounce):
    """run() liefert identisches Ergebnis wie compute_ehpda_bounce()."""
    df = _base_df(
        [
            _bar("2024-01-03 15:00", 20010, 20015, 20009, 20015),
        ]
    )
    r1 = bounce.compute_ehpda_bounce(df)
    r2 = bounce.run(df)
    pd.testing.assert_frame_equal(r1, r2)
