"""
Testfeld: david_bibliothek/11_ICT_Konzepte/manip_liquidity_sweep.py

Was getestet wird:
- Bullish Sweep: Low fällt >= 3 Punkte unter Asia_Low + Recovery → manip_bull=True
- Bearish Sweep: High steigt >= 3 Punkte über Asia_High + Recovery → manip_bear=True
- Kein Signal ohne Recovery (Preis kommt nicht zurück)
- Min-Tiefe-Filter: Sweep < 3 Punkte → kein Signal
- Erstes Signal pro Tag (zweiter Sweep gleicher Tag → ignoriert)
- Multi-Bar Recovery: Recovery darf auf nachfolgendem Bar passieren
- tz-naive Index: kein Crash, wird automatisch als UTC behandelt
- Leerer Input: gibt leeren DataFrame zurück

Zeitzonen-Notiz:
    Tests nutzen UTC-Timestamps. New York im Sommer = UTC-4 (EDT).
    Asia-Phase: 00:00–08:00 ET = 04:00–12:00 UTC
    Manip-Phase: 08:00–10:00 ET = 12:00–14:00 UTC
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_ALGO = (
    Path(__file__).parent.parent
    / "david_bibliothek"
    / "11_ICT_Konzepte"
    / "manip_liquidity_sweep.py"
)


def _load():
    if not _ALGO.exists():
        pytest.skip(f"Algo nicht gefunden: {_ALGO}")
    spec = importlib.util.spec_from_file_location("manip_liquidity_sweep", _ALGO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manip():
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


# ── Bullish Sweep ────────────────────────────────────────────────────────────────


def test_bull_sweep_same_bar_recovery(manip):
    """Sweep + Recovery auf demselben Bar → manip_bull=True.

    Asia-Bar (06:00 UTC = 02:00 ET): Low=20000 → Asia_Low = 20000
    Manip-Bar (13:00 UTC = 09:00 ET): Low=19994 (6 Punkte unter Asia_Low),
                                       Close=20005 (über Asia_Low → Recovery)
    """
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20010, 20020, 20000, 20010),  # Asia-Bar
            _bar(
                "2024-06-03 13:00", 20000, 20006, 19994, 20005
            ),  # Manip: Sweep + Recovery
        ]
    )
    result = manip.run(df)
    assert result["manip_bull"].iloc[1] is True or result["manip_bull"].iloc[1]


def test_bull_sweep_multibar_recovery(manip):
    """Recovery erst auf nachfolgendem Bar → manip_bull=True auf Recovery-Bar.

    Sweep auf Bar 1 (Close noch unter Asia_Low), Recovery auf Bar 2.
    """
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20010, 20020, 20000, 20010),  # Asia: Low=20000
            _bar(
                "2024-06-03 13:00", 20000, 20002, 19993, 19998
            ),  # Sweep (7 Punkte), kein Recovery
            _bar(
                "2024-06-03 13:01", 19998, 20015, 19995, 20010
            ),  # Recovery: Close > 20000
        ]
    )
    result = manip.run(df)
    # Signal muss irgendwo auf den Manip-Bars aufleuchten
    assert result["manip_bull"].any(), (
        "Kein Bull-Signal obwohl Sweep + Recovery vorhanden"
    )


def test_bull_no_signal_without_recovery(manip):
    """Sweep ohne Recovery (Preis bleibt unter Asia_Low) → kein Signal."""
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20010, 20020, 20000, 20010),  # Asia: Low=20000
            _bar(
                "2024-06-03 13:00", 20000, 20002, 19990, 19995
            ),  # Sweep, Close=19995 < 20000
            _bar("2024-06-03 13:01", 19995, 19998, 19985, 19992),  # immer noch drunter
        ]
    )
    result = manip.run(df)
    assert not result["manip_bull"].any(), "Signal ohne Recovery erwartet False"


def test_bull_sweep_too_shallow(manip):
    """Sweep nur 2 Punkte (< min_sweep_pts=3) → kein Signal."""
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20010, 20020, 20000, 20010),  # Asia: Low=20000
            _bar("2024-06-03 13:00", 20000, 20006, 19998, 20005),  # Sweep nur 2 Punkte
        ]
    )
    result = manip.run(df)
    assert not result["manip_bull"].any(), (
        "Schwacher Sweep (<3 Punkte) darf kein Signal feuern"
    )


def test_bull_first_signal_per_day_only(manip):
    """Zweiter Sweep am gleichen Tag → wird ignoriert (max. 1 Signal/Tag)."""
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20010, 20020, 20000, 20010),  # Asia: Low=20000
            _bar("2024-06-03 13:00", 20000, 20010, 19990, 20005),  # Sweep 1 → Signal
            _bar(
                "2024-06-03 13:30", 20000, 20010, 19985, 20008
            ),  # Sweep 2 → soll ignoriert werden
        ]
    )
    result = manip.run(df)
    assert result["manip_bull"].sum() <= 1, "Nur max. 1 Bull-Signal pro Tag erlaubt"


# ── Bearish Sweep ────────────────────────────────────────────────────────────────


def test_bear_sweep_same_bar_recovery(manip):
    """Bear Sweep + Recovery auf demselben Bar → manip_bear=True.

    Asia-Bar: High=20100 → Asia_High = 20100
    Manip-Bar: High=20107 (7 Punkte über Asia_High → Sweep),
               Close=20095 (unter Asia_High → Recovery)
    """
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20090, 20100, 20080, 20090),  # Asia: High=20100
            _bar(
                "2024-06-03 13:00", 20100, 20107, 20093, 20095
            ),  # Bear: Sweep + Recovery
        ]
    )
    result = manip.run(df)
    assert result["manip_bear"].iloc[1] is True or result["manip_bear"].iloc[1]


def test_bear_no_signal_without_recovery(manip):
    """Bear Sweep ohne Recovery → kein Signal."""
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20090, 20100, 20080, 20090),  # Asia: High=20100
            _bar(
                "2024-06-03 13:00", 20100, 20108, 20098, 20105
            ),  # Sweep, Close=20105 > 20100
        ]
    )
    result = manip.run(df)
    assert not result["manip_bear"].any(), "Bear-Signal ohne Recovery erwartet False"


def test_bear_sweep_too_shallow(manip):
    """Bear Sweep nur 2 Punkte → kein Signal."""
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 20090, 20100, 20080, 20090),  # Asia: High=20100
            _bar("2024-06-03 13:00", 20100, 20102, 20094, 20095),  # Sweep nur 2 Punkte
        ]
    )
    result = manip.run(df)
    assert not result["manip_bear"].any()


# ── Output-Struktur ──────────────────────────────────────────────────────────────


def test_output_columns(manip):
    """Beide Output-Spalten vorhanden."""
    df = _make_df([_bar("2024-06-03 06:00", 100, 101, 99, 100)])
    result = manip.run(df)
    assert "manip_bull" in result.columns
    assert "manip_bear" in result.columns


def test_output_same_length(manip):
    """Output hat gleich viele Zeilen wie Input."""
    df = _make_df(
        [
            _bar("2024-06-03 06:00", 100, 101, 99, 100),
            _bar("2024-06-03 13:00", 100, 101, 99, 100),
        ]
    )
    result = manip.run(df)
    assert len(result) == len(df)


# ── Edge Cases ───────────────────────────────────────────────────────────────────


def test_tz_naive_no_crash(manip):
    """tz-naive Index wird graceful behandelt (kein ValueError)."""
    idx = pd.to_datetime(["2024-06-03 06:00", "2024-06-03 13:00"])  # tz-naive
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [101.0, 101.0],
            "Low": [99.0, 99.0],
            "Close": [100.0, 100.0],
        },
        index=idx,
    )
    # Darf nicht crashen
    result = manip.run(df)
    assert "manip_bull" in result.columns


def test_empty_dataframe(manip):
    """Leerer DataFrame → leerer Output, kein Crash."""
    df = pd.DataFrame(
        columns=["Open", "High", "Low", "Close"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    result = manip.run(df)
    assert len(result) == 0


def test_no_asia_bars(manip):
    """Nur Bars in der Manip-Phase (kein Asia) → Asia_Low=NaN → kein Signal."""
    df = _make_df(
        [
            _bar("2024-06-03 13:00", 100, 101, 94, 102),  # nur Manip-Phase
            _bar("2024-06-03 13:01", 102, 105, 99, 103),
        ]
    )
    result = manip.run(df)
    assert not result["manip_bull"].any()
    assert not result["manip_bear"].any()
