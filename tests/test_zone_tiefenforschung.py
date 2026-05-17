"""Tests für Zone-Tiefenforschung: visit-duration, return-excursion, manip-day-stats, session-return."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_bull_zone_df(bars_spec):
    """Erzeugt einen DataFrame für eine einzelne Bull-Zone.

    bars_spec: Liste von Dicts mit keys: High, Low, signal, filled
    Erste Bar mit signal=True = Zone-Entstehung.
    Zone high/low werden aus erster Signal-Bar genommen.

    Beispiel:
        bars_spec = [
            {"High": 105, "Low": 104, "signal": True,  "filled": False},  # Bar 0: signal
            {"High": 102, "Low": 101, "signal": False, "filled": False},  # Bar 1: in-zone
            ...
        ]
    """
    n = len(bars_spec)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")

    # Zone high/low aus Signal-Bar (erster Bar mit signal=True)
    z_high, z_low = None, None
    for b in bars_spec:
        if b.get("signal"):
            z_high = b.get("zone_high", 103.0)
            z_low = b.get("zone_low", 100.0)
            break

    fvg_bull = [bool(b.get("signal", False)) for b in bars_spec]
    fvg_bull_filled = [bool(b.get("filled", False)) for b in bars_spec]
    # ffill zone high/low after signal bar
    bull_high = []
    bull_low = []
    active = False
    for b in bars_spec:
        if b.get("signal"):
            active = True
        if active:
            bull_high.append(z_high)
            bull_low.append(z_low)
        else:
            bull_high.append(None)
            bull_low.append(None)

    data = {
        "High": [b["High"] for b in bars_spec],
        "Low": [b["Low"] for b in bars_spec],
        "Open": [b["Low"] for b in bars_spec],
        "Close": [b["High"] for b in bars_spec],
        "fvg_bull": fvg_bull,
        "fvg_bear": [False] * n,
        "fvg_bull_high": bull_high,
        "fvg_bull_low": bull_low,
        "fvg_bull_filled": fvg_bull_filled,
        "fvg_bear_high": [None] * n,
        "fvg_bear_low": [None] * n,
        "fvg_bear_filled": [False] * n,
    }
    return pd.DataFrame(data, index=idx)


def test_visit_duration_two_visits_return_then_break():
    """1. Besuch: Return-Exit. 2. Besuch: Break-Exit. Post-Death: 1 Bar bis Re-Touch."""
    # Zone bull_high=103, bull_low=100
    bars = [
        {"High": 105, "Low": 104, "signal": True, "filled": False},  # Bar 0: signal
        {"High": 102, "Low": 101, "signal": False, "filled": False},  # Bar 1: in-zone
        {
            "High": 106,
            "Low": 104,
            "signal": False,
            "filled": False,
        },  # Bar 2: exit return (H>103)
        {
            "High": 102,
            "Low": 101,
            "signal": False,
            "filled": False,
        },  # Bar 3: in-zone (visit 2)
        {
            "High": 101,
            "Low": 98,
            "signal": False,
            "filled": True,
        },  # Bar 4: exit break (L<100)
        {
            "High": 105,
            "Low": 101,
            "signal": False,
            "filled": True,
        },  # Bar 5: re-touch (L<=103)
    ]
    df = _make_bull_zone_df(bars)
    from sb.inspect import analyze_zone_visit_duration

    result = analyze_zone_visit_duration(df, "fvg", max_visits=4, post_death_window=10)

    bull = result["bull"]
    v1 = bull["visit_1"]
    v2 = bull["visit_2"]
    pd_stats = bull["post_death"]

    assert v1["n"] == 1
    assert v1["median_duration_inside"] == 1
    assert v1["exit_return_pct"] == 100.0
    assert v1["exit_break_pct"] == 0.0

    assert v2["n"] == 1
    assert v2["median_duration_inside"] == 1
    assert (
        v2["median_outside_before"] == 1
    )  # 1 Bar zwischen visit 1 Ende und visit 2 Start
    assert v2["exit_break_pct"] == 100.0
    assert v2["exit_return_pct"] == 0.0

    assert pd_stats["n"] == 1
    assert pd_stats["permanent"] == 0
    assert pd_stats["permanent_pct"] == 0.0
    assert pd_stats["median_bars_to_retouch"] == 1


def test_visit_duration_permanent_post_death():
    """Nach Break-Exit: Preis berührt Zone nicht mehr → permanent=True."""
    bars = [
        {"High": 105, "Low": 104, "signal": True, "filled": False},  # Bar 0: signal
        {"High": 102, "Low": 101, "signal": False, "filled": False},  # Bar 1: in-zone
        {"High": 101, "Low": 98, "signal": False, "filled": True},  # Bar 2: exit break
        {
            "High": 97,
            "Low": 95,
            "signal": False,
            "filled": True,
        },  # Bar 3: kein Re-Touch
        {
            "High": 96,
            "Low": 94,
            "signal": False,
            "filled": True,
        },  # Bar 4: kein Re-Touch
    ]
    df = _make_bull_zone_df(bars)
    from sb.inspect import analyze_zone_visit_duration

    result = analyze_zone_visit_duration(df, "fvg", max_visits=4, post_death_window=5)

    pd_stats = result["bull"]["post_death"]
    assert pd_stats["n"] == 1
    assert pd_stats["permanent"] == 1
    assert pd_stats["permanent_pct"] == 100.0


def test_return_excursion_bull():
    """Fake-Out Bull: Preis geht 5pts unter bull_low bevor Rückkehr."""
    # Zone: bull_high=103, bull_low=100
    # Break: Low=95 (5pts unter bull_low=100)
    # Return: High=101 >= bull_low=100
    bars = [
        {"High": 105, "Low": 104, "signal": True, "filled": False},  # Bar 0: signal
        {
            "High": 99,
            "Low": 95,
            "signal": False,
            "filled": True,
        },  # Bar 1: break (L=95 < 100)
        {
            "High": 101,
            "Low": 98,
            "signal": False,
            "filled": True,
        },  # Bar 2: return (H=101 >= 100)
        {
            "High": 99,
            "Low": 96,
            "signal": False,
            "filled": True,
        },  # Bar 3: no longer relevant
    ]
    df = _make_bull_zone_df(bars)
    from sb.inspect import analyze_zone_return_excursion

    result = analyze_zone_return_excursion(df, "fvg", return_window=5)

    bull = result["bull"]
    assert bull["n_fake_outs"] == 1
    # excursion = bull_low - min(Low) = 100 - 95 = 5pts
    assert bull["p50"] == 5.0


def test_return_excursion_no_return():
    """Echter Break (kein Return) → wird nicht gezählt."""
    bars = [
        {"High": 105, "Low": 104, "signal": True, "filled": False},
        {"High": 99, "Low": 95, "signal": False, "filled": True},
        {"High": 94, "Low": 91, "signal": False, "filled": True},  # kein Return
    ]
    df = _make_bull_zone_df(bars)
    from sb.inspect import analyze_zone_return_excursion

    result = analyze_zone_return_excursion(df, "fvg", return_window=2)

    assert result["bull"]["n_fake_outs"] == 0


def _make_manip_df(days_spec):
    """Erzeugt Intraday-DataFrame mit manip_bear Signal.

    days_spec: Liste von Dicts:
      date_str: "2024-01-02"
      bars: Liste von (hour_ny, close, manip_bear)
    """
    import pytz

    ny = pytz.timezone("US/Eastern")
    rows = []
    for day in days_spec:
        for hour, close, mb in day["bars"]:
            h = int(hour)
            m = int((hour - h) * 60)
            dt_ny = pd.Timestamp(f"{day['date_str']} {h:02d}:{m:02d}:00").tz_localize(
                ny
            )
            dt_utc = dt_ny.tz_convert("UTC")
            rows.append(
                {
                    "ts": dt_utc,
                    "Close": close,
                    "High": close + 1,
                    "Low": close - 1,
                    "Open": close,
                    "manip_bear": mb,
                }
            )
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df


def test_manip_day_bias_basic():
    """MANIP-Tage bearish, Non-MANIP-Tage bullish."""
    days = [
        # MANIP-Tag: Open=100 (09:30), Close=90 (15:59) → delta=-10
        {
            "date_str": "2024-01-02",
            "bars": [
                (9.5, 100, True),  # 09:30 manip_bear
                (10.0, 95, False),
                (15.9, 90, False),  # RTH close
            ],
        },
        # Non-MANIP-Tag: Open=100, Close=120 → delta=+20
        {
            "date_str": "2024-01-03",
            "bars": [
                (9.5, 100, False),
                (10.0, 110, False),
                (15.9, 120, False),
            ],
        },
    ]
    df = _make_manip_df(days)
    from sb.inspect import analyze_manip_day_bias

    result = analyze_manip_day_bias(df, signal_col="manip_bear")

    assert result["manip_active"]["n"] == 1
    assert result["manip_active"]["avg_delta"] == -10.0
    assert result["manip_active"]["pct_bearish"] == 100.0

    assert result["no_manip"]["n"] == 1
    assert result["no_manip"]["avg_delta"] == 20.0
    assert result["no_manip"]["pct_bullish"] == 100.0


def _make_session_zone_df(signal_hour_ny, break_offset, return_offset_or_none):
    """Eine Zone die in signal_hour_ny entsteht, bei break_offset bricht.

    return_offset_or_none: Bars nach break bis Return, oder None (kein Return).
    """
    import pytz

    ny = pytz.timezone("US/Eastern")

    # Signal-Bar
    base = pd.Timestamp(
        f"2024-01-02 {int(signal_hour_ny):02d}:{int((signal_hour_ny % 1) * 60):02d}:00"
    )
    base = base.tz_localize(ny).tz_convert("UTC")

    n = break_offset + (return_offset_or_none or 5) + 5
    idx = pd.date_range(base, periods=n, freq="1min")

    fvg_bull = [False] * n
    fvg_bull[0] = True
    z_high, z_low = 103.0, 100.0

    highs = [104.0] * n
    lows = [104.0] * n
    filled = [False] * n

    # Break at break_offset: Low goes below z_low
    lows[break_offset] = 98.0
    highs[break_offset] = 101.0
    filled[break_offset] = True

    # Subsequent bars: below zone
    for i in range(break_offset + 1, n):
        lows[i] = 96.0
        highs[i] = 99.0
        filled[i] = True

    # Return if specified
    if return_offset_or_none is not None:
        ret_i = break_offset + return_offset_or_none
        highs[ret_i] = 101.0  # High >= bull_low = 100

    bull_high = [z_high if i >= 0 else None for i in range(n)]
    bull_low = [z_low if i >= 0 else None for i in range(n)]

    data = {
        "High": highs,
        "Low": lows,
        "Open": lows,
        "Close": highs,
        "fvg_bull": fvg_bull,
        "fvg_bear": [False] * n,
        "fvg_bull_high": bull_high,
        "fvg_bull_low": bull_low,
        "fvg_bull_filled": filled,
        "fvg_bear_high": [None] * n,
        "fvg_bear_low": [None] * n,
        "fvg_bear_filled": [False] * n,
    }
    return pd.DataFrame(data, index=idx)


def test_session_break_return_am_returns():
    """AM-Zone bricht und kehrt innerhalb 30 Bars zurück."""
    df = _make_session_zone_df(
        signal_hour_ny=9.5, break_offset=5, return_offset_or_none=20
    )
    from sb.inspect import analyze_zone_session_break_return

    result = analyze_zone_session_break_return(
        df, "fvg", return_windows=[15, 30, 60, 120]
    )

    am = result["bull"]["AM"]
    assert am["n_through"] == 1
    assert am["return_pct_30"] == 100.0
    assert am["return_pct_15"] == 0.0  # return at bar 20, not within 15


def test_session_break_return_london_permanent():
    """London-Zone bricht und kehrt NICHT zurück."""
    df = _make_session_zone_df(
        signal_hour_ny=3.0, break_offset=5, return_offset_or_none=None
    )
    from sb.inspect import analyze_zone_session_break_return

    result = analyze_zone_session_break_return(
        df, "fvg", return_windows=[15, 30, 60, 120]
    )

    london = result["bull"]["London"]
    assert london["n_through"] == 1
    assert london["return_pct_120"] == 0.0
