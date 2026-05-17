"""Tests für sb/inspect.py – Baustein-Inspektor."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


import numpy as np  # noqa: F401 (used in test_zone_near_level_pct_correct)

from sb.inspect import (
    InspectResult,
    analyze_fvg_outcomes,
    analyze_level_outcomes,
    analyze_zone_near_level,
    analyze_zone_outcomes,  # noqa: F401
    analyze_zone_overlap_outcomes,
    build_heatmap,
    detect_level_columns,
    detect_signal_columns,
    detect_zone_prefixes,
    find_algo_file,
    group_signals_by_window,
    inspect_algo,
    run_algo,
    save_zone_research,
)


# ── find_algo_file ─────────────────────────────────────────────────────────────


def test_find_algo_file_exact_match(tmp_path):
    d = tmp_path / "lib"
    d.mkdir()
    (d / "5a. FVG 2Tage.py").write_text("# algo")
    (d / "1. ATR Standard.py").write_text("# algo")

    result = find_algo_file("FVG 2Tage", [d])
    assert result is not None
    assert result.name == "5a. FVG 2Tage.py"


def test_find_algo_file_case_insensitive(tmp_path):
    d = tmp_path / "lib"
    d.mkdir()
    (d / "3. FVG Standard.py").write_text("# algo")

    result = find_algo_file("fvg standard", [d])
    assert result is not None
    assert result.name == "3. FVG Standard.py"


def test_find_algo_file_not_found(tmp_path):
    d = tmp_path / "lib"
    d.mkdir()
    (d / "1. ATR Standard.py").write_text("# algo")

    result = find_algo_file("FVG", [d])
    assert result is None


def test_find_algo_file_multiple_dirs(tmp_path):
    d1 = tmp_path / "lib1"
    d2 = tmp_path / "lib2"
    d1.mkdir()
    d2.mkdir()
    (d2 / "2. ATR Session.py").write_text("# algo")

    result = find_algo_file("ATR Session", [d1, d2])
    assert result is not None
    assert result.name == "2. ATR Session.py"


# ── run_algo ───────────────────────────────────────────────────────────────────


def _make_algo(tmp_path: Path, content: str, name: str = "test_algo.py") -> Path:
    f = tmp_path / name
    f.write_text(content)
    return f


def test_run_algo_calls_run(tmp_path):
    algo = _make_algo(
        tmp_path,
        """
import pandas as pd
def run(df):
    out = df.copy()
    out["test_signal"] = 1
    return out
""",
    )
    df = pd.DataFrame({"High": [1.0], "Low": [0.5], "Close": [0.8]})
    result = run_algo(algo, df)
    assert "test_signal" in result.columns
    assert result["test_signal"].iloc[0] == 1


def test_run_algo_missing_run_raises(tmp_path):
    algo = _make_algo(tmp_path, "# kein run()")
    df = pd.DataFrame({"High": [1.0], "Low": [0.5], "Close": [0.8]})
    with pytest.raises(AttributeError, match="kein run"):
        run_algo(algo, df)


# ── detect_signal_columns ──────────────────────────────────────────────────────


def test_detect_signal_columns_bool():
    df = pd.DataFrame(
        {
            "High": [1.0, 2.0],
            "Low": [0.5, 1.0],
            "Close": [0.8, 1.5],
            "Open": [0.6, 1.2],
            "fvg_bull": [True, False],
            "fvg_bear": [False, True],
        }
    )
    result = detect_signal_columns(df)
    assert set(result) == {"fvg_bull", "fvg_bear"}


def test_detect_signal_columns_binary_int():
    df = pd.DataFrame(
        {
            "High": [1.0],
            "Low": [0.5],
            "Close": [0.8],
            "Open": [0.6],
            "signal_bull": [1],
            "signal_bear": [0],
            "atr": [12.5],
        }
    )
    result = detect_signal_columns(df)
    assert "signal_bull" in result
    assert "atr" not in result


def test_detect_signal_columns_excludes_ohlcv():
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [0.0],
            "close": [1.0],
            "volume": [1000.0],
            "Open": [1.0],
            "High": [1.0],
            "Low": [0.0],
            "Close": [1.0],
            "fvg_bull": [1],
        }
    )
    result = detect_signal_columns(df)
    assert result == ["fvg_bull"]


def test_detect_signal_columns_empty():
    df = pd.DataFrame(
        {
            "High": [1.0],
            "Low": [0.5],
            "Close": [0.8],
            "atr": [12.5],
        }
    )
    result = detect_signal_columns(df)
    assert result == []


# ── group_signals_by_window ────────────────────────────────────────────────────


def _make_signal_df(
    timestamps_ny: list[str], bull: list[int], bear: list[int]
) -> pd.DataFrame:
    idx = pd.to_datetime(timestamps_ny).tz_localize("US/Eastern")
    return pd.DataFrame({"fvg_bull": bull, "fvg_bear": bear}, index=idx)


def test_group_signals_by_window_basic():
    df = _make_signal_df(
        ["2024-01-02 09:35", "2024-01-02 09:45", "2024-01-02 10:05"],
        bull=[1, 1, 0],
        bear=[0, 0, 1],
    )
    rows = group_signals_by_window(df, ["fvg_bull", "fvg_bear"], window_minutes=30)
    window_map = {r["window"]: r for r in rows}
    assert window_map["09:30"]["bull"] == 2
    assert window_map["09:30"]["bear"] == 0
    assert window_map["10:00"]["bull"] == 0
    assert window_map["10:00"]["bear"] == 1


def test_group_signals_by_window_rate():
    """rate_per_day = total / Anzahl_Tage."""
    df = _make_signal_df(
        ["2024-01-02 09:35", "2024-01-03 09:40"],  # 2 verschiedene Tage
        bull=[1, 1],
        bear=[0, 0],
    )
    rows = group_signals_by_window(df, ["fvg_bull", "fvg_bear"], window_minutes=30)
    w = next(r for r in rows if r["window"] == "09:30")
    # 2 Signale über 2 Tage → rate = 1.0
    assert w["rate_per_day"] == 1.0


def test_group_signals_by_window_total():
    df = _make_signal_df(["2024-01-02 09:35"], bull=[1], bear=[1])
    rows = group_signals_by_window(df, ["fvg_bull", "fvg_bear"], window_minutes=30)
    w = next(r for r in rows if r["window"] == "09:30")
    assert w["total"] == 2


def test_group_signals_by_window_empty_windows_excluded():
    df = _make_signal_df(["2024-01-02 09:35"], bull=[1], bear=[0])
    rows = group_signals_by_window(df, ["fvg_bull", "fvg_bear"], window_minutes=30)
    assert all(r["total"] > 0 for r in rows)
    assert len(rows) == 1


# ── build_heatmap ──────────────────────────────────────────────────────────────


def test_build_heatmap_weekday():
    """Heatmap gruppiert nach Wochentag."""
    # 2024-01-02 = Dienstag, 2024-01-03 = Mittwoch
    df = _make_signal_df(
        ["2024-01-02 09:35", "2024-01-03 09:40"],
        bull=[1, 1],
        bear=[0, 0],
    )
    hm = build_heatmap(df, ["fvg_bull", "fvg_bear"], window_minutes=30)
    assert "09:30" in hm
    assert hm["09:30"].get("Di", 0) == 1
    assert hm["09:30"].get("Mi", 0) == 1


# ── inspect_algo ──────────────────────────────────────────────────────────────


def _make_parquet(tmp_path: Path) -> Path:
    idx = pd.to_datetime(
        [
            "2024-01-02 14:35",
            "2024-01-02 14:36",
            "2024-01-03 15:05",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=idx,
    )
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    return p


def test_inspect_algo_returns_result(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    algo_dir = tmp_path / "lib"
    algo_dir.mkdir()
    (algo_dir / "3. FVG Standard.py").write_text("""
import pandas as pd
def run(df):
    out = df.copy()
    out["fvg_bull"] = 1
    out["fvg_bear"] = 0
    return out
""")
    result = inspect_algo("FVG Standard", [algo_dir], parquet_path, window_minutes=30)
    assert isinstance(result, InspectResult)
    assert result.algo_file.name == "3. FVG Standard.py"
    assert result.total_bars == 3
    assert result.total_days == 2
    # fvg_bull und fvg_bear sind beide Binärspalten → beide erkannt
    assert "fvg_bull" in result.signal_cols
    total = sum(w["total"] for w in result.windows)
    assert total == 3


def test_inspect_algo_with_heatmap(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    algo_dir = tmp_path / "lib"
    algo_dir.mkdir()
    (algo_dir / "test_algo.py").write_text("""
def run(df):
    out = df.copy()
    out["sig_bull"] = 1
    return out
""")
    result = inspect_algo(
        "test_algo", [algo_dir], parquet_path, window_minutes=30, with_heatmap=True
    )
    assert isinstance(result.heatmap, dict)
    assert len(result.heatmap) > 0


def test_inspect_algo_not_found(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    with pytest.raises(FileNotFoundError, match="kein Algo"):
        inspect_algo("NONEXISTENT_ALGO", [tmp_path], parquet_path)


# ── _make_event_masks + events_only ───────────────────────────────────────────


def test_make_event_masks_rising_edge():
    """Event-Maske ist nur beim 0→1 Übergang True."""
    from sb.inspect import _make_event_masks  # noqa: PLC0415

    idx = pd.to_datetime(
        ["2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:32", "2024-01-02 09:33"]
    ).tz_localize("US/Eastern")
    df = pd.DataFrame({"fvg_bull": [0, 1, 1, 0]}, index=idx)
    masks = _make_event_masks(df, ["fvg_bull"])
    # Nur Bar 1 (0→1) ist True, Bar 2 (1→1) ist False
    assert masks["fvg_bull"].tolist() == [False, True, False, False]


def test_group_signals_events_only_less_than_normal():
    """events_only zählt weniger Signale als Standardmodus bei persistentem Signal."""
    # 3 Bars alle fvg_bull=1 → Standard: 3, Events-Only: 1
    idx = pd.to_datetime(
        ["2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:32"]
    ).tz_localize("US/Eastern")
    df = pd.DataFrame({"fvg_bull": [1, 1, 1]}, index=idx)

    normal = group_signals_by_window(
        df, ["fvg_bull"], window_minutes=30, events_only=False
    )
    events = group_signals_by_window(
        df, ["fvg_bull"], window_minutes=30, events_only=True
    )

    normal_total = sum(w["total"] for w in normal)
    events_total = sum(w["total"] for w in events)
    assert normal_total == 3
    assert events_total == 1  # nur der erste Bar


def test_group_signals_events_only_two_separate_events():
    """Zwei getrennte Signal-Blöcke → events_only zählt 2."""
    idx = pd.to_datetime(
        [
            "2024-01-02 09:30",
            "2024-01-02 09:31",  # Block 1
            "2024-01-02 09:32",  # Lücke (0)
            "2024-01-02 09:33",
            "2024-01-02 09:34",  # Block 2
        ]
    ).tz_localize("US/Eastern")
    df = pd.DataFrame({"fvg_bull": [1, 1, 0, 1, 1]}, index=idx)

    events = group_signals_by_window(
        df, ["fvg_bull"], window_minutes=30, events_only=True
    )
    total = sum(w["total"] for w in events)
    assert total == 2  # je 1 Event pro Block


# ── analyze_fvg_outcomes ───────────────────────────────────────────────────────


def _make_fvg_df(
    n_bars: int = 5,
    bull_formation_bar: int = 1,
    bear_formation_bar: int | None = None,
    low_values: list[float] | None = None,
    high_values: list[float] | None = None,
    bull_zone_high: float = 103.0,
    bull_zone_low: float = 100.0,
    bear_zone_low: float = 97.0,
    bear_zone_high: float = 100.0,
    bull_filled: list[bool] | None = None,
    bear_filled: list[bool] | None = None,
) -> pd.DataFrame:
    """Erstellt DataFrame mit FVG-Spalten (manuell gesetzt, kein Algo-Lauf nötig)."""
    idx = pd.date_range("2024-01-02 09:30", periods=n_bars, freq="1min", tz="UTC")
    lows = low_values or [105.0] * n_bars
    highs = high_values or [110.0] * n_bars

    fvg_bull = [False] * n_bars
    fvg_bull[bull_formation_bar] = True

    bull_high_col = [None] * n_bars
    bull_low_col = [None] * n_bars
    for i in range(bull_formation_bar, n_bars):
        bull_high_col[i] = bull_zone_high
        bull_low_col[i] = bull_zone_low

    fvg_bear = [False] * n_bars
    bear_low_col = [None] * n_bars
    bear_high_col = [None] * n_bars
    if bear_formation_bar is not None:
        fvg_bear[bear_formation_bar] = True
        for i in range(bear_formation_bar, n_bars):
            bear_low_col[i] = bear_zone_low
            bear_high_col[i] = bear_zone_high

    bf = bull_filled or [False] * n_bars
    bef = bear_filled or [False] * n_bars

    return pd.DataFrame(
        {
            "High": highs,
            "Low": lows,
            "Open": [100.0] * n_bars,
            "Close": [100.0] * n_bars,
            "fvg_bull": fvg_bull,
            "fvg_bear": fvg_bear,
            "fvg_bull_high": bull_high_col,
            "fvg_bull_low": bull_low_col,
            "fvg_bull_ce": [None] * n_bars,
            "fvg_bull_filled": bf,
            "fvg_bull_active": [False] * n_bars,
            "fvg_bear_low": bear_low_col,
            "fvg_bear_high": bear_high_col,
            "fvg_bear_ce": [None] * n_bars,
            "fvg_bear_filled": bef,
            "fvg_bear_active": [False] * n_bars,
        },
        index=idx,
    )


def test_analyze_fvg_outcomes_bull_bounce():
    """Preis berührt Bull-Zone von oben aber füllt sie nicht → Bounce."""
    # Zone: high=103, low=100
    # Bar 0: Formation (fvg_bull=True)
    # Bar 1: Low=102 → berührt (102 <= 103), nicht gefüllt (102 > 100)
    # Bar 2: Low=104 → wieder über Zone
    df = _make_fvg_df(
        n_bars=3,
        bull_formation_bar=0,
        low_values=[105.0, 102.0, 104.0],
        bull_zone_high=103.0,
        bull_zone_low=100.0,
        bull_filled=[False, False, False],
    )
    stats = analyze_fvg_outcomes(df)
    assert stats["bull"]["total_zones"] == 1
    assert stats["bull"]["bounce"] == 1
    assert stats["bull"]["through"] == 0
    assert stats["bull"]["bounce_pct"] == 100.0


def test_analyze_fvg_outcomes_bull_through():
    """Preis geht durch Bull-Zone → Durch."""
    # Bar 0: Formation
    # Bar 1: Low=102 → berührt Zone
    # Bar 2: Low=99 → unter zone_low=100 → filled
    df = _make_fvg_df(
        n_bars=3,
        bull_formation_bar=0,
        low_values=[105.0, 102.0, 99.0],
        bull_zone_high=103.0,
        bull_zone_low=100.0,
        bull_filled=[False, False, True],
    )
    stats = analyze_fvg_outcomes(df)
    assert stats["bull"]["through"] == 1
    assert stats["bull"]["bounce"] == 0
    assert stats["bull"]["through_pct"] == 100.0


def test_analyze_fvg_outcomes_bull_no_touch():
    """Preis kommt nie in die Zone → keine Berührung."""
    # Low bleibt immer >= 104 (über zone_high=103 → keine Berührung)
    df = _make_fvg_df(
        n_bars=3,
        bull_formation_bar=0,
        low_values=[105.0, 104.0, 106.0],
        bull_zone_high=103.0,
        bull_zone_low=100.0,
        bull_filled=[False, False, False],
    )
    stats = analyze_fvg_outcomes(df)
    assert stats["bull"]["no_touch"] == 1
    assert stats["bull"]["touches"] == 0


def test_analyze_fvg_outcomes_bear_bounce():
    """Preis berührt Bear-Zone von unten aber füllt sie nicht → Bounce."""
    # Bear Zone: low=97 (Boden), high=100 (Decke)
    # Bar 0: Formation (fvg_bear=True)
    # Bar 1: High=98 → berührt (98 >= 97), nicht gefüllt (98 < 100)
    df = _make_fvg_df(
        n_bars=3,
        bull_formation_bar=0,  # dummy bull
        bear_formation_bar=0,
        high_values=[90.0, 98.0, 95.0],
        low_values=[85.0, 93.0, 90.0],
        bear_zone_low=97.0,
        bear_zone_high=100.0,
        bear_filled=[False, False, False],
    )
    stats = analyze_fvg_outcomes(df)
    assert stats["bear"]["bounce"] == 1
    assert stats["bear"]["through"] == 0


def test_analyze_fvg_outcomes_bear_through():
    """Preis geht durch Bear-Zone → Durch."""
    df = _make_fvg_df(
        n_bars=3,
        bull_formation_bar=0,
        bear_formation_bar=0,
        high_values=[90.0, 98.0, 101.0],
        low_values=[85.0, 93.0, 96.0],
        bear_zone_low=97.0,
        bear_zone_high=100.0,
        bear_filled=[False, False, True],
    )
    stats = analyze_fvg_outcomes(df)
    assert stats["bear"]["through"] == 1
    assert stats["bear"]["bounce"] == 0


def test_analyze_fvg_outcomes_two_zones():
    """Zwei aufeinanderfolgende Zonen werden separat gezählt."""
    # Zone 1: Bar 0, berührt und Bounce (Bar 1: Low=102)
    # Zone 2: Bar 2, berührt und Durch (Bar 3: Low=99, filled)
    n = 5
    df = _make_fvg_df(
        n_bars=n,
        bull_formation_bar=0,
        low_values=[105.0, 102.0, 105.0, 99.0, 110.0],
        bull_zone_high=103.0,
        bull_zone_low=100.0,
        bull_filled=[False, False, False, True, False],
    )
    # Zweite Zone manuell einsetzen
    df["fvg_bull"] = [True, False, True, False, False]
    df["fvg_bull_high"] = [103.0, 103.0, 103.0, 103.0, 103.0]
    df["fvg_bull_low"] = [100.0, 100.0, 100.0, 100.0, 100.0]

    stats = analyze_fvg_outcomes(df)
    assert stats["bull"]["total_zones"] == 2
    assert stats["bull"]["bounce"] == 1
    assert stats["bull"]["through"] == 1


def test_analyze_fvg_outcomes_missing_columns():
    """ValueError wenn FVG-Spalten fehlen."""
    df = pd.DataFrame({"High": [1.0], "Low": [0.5], "fvg_bull": [False]})
    with pytest.raises(ValueError, match="Spalten fehlen"):
        analyze_fvg_outcomes(df)


def test_inspect_algo_events_only_flag(tmp_path):
    """inspect_algo mit events_only=True liefert weniger Signale als ohne."""
    parquet_path = _make_parquet(tmp_path)
    algo_dir = tmp_path / "lib"
    algo_dir.mkdir()
    # Algo setzt fvg_bull auf allen 3 Bars = 1 (persistent)
    (algo_dir / "fvg.py").write_text("""
def run(df):
    out = df.copy()
    out["fvg_bull"] = 1
    return out
""")
    normal = inspect_algo("fvg", [algo_dir], parquet_path, events_only=False)
    events = inspect_algo("fvg", [algo_dir], parquet_path, events_only=True)

    normal_total = sum(w["total"] for w in normal.windows)
    events_total = sum(w["total"] for w in events.windows)

    assert normal_total == 3  # alle 3 Bars
    assert events_total == 1  # nur erster Bar
    assert events.events_only is True
    assert normal.events_only is False


# ── detect_zone_prefixes ────────────────────────────────────────────────────────


def test_detect_zone_prefixes_finds_fvg():
    """fvg-Prefix wird erkannt wenn alle 4 Pflicht-Spalten vorhanden sind."""
    df = pd.DataFrame(
        {
            "fvg_bull": [False],
            "fvg_bull_high": [103.0],
            "fvg_bull_low": [100.0],
            "fvg_bull_filled": [False],
            "fvg_bear": [False],
            "fvg_bear_low": [97.0],
            "fvg_bear_high": [100.0],
            "fvg_bear_filled": [False],
            "Close": [101.0],
        }
    )
    prefixes = detect_zone_prefixes(df)
    assert prefixes == ["fvg"]


def test_detect_zone_prefixes_incomplete_columns():
    """Prefix wird NICHT erkannt wenn eine Pflicht-Spalte fehlt."""
    df = pd.DataFrame(
        {
            "fvg_bull": [False],
            "fvg_bull_high": [103.0],
            # fvg_bull_low fehlt absichtlich
            "fvg_bull_filled": [False],
            "Close": [101.0],
        }
    )
    prefixes = detect_zone_prefixes(df)
    assert prefixes == []


def test_detect_zone_prefixes_multiple():
    """Zwei Prefixe (fvg + ifvg) werden beide erkannt."""
    df = pd.DataFrame(
        {
            "fvg_bull": [False],
            "fvg_bull_high": [103.0],
            "fvg_bull_low": [100.0],
            "fvg_bull_filled": [False],
            "fvg_bear": [False],
            "fvg_bear_low": [97.0],
            "fvg_bear_high": [100.0],
            "fvg_bear_filled": [False],
            "ifvg_bull": [False],
            "ifvg_bull_high": [105.0],
            "ifvg_bull_low": [102.0],
            "ifvg_bull_filled": [False],
            "ifvg_bear": [False],
            "ifvg_bear_low": [95.0],
            "ifvg_bear_high": [98.0],
            "ifvg_bear_filled": [False],
        }
    )
    prefixes = detect_zone_prefixes(df)
    assert set(prefixes) == {"fvg", "ifvg"}
    assert len(prefixes) == 2


# ── Penetrations-Tiefe ──────────────────────────────────────────────────────────


def test_depth_stops_at_fill_bar():
    """depth_pts endet am ersten Fill-Bar, nicht am Ende der Episode.

    Zone: high=108, low=105 (Größe = 3 Punkte)
    Bar 0: Formation
    Bar 1: Low=107 → in Zone (107 <= 108), nicht gefüllt
    Bar 2: Low=104 → unter Zone → filled=True  (Tiefe = 108-104 = 4 Punkte)
    Bar 3: Low=100 → weiter unten, aber NACH Fill → darf NICHT zählen
    """
    df = _make_fvg_df(
        n_bars=4,
        bull_formation_bar=0,
        low_values=[110.0, 107.0, 104.0, 100.0],
        bull_zone_high=108.0,
        bull_zone_low=105.0,
        bull_filled=[False, False, True, True],
    )
    stats = analyze_fvg_outcomes(df)
    # Tiefe = entry(108) - min_low_bis_fill(104) = 4 Punkte
    assert stats["bull"]["depth"]["depth_pts_mean"] == 4.0
    # Zone-Größe = 3, depth_pct = 4/3*100 = 133.3%
    assert stats["bull"]["depth"]["depth_pct_mean"] == pytest.approx(133.3, abs=0.1)


def test_depth_bounce_zone_full_episode():
    """Bounce-Zone (nie gefüllt): Tiefe über gesamte Episode gemessen.

    Zone: high=108, low=105 (Größe = 3 Punkte)
    Bar 0: Formation
    Bar 1: Low=107 → berührt (bounce)
    Bar 2: Low=106 → noch tiefer, nie gefüllt
    Tiefe = 108-106 = 2 Punkte = 66.7%
    """
    df = _make_fvg_df(
        n_bars=3,
        bull_formation_bar=0,
        low_values=[110.0, 107.0, 106.0],
        bull_zone_high=108.0,
        bull_zone_low=105.0,
        bull_filled=[False, False, False],
    )
    stats = analyze_fvg_outcomes(df)
    assert stats["bull"]["bounce"] == 1
    assert stats["bull"]["depth"]["depth_pts_mean"] == 2.0
    assert stats["bull"]["depth"]["depth_pct_mean"] == pytest.approx(66.7, abs=0.1)


def test_depth_buckets_lt25():
    """Zone nur minimal berührt (<25%) → landet in pct_lt25.

    Zone: high=108, low=100 (Größe = 8 Punkte)
    Bar 1: Low=107.5 → Tiefe = 0.5 Punkte = 6.25% → Bucket <25%
    """
    df = _make_fvg_df(
        n_bars=3,
        bull_formation_bar=0,
        low_values=[110.0, 107.5, 110.0],
        bull_zone_high=108.0,
        bull_zone_low=100.0,
        bull_filled=[False, False, False],
    )
    stats = analyze_fvg_outcomes(df)
    assert stats["bull"]["depth"]["pct_lt25"] == 100.0
    assert stats["bull"]["depth"]["pct_25_50"] == 0.0
    assert stats["bull"]["depth"]["pct_gte100"] == 0.0


# ── save_zone_research ──────────────────────────────────────────────────────────


def test_save_zone_research_creates_files(tmp_path):
    """save_zone_research erstellt .json und .md in _research/ neben algo_file."""
    algo_file = tmp_path / "lib" / "3. FVG Standard.py"
    algo_file.parent.mkdir()
    algo_file.write_text("# dummy")

    stats = {
        "fvg": {
            "bull": {
                "total_zones": 10,
                "no_touch": 3,
                "touches": 7,
                "bounce": 2,
                "through": 5,
                "bounce_pct": 28.6,
                "through_pct": 71.4,
                "depth": {
                    "depth_pts_mean": 4.0,
                    "depth_pts_median": 3.5,
                    "depth_pct_mean": 133.3,
                    "depth_pct_median": 116.7,
                    "pct_lt25": 10.0,
                    "pct_25_50": 5.0,
                    "pct_50_75": 5.0,
                    "pct_75_100": 8.0,
                    "pct_gte100": 72.0,
                },
            },
            "bear": {
                "total_zones": 8,
                "no_touch": 2,
                "touches": 6,
                "bounce": 1,
                "through": 5,
                "bounce_pct": 16.7,
                "through_pct": 83.3,
                "depth": {},
            },
        }
    }
    data_info = {"bars": 1000, "from": "2024-01-01", "to": "2026-01-01", "days": 500}

    research_dir = save_zone_research(stats, algo_file, data_info)

    assert research_dir.exists()
    assert research_dir.is_dir()

    json_files = list(research_dir.glob("*.json"))
    md_files = list(research_dir.glob("*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1


def test_save_zone_research_json_content(tmp_path):
    """Die JSON-Datei enthält algo, generated, data und zones korrekt."""
    import json as _json

    algo_file = tmp_path / "lib" / "myalgo.py"
    algo_file.parent.mkdir()
    algo_file.write_text("# dummy")

    stats = {
        "fvg": {
            "bull": {
                "total_zones": 5,
                "no_touch": 1,
                "touches": 4,
                "bounce": 2,
                "through": 2,
                "bounce_pct": 50.0,
                "through_pct": 50.0,
                "depth": {},
            },
            "bear": {
                "total_zones": 3,
                "no_touch": 0,
                "touches": 3,
                "bounce": 1,
                "through": 2,
                "bounce_pct": 33.3,
                "through_pct": 66.7,
                "depth": {},
            },
        }
    }
    data_info = {"bars": 500, "from": "2025-01-01", "to": "2026-01-01", "days": 250}

    research_dir = save_zone_research(stats, algo_file, data_info)

    json_file = next(research_dir.glob("*.json"))
    payload = _json.loads(json_file.read_text())

    assert payload["algo"] == "myalgo.py"
    assert payload["data"]["bars"] == 500
    assert payload["zones"]["fvg"]["bull"]["total_zones"] == 5
    assert "generated" in payload


# ── detect_level_columns ────────────────────────────────────────────────────────


def test_detect_level_columns_finds_prev_day():
    """Erkennt prev_day_high und prev_day_low als Level-Spalten."""
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "prev_day_high": [100.0],
            "prev_day_low": [90.0],
        }
    )
    cols = detect_level_columns(df)
    assert "prev_day_high" in cols
    assert "prev_day_low" in cols


def test_detect_level_columns_excludes_ohlcv():
    """OHLCV-Spalten werden nicht als Level erkannt."""
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [1000.0],
        }
    )
    cols = detect_level_columns(df)
    assert cols == []


def test_detect_level_columns_excludes_zone_columns():
    """Zone-Spalten (fvg_bull_high) werden nicht als Level erkannt."""
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "fvg_bull_high": [100.0],
            "fvg_bull_low": [90.0],
            "prev_week_high": [105.0],
        }
    )
    cols = detect_level_columns(df)
    assert "fvg_bull_high" not in cols
    assert "prev_week_high" in cols


# ── analyze_level_outcomes ──────────────────────────────────────────────────────


def test_analyze_level_bounce_from_below():
    """Preis kommt von unten, berührt Level, schließt darunter → Bounce."""
    df = pd.DataFrame(
        {
            "High": [95.0, 102.0, 96.0],
            "Low": [85.0, 98.0, 88.0],
            "Close": [90.0, 95.0, 92.0],
            "prev_day_high": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    stats = analyze_level_outcomes(df, "prev_day_high")
    assert stats["touches"] == 1
    assert stats["bounce"] == 1
    assert stats["through"] == 0


def test_analyze_level_through_from_below():
    """Preis kommt von unten, schließt über Level → Durch."""
    df = pd.DataFrame(
        {
            "High": [95.0, 105.0, 108.0],
            "Low": [85.0, 98.0, 99.0],
            "Close": [90.0, 103.0, 106.0],
            "prev_day_high": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    stats = analyze_level_outcomes(df, "prev_day_high")
    assert stats["touches"] == 1
    assert stats["bounce"] == 0
    assert stats["through"] == 1


def test_analyze_level_bounce_from_above():
    """Preis kommt von oben, berührt Level, schließt darüber → Bounce."""
    df = pd.DataFrame(
        {
            "High": [115.0, 102.0, 112.0],
            "Low": [105.0, 98.0, 102.0],
            "Close": [110.0, 105.0, 108.0],
            "prev_day_low": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    stats = analyze_level_outcomes(df, "prev_day_low")
    assert stats["touches"] == 1
    assert stats["bounce"] == 1
    assert stats["through"] == 0


def test_analyze_level_no_touch():
    """Preis berührt Level nie → 0 Touches."""
    df = pd.DataFrame(
        {
            "High": [80.0, 82.0, 81.0],
            "Low": [70.0, 71.0, 70.0],
            "Close": [75.0, 76.0, 75.0],
            "prev_day_high": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    stats = analyze_level_outcomes(df, "prev_day_high")
    assert stats["touches"] == 0


def test_analyze_level_episode_starts_as_touch():
    """Erste Bar der Episode ist bereits eine Touch-Bar (k==i).

    Level ändert sich in Bar2 von 100→110.
    Bar0+Bar1: Low=85,85 < 100 → kein Touch
    Bar2: erste Bar der neuen Episode (k==i), Level=110
          Hi=112 >= 110 → Touch (von oben), Lo=108 < 110 → Wick unter Level
          Close=105 < 110 → Bounce von oben
    """
    df = pd.DataFrame(
        {
            "High": [95.0, 95.0, 112.0],
            "Low": [85.0, 85.0, 108.0],
            "Close": [90.0, 90.0, 105.0],
            "prev_day_high": [100.0, 100.0, 110.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    stats = analyze_level_outcomes(df, "prev_day_high")
    # Bar0+Bar1: kein Touch an 100 (Hi=95 < 100)
    # Bar2: erste Bar der neuen Episode (k==i), Hi=112 >= 110 → Touch,
    #       Lo=108 < 110 → start_above = False (von unten?!)
    #       Nein! Von oben: High > Level, Low < Level → Wick eindringend
    #       Close=105 < 110 → Bounce nach unten
    assert stats["touches"] == 1
    assert stats["bounce"] == 1
    assert stats["through"] == 0


def test_analyze_level_new_episode_resets():
    """Wenn Level-Wert sich ändert (neuer Tag), ist das eine neue Episode."""
    df = pd.DataFrame(
        {
            "High": [102.0, 102.0, 112.0],
            "Low": [98.0, 98.0, 108.0],
            "Close": [103.0, 96.0, 110.0],
            "prev_day_high": [100.0, 100.0, 110.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    stats = analyze_level_outcomes(df, "prev_day_high")
    assert stats["touches"] == 2
    assert stats["through"] == 2


# ── analyze_zone_near_level Tests ───────────────────────────────────────────


def test_zone_near_level_near_bounce():
    """FVG-Zone nahe am Level → bounce wird in 'near' gezählt."""
    # Zone: bull_high=105, bull_low=95 → mid=100, level=102 → diff=2 ≤ 20 → near
    # Bar0: Zone aktiv, kein Touch
    # Bar1: Low=97, High=107, Close=96 → Touch (97<=100<=107), Close<zone_mid → Bounce
    df = pd.DataFrame(
        {
            "High": [90.0, 107.0, 88.0],
            "Low": [80.0, 97.0, 78.0],
            "Close": [85.0, 96.0, 83.0],
            "fvg_bull_high": [105.0, 105.0, 105.0],
            "fvg_bull_low": [95.0, 95.0, 95.0],
            "fvg_bull_filled": [0.0, 0.0, 0.0],
            "prev_day_high": [102.0, 102.0, 102.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    result = analyze_zone_near_level(df, "fvg", "prev_day_high", proximity_pts=20.0)
    assert result["near"]["touches"] == 1
    assert result["near"]["bounce"] == 1
    assert result["near"]["through"] == 0
    assert result["far"]["touches"] == 0


def test_zone_near_level_far_through():
    """FVG-Zone weit vom Level → through wird in 'far' gezählt."""
    # Zone mid=100, level=200 → diff=100 > 20 → far
    # Bar1: Touch, Close > zone_mid → Durch
    df = pd.DataFrame(
        {
            "High": [90.0, 107.0, 110.0],
            "Low": [80.0, 97.0, 100.0],
            "Close": [85.0, 105.0, 108.0],
            "fvg_bull_high": [105.0, 105.0, 105.0],
            "fvg_bull_low": [95.0, 95.0, 95.0],
            "fvg_bull_filled": [0.0, 0.0, 0.0],
            "prev_day_high": [200.0, 200.0, 200.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    result = analyze_zone_near_level(df, "fvg", "prev_day_high", proximity_pts=20.0)
    assert result["far"]["touches"] == 1
    assert result["far"]["through"] == 1
    assert result["near"]["touches"] == 0


def test_zone_near_level_nan_level_counts_as_far():
    """NaN im Level → zählt als 'far'."""
    df = pd.DataFrame(
        {
            "High": [90.0, 107.0, 88.0],
            "Low": [80.0, 97.0, 78.0],
            "Close": [85.0, 96.0, 83.0],
            "fvg_bull_high": [105.0, 105.0, 105.0],
            "fvg_bull_low": [95.0, 95.0, 95.0],
            "fvg_bull_filled": [0.0, 0.0, 0.0],
            "prev_day_high": [float("nan"), float("nan"), float("nan")],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    result = analyze_zone_near_level(df, "fvg", "prev_day_high", proximity_pts=20.0)
    assert result["near"]["touches"] == 0
    assert result["far"]["touches"] == 1


def test_zone_near_level_pct_correct():
    """bounce_pct und through_pct werden korrekt berechnet."""
    # 2 Episoden: beide near, 1 bounce + 1 through
    df = pd.DataFrame(
        {
            "High": [90.0, 107.0, 90.0, 90.0, 107.0, 90.0],
            "Low": [80.0, 97.0, 80.0, 80.0, 97.0, 80.0],
            "Close": [85.0, 96.0, 85.0, 85.0, 105.0, 85.0],  # Bar1=bounce, Bar4=through
            "fvg_bull_high": [105.0, 105.0, float("nan"), 105.0, 105.0, float("nan")],
            "fvg_bull_low": [95.0, 95.0, float("nan"), 95.0, 95.0, float("nan")],
            "fvg_bull_filled": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "prev_day_high": [102.0, 102.0, 102.0, 102.0, 102.0, 102.0],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC"),
    )
    result = analyze_zone_near_level(df, "fvg", "prev_day_high", proximity_pts=20.0)
    assert result["near"]["touches"] == 2
    assert result["near"]["bounce_pct"] == 50.0
    assert result["near"]["through_pct"] == 50.0


# ── analyze_zone_overlap_outcomes ────────────────────────────────────────────


def test_zone_overlap_single_bull():
    """Einzelne Bull-Zone ohne Überlappung → single."""
    df = pd.DataFrame(
        {
            "High": [90.0, 107.0, 88.0],
            "Low": [80.0, 97.0, 78.0],
            "Close": [85.0, 96.0, 83.0],
            "fvg_bull_high": [105.0, 105.0, np.nan],
            "fvg_bull_low": [95.0, 95.0, np.nan],
            "fvg_bull_filled": [0.0, 0.0, 1.0],
            "fvg_bear_high": [np.nan, np.nan, np.nan],
            "fvg_bear_low": [np.nan, np.nan, np.nan],
            "fvg_bear_filled": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    result = analyze_zone_overlap_outcomes(df, "fvg")
    assert result["bull"]["single"]["touches"] == 1
    assert result["bull"]["double"]["touches"] == 0


def test_zone_overlap_double_bull_same():
    """Bull×Bull auf gleichem TF nicht möglich → double_same immer 0.

    Hintergrund: bh[k] hat pro Bar nur einen Wert, daher kann nie eine
    ZWEITE Bull-Episode gleichzeitig aktiv sein. Getestet wird stattdessen,
    dass double_same korrekt auf 0 bleibt auch wenn zwei aufeinanderfolgende
    Bull-Episoden existieren (alte Episode endet bevor neue beginnt).
    """
    # Zone A: high=110, low=100 (Bars 0-1, dann filled)
    # Zone B: high=107, low=97  (Bars 2-3, neue Episode)
    # Auf keinem Bar sind beide gleichzeitig aktiv → double_same muss 0 sein
    df = pd.DataFrame(
        {
            "High": [95.0, 95.0, 108.0, 108.0],
            "Low": [85.0, 85.0, 99.0, 99.0],
            "Close": [90.0, 90.0, 96.0, 96.0],
            "fvg_bull_high": [
                110.0,
                110.0,
                107.0,
                107.0,
            ],  # Wert ändert sich → neue Episode ab Bar2
            "fvg_bull_low": [100.0, 100.0, 97.0, 97.0],
            "fvg_bull_filled": [0.0, 0.0, 0.0, 0.0],
            "fvg_bear_high": [np.nan, np.nan, np.nan, np.nan],
            "fvg_bear_low": [np.nan, np.nan, np.nan, np.nan],
            "fvg_bear_filled": [0.0, 0.0, 0.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC"),
    )
    result = analyze_zone_overlap_outcomes(df, "fvg")
    # Bull×Bull auf gleichem TF: double_same ist immer 0
    assert result["bull"]["double_same"]["touches"] == 0
    # Keine Bear-Zone aktiv → kein double_opposite → alles landet in single
    assert result["bull"]["single"]["touches"] >= 1
    assert result["bull"]["double"]["touches"] == 0


def test_zone_overlap_bull_bear_opposite():
    """Bull-Zone überlappt mit aktiver Bear-Zone → double_opposite."""
    # Bull: high=105, low=95 (aktiv)
    # Bear: high=102, low=92 (aktiv, überlappt mit Bull)
    # Bar1: Touch am Bull (low=97 <= 100 <= high=103), Bear aktiv und überlappt
    df = pd.DataFrame(
        {
            "High": [90.0, 103.0, 88.0],
            "Low": [80.0, 97.0, 78.0],
            "Close": [85.0, 93.0, 83.0],  # Close < zone_mid → Bounce
            "fvg_bull_high": [105.0, 105.0, 105.0],
            "fvg_bull_low": [95.0, 95.0, 95.0],
            "fvg_bull_filled": [0.0, 0.0, 0.0],
            "fvg_bear_high": [102.0, 102.0, 102.0],
            "fvg_bear_low": [92.0, 92.0, 92.0],
            "fvg_bear_filled": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    result = analyze_zone_overlap_outcomes(df, "fvg")
    assert result["bull"]["double_opposite"]["touches"] == 1
    assert result["bull"]["double"]["touches"] == 1
    assert result["bull"]["single"]["touches"] == 0


def test_zone_overlap_no_overlap():
    """Bull-Zone und Bear-Zone überlappen sich NICHT → beide single."""
    # Bull: high=110, low=100
    # Bear: high=85, low=75 (weit entfernt, keine Überlappung)
    df = pd.DataFrame(
        {
            "High": [95.0, 107.0, 88.0],
            "Low": [85.0, 97.0, 78.0],
            "Close": [90.0, 96.0, 83.0],
            "fvg_bull_high": [110.0, 110.0, 110.0],
            "fvg_bull_low": [100.0, 100.0, 100.0],
            "fvg_bull_filled": [0.0, 0.0, 0.0],
            "fvg_bear_high": [85.0, 85.0, 85.0],
            "fvg_bear_low": [75.0, 75.0, 75.0],
            "fvg_bear_filled": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    result = analyze_zone_overlap_outcomes(df, "fvg")
    assert result["bull"]["single"]["touches"] == 1
    assert result["bull"]["double"]["touches"] == 0


# ── CLI fvg-overlap Tests ────────────────────────────────────────────────────────


def _make_overlap_algo(tmp_path: Path, name: str = "test_fvg") -> Path:
    """Minimales Algo das Bull + Bear Zone-Spalten zurückgibt."""
    algo_dir = tmp_path / "algos"
    algo_dir.mkdir(exist_ok=True)
    algo_file = algo_dir / f"{name}.py"
    algo_file.write_text(
        "import pandas as pd\n"
        "\n"
        "def run(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df = df.copy()\n"
        "    # detect_zone_prefixes braucht _bull / _bear Spalten als Marker\n"
        "    df['fvg_bull'] = 1\n"
        "    df['fvg_bear'] = 1\n"
        "    # Bull zone: high=105, low=95 (never filled)\n"
        "    df['fvg_bull_high'] = 105.0\n"
        "    df['fvg_bull_low'] = 95.0\n"
        "    df['fvg_bull_filled'] = 0.0\n"
        "    # Bear zone: high=102, low=92 (overlaps bull zone)\n"
        "    df['fvg_bear_high'] = 102.0\n"
        "    df['fvg_bear_low'] = 92.0\n"
        "    df['fvg_bear_filled'] = 0.0\n"
        "    return df\n"
    )
    return algo_dir


def _make_overlap_data(tmp_path: Path) -> Path:
    """Minimale Preis-Daten mit einem Touch in der Zone."""
    dates = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 10,
            # Bar 3: High=103, Low=97 → Touch (97 <= 100 <= 103), Close=93 → Bounce
            "high": [90.0, 90.0, 90.0, 103.0] + [90.0] * 6,
            "low": [80.0, 80.0, 80.0, 97.0] + [80.0] * 6,
            "close": [85.0, 85.0, 85.0, 93.0] + [85.0] * 6,
            "volume": [100] * 10,
        },
        index=dates,
    )
    data_path = tmp_path / "bars.parquet"
    df.to_parquet(data_path)
    return data_path


def _make_sources_yaml(tmp_path: Path, data_path: Path) -> Path:
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        f"backtest_data:\n  path: {data_path}\n  holdout_start: '2025-01-01'\n"
    )
    return sources


def test_fvg_overlap_cli_basic(tmp_path):
    """fvg-overlap Command läuft durch und zeigt Tabelle."""
    from typer.testing import CliRunner
    from sb.cli import app

    algo_dir = _make_overlap_algo(tmp_path)
    data_path = _make_overlap_data(tmp_path)
    sources = _make_sources_yaml(tmp_path, data_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["fvg-overlap", "test_fvg", "--dir", str(algo_dir), "--sources", str(sources)],
    )
    assert result.exit_code == 0, result.output
    assert "Bull" in result.output
    assert "Bear" in result.output
    assert "Double" in result.output


def test_fvg_overlap_cli_data_flag(tmp_path):
    """fvg-overlap Command funktioniert mit --data Flag."""
    from typer.testing import CliRunner
    from sb.cli import app

    algo_dir = _make_overlap_algo(tmp_path)
    data_path = _make_overlap_data(tmp_path)
    sources = _make_sources_yaml(tmp_path, data_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "fvg-overlap",
            "test_fvg",
            "--dir",
            str(algo_dir),
            "--sources",
            str(sources),
            "--data",
            str(data_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "fvg" in result.output.lower()


def test_fvg_overlap_cli_algo_not_found(tmp_path):
    """fvg-overlap gibt Fehler wenn Algo nicht gefunden."""
    from typer.testing import CliRunner
    from sb.cli import app

    data_path = _make_overlap_data(tmp_path)
    sources = _make_sources_yaml(tmp_path, data_path)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "fvg-overlap",
            "nichtexistent",
            "--dir",
            str(empty_dir),
            "--sources",
            str(sources),
        ],
    )
    assert result.exit_code != 0
    assert "gefunden" in result.output.lower() or "kein" in result.output.lower()


# ── analyze_zone_mtf_nesting Tests ───────────────────────────────────────────────


def _make_mtf_df(
    n: int,
    bull_high: float,
    bull_low: float,
    bull_filled: float,
    bear_high: float,
    bear_low: float,
    bear_filled: float,
    freq: str = "1min",
    prefix: str = "fvg",
) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 09:30", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "High": [bull_high + 2] * n,
            "Low": [bull_low - 2] * n,
            "Close": [(bull_high + bull_low) / 2 - 1] * n,  # below zone_mid → Bounce
            f"{prefix}_bull_high": [bull_high] * n,
            f"{prefix}_bull_low": [bull_low] * n,
            f"{prefix}_bull_filled": [bull_filled] * n,
            f"{prefix}_bear_high": [bear_high] * n,
            f"{prefix}_bear_low": [bear_low] * n,
            f"{prefix}_bear_filled": [bear_filled] * n,
        },
        index=idx,
    )


def test_mtf_nesting_bull_nested_contained():
    """5min Bull-Zone enthält 1min Bull-Zone (contained) → nested-Bucket."""
    from sb.inspect import analyze_zone_mtf_nesting

    # HTF (5min): Bull zone 100-110
    df_htf = _make_mtf_df(3, bull_high=110.0, bull_low=100.0, bull_filled=0.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="5min")
    # LTF (1min): Bull zone 103-107 (inside 100-110)
    df_ltf = _make_mtf_df(15, bull_high=107.0, bull_low=103.0, bull_filled=0.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="1min")

    result = analyze_zone_mtf_nesting(df_htf, df_ltf, "fvg", nesting="contained")
    assert result["bull"]["nested"]["touches"] == 1
    assert result["bull"]["single"]["touches"] == 0


def test_mtf_nesting_bull_not_contained():
    """LTF Zone größer als HTF Zone → nicht nested (contained)."""
    from sb.inspect import analyze_zone_mtf_nesting

    # HTF: Bull zone 103-107 (klein)
    df_htf = _make_mtf_df(3, bull_high=107.0, bull_low=103.0, bull_filled=0.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="5min")
    # LTF: Bull zone 100-110 (größer → nicht contained)
    df_ltf = _make_mtf_df(15, bull_high=110.0, bull_low=100.0, bull_filled=0.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="1min")

    result = analyze_zone_mtf_nesting(df_htf, df_ltf, "fvg", nesting="contained")
    assert result["bull"]["nested"]["touches"] == 0
    assert result["bull"]["single"]["touches"] == 1


def test_mtf_nesting_ltf_filled():
    """LTF Zone gefüllt (filled=1.0) → zählt nicht als nested."""
    from sb.inspect import analyze_zone_mtf_nesting

    df_htf = _make_mtf_df(3, bull_high=110.0, bull_low=100.0, bull_filled=0.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="5min")
    # LTF zone filled → inactive
    df_ltf = _make_mtf_df(15, bull_high=107.0, bull_low=103.0, bull_filled=1.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="1min")

    result = analyze_zone_mtf_nesting(df_htf, df_ltf, "fvg", nesting="contained")
    assert result["bull"]["nested"]["touches"] == 0
    assert result["bull"]["single"]["touches"] == 1


def test_mtf_nesting_overlap_mode():
    """overlap-Modus: LTF überlappt teilweise → nested."""
    from sb.inspect import analyze_zone_mtf_nesting

    # HTF: Bull zone 103-107
    df_htf = _make_mtf_df(3, bull_high=107.0, bull_low=103.0, bull_filled=0.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="5min")
    # LTF: Bull zone 100-105 (überlappt mit 103-107 aber nicht contained)
    df_ltf = _make_mtf_df(15, bull_high=105.0, bull_low=100.0, bull_filled=0.0,
                          bear_high=np.nan, bear_low=np.nan, bear_filled=0.0, freq="1min")

    result_contained = analyze_zone_mtf_nesting(df_htf, df_ltf, "fvg", nesting="contained")
    result_overlap = analyze_zone_mtf_nesting(df_htf, df_ltf, "fvg", nesting="overlap")

    assert result_contained["bull"]["nested"]["touches"] == 0  # nicht contained
    assert result_overlap["bull"]["nested"]["touches"] == 1    # aber overlap
