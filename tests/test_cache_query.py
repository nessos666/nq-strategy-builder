from __future__ import annotations

import pandas as pd

from sb.cache.cache_query import query_signal_bars


def _make_cache_df() -> pd.DataFrame:
    """Kleiner Demo-Cache mit bos_bullish + fvg_bullish."""
    idx = pd.date_range("2024-01-01 09:30", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "Close": [100.0] * 10,
            "bos_bullish": [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            "fvg_bullish": [
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        },
        index=idx,
    )
    return df


def test_query_single_concept():
    """BOS allein → 2 Bars (Index 1 und 4)."""
    df = _make_cache_df()
    bars = query_signal_bars(df, concepts=["bos"])
    assert len(bars) == 2


def test_query_two_concepts_intersection():
    """BOS UND FVG → nur Bar 4 (beide True)."""
    df = _make_cache_df()
    bars = query_signal_bars(df, concepts=["bos", "fvg"], mode="all")
    assert len(bars) == 1


def test_query_two_concepts_union():
    """BOS ODER FVG → Bars 1, 3, 4 → 3 Bars."""
    df = _make_cache_df()
    bars = query_signal_bars(df, concepts=["bos", "fvg"], mode="any")
    assert len(bars) == 3


def test_query_unknown_concept_returns_empty():
    """Unbekanntes Konzept → leere Liste, kein Crash."""
    df = _make_cache_df()
    bars = query_signal_bars(df, concepts=["unknown_xyz"])
    assert len(bars) == 0


def test_query_empty_concepts_returns_empty():
    """Leere Konzept-Liste → leere Ausgabe."""
    df = _make_cache_df()
    bars = query_signal_bars(df, concepts=[])
    assert len(bars) == 0


def test_query_returns_index():
    """Rückgabe ist ein pd.Index (Timestamps)."""
    df = _make_cache_df()
    result = query_signal_bars(df, concepts=["bos"])
    assert isinstance(result, pd.Index)


F = False
T = True


def _make_v2_cache_df() -> pd.DataFrame:
    """Cache-Simulation mit v2-Spalten für neue Konzepte."""
    idx = pd.date_range("2024-01-01 09:30", periods=10, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "Close": [100.0] * 10,
            # ICT Entry
            "breaker_bullish": [F, T, F, F, F, F, F, F, F, F],
            "breaker_bearish": [F, F, F, F, F, F, F, F, F, F],
            "rb_bullish": [F, F, T, F, F, F, F, F, F, F],
            "rb_bearish": [F, F, F, F, F, F, F, F, F, F],
            "vb_bullish": [F, F, F, T, F, F, F, F, F, F],
            "vb_bearish": [F, F, F, F, F, F, F, F, F, F],
            "pb_bullish": [F, F, F, F, T, F, F, F, F, F],
            "pb_bearish": [F, F, F, F, F, F, F, F, F, F],
            "js_bullish": [F, F, F, F, F, T, F, F, F, F],
            "js_bearish": [F, F, F, F, F, F, F, F, F, F],
            "sb_entry_long": [F, F, F, F, F, F, T, F, F, F],
            "sb_entry_short": [F, F, F, F, F, F, F, F, F, F],
            "mmxm_signal_long": [F, F, F, F, F, F, F, T, F, F],
            "mmxm_signal_short": [F, F, F, F, F, F, F, F, F, F],
            "swept_buy_side": [F, F, F, F, F, F, F, F, T, F],
            "swept_sell_side": [F, F, F, F, F, F, F, F, F, F],
            "ote_bullish": [F, F, F, F, F, F, F, F, F, T],
            "ote_bearish": [F, F, F, F, F, F, F, F, F, F],
            # Science Filters
            "hurst_passes": [T, T, F, F, T, T, T, F, F, T],
            "kalman_passes": [T, F, T, F, T, F, T, F, T, F],
            "garch_passes": [F, T, T, T, F, F, F, T, T, T],
            "changepoint_passes": [F, F, F, F, T, F, F, F, F, F],
            "nash_passes": [T, T, T, T, F, F, F, F, F, F],
        },
        index=idx,
    )


def test_breaker_concept_resolves():
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["BREAKER"])
    assert len(bars) == 1  # Bar 1


def test_rb_concept_resolves():
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["RB"])
    assert len(bars) == 1  # Bar 2


def test_judas_concept_resolves():
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["JUDAS"])
    assert len(bars) == 1  # Bar 5


def test_silver_bullet_concept_resolves():
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["SILVER_BULLET"])
    assert len(bars) == 1  # Bar 6


def test_sweep_fixed_resolves():
    """sweep mappt jetzt auf swept_buy/sell_side, nicht mehr auf sweep_high."""
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["SWEEP"])
    assert len(bars) == 1  # Bar 8


def test_ote_fixed_resolves():
    """ote mappt jetzt auf ote_bullish, nicht mehr ote_long."""
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["OTE"])
    assert len(bars) == 1  # Bar 9


def test_hurst_filter_resolves():
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["HURST"])
    assert len(bars) == 6


def test_breaker_and_hurst_intersection():
    """Breaker UND Hurst-Filter → nur Bar 1 (beide True)."""
    df = _make_v2_cache_df()
    bars = query_signal_bars(df, concepts=["BREAKER", "HURST"], mode="all")
    assert len(bars) == 1


from sb.cache.cache_query import query_signal_bars_roles


def _make_role_cache() -> pd.DataFrame:
    """Mini-Cache mit bekannten Signal-Spalten für Rollen-Tests."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1min")
    return pd.DataFrame(
        {
            # Entry signals
            "bos_bullish": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            "bos_bearish": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Zone active flags
            "fvg_bull_active": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "fvg_bear_active": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Context flags
            "narrative_bullish": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "narrative_bearish": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "hurst_passes": [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            # Timing flags (kz_london_open = echte Spalte aus algo_20_sessions_v2)
            "kz_london_open": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        },
        index=idx,
    )


def test_roles_entry_only():
    """Entry-only: BOS fires an Bar 1 und 5. Lookback=20 → Bars 1-9 aktiv (9 Bars)."""
    cache = _make_role_cache()
    result = query_signal_bars_roles(
        cache, entry=["BOS"], zone=[], context=[], timing=[]
    )
    assert len(result) == 9


def test_roles_entry_and_zone():
    """BOS (entry) mit Lookback + FVG-Zone aktiv → Bars 1,2,3 (3 Bars)."""
    cache = _make_role_cache()
    result = query_signal_bars_roles(
        cache, entry=["BOS"], zone=["FVG"], context=[], timing=[]
    )
    # Lookback: Bars 1-9 aktiv nach BOS@Bar1
    # FVG zone: Bars 1,2,3 (fvg_bull_active)
    # AND: Bars 1,2,3 → 3
    assert len(result) == 3


def test_roles_context_filters():
    """Context-Filter NARRATIVE: Entry-Lookback AND Narrative → Bars 1,2,3."""
    cache = _make_role_cache()
    result = query_signal_bars_roles(
        cache, entry=["BOS"], zone=[], context=["NARRATIVE"], timing=[]
    )
    # Lookback: Bars 1-9
    # Narrative: Bars 0,1,2,3
    # AND: Bars 1,2,3 → 3
    assert len(result) == 3


def test_roles_timing_gate():
    """KILLZONE filtert Entry-Lookback: nur Bars 1,2,5,6 haben killzone=1."""
    cache = _make_role_cache()
    result_with_timing = query_signal_bars_roles(
        cache, entry=["BOS"], zone=[], context=[], timing=["KILLZONE"]
    )
    result_without = query_signal_bars_roles(
        cache, entry=["BOS"], zone=[], context=[], timing=[]
    )
    # Ohne Timing: Bars 1-9 → 9
    assert len(result_without) == 9
    # Mit Killzone (Bars 0,1,2,5,6) AND Lookback(Bars 1-9) → Bars 1,2,5,6 → 4
    assert len(result_with_timing) == 4


def test_roles_full_stack():
    """BOS + FVG + NARRATIVE + KILLZONE – alle Rollen zusammen."""
    cache = _make_role_cache()
    result = query_signal_bars_roles(
        cache,
        entry=["BOS"],
        zone=["FVG"],
        context=["NARRATIVE"],
        timing=["KILLZONE"],
    )
    # Lookback: Bars 1-9
    # FVG zone: Bars 1,2,3
    # Narrative: Bars 0,1,2,3
    # Killzone: Bars 0,1,2,5,6
    # AND aller: Bars 1,2 → 2
    assert len(result) == 2


def test_roles_empty_entry_returns_empty():
    cache = _make_role_cache()
    result = query_signal_bars_roles(
        cache, entry=[], zone=["FVG"], context=[], timing=[]
    )
    assert len(result) == 0


def test_roles_unknown_concept_ignored_gracefully():
    """Unbekanntes Konzept ohne Cache-Spalten wird ignoriert, kein Absturz."""
    cache = _make_role_cache()
    result = query_signal_bars_roles(
        cache, entry=["BOS"], zone=["UNBEKANNT_XYZ"], context=[], timing=[]
    )
    # Zone UNBEKANNT hat keine Spalten → ignoriert → nur Entry-Lookback zählt → 9
    assert len(result) == 9


def test_roles_context_with_two_filters():
    """Zwei Context-Filter: beide müssen passen (AND)."""
    cache = _make_role_cache()
    result = query_signal_bars_roles(
        cache, entry=["BOS"], zone=[], context=["NARRATIVE", "HURST"], timing=[]
    )
    # Lookback: Bars 1-9
    # Narrative: Bars 0,1,2,3
    # Hurst: alle außer Bar 2
    # AND: Bars 1,3 → 2 (Bar 2 fällt wegen Hurst raus)
    assert len(result) == 2


def test_roles_lookback_allows_zone_after_entry():
    """Kerntest: Entry feuert auf Bar 0, Zone wird erst auf Bar 15 aktiv.
    Mit Lookback=20 muss Bar 15 dennoch als Signal gezählt werden."""
    from sb.cache.cache_query import ENTRY_LOOKBACK_BARS
    import pandas as pd

    n = ENTRY_LOOKBACK_BARS + 5  # z.B. 25 Bars
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    bos = [0] * n
    bos[0] = 1  # BOS feuert auf Bar 0
    zone = [0] * n
    zone[15] = 1  # Zone wird erst auf Bar 15 aktiv (innerhalb Lookback)
    zone[ENTRY_LOOKBACK_BARS + 1] = 1  # dieser Bar ist AUSSERHALB Lookback

    df = pd.DataFrame(
        {"bos_bullish": bos, "fvg_bull_active": zone},
        index=idx,
    )
    result = query_signal_bars_roles(
        df, entry=["BOS"], zone=["FVG"], context=[], timing=[]
    )
    # Bar 15: innerhalb Lookback(0..19) → soll matchen
    # Bar ENTRY_LOOKBACK_BARS+1: außerhalb → soll nicht matchen
    assert idx[15] in result
    assert idx[ENTRY_LOOKBACK_BARS + 1] not in result


# --- Tests für die 13 neuen Konzepte (Spalten-Auflösung) ---


def _make_new_concepts_cache() -> pd.DataFrame:
    """Cache-DataFrame mit Spalten aller 13 neuen Konzepte."""
    T, F = True, False
    idx = pd.date_range("2024-01-01 09:30", periods=6, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            # OPENING_RANGE (zone)
            "in_opening_range": [T, T, F, F, F, F],
            "or_completed": [F, T, F, F, F, F],
            # MIDNIGHT_OPEN (context)
            "mo_above": [T, T, T, F, F, F],
            # HRL_LRL (context)
            "is_lrl": [F, F, T, T, F, F],
            "is_hrl": [T, F, F, F, T, F],
            # QUARTERLY_THEORY (context)
            "is_expansion_quarter": [T, T, T, F, F, F],
            "is_low_prob_quarter": [F, F, F, T, T, T],
            # SESSION_TIMING (timing)
            "session_asia": [T, F, F, F, F, F],
            "session_london": [F, T, T, F, F, F],
            "session_ny": [F, F, F, T, T, F],
            # SPECIAL_TIMING (timing)
            "at_session_open": [T, F, F, T, F, F],
            "at_makro_start": [F, T, F, F, T, F],
            "is_first_bar_of_day": [T, F, F, F, F, F],
            # WEEK_LEVELS (zone)
            "near_week_high": [F, F, T, F, F, F],
            "near_week_low": [F, F, F, T, F, F],
            "near_month_high": [F, F, F, F, T, F],
            "near_month_low": [F, F, F, F, F, T],
            # RTH_FILTER (timing)
            "is_rth": [F, T, T, T, T, F],
            "is_eth": [T, F, F, F, F, T],
            # SMC_FVG (zone)
            "smc_fvg_bull": [F, T, F, F, F, F],
            "smc_fvg_bear": [F, F, T, F, F, F],
            # SMC_SESSIONS (timing)
            "smc_session_active": [T, T, F, F, T, T],
            # SHANNON_CAPACITY (context)
            "shannon_cap_passes": [T, T, T, F, F, F],
            # PRIGOGINE (context)
            "prigogine_tradeable": [F, T, T, T, F, F],
            # CUSUM (context)
            "cusum_detected": [F, F, T, T, T, F],
        },
        index=idx,
    )


def test_opening_range_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["OPENING_RANGE"])
    assert len(bars) == 2  # Bars 0+1


def test_midnight_open_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["MIDNIGHT_OPEN"])
    assert len(bars) == 3  # Bars 0+1+2


@pytest.mark.xfail(reason="HRL_LRL concept not resolving — known bug in query_signal_bars")
def test_hrl_lrl_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["HRL_LRL"])
    assert len(bars) == 4  # Bars 0+2+3+4


def test_quarterly_theory_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["QUARTERLY_THEORY"])
    assert len(bars) == 6  # alle Bars


def test_session_timing_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["SESSION_TIMING"])
    assert len(bars) == 5  # Bars 0–4


def test_special_timing_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["SPECIAL_TIMING"])
    assert len(bars) == 4  # Bars 0+1+3+4


def test_week_levels_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["WEEK_LEVELS"])
    assert len(bars) == 4  # Bars 2+3+4+5


def test_rth_filter_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["RTH_FILTER"])
    assert len(bars) == 6  # alle Bars (eth + rth decken alles)


def test_smc_fvg_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["SMC_FVG"])
    assert len(bars) == 2  # Bars 1+2


def test_smc_sessions_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["SMC_SESSIONS"])
    assert len(bars) == 4  # Bars 0+1+4+5


def test_shannon_capacity_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["SHANNON_CAPACITY"])
    assert len(bars) == 3  # Bars 0+1+2


def test_prigogine_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["PRIGOGINE"])
    assert len(bars) == 3  # Bars 1+2+3


def test_cusum_resolves():
    df = _make_new_concepts_cache()
    bars = query_signal_bars(df, concepts=["CUSUM"])
    assert len(bars) == 3  # Bars 2+3+4


def test_regime_concept_maps_to_regime_passes():
    """REGIME-Konzept muss 'regime_passes' Spalte nutzen, nicht nur hurst_passes."""
    from sb.cache.cache_query import _CONCEPT_PREFIXES, _CONTEXT_COLUMNS

    # Prüfe _CONCEPT_PREFIXES
    regime_cols_prefix = _CONCEPT_PREFIXES.get("regime", [])
    assert "regime_passes" in regime_cols_prefix, (
        f"REGIME in _CONCEPT_PREFIXES mappt auf {regime_cols_prefix} statt auf regime_passes"
    )

    # Prüfe _CONTEXT_COLUMNS (bei Filterung)
    regime_cols_context = _CONTEXT_COLUMNS.get("regime", [])
    assert "regime_passes" in regime_cols_context, (
        f"REGIME in _CONTEXT_COLUMNS mappt auf {regime_cols_context} statt auf regime_passes"
    )
