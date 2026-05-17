from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# Warnung pro Konzept nur einmal ausgeben
_warned_missing_concepts: set[str] = set()

# Konzept-Namen → Spalten-Namen im Signal-Cache (v2-Bibliothek)
_CONCEPT_PREFIXES: dict[str, list[str]] = {
    # --- ICT Entry Signals – Struktur ---
    "bos": ["bos_bullish", "bos_bearish"],
    "choch": ["choch_bullish", "choch_bearish"],
    "mss": ["mss_bull", "mss_bear"],
    "displacement": ["displacement_bullish", "displacement_bearish", "displacement"],
    "failure_swing": ["failure_swing_bullish", "failure_swing_bearish"],
    # --- ICT Entry Signals – Zonen ---
    "ob": ["ob_bullish", "ob_bearish"],
    "ob_kl": ["ob_kl_bullish", "ob_kl_bearish"],
    "mb": ["mb_bullish", "mb_bearish"],
    # ob_kl prefix already in _CONCEPT_PREFIXES above
    "fvg": [
        "fvg_bullish",
        "fvg_bearish",
        "fvg_bull_active",
        "fvg_bear_active",
        "smc_fvg_bull",
        "smc_fvg_bear",
    ],
    "ifvg": ["ifvg_bullish", "ifvg_bearish"],
    "bpr": ["bpr_bullish", "bpr_bearish"],
    "breaker": ["breaker_bullish", "breaker_bearish"],
    "rb": ["rb_bullish", "rb_bearish"],
    "vb": ["vb_bullish", "vb_bearish"],
    "pb": ["pb_bullish", "pb_bearish"],
    "rc": ["rc_bullish", "rc_bearish"],
    "cesd": ["cesd_bullish", "cesd_bearish", "cesd_any"],
    "breakaway": ["breakaway_bullish", "breakaway_bearish"],
    "post_fvg": ["post_swing_fvg_bullish", "post_swing_fvg_bearish"],
    "first_fvg": [
        "fvg_first_bull_london",
        "fvg_first_bear_london",
        "fvg_first_bull_ny_am",
        "fvg_first_bear_ny_am",
        "fvg_first_bull_any",
        "fvg_first_bear_any",
    ],
    "vi": ["vi_bullish", "vi_bearish"],
    # --- ICT Entry Signals – Liquidität ---
    "sweep": ["swept_buy_side", "swept_sell_side", "bsl_formed", "ssl_formed"],
    "erl": ["erl_buy_side", "erl_sell_side"],
    "irl": ["irl_active", "in_irl"],
    "liquidity_bias": ["liquidity_bias_bullish", "liquidity_bias_bearish"],
    "reh": ["reh_active"],
    "rel": ["rel_active"],
    "ce": ["ce_bull_active", "ce_bear_active"],
    # --- ICT Patterns – komplex ---
    "ote": ["ote_bullish", "ote_bearish"],
    "judas": ["js_bullish", "js_bearish"],
    "silver_bullet": ["sb_entry_long", "sb_entry_short"],
    "mmxm": ["mmxm_signal_long", "mmxm_signal_short"],
    "narrative": ["narrative_bullish", "narrative_bearish"],
    "amd": ["signal_long", "signal_short"],
    "manip": ["manip_bull", "manip_bear"],
    "inside_day": ["inside_day_breakout_bull", "inside_day_breakout_bear"],
    "rth_gap": ["rth_gap_bull", "rth_gap_bear"],
    "session_bias": ["session_bias_bullish", "session_bias_bearish"],
    "ndog": ["ndog_bullish", "ndog_bearish"],
    "nwog": ["nwog_bullish", "nwog_bearish"],
    "ehpda_bounce": ["ehpda_bounce_bullish", "ehpda_bounce_bearish"],
    "smc_bos": ["smc_previous_high_broken", "smc_previous_low_broken"],
    # --- Science Filter ---
    "hurst": ["hurst_passes"],
    "garch": ["garch_passes"],
    "kalman": ["kalman_passes"],
    "changepoint": ["changepoint_passes", "changepoint_detected"],
    "nash": ["nash_passes"],
    "shannon": ["shannon_cap_passes", "shannon_ordered"],
    "wiener": ["wiener_passes"],
    "pearson": ["pearson_anomaly"],
    "markov": ["markov_stable"],
    "turing": ["signal_long", "signal_short"],
    "entropy": ["shannon_ordered"],
    "regime": ["regime_passes"],
    # --- Neue PDA-Konzepte ---
    "opening_range": ["in_opening_range", "or_completed"],
    "midnight_open": ["mo_above"],
    "hrl_lrl": ["is_pda_lrl", "is_pda_hrl", "is_stat_lrl", "is_stat_hrl"],
    "pda_regime": ["is_pda_lrl", "is_pda_hrl"],
    "stat_regime": ["is_stat_lrl", "is_stat_hrl"],
    "quarterly_theory": ["is_expansion_quarter", "is_low_prob_quarter"],
    "session_timing": ["session_asia", "session_london", "session_ny"],
    "special_timing": ["at_session_open", "at_makro_start", "is_first_bar_of_day"],
    "week_levels": [
        "near_week_high",
        "near_week_low",
        "near_month_high",
        "near_month_low",
    ],
    "rth_filter": ["is_rth", "is_eth"],
    # --- Neue SMC-Konzepte ---
    "smc_fvg": ["smc_fvg_bull", "smc_fvg_bear"],
    "smc_sessions": ["smc_session_active"],
    "drawn_liquidity": ["drawn_direction_bullish", "drawn_direction_bearish"],
    "dwm_levels": ["near_day_high", "near_day_low"],
    # --- Neue Science-Konzepte ---
    "shannon_capacity": ["shannon_cap_passes"],
    "prigogine": ["prigogine_tradeable"],
    "marktphase": ["marktphase_expansion", "marktphase_ranging"],
    # --- Neue LDP-Konzepte ---
    "cusum": ["cusum_detected"],
    # --- Timing / Zeitfenster ---
    "macro_timing": ["in_macro"],
    "makro_zeitfenster": ["in_makro"],
    # --- Kontext / Marktstruktur ---
    "killzone": ["kz_london_open", "kz_ny_open", "kz_ny_pm"],
    "premium": ["pd_in_premium"],
    "discount": ["pd_in_discount"],
    "turtle_soup": ["ts_bull", "ts_bear"],
    "equilibrium": ["eq_above", "eq_below"],
    "dealing_range": ["dr_in_range"],
    "consolidation": ["in_consolidation"],
    "expansion": ["range_expansion"],
    "strong_candle": ["strong_candle"],
    "cbdr": ["cbdr_above", "cbdr_below"],
    # --- Session-spezifische Sweeps ---
    "london_sweep": ["ldn_high_swept", "ldn_low_swept"],
}

# Lookback-Fenster für Entry-Signale (Bars).
# BOS/CHOCH/SWEEP feuern auf einem Bar; danach gilt der Entry-Kontext
# für diese viele Bars als aktiv, damit Zone-Filter greifen können.
ENTRY_LOOKBACK_BARS = 20

# Rollen-spezifische Spalten-Mappings.
# Entry: Spalten die feuern wenn das Signal aktiv ist.
_ENTRY_COLUMNS: dict[str, list[str]] = {
    "bos": ["bos_bullish", "bos_bearish"],
    "choch": ["choch_bullish", "choch_bearish"],
    "mss": ["mss_bull", "mss_bear"],
    "displacement": ["displacement_bullish", "displacement_bearish"],
    "failure_swing": ["failure_swing_bullish", "failure_swing_bearish"],
    "sweep": ["swept_buy_side", "swept_sell_side"],
    "erl": ["erl_buy_side", "erl_sell_side"],
    "irl": ["irl_active"],
    "reh": ["reh_active"],
    "rel": ["rel_active"],
    "ce": ["ce_bull_active", "ce_bear_active"],
    "ote": ["ote_bullish", "ote_bearish"],
    "judas": ["js_bullish", "js_bearish"],
    "silver_bullet": ["sb_entry_long", "sb_entry_short"],
    "mmxm": ["mmxm_signal_long", "mmxm_signal_short"],
    "amd": ["signal_long", "signal_short"],
    "manip": ["manip_bull", "manip_bear"],
    "manip_bear": ["manip_bear"],
    "manip_bull": ["manip_bull"],
    "inside_day": ["inside_day_breakout_bull", "inside_day_breakout_bear"],
    "rth_gap": ["rth_gap_bull", "rth_gap_bear"],
    "ndog": ["ndog_bullish", "ndog_bearish"],
    "nwog": ["nwog_bullish", "nwog_bearish"],
    "ehpda_bounce": ["ehpda_bounce_bullish", "ehpda_bounce_bearish"],
    "smc_bos": ["smc_previous_high_broken", "smc_previous_low_broken"],
    "cbdr": ["cbdr_above", "cbdr_below"],
    "cisd": ["cisd_bullish", "cisd_bearish"],
    "fvg": ["fvg_bullish", "fvg_bearish"],
    "ob": ["ob_bullish", "ob_bearish"],
    "ob_kl": ["ob_kl_bullish", "ob_kl_bearish"],
    "ifvg": ["ifvg_bullish", "ifvg_bearish"],
    "bpr": ["bpr_bullish", "bpr_bearish"],
    "breaker": ["breaker_bullish", "breaker_bearish"],
    "rb": ["rb_bullish", "rb_bearish"],
    "vb": ["vb_bullish", "vb_bearish"],
    "pb": ["pb_bullish", "pb_bearish"],
    "vi": ["vi_bullish", "vi_bearish"],
    "london_sweep": ["ldn_high_swept", "ldn_low_swept"],
    # --- david_bibliothek Zonen (Zone-Creation als Entry) ---
    "fvg_standard": ["fvg_bull", "fvg_bear"],
    "ifvg_1woche": ["ifvg_bullish", "ifvg_bearish"],
    "ifvg_sameday": ["ifvg_bullish", "ifvg_bearish"],
    "fvg_2tage": ["fvg_bull", "fvg_bear"],
    "fvg_1_2wochen": ["fvg_bull", "fvg_bear"],
    "ob_chaos": ["ob_bull", "ob_bear"],
    "ob_tageshoch": ["ob_bull", "ob_bear"],
    "ob_session": ["ob_bull", "ob_bear", "s_ob_bull", "s_ob_bear"],
    "ob_session_htief": ["s_ob_bull", "s_ob_bear"],
    # --- Entry-Logik: First Touch ---
    "fvg_std_ft": ["fvg_std_bull_ft", "fvg_std_bear_ft"],
    "ifvg_1w_ft": ["ifvg_1w_bull_ft", "ifvg_1w_bear_ft"],
    "ifvg_sd_ft": ["ifvg_sd_bull_ft", "ifvg_sd_bear_ft"],
    "fvg_2t_ft": ["fvg_2t_bull_ft", "fvg_2t_bear_ft"],
    "fvg_12w_ft": ["fvg_12w_bull_ft", "fvg_12w_bear_ft"],
    "ob_chaos_ft": ["ob_chaos_bull_ft", "ob_chaos_bear_ft"],
    "ob_tagh_ft": ["ob_tagh_bull_ft", "ob_tagh_bear_ft"],
    "ob_sess_ft": ["ob_sess_bull_ft", "ob_sess_bear_ft"],
    "s_ob_sess_ft": ["s_ob_sess_bull_ft", "s_ob_sess_bear_ft"],
    # --- Entry-Logik: Displacement ---
    "fvg_std_dp": ["fvg_std_bull_dp", "fvg_std_bear_dp"],
    "ifvg_1w_dp": ["ifvg_1w_bull_dp", "ifvg_1w_bear_dp"],
    "ifvg_sd_dp": ["ifvg_sd_bull_dp", "ifvg_sd_bear_dp"],
    "fvg_2t_dp": ["fvg_2t_bull_dp", "fvg_2t_bear_dp"],
    "fvg_12w_dp": ["fvg_12w_bull_dp", "fvg_12w_bear_dp"],
    "ob_chaos_dp": ["ob_chaos_bull_dp", "ob_chaos_bear_dp"],
    "ob_tagh_dp": ["ob_tagh_bull_dp", "ob_tagh_bear_dp"],
    "ob_sess_dp": ["ob_sess_bull_dp", "ob_sess_bear_dp"],
    "s_ob_sess_dp": ["s_ob_sess_bull_dp", "s_ob_sess_bear_dp"],
    # --- Second Touch 50% ---
    "fvg_std_st50": ["fvg_std_bull_st50", "fvg_std_bear_st50"],
    "ifvg_1w_st50": ["ifvg_1w_bull_st50", "ifvg_1w_bear_st50"],
    "ifvg_sd_st50": ["ifvg_sd_bull_st50", "ifvg_sd_bear_st50"],
    "fvg_2t_st50": ["fvg_2t_bull_st50", "fvg_2t_bear_st50"],
    "fvg_12w_st50": ["fvg_12w_bull_st50", "fvg_12w_bear_st50"],
    "ob_chaos_st50": ["ob_chaos_bull_st50", "ob_chaos_bear_st50"],
    "ob_tagh_st50": ["ob_tagh_bull_st50", "ob_tagh_bear_st50"],
    "ob_sess_st50": ["ob_sess_bull_st50", "ob_sess_bear_st50"],
    "s_ob_sess_st50": ["s_ob_sess_bull_st50", "s_ob_sess_bear_st50"],
}

# Zone: Spalten die anzeigen dass die Zone an diesem Bar AKTIV ist.
_ZONE_COLUMNS: dict[str, list[str]] = {
    "fvg": ["fvg_bull_active", "fvg_bear_active"],
    "ob": ["ob_validated"],
    "ob_kl": ["ob_kl_validated"],
    "mb": ["mb_bullish", "mb_bearish"],
    "ote": ["ote_bullish", "ote_bearish"],
    "ifvg": ["ifvg_bullish", "ifvg_bearish"],
    "bpr": ["bpr_bullish", "bpr_bearish"],
    "breaker": ["breaker_bullish", "breaker_bearish"],
    "rb": ["rb_bullish", "rb_bearish"],
    "vb": ["vb_bullish", "vb_bearish"],
    "ce": ["ce_bull_active", "ce_bear_active"],
    "opening_range": ["in_opening_range", "or_completed"],
    "smc_fvg": ["smc_fvg_bull", "smc_fvg_bear"],
    "week_levels": [
        "near_week_high",
        "near_week_low",
        "near_month_high",
        "near_month_low",
    ],
    "drawn_liquidity": ["drawn_direction_bullish", "drawn_direction_bearish"],
    "dwm_levels": ["near_day_high", "near_day_low"],
    "premium": ["pd_in_premium"],
    "discount": ["pd_in_discount"],
    "equilibrium": ["eq_above", "eq_below"],
    "dealing_range": ["dr_in_range"],
}

# Context: Filter die passen müssen (jeder AND-verknüpft).
_CONTEXT_COLUMNS: dict[str, list[str]] = {
    "narrative": ["narrative_bullish", "narrative_bearish"],
    "liquidity_bias": ["liquidity_bias_bullish", "liquidity_bias_bearish"],
    "session_bias": ["session_bias_bullish", "session_bias_bearish"],
    "premium": ["pd_in_premium"],
    "discount": ["pd_in_discount"],
    "turtle_soup": ["ts_bull", "ts_bear"],
    "equilibrium": ["eq_above", "eq_below"],
    "dealing_range": ["dr_in_range"],
    "consolidation": ["in_consolidation"],
    "expansion": ["range_expansion"],
    "hurst": ["hurst_passes"],
    "garch": ["garch_passes"],
    "kalman": ["kalman_passes"],
    "regime": ["regime_passes"],
    "markov": ["markov_stable"],
    "changepoint": ["changepoint_passes"],
    "nash": ["nash_passes"],
    "shannon": ["shannon_cap_passes"],
    "wiener": ["wiener_passes"],
    "pearson": ["pearson_anomaly"],
    "turing": ["signal_long", "signal_short"],
    "entropy": ["shannon_ordered"],
    "midnight_open": ["mo_above"],
    "hrl_lrl": ["is_pda_lrl", "is_pda_hrl", "is_stat_lrl", "is_stat_hrl"],
    "pda_regime": ["is_pda_lrl", "is_pda_hrl"],
    "stat_regime": ["is_stat_lrl", "is_stat_hrl"],
    "quarterly_theory": ["is_expansion_quarter", "is_low_prob_quarter"],
    "shannon_capacity": ["shannon_cap_passes"],
    "prigogine": ["prigogine_tradeable"],
    "cusum": ["cusum_detected"],
    "marktphase": ["marktphase_expansion", "marktphase_ranging"],
}

# Timing: Zeitfenster-Gates (jeder AND-verknüpft).
_TIMING_COLUMNS: dict[str, list[str]] = {
    "killzone": ["kz_london_open", "kz_ny_open", "kz_ny_pm"],
    "macro_timing": ["in_macro"],
    "makro_zeitfenster": ["in_makro"],
    "session_timing": ["session_asia", "session_london", "session_ny"],
    "special_timing": ["at_session_open", "at_makro_start", "is_first_bar_of_day"],
    "smc_sessions": ["smc_session_active"],
    "rth_filter": ["is_rth", "is_eth"],
    # --- david_bibliothek Timing ---
    "macro_short": ["in_macro_short"],
    "macro_long": ["in_macro_long"],
}


def _build_concept_mask(
    cache_df: pd.DataFrame,
    concept: str,
    role_cols: dict[str, list[str]],
) -> pd.Series | None:
    """Baut eine Boolean-Maske für ein einzelnes Konzept.

    Priorität: (1) role_cols-Spalten, (2) direkte Spalte im Cache.
    Kein Fallback auf _CONCEPT_PREFIXES – die Rolle muss explizit konfiguriert sein.
    Gibt None zurück wenn keine Spalten gefunden wurden.
    """
    concept_lower = concept.lower().strip()

    # 1. Rollen-spezifische Spalten haben höchste Priorität
    cols = role_cols.get(concept_lower, [])
    found = [c for c in cols if c in cache_df.columns]

    if not found:
        # 2. Direkte Spalte als letzter Ausweg (z.B. berechnete Summenspalten)
        if concept_lower in cache_df.columns:
            return cache_df[concept_lower].astype(bool)  # type: ignore[return-value]
        if concept not in _warned_missing_concepts:
            logger.warning("Konzept '%s' hat keine Spalten im Cache.", concept)
            _warned_missing_concepts.add(concept)
        return None

    mask = pd.Series(False, index=cache_df.index)
    for col in found:
        mask = mask | cache_df[col].astype(bool)
    return mask  # type: ignore[return-value]


def query_signal_bars_roles(
    cache_df: pd.DataFrame,
    entry: list[str],
    zone: list[str],
    context: list[str],
    timing: list[str],
) -> pd.Index:
    """Rollen-bewusste Signal-Abfrage.

    Kombinationslogik:
    - Entry:   OR innerhalb (mindestens 1 Entry-Signal feuert)
    - Zone:    OR innerhalb (mindestens 1 Zone aktiv), AND mit Entry-Maske
    - Context: jedes Konzept einzeln AND mit Entry-Maske (alle müssen passen)
    - Timing:  jedes Konzept einzeln AND mit Entry-Maske (alle müssen offen sein)

    Unbekannte Konzepte (keine Spalten im Cache) werden ignoriert.
    Gibt pd.Index([]) zurück wenn entry leer ist.
    """
    if not entry:
        return pd.Index([])

    # 1. Entry-Maske: OR über alle Entry-Konzepte, mit Lookback.
    # Entry-Signale (BOS, CHOCH, SWEEP...) feuern auf einem einzelnen Bar.
    # Damit eine Zone (FVG, OB) in den Folgebars greift, gilt das Signal
    # für ENTRY_LOOKBACK_BARS Bars nach dem Feuern als aktiv.
    entry_masks = [_build_concept_mask(cache_df, c, _ENTRY_COLUMNS) for c in entry]
    valid_entry = [m for m in entry_masks if m is not None]
    if not valid_entry:
        return pd.Index([])

    def _with_lookback(m: pd.Series) -> pd.Series:
        return m.rolling(ENTRY_LOOKBACK_BARS, min_periods=1).max().astype(bool)  # type: ignore[return-value]

    mask = _with_lookback(valid_entry[0])
    for m in valid_entry[1:]:
        mask = mask | _with_lookback(m)

    # 2. Zone-Maske: OR über alle Zone-Konzepte, AND mit Entry
    if zone:
        zone_masks = [_build_concept_mask(cache_df, c, _ZONE_COLUMNS) for c in zone]
        valid_zones = [m for m in zone_masks if m is not None]
        if valid_zones:
            zone_mask = valid_zones[0]
            for m in valid_zones[1:]:
                zone_mask = zone_mask | m
            mask = mask & zone_mask

    # 3. Context-Filter: jeder einzeln AND
    for concept in context:
        ctx_mask = _build_concept_mask(cache_df, concept, _CONTEXT_COLUMNS)
        if ctx_mask is not None:
            mask = mask & ctx_mask

    # 4. Timing-Gates: jeder einzeln AND
    for concept in timing:
        tmg_mask = _build_concept_mask(cache_df, concept, _TIMING_COLUMNS)
        if tmg_mask is not None:
            mask = mask & tmg_mask

    return cache_df.index[mask]


def query_signal_bars(
    cache_df: pd.DataFrame,
    concepts: list[str],
    mode: Literal["any", "all"] = "any",
) -> pd.Index:
    """Gibt Timestamps zurück wo die angegebenen Konzepte Signale haben.

    Args:
        cache_df: Signal-Cache DataFrame (Index = Timestamps)
        concepts: Liste von Konzept-Namen (z.B. ["bos", "fvg"])
        mode: "any" = mindestens 1 Konzept aktiv, "all" = alle Konzepte aktiv

    Returns:
        pd.Index mit Timestamps der Signal-Bars
    """
    if not concepts:
        return pd.Index([])

    signal_masks: list[pd.Series] = []

    for concept in concepts:
        concept_lower = concept.lower().strip()

        # Direkte Spalte vorhanden?
        if concept_lower in cache_df.columns:
            signal_masks.append(cache_df[concept_lower].astype(bool))  # type: ignore[arg-type]
            continue

        # Über Präfix-Map suchen
        prefixes = _CONCEPT_PREFIXES.get(concept_lower, [])
        found_cols = [c for c in prefixes if c in cache_df.columns]

        if found_cols:
            combined = pd.Series(False, index=cache_df.index)
            for col in found_cols:
                combined = combined | cache_df[col].astype(bool)
            signal_masks.append(combined)
        else:
            if concept not in _warned_missing_concepts:
                logger.warning("Konzept '%s' hat keine Spalten im Cache.", concept)
                _warned_missing_concepts.add(concept)

    if not signal_masks:
        return pd.Index([])

    if mode == "any":
        mask = signal_masks[0]
        for m in signal_masks[1:]:
            mask = mask | m
    else:  # "all"
        mask = signal_masks[0]
        for m in signal_masks[1:]:
            mask = mask & m

    return cache_df.index[mask]
