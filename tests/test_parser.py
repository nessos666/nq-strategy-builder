from __future__ import annotations

from sb.engine.parser import parse_idea
from sb.models import ParsedIdea


def test_parse_bos_fvg_london():
    result = parse_idea("BOS + FVG London Open")
    assert isinstance(result, ParsedIdea)
    assert "BOS" in result.concepts
    assert "FVG" in result.concepts
    assert result.session == "london"


def test_parse_ny_session():
    result = parse_idea("Order Block Entry New York")
    assert "OB" in result.concepts
    assert result.session == "ny"


def test_parse_sl_hint():
    result = parse_idea("FVG Entry, SL 12 Punkte")
    assert result.sl_hint_points == 12.0


def test_parse_ote():
    result = parse_idea("OTE Zone nach Sweep")
    assert "OTE" in result.concepts
    assert "SWEEP" in result.concepts


def test_parse_unknown_returns_all_session():
    result = parse_idea("irgendwas unbekanntes")
    assert result.concepts == []
    assert result.session == "all"


def test_parse_case_insensitive():
    result = parse_idea("bos entry fvg fill london open")
    assert "BOS" in result.concepts
    assert "FVG" in result.concepts
    assert result.session == "london"


def test_parse_judas_swing():
    result = parse_idea("Judas Swing London Entry")
    assert "JUDAS" in result.concepts
    assert result.session == "london"


def test_parse_silver_bullet():
    result = parse_idea("Silver Bullet NY Setup")
    assert "SILVER_BULLET" in result.concepts
    assert result.session == "ny"


def test_parse_breaker_block():
    result = parse_idea("Breaker Block nach BOS London")
    assert "BREAKER" in result.concepts
    assert "BOS" in result.concepts


def test_parse_rejection_block():
    result = parse_idea("RB Entry mit FVG")
    assert "RB" in result.concepts
    assert "FVG" in result.concepts


def test_parse_vacuum_block():
    result = parse_idea("Vacuum Block Fill")
    assert "VB" in result.concepts


def test_parse_mmxm():
    result = parse_idea("MMXM Pattern London")
    assert "MMXM" in result.concepts


def test_parse_judas_swing_keyword():
    result = parse_idea("judas swing setup london")
    assert "JUDAS" in result.concepts


def test_parse_hurst_filter():
    result = parse_idea("FVG London Hurst Filter")
    assert "FVG" in result.concepts
    assert "HURST" in result.concepts


def test_parse_displacement():
    result = parse_idea("Displacement + FVG London")
    assert "DISPLACEMENT" in result.concepts
    assert "FVG" in result.concepts


def test_parse_choch():
    result = parse_idea("CHoCH Entry NY")
    assert "CHOCH" in result.concepts
    assert result.session == "ny"


def test_parse_mb():
    result = parse_idea("Mitigation Block London Open")
    assert "MB" in result.concepts


def test_parse_ndog():
    result = parse_idea("NDOG Bearish Setup")
    assert "NDOG" in result.concepts


def test_parse_narrative():
    result = parse_idea("Narrative Bullish FVG London")
    assert "NARRATIVE" in result.concepts
    assert "FVG" in result.concepts


def test_bos_fvg_ny_gets_roles():
    idea = parse_idea("BOS + FVG NY")
    assert idea.entry == ["BOS"]
    assert idea.zone == ["FVG"]
    assert idea.context == []
    assert idea.timing == []


def test_bos_fvg_hurst_ny_gets_roles():
    idea = parse_idea("BOS + FVG + Hurst NY")
    assert idea.entry == ["BOS"]
    assert idea.zone == ["FVG"]
    assert idea.context == ["HURST"]
    assert idea.timing == []


def test_choch_ob_narrative_killzone_ny():
    idea = parse_idea("CHoCH + OB + Narrative + Killzone NY")
    assert "CHOCH" in idea.entry
    assert "OB" in idea.zone
    assert "NARRATIVE" in idea.context
    assert "KILLZONE" in idea.timing


def test_single_entry_no_zone():
    """Einzelne Entry ohne Zone – zone bleibt leer."""
    idea = parse_idea("Judas Swing NY")
    assert "JUDAS" in idea.entry
    assert idea.zone == []


def test_concepts_still_populated():
    """concepts-Feld muss weiterhin alle Konzepte enthalten (Backward-Compat)."""
    idea = parse_idea("BOS + FVG NY")
    assert "BOS" in idea.concepts
    assert "FVG" in idea.concepts


def test_macro_timing_is_timing_role():
    idea = parse_idea("BOS + FVG + macro timing NY")
    assert "MACRO_TIMING" in idea.timing


def test_premium_is_context_role():
    idea = parse_idea("BOS + FVG + premium NY")
    assert "PREMIUM" in idea.context


def test_dealing_range_is_context_role():
    idea = parse_idea("OTE + OB + dealing range NY")
    assert "DEALING_RANGE" in idea.context


def test_auto_sl_fvg_no_explicit_sl():
    """FVG ohne SL im Text → automatisch 5pts."""
    result = parse_idea("FVG Standard AM session bear")
    assert result.sl_hint_points == 5.0


def test_auto_sl_ob_bear():
    """OB Bear ohne SL im Text → automatisch 10pts."""
    result = parse_idea("OB Session short AM")
    assert result.sl_hint_points == 10.0


def test_auto_sl_ob_bull():
    """OB Bull ohne SL im Text → automatisch 20pts (2× Bear)."""
    result = parse_idea("OB Session long AM")
    assert result.sl_hint_points == 20.0


def test_explicit_sl_overrides_auto():
    """Expliziter SL im Text überschreibt Auto-Wert."""
    result = parse_idea("FVG Standard, SL 15 pts")
    assert result.sl_hint_points == 15.0


def test_parser_recognizes_additional_mapped_concepts():
    idea = parse_idea(
        "CISD + Opening Range + Midnight Open + Quarterly Theory + "
        "Shannon Capacity + Session Timing + Week Levels + RTH Filter + "
        "SMC FVG + SMC Sessions + CUSUM NY"
    )
    assert "CISD" in idea.entry
    assert "OPENING_RANGE" in idea.zone
    assert "MIDNIGHT_OPEN" in idea.context
    assert "QUARTERLY_THEORY" in idea.context
    assert "SHANNON_CAPACITY" in idea.context
    assert "SESSION_TIMING" in idea.timing
    assert "WEEK_LEVELS" in idea.zone
    assert "RTH_FILTER" in idea.timing
    assert "SMC_FVG" in idea.zone
    assert "SMC_SESSIONS" in idea.timing
    assert "CUSUM" in idea.context


def test_parser_recognizes_additional_keyword_variants():
    idea = parse_idea("Makro-Zeitfenster + HRL/LRL + Special Timing + Prigogine")
    assert "MAKRO_ZEITFENSTER" in idea.timing
    assert "HRL_LRL" in idea.context
    assert "SPECIAL_TIMING" in idea.timing
    assert "PRIGOGINE" in idea.context


def test_parser_keeps_exit_mode_and_exit_concept_in_sync():
    idea = parse_idea("OB Session long breakeven trail")
    assert idea.exit_mode == "breakeven_trail"
    assert "EXIT_BREAKEVEN" in idea.concepts
