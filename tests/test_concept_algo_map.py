from sb.cache.concept_algo_map import CONCEPT_DEFAULT_ROLE, CONCEPT_TO_ALGO_KEYWORDS


def test_all_concept_keywords_have_default_role():
    """Jedes Konzept in CONCEPT_TO_ALGO_KEYWORDS muss eine Rolle haben."""
    for concept in CONCEPT_TO_ALGO_KEYWORDS:
        assert concept in CONCEPT_DEFAULT_ROLE, (
            f"Konzept '{concept}' fehlt in CONCEPT_DEFAULT_ROLE"
        )


def test_roles_are_valid():
    valid_roles = {"entry", "zone", "context", "timing", "exit"}
    for concept, role in CONCEPT_DEFAULT_ROLE.items():
        assert role in valid_roles, f"Ungültige Rolle '{role}' für Konzept '{concept}'"


def test_sweep_keyword_is_specific():
    """SWEEP darf nicht 'liquidity' als Keyword haben – zu breit."""
    keywords = CONCEPT_TO_ALGO_KEYWORDS.get("SWEEP", [])
    assert "liquidity" not in keywords, (
        "SWEEP-Keyword 'liquidity' ist zu breit und lädt falsche Algos"
    )


def test_cisd_in_map():
    assert "CISD" in CONCEPT_TO_ALGO_KEYWORDS
    assert "CISD" in CONCEPT_DEFAULT_ROLE
    assert CONCEPT_DEFAULT_ROLE["CISD"] == "entry"


def test_macro_timing_in_map():
    assert "MACRO_TIMING" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["MACRO_TIMING"] == "timing"


def test_makro_zeitfenster_in_map():
    assert "MAKRO_ZEITFENSTER" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["MAKRO_ZEITFENSTER"] == "timing"


def test_all_default_roles_have_keyword_entry():
    """Jedes Konzept in CONCEPT_DEFAULT_ROLE muss auch in CONCEPT_TO_ALGO_KEYWORDS stehen."""
    for concept in CONCEPT_DEFAULT_ROLE:
        assert concept in CONCEPT_TO_ALGO_KEYWORDS, (
            f"Konzept '{concept}' hat Rolle aber fehlt in CONCEPT_TO_ALGO_KEYWORDS"
        )


# --- Tests für die 13 neuen Konzepte ---


def test_opening_range_in_map():
    assert "OPENING_RANGE" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["OPENING_RANGE"] == "zone"


def test_midnight_open_in_map():
    assert "MIDNIGHT_OPEN" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["MIDNIGHT_OPEN"] == "context"


def test_hrl_lrl_in_map():
    assert "HRL_LRL" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["HRL_LRL"] == "context"


def test_quarterly_theory_in_map():
    assert "QUARTERLY_THEORY" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["QUARTERLY_THEORY"] == "context"


def test_session_timing_in_map():
    assert "SESSION_TIMING" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["SESSION_TIMING"] == "timing"


def test_special_timing_in_map():
    assert "SPECIAL_TIMING" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["SPECIAL_TIMING"] == "timing"


def test_week_levels_in_map():
    assert "WEEK_LEVELS" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["WEEK_LEVELS"] == "zone"


def test_rth_filter_in_map():
    assert "RTH_FILTER" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["RTH_FILTER"] == "timing"


def test_smc_fvg_in_map():
    assert "SMC_FVG" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["SMC_FVG"] == "zone"


def test_smc_sessions_in_map():
    assert "SMC_SESSIONS" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["SMC_SESSIONS"] == "timing"


def test_shannon_capacity_in_map():
    assert "SHANNON_CAPACITY" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["SHANNON_CAPACITY"] == "context"


def test_prigogine_in_map():
    assert "PRIGOGINE" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["PRIGOGINE"] == "context"


def test_cusum_in_map():
    assert "CUSUM" in CONCEPT_TO_ALGO_KEYWORDS
    assert CONCEPT_DEFAULT_ROLE["CUSUM"] == "context"


from sb.cache.concept_algo_map import CONCEPT_SL_POINTS


def test_concept_sl_points_fvg():
    assert CONCEPT_SL_POINTS["FVG_STANDARD"] == 5.0
    assert CONCEPT_SL_POINTS["IFVG_1WOCHE"] == 5.0
    assert CONCEPT_SL_POINTS["FVG_2TAGE"] == 5.0


def test_concept_sl_points_ob_bear_base():
    assert CONCEPT_SL_POINTS["OB_SESSION"] == 10.0
    assert CONCEPT_SL_POINTS["OB_TAGESHOCH"] == 10.0
    assert CONCEPT_SL_POINTS["OB_CHAOS"] == 10.0
