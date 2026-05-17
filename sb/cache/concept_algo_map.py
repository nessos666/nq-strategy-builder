from __future__ import annotations

from pathlib import Path

# Concept-Name (uppercase) → Teilstrings die im Algo-Dateinamen vorkommen müssen
# Matching: Dateiname-Stem enthält mindestens einen der Keyword-Strings (case-insensitive)
CONCEPT_TO_ALGO_KEYWORDS: dict[str, list[str]] = {
    # --- Legacy Core-Konzepte ---
    "BOS": ["bos"],
    "CHOCH": ["choch"],
    "DISPLACEMENT": ["displacement"],
    "FVG": ["fvg"],
    "OB": ["ob"],
    "OTE": ["ote"],
    "SWEEP": ["sweep"],
    "JUDAS": ["judas"],
    "SILVER_BULLET": ["silver_bullet"],
    "BREAKER": ["breaker"],
    "RB": ["rb"],
    "VB": ["vb"],
    "MMXM": ["mmxm"],
    "MB": ["mb"],
    "NDOG": ["ndog"],
    "NWOG": ["nwog"],
    "EHPDA_BOUNCE": ["ehpda_bounce"],
    "NARRATIVE": ["narrative"],
    "KILLZONE": ["killzone"],
    "PREMIUM": ["premium"],
    "DISCOUNT": ["discount"],
    "CBDR": ["cbdr"],
    "DEALING_RANGE": ["dealing_range"],
    # --- Zonen: FVG-Familie ---
    "FVG_STANDARD": ["FVG Standard"],
    "IFVG_1WOCHE": ["iFVG 1Woche"],
    "IFVG_SAMEDAY": ["iFVG SameDay"],
    "FVG_2TAGE": ["FVG 2Tage"],
    "FVG_1_2WOCHEN": ["FVG 1-2Wochen"],
    # --- Zonen: Order Block-Familie ---
    "OB_CHAOS": ["Chaos Order Block"],
    "OB_TAGESHOCH": ["Tageshoch-Tief Orderblock"],
    "OB_SESSION": ["Session Orderblock"],
    "OB_SESSION_HTIEF": ["Session Hoch-Tief Orderblock"],
    # --- ICT Konzepte ---
    "MANIP": ["manip_liquidity_sweep"],
    "MANIP_BEAR": ["manip_liquidity_sweep"],
    "TURTLE_SOUP": ["turtle_soup"],
    # --- Wissenschaftliche Filter ---
    "HURST": ["hurst_exponent"],
    # --- Timing ---
    "MACRO_SHORT": ["Macro Time Short"],
    "MACRO_LONG": ["Macro Time Long"],
    # --- Entry-Logik: First Touch (ft) ---
    "FVG_STD_FT": ["entry_first_touch"],
    "IFVG_1W_FT": ["entry_first_touch"],
    "IFVG_SD_FT": ["entry_first_touch"],
    "FVG_2T_FT": ["entry_first_touch"],
    "FVG_12W_FT": ["entry_first_touch"],
    "OB_CHAOS_FT": ["entry_first_touch"],
    "OB_TAGH_FT": ["entry_first_touch"],
    "OB_SESS_FT": ["entry_first_touch"],
    "S_OB_SESS_FT": ["entry_first_touch"],
    # --- Entry-Logik: Displacement (dp) ---
    "FVG_STD_DP": ["entry_displacement"],
    "IFVG_1W_DP": ["entry_displacement"],
    "IFVG_SD_DP": ["entry_displacement"],
    "FVG_2T_DP": ["entry_displacement"],
    "FVG_12W_DP": ["entry_displacement"],
    "OB_CHAOS_DP": ["entry_displacement"],
    "OB_TAGH_DP": ["entry_displacement"],
    "OB_SESS_DP": ["entry_displacement"],
    "S_OB_SESS_DP": ["entry_displacement"],
    # --- Entry-Logik: Second Touch 50% (st50) ---
    "FVG_STD_ST50": ["entry_second_touch_50"],
    "IFVG_1W_ST50": ["entry_second_touch_50"],
    "IFVG_SD_ST50": ["entry_second_touch_50"],
    "FVG_2T_ST50": ["entry_second_touch_50"],
    "FVG_12W_ST50": ["entry_second_touch_50"],
    "OB_CHAOS_ST50": ["entry_second_touch_50"],
    "OB_TAGH_ST50": ["entry_second_touch_50"],
    "OB_SESS_ST50": ["entry_second_touch_50"],
    "S_OB_SESS_ST50": ["entry_second_touch_50"],
    # --- Weitere Konzepte ---
    "CISD": ["cisd"],
    "MACRO_TIMING": ["macro_timing"],
    "MAKRO_ZEITFENSTER": ["makro_zeitfenster"],
    "OPENING_RANGE": ["opening_range"],
    "MIDNIGHT_OPEN": ["midnight_open"],
    "HRL_LRL": [
        "HRL LRL Petar Drawdown Filter",
        "HRL LRL Internet 15min Trend Filter",
    ],
    "PDA_REGIME": ["HRL LRL Petar Drawdown Filter"],
    "STAT_REGIME": ["HRL LRL Internet 15min Trend Filter"],
    "QUARTERLY_THEORY": ["quarterly_theory"],
    "SESSION_TIMING": ["session_timing"],
    "SPECIAL_TIMING": ["special_timing"],
    "WEEK_LEVELS": ["week_levels"],
    "RTH_FILTER": ["rth_filter"],
    "SMC_FVG": ["smc_fvg"],
    "SMC_SESSIONS": ["smc_sessions"],
    "SHANNON_CAPACITY": ["shannon_capacity"],
    "PRIGOGINE": ["prigogine"],
    "CUSUM": ["cusum"],
    # --- Exit-Logik ---
    "EXIT_ATR_TRAIL": ["exit_atr_trail"],
    "EXIT_BREAKEVEN": ["exit_breakeven"],
    "EXIT_NEXT_ZONE": ["exit_next_zone"],
    "EXIT_SESSION_LEVEL": ["exit_session_level"],
}

# Rolle jedes Konzepts im Strategie-Aufbau.
# entry  = feuert direkt als Entry-Signal (FVG/OB = Zone wird betreten)
# timing = Zeitfenster-Gate (AND-verknüpft mit Entry)
CONCEPT_DEFAULT_ROLE: dict[str, str] = {
    # Legacy Core-Konzepte
    "BOS": "entry",
    "CHOCH": "entry",
    "DISPLACEMENT": "entry",
    "FVG": "zone",
    "OB": "zone",
    "OTE": "zone",
    "SWEEP": "entry",
    "JUDAS": "entry",
    "SILVER_BULLET": "entry",
    "BREAKER": "zone",
    "RB": "zone",
    "VB": "zone",
    "MMXM": "entry",
    "MB": "zone",
    "NDOG": "entry",
    "NWOG": "entry",
    "EHPDA_BOUNCE": "entry",
    "NARRATIVE": "context",
    "KILLZONE": "timing",
    "PREMIUM": "context",
    "DISCOUNT": "context",
    "CBDR": "context",
    "DEALING_RANGE": "context",
    # FVG/OB → entry (direkte Entry-Signale)
    "FVG_STANDARD": "entry",
    "IFVG_1WOCHE": "entry",
    "IFVG_SAMEDAY": "entry",
    "FVG_2TAGE": "entry",
    "FVG_1_2WOCHEN": "entry",
    "OB_CHAOS": "entry",
    "OB_TAGESHOCH": "entry",
    "OB_SESSION": "entry",
    "OB_SESSION_HTIEF": "entry",
    # ICT Konzepte
    "MANIP": "entry",
    "MANIP_BEAR": "entry",
    "TURTLE_SOUP": "context",
    # Wissenschaftliche Filter
    "HURST": "context",
    # Timing
    "MACRO_SHORT": "timing",
    "MACRO_LONG": "timing",
    # Entry-Logik First Touch
    "FVG_STD_FT": "entry",
    "IFVG_1W_FT": "entry",
    "IFVG_SD_FT": "entry",
    "FVG_2T_FT": "entry",
    "FVG_12W_FT": "entry",
    "OB_CHAOS_FT": "entry",
    "OB_TAGH_FT": "entry",
    "OB_SESS_FT": "entry",
    "S_OB_SESS_FT": "entry",
    # Entry-Logik Displacement
    "FVG_STD_DP": "entry",
    "IFVG_1W_DP": "entry",
    "IFVG_SD_DP": "entry",
    "FVG_2T_DP": "entry",
    "FVG_12W_DP": "entry",
    "OB_CHAOS_DP": "entry",
    "OB_TAGH_DP": "entry",
    "OB_SESS_DP": "entry",
    "S_OB_SESS_DP": "entry",
    # Entry-Logik Second Touch 50%
    "FVG_STD_ST50": "entry",
    "IFVG_1W_ST50": "entry",
    "IFVG_SD_ST50": "entry",
    "FVG_2T_ST50": "entry",
    "FVG_12W_ST50": "entry",
    "OB_CHAOS_ST50": "entry",
    "OB_TAGH_ST50": "entry",
    "OB_SESS_ST50": "entry",
    "S_OB_SESS_ST50": "entry",
    # Weitere Konzepte
    "CISD": "entry",
    "MACRO_TIMING": "timing",
    "MAKRO_ZEITFENSTER": "timing",
    "OPENING_RANGE": "zone",
    "MIDNIGHT_OPEN": "context",
    "HRL_LRL": "context",
    "PDA_REGIME": "context",
    "STAT_REGIME": "context",
    "QUARTERLY_THEORY": "context",
    "SESSION_TIMING": "timing",
    "SPECIAL_TIMING": "timing",
    "WEEK_LEVELS": "zone",
    "RTH_FILTER": "timing",
    "SMC_FVG": "zone",
    "SMC_SESSIONS": "timing",
    "SHANNON_CAPACITY": "context",
    "PRIGOGINE": "context",
    "CUSUM": "context",
    # Exit-Logik
    "EXIT_ATR_TRAIL": "exit",
    "EXIT_BREAKEVEN": "exit",
    "EXIT_NEXT_ZONE": "exit",
    "EXIT_SESSION_LEVEL": "exit",
}

# SL-Werte aus empirischer Forschung (p80 Return-Excursion, 717k 1min Bars)
# OB: Bear-Basiswert gespeichert. Bull = 2× (wird in parser.py angewendet).
CONCEPT_SL_POINTS: dict[str, float] = {
    "FVG_STANDARD": 5.0,
    "IFVG_1WOCHE": 5.0,
    "IFVG_SAMEDAY": 5.0,
    "FVG_2TAGE": 5.0,
    "FVG_1_2WOCHEN": 5.0,
    "OB_CHAOS": 10.0,
    "OB_TAGESHOCH": 10.0,
    "OB_SESSION": 10.0,
    "OB_SESSION_HTIEF": 10.0,
}

# Empirisch bester Entry-Typ pro Zone (Chat 414, 717k 1min Bars)
# MFE/MAE Ratio — höher = bessere Entry-Qualität
CONCEPT_PREFERRED_ENTRY: dict[str, str] = {
    # OBs: Entry-Typ macht grossen Unterschied
    "OB_CHAOS": "ft",  # 1.15 (FT) vs 0.83 (DP) vs 0.76 (ST50)
    "OB_TAGESHOCH": "dp",  # 1.00 (DP) vs 0.77 (FT) vs 0.96 (ST50)
    "OB_SESSION": "st50",  # 1.07 (ST50) vs 0.82 (FT) vs 0.81 (DP)
    "OB_SESSION_HTIEF": "st50",  # analog zu OB_SESSION
    # FVGs: kein Unterschied (alle ~1.00), kein Preferred
}

# Welche Konzepte profitieren von Premium/Discount-Filter? (Chat 414)
# P/D ist ein Choppy-Market-Filter, kein Trend-Filter!
# Gilt NUR für OBs (S_OB_SESS Δ+0.94), FVGs zeigen keinen Effekt.
CONCEPT_PD_BENEFIT: dict[str, bool] = {
    "OB_CHAOS": True,  # Δ+0.57
    "OB_TAGESHOCH": True,
    "OB_SESSION": True,
    "OB_SESSION_HTIEF": True,
    "FVG_STANDARD": False,
    "IFVG_1WOCHE": False,
    "IFVG_SAMEDAY": False,
    "FVG_2TAGE": False,
    "FVG_1_2WOCHEN": False,
}

# ATR-Multiplier Forschung (Chat 414, FVG_STD_FT, RR 2:1)
# Kein Multiplier hat PF > 1.0 allein — Entries brauchen Filter!
RESEARCH_ATR_OPTIMAL: float = 1.25  # PF 0.98 (bester von 1.0–3.0)
RESEARCH_ATR_RANGE: tuple[float, float] = (1.0, 2.0)  # eingeengter Suchraum


def get_algo_files_for_concepts(
    concepts: list[str],
    algo_dirs: list[Path],
    algo_pattern: str = "*.py",
) -> list[Path]:
    """Gibt Algo-Dateien zurück die für die angegebenen Konzepte benötigt werden.

    Concept-Namen müssen uppercase sein (wie ParsedIdea.concepts liefert).
    Matching: Dateiname-Stem enthält mindestens einen der Keyword-Strings.
    """
    needed_keywords: set[str] = set()
    for concept in concepts:
        keywords = CONCEPT_TO_ALGO_KEYWORDS.get(concept.upper(), [])
        needed_keywords.update(keywords)

    if not needed_keywords:
        return []

    result: list[Path] = []
    seen: set[str] = set()

    for algo_dir in algo_dirs:
        algo_dir = Path(algo_dir)
        if not algo_dir.exists():
            continue
        for algo_file in sorted(algo_dir.glob(algo_pattern)):
            if algo_file.stem in seen:
                continue
            stem_lower = algo_file.stem.lower()
            if any(kw.lower() in stem_lower for kw in needed_keywords):
                result.append(algo_file)
                seen.add(algo_file.stem)

    return result


def get_all_algo_files(
    algo_dirs: list[Path],
    algo_pattern: str = "*.py",
    include_science: bool = True,
) -> list[Path]:
    """Gibt alle Algo-Dateien zurück – optional ohne Science-Algos."""
    result: list[Path] = []
    seen: set[str] = set()
    for algo_dir in algo_dirs:
        algo_dir = Path(algo_dir)
        if not algo_dir.exists():
            continue
        for algo_file in sorted(algo_dir.glob(algo_pattern)):
            if algo_file.stem in seen:
                continue
            if not include_science and "science_" in algo_file.stem.lower():
                continue
            result.append(algo_file)
            seen.add(algo_file.stem)
    return result
