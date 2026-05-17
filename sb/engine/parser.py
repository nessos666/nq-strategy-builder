from __future__ import annotations

import re

import logging

from sb.cache.concept_algo_map import (  # noqa: E402
    CONCEPT_DEFAULT_ROLE,
    CONCEPT_PREFERRED_ENTRY,
    CONCEPT_SL_POINTS,
)  # CONCEPT_PD_BENEFIT entfernt – Auto-P/D schadet empirisch (Chat 414)
from sb.models import ParsedIdea

logger = logging.getLogger(__name__)

CONCEPT_KEYWORDS: dict[str, list[str]] = {
    # --- Legacy Core-Konzepte ---
    "BOS": ["bos"],
    "CHOCH": ["choch"],
    "DISPLACEMENT": ["displacement"],
    "FVG": ["fvg"],
    "OB": ["order block", "ob"],
    "OTE": ["ote"],
    "SWEEP": ["sweep"],
    "JUDAS": ["judas swing", "judas"],
    "SILVER_BULLET": ["silver bullet"],
    "BREAKER": ["breaker block", "breaker"],
    "RB": ["rejection block", "rb"],
    "VB": ["vacuum block", "vb"],
    "MMXM": ["mmxm"],
    "MB": ["mitigation block", "mb"],
    "NDOG": ["ndog"],
    "NWOG": ["nwog"],
    "EHPDA_BOUNCE": ["ehpda bounce", "ehpda_bounce"],
    "NARRATIVE": ["narrative"],
    "KILLZONE": ["killzone"],
    "MACRO_TIMING": ["macro timing"],
    "MAKRO_ZEITFENSTER": ["makro zeitfenster", "makro-zeitfenster"],
    "PREMIUM": ["premium"],
    "DISCOUNT": ["discount"],
    "CBDR": ["cbdr"],
    "DEALING_RANGE": ["dealing range"],
    "CISD": ["cisd"],
    "OPENING_RANGE": ["opening range"],
    "MIDNIGHT_OPEN": ["midnight open"],
    "HRL_LRL": ["hrl lrl", "hrl/lrl"],
    "PDA_REGIME": ["pd array regime", "pda regime", "roentgenblick"],
    "STAT_REGIME": ["stat regime", "statistisches regime", "oberflaeche"],
    "QUARTERLY_THEORY": ["quarterly theory"],
    "SESSION_TIMING": ["session timing"],
    "SPECIAL_TIMING": ["special timing"],
    "WEEK_LEVELS": ["week levels"],
    "RTH_FILTER": ["rth filter"],
    "SMC_FVG": ["smc fvg"],
    "SMC_SESSIONS": ["smc sessions"],
    "SHANNON_CAPACITY": ["shannon capacity"],
    "PRIGOGINE": ["prigogine"],
    "CUSUM": ["cusum"],
    # --- Zonen: FVG-Familie ---
    "FVG_STANDARD": ["fvg standard", "fvg std", "3. fvg", "3 fvg"],
    "IFVG_1WOCHE": ["ifvg 1woche", "ifvg woche", "4a", "ifvg 1w"],
    "IFVG_SAMEDAY": ["ifvg sameday", "ifvg same", "4b"],
    "FVG_2TAGE": ["fvg 2tage", "fvg 2t", "5a"],
    "FVG_1_2WOCHEN": ["fvg 1-2wochen", "fvg wochen", "5b"],
    # --- Zonen: Order Block-Familie ---
    "OB_CHAOS": ["ob chaos", "chaos ob", "chaos orderblock", "chaos order block", "7a"],
    "OB_TAGESHOCH": ["ob tageshoch", "tageshoch ob", "tageshoch orderblock", "7b"],
    "OB_SESSION": ["ob session", "session ob", "session orderblock", "7c"],
    "OB_SESSION_HTIEF": [
        "ob session hoch-tief",
        "session hoch-tief",
        "7d",
        "s_ob htief",
    ],
    # --- Timing ---
    "MACRO_SHORT": ["macro short", "macro time short", "6a"],
    "MACRO_LONG": ["macro long", "macro time long", "6b"],
    # --- Entry-Logik: First Touch ---
    "FVG_STD_FT": ["fvg standard first touch", "fvg std ft", "3 ft"],
    "IFVG_1W_FT": ["ifvg 1woche first touch", "ifvg 1w ft", "4a ft"],
    "IFVG_SD_FT": ["ifvg sameday first touch", "ifvg sd ft", "4b ft"],
    "FVG_2T_FT": ["fvg 2tage first touch", "fvg 2t ft", "5a ft"],
    "FVG_12W_FT": ["fvg 1-2wochen first touch", "fvg 12w ft", "5b ft"],
    "OB_CHAOS_FT": ["ob chaos first touch", "chaos ob ft", "7a ft"],
    "OB_TAGH_FT": ["ob tageshoch first touch", "tageshoch ob ft", "7b ft"],
    "OB_SESS_FT": ["ob session first touch", "session ob ft", "7c ft"],
    "S_OB_SESS_FT": ["s ob sess ft", "session ob ft small"],
    # --- Entry-Logik: Second Touch 50% ---
    "FVG_STD_ST50": ["fvg standard second touch", "fvg std st50", "3 st50"],
    "IFVG_1W_ST50": ["ifvg 1woche second touch", "ifvg 1w st50", "4a st50"],
    "IFVG_SD_ST50": ["ifvg sameday second touch", "ifvg sd st50", "4b st50"],
    "FVG_2T_ST50": ["fvg 2tage second touch", "fvg 2t st50", "5a st50"],
    "FVG_12W_ST50": ["fvg 1-2wochen second touch", "fvg 12w st50", "5b st50"],
    "OB_CHAOS_ST50": ["ob chaos second touch", "chaos ob st50", "7a st50"],
    "OB_TAGH_ST50": ["ob tageshoch second touch", "tageshoch ob st50", "7b st50"],
    "OB_SESS_ST50": ["ob session second touch", "session ob st50", "7c st50"],
    "S_OB_SESS_ST50": ["s ob sess st50", "session ob st50 small"],
    # --- ICT Konzepte ---
    "MANIP": ["manip", "manipulation", "liquidity sweep", "liquidity grab"],
    "MANIP_BEAR": ["manip bear", "manip short", "bear manip", "liquidity sweep short"],
    "TURTLE_SOUP": ["turtle soup", "turtle_soup", "ts filter"],
    # --- Exit-Logik ---
    "EXIT_ATR_TRAIL": ["exit atr trail", "atr trail"],
    "EXIT_BREAKEVEN": ["exit breakeven", "breakeven"],
    "EXIT_NEXT_ZONE": ["exit next zone", "next zone"],
    "EXIT_SESSION_LEVEL": ["exit session level", "session level"],
    # --- Wissenschaftliche Filter ---
    "HURST": ["hurst", "hurst exp", "mean reversion filter"],
    # --- Entry-Logik: Displacement ---
    "FVG_STD_DP": ["fvg standard displacement", "fvg std dp", "3 dp"],
    "IFVG_1W_DP": ["ifvg 1woche displacement", "ifvg 1w dp", "4a dp"],
    "IFVG_SD_DP": ["ifvg sameday displacement", "ifvg sd dp", "4b dp"],
    "FVG_2T_DP": ["fvg 2tage displacement", "fvg 2t dp", "5a dp"],
    "FVG_12W_DP": ["fvg 1-2wochen displacement", "fvg 12w dp", "5b dp"],
    "OB_CHAOS_DP": ["ob chaos displacement", "chaos ob dp", "7a dp"],
    "OB_TAGH_DP": ["ob tageshoch displacement", "tageshoch ob dp", "7b dp"],
    "OB_SESS_DP": ["ob session displacement", "session ob dp", "7c dp"],
    "S_OB_SESS_DP": ["s ob sess dp", "session ob dp small"],
}

SESSION_KEYWORDS: dict[str, list[str]] = {
    "london": ["london", "european"],
    "ny": ["new york", "ny", "nyo", "american"],
    "asia": ["asia", "asian", "tokyo"],
}

SL_PATTERN = re.compile(
    r"(?:sl|stop|stoploss|stop.loss)\s*[:\s]?\s*(\d+(?:\.\d+)?)\s*(?:punkte|pts|points|p\b)",
    re.IGNORECASE,
)


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword)
    if re.search(r"[A-Za-z0-9]", keyword):
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


CONCEPT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    concept: [_keyword_pattern(keyword) for keyword in keywords]
    for concept, keywords in CONCEPT_KEYWORDS.items()
}

SESSION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    session: [_keyword_pattern(keyword) for keyword in keywords]
    for session, keywords in SESSION_KEYWORDS.items()
}


def parse_idea(text: str) -> ParsedIdea:
    """Wandelt Freitext in ParsedIdea um.

    Konzepte werden automatisch einer Rolle zugewiesen (entry/zone/context/timing)
    basierend auf CONCEPT_DEFAULT_ROLE. Das concepts-Feld enthält weiterhin alle
    erkannten Konzepte für Backward-Kompatibilität.
    """
    concepts: list[str] = []
    entry: list[str] = []
    zone: list[str] = []
    context: list[str] = []
    timing: list[str] = []

    for concept, patterns in CONCEPT_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            concepts.append(concept)
            role = CONCEPT_DEFAULT_ROLE.get(concept)
            if role == "entry":
                entry.append(concept)
            elif role == "zone":
                zone.append(concept)
            elif role == "context":
                context.append(concept)
            elif role == "timing":
                timing.append(concept)

    # Bereinigung: wenn FT/DP-Variante erkannt, Base-Zone entfernen
    _FT_DP_TO_BASE = {
        "FVG_STD_FT": "FVG_STANDARD",
        "FVG_STD_DP": "FVG_STANDARD",
        "FVG_STD_ST50": "FVG_STANDARD",
        "IFVG_1W_FT": "IFVG_1WOCHE",
        "IFVG_1W_DP": "IFVG_1WOCHE",
        "IFVG_1W_ST50": "IFVG_1WOCHE",
        "IFVG_SD_FT": "IFVG_SAMEDAY",
        "IFVG_SD_DP": "IFVG_SAMEDAY",
        "IFVG_SD_ST50": "IFVG_SAMEDAY",
        "FVG_2T_FT": "FVG_2TAGE",
        "FVG_2T_DP": "FVG_2TAGE",
        "FVG_2T_ST50": "FVG_2TAGE",
        "FVG_12W_FT": "FVG_1_2WOCHEN",
        "FVG_12W_DP": "FVG_1_2WOCHEN",
        "FVG_12W_ST50": "FVG_1_2WOCHEN",
        "OB_CHAOS_FT": "OB_CHAOS",
        "OB_CHAOS_DP": "OB_CHAOS",
        "OB_CHAOS_ST50": "OB_CHAOS",
        "OB_TAGH_FT": "OB_TAGESHOCH",
        "OB_TAGH_DP": "OB_TAGESHOCH",
        "OB_TAGH_ST50": "OB_TAGESHOCH",
        "OB_SESS_FT": "OB_SESSION",
        "OB_SESS_DP": "OB_SESSION",
        "OB_SESS_ST50": "OB_SESSION",
        "S_OB_SESS_FT": "OB_SESSION",
        "S_OB_SESS_DP": "OB_SESSION",
        "S_OB_SESS_ST50": "OB_SESSION",
    }
    for ft_dp, base in _FT_DP_TO_BASE.items():
        if ft_dp in concepts and base in concepts:
            concepts.remove(base)
            if base in entry:
                entry.remove(base)

    session = "all"
    for sess, patterns in SESSION_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            session = sess
            break

    sl_hint: float | None = None
    match = SL_PATTERN.search(text)
    if match:
        sl_hint = float(match.group(1))

    # Wenn MANIP_BEAR erkannt → MANIP aus entry entfernen (nur bear signal)
    if "MANIP_BEAR" in concepts and "MANIP" in entry:
        entry.remove("MANIP")

    # Direction: "MANIP_BEAR" oder explizit "short" → -1, sonst 0 (beide)
    # Erst erkannte Konzept-Keywords aus dem Text entfernen, damit z.B.
    # "macro short" (= Konzeptname MACRO_SHORT) nicht als direction=-1 gilt.
    text_for_dir = text.lower()
    for _concept in concepts:
        for kw in CONCEPT_KEYWORDS.get(_concept, []):
            text_for_dir = re.sub(
                rf"(?<!\w){re.escape(kw.lower())}(?!\w)", "", text_for_dir
            )
    if "MANIP_BEAR" in concepts or re.search(r"\bshort\b", text_for_dir):
        direction = -1
    elif re.search(r"\blong\b", text_for_dir):
        direction = 1
    else:
        direction = 0

    # Auto-SL: wenn kein expliziter SL-Wert im Text, Forschungswert aus Konzept nehmen
    _FVG_CONCEPTS = {
        "FVG_STANDARD",
        "IFVG_1WOCHE",
        "IFVG_SAMEDAY",
        "FVG_2TAGE",
        "FVG_1_2WOCHEN",
    }
    if sl_hint is None:
        for concept in concepts:
            if concept in CONCEPT_SL_POINTS:
                sl_hint = CONCEPT_SL_POINTS[concept]
                # Bull-OBs brauchen 2× mehr SL als Bear-OBs
                if concept not in _FVG_CONCEPTS and direction == 1:
                    sl_hint *= 2.0
                break

    use_trail = bool(re.search(r"\btrail\b", text, re.IGNORECASE))

    # Exit-Mode erkennen
    exit_mode = "fixed"
    if re.search(r"\bbreakeven\s+trail\b", text, re.IGNORECASE):
        exit_mode = "breakeven_trail"
    elif re.search(r"\bbreakeven\b", text, re.IGNORECASE):
        exit_mode = "breakeven"
    elif re.search(r"\b(?:next|nächste)\s+zone\b", text, re.IGNORECASE):
        exit_mode = "next_zone"
    elif re.search(r"\bsession\s+level\b", text, re.IGNORECASE):
        exit_mode = "session_level"
    elif re.search(r"\batr\s+trail\b", text, re.IGNORECASE):
        exit_mode = "atr_trail"

    # Entry-Typ-Empfehlung loggen + Entry-Varianten zum Cache-Bau hinzufügen
    # Wenn OB-Konzepte erkannt → FT/DP/ST50 Varianten in concepts einfügen,
    # damit der Signal-Cache alle drei Entry-Algos vorberechnet (Chat 414).
    _ENTRY_VARIANTS = {
        "OB_CHAOS": ["OB_CHAOS_FT", "OB_CHAOS_DP", "OB_CHAOS_ST50"],
        "OB_TAGESHOCH": ["OB_TAGH_FT", "OB_TAGH_DP", "OB_TAGH_ST50"],
        "OB_SESSION": [
            "OB_SESS_FT",
            "OB_SESS_DP",
            "OB_SESS_ST50",
            "S_OB_SESS_FT",
            "S_OB_SESS_DP",
            "S_OB_SESS_ST50",
        ],
        "OB_SESSION_HTIEF": [
            "OB_SESS_FT",
            "OB_SESS_DP",
            "OB_SESS_ST50",
            "S_OB_SESS_FT",
            "S_OB_SESS_DP",
            "S_OB_SESS_ST50",
        ],
        "FVG_STANDARD": ["FVG_STD_FT", "FVG_STD_DP", "FVG_STD_ST50"],
        "IFVG_1WOCHE": ["IFVG_1W_FT", "IFVG_1W_DP", "IFVG_1W_ST50"],
        "IFVG_SAMEDAY": ["IFVG_SD_FT", "IFVG_SD_DP", "IFVG_SD_ST50"],
        "FVG_2TAGE": ["FVG_2T_FT", "FVG_2T_DP", "FVG_2T_ST50"],
        "FVG_1_2WOCHEN": ["FVG_12W_FT", "FVG_12W_DP", "FVG_12W_ST50"],
    }
    for c in list(entry):
        pref = CONCEPT_PREFERRED_ENTRY.get(c)
        if pref:
            logger.info("Empfohlener Entry fuer %s: %s (empirisch)", c, pref.upper())
        variants = _ENTRY_VARIANTS.get(c, [])
        for v in variants:
            if v not in concepts:
                concepts.append(v)

    return ParsedIdea(
        raw=text,
        concepts=concepts,
        session=session,
        sl_hint_points=sl_hint,
        entry=entry,
        zone=zone,
        context=context,
        timing=timing,
        direction=direction,
        use_trail=use_trail,
        exit_mode=exit_mode,
    )
