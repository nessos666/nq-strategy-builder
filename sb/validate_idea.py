#!/usr/bin/env python3
"""
validate_idea.py — Validierungs-Pipeline (Ebene 3) für Ideen vor Batch-Start.

Prüft eine Idee gegen 5 Gates:
  1. SYNTAX        — Ist die Idee lesbar? (Nicht leer, keine Sonderzeichen)
  2. NEGATIV-DB    — Ist diese Kombination bereits als failed bekannt?
  3. DUPLIKAT      — Wurde diese Idee schonmal getestet?
  4. BROKEN-ALGO   — Nutzt die Idee einen der 4 Config-Only-Stubs?
  5. LOCK-STATUS   — Ist mindestens ein benötigter Algo unlocked?

CLI:
    python -m sb.validate_idea "fvg standard bull NY trail"
    python -m sb.validate_idea file.txt
    python -m sb.validate_idea --json "fvg standard bull NY trail"

Exit codes:
    0 = ALLE GATES PASSED
    1 = EIN GATE FAILED
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s",
)
log = logging.getLogger("validate_idea")

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

SB_DIR = Path(__file__).resolve().parent
NEGATIV_INDEX = SB_DIR / "knowledge" / "negativ" / "INDEX.md"
FAILED_COMBOS_DIR = SB_DIR / "knowledge" / "negativ" / "failed_combinations"
DONE_DIR = SB_DIR / "ideas" / "queue" / "done"
OUTPUT_DIRS = [
    SB_DIR / "output_worker_1",
    SB_DIR / "output_worker_2",
    SB_DIR / "output_worker_3",
    SB_DIR / "output_david_1",
]

# Die 4 bekannten Config-Only-Stubs (kein run(), nur Dataclass)
BROKEN_ALGO_FILES: list[tuple[str, str]] = [
    ("03_Context", "ict_fibonacci_levels"),
    ("10_Exit_Logik", "ict_trailing_stop"),
    ("11_ICT_Konzepte", "ict_turtle_soup_multi_tf"),
    ("12_Position_Management", "ict_partial_close"),
]

BROKEN_ALGO_KEYWORDS: list[str] = [
    "ict_fibonacci_levels",
    "ict_trailing_stop",
    "ict_turtle_soup",
    "ict_turtle_soup_multi_tf",
    "ict_partial_close",
    "fibonacci",
    "fibonacci_levels",
]

# Erkennbare Konzepte aus dem Parser (vereinfacht für Lock-Check)
BIBLIOTHEK_ALGO_DIRS: list[Path] = [
    SB_DIR / "david_bibliothek/02_FVG_Zonen",
    SB_DIR / "david_bibliothek/03_Order_Blocks",
    SB_DIR / "david_bibliothek/03_Context",
    SB_DIR / "david_bibliothek/05_Stoploss_TakeProfit",
    SB_DIR / "david_bibliothek/06_Time_Zeit",
    SB_DIR / "david_bibliothek/09_Concept_Algos",
    SB_DIR / "david_bibliothek/09_Entry_Logik",
    SB_DIR / "david_bibliothek/10_Exit_Logik",
    SB_DIR / "david_bibliothek/11_ICT_Konzepte",
    SB_DIR / "david_bibliothek/12_Position_Management",
    SB_DIR / "david_bibliothek/00_Nicht_Funktionierend",
    SB_DIR / "david_bibliothek/04_Opening_Gaps_NDOG_NWOG",
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Ergebnis eines einzelnen Gate-Checks."""

    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """Gesamtergebnis der Validierung."""

    idea: str
    gates: list[GateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failed_gates(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea,
            "all_passed": self.all_passed,
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "detail": g.detail,
                    "severity": g.severity,
                }
                for g in self.gates
            ],
        }


# ---------------------------------------------------------------------------
# Gate 1: SYNTAX
# ---------------------------------------------------------------------------


def _check_syntax(idea: str) -> GateResult:
    """Gate 1: Prüft ob die Idee lesbar ist."""
    if not idea or not idea.strip():
        return GateResult(
            name="SYNTAX",
            passed=False,
            detail="Idea is empty or whitespace only.",
        )

    text = idea.strip()

    # Minimale Länge: mindestens 2 Wörter
    words = text.split()
    if len(words) < 2:
        return GateResult(
            name="SYNTAX",
            passed=False,
            detail=f"Too short ({len(words)} words) — need at least 2 words.",
        )

    # Längenlimit: 500 Zeichen
    if len(text) > 500:
        return GateResult(
            name="SYNTAX",
            passed=False,
            detail=f"Too long ({len(text)} chars); max 500.",
        )

    # Blockierte Sonderzeichen (würden Parser crashen)
    blocked_pattern = re.compile(r"[<>|&;`$(){}\[\]!#~]")
    if blocked_pattern.search(text):
        return GateResult(
            name="SYNTAX",
            passed=False,
            detail=f"Contains blocked special characters: {blocked_pattern.findall(text)}",
        )

    # Keine reinen Zahlen
    if re.match(r"^[\d\s.,]+$", text):
        return GateResult(
            name="SYNTAX",
            passed=False,
            detail="Only numbers — not a valid idea.",
        )

    return GateResult(name="SYNTAX", passed=True, detail="Syntax OK.")


# ---------------------------------------------------------------------------
# Gate 2: NEGATIV-DB
# ---------------------------------------------------------------------------


def _load_negativ_index() -> dict[str, dict[str, Any]]:
    """Lädt die Negativ-INDEX.md und extrahiert bekannte failed-Kombinationen."""
    entries: dict[str, dict[str, Any]] = {}
    if not NEGATIV_INDEX.exists():
        return entries

    try:
        text = NEGATIV_INDEX.read_text(encoding="utf-8")
        # Parse Markdown-Tabelle
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("|") and "NEG-" in line:
                cols = [c.strip() for c in line.split("|")]
                if len(cols) >= 6:
                    neg_id = cols[1]
                    title = cols[2]
                    category = cols[3]
                    description = cols[5]
                    entries[neg_id] = {
                        "title": title,
                        "category": category,
                        "description": description,
                    }
    except Exception as e:
        log.warning("Could not read NEGATIV INDEX: %s", e)

    return entries


def _load_failed_combinations() -> list[str]:
    """Lädt alle failed_combination-Dateien und extrahiert related_ideas."""
    failed_ideas: list[str] = []
    if not FAILED_COMBOS_DIR.exists():
        return failed_ideas

    for f in sorted(FAILED_COMBOS_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        # Suche nach related_ideas Feld
        m = re.search(r"related_ideas:\s*\[(.*?)\]", text, re.DOTALL)
        if m:
            raw = m.group(1)
            ideas = re.findall(r"\"([^\"]+)\"", raw)
            for idea_text in ideas:
                # Versuche aus dem Kontext zu extrahieren: manche related_ideas
                # sind Einträge wie "Queue-Runner Struktur" — keinen konkreten
                # Ideen-Text. Deswegen generische Kombinationen aus dem Dateinamen
                # ableiten.
                if idea_text and not idea_text.startswith("NEG-"):
                    failed_ideas.append(idea_text.strip().lower())

        # Suche nach symptoms: — dort stehen oft Fehlerbeschreibungen
        m = re.search(r"symptoms:\s*\n((?:\s*-\s*\"[^\"]*\"\n?)*)", text)
        if m:
            symptoms_text = m.group(1)
            # Extrahiere quoted strings
            symptoms = re.findall(r"\"([^\"]+)\"", symptoms_text)
            for symptom in symptoms:
                # Nur Ideen-ähnliche Strings (enthalten Konzeptnamen)
                if any(kw in symptom.lower() for kw in ["fvg", "ob ", "ifvg", "nwog", "ndog", "manip", "sweep"]):
                    failed_ideas.append(symptom.strip().lower())

    return failed_ideas


def _check_negativ_db(idea: str) -> GateResult:
    """Gate 2: Prüft ob die Idee in der Negativ-DB als failed bekannt ist."""
    idea_lower = idea.strip().lower()
    failed_ideas = _load_failed_combinations()
    negativ_entries = _load_negativ_index()

    # Check 1: Direkter Match gegen failed_combinations
    for failed in failed_ideas:
        if failed == idea_lower or idea_lower.startswith(failed) or failed.startswith(idea_lower):
            return GateResult(
                name="NEGATIV-DB",
                passed=False,
                detail=f"Known failed combination: '{failed}'",
            )

    # Check 2: Prüfe ob in NEG-003 (ImportError) genannte Keywords enthalten sind
    # NEG-003 = dynamic_sl_tp_combined ImportError
    if "dynamic_sl_tp" in idea_lower or "dynamic_sl_tp_combined" in idea_lower:
        return GateResult(
            name="NEGATIV-DB",
            passed=False,
            detail="NEG-003: dynamic_sl_tp_combined is a known failed combination (ImportError).",
        )

    # Check 3: Prüfe auf false_assumptions (NEG-005, NEG-010)
    for neg_id, entry in negativ_entries.items():
        if entry.get("category") == "failed_combination":
            # Prüfe ob der Ideen-Titel im Ideen-Text vorkommt
            title_lower = entry.get("title", "").lower()
            if title_lower and title_lower in idea_lower:
                return GateResult(
                    name="NEGATIV-DB",
                    passed=False,
                    detail=f"{neg_id}: '{entry['title']}' — {entry.get('description', '')}",
                )

    return GateResult(
        name="NEGATIV-DB",
        passed=True,
        detail="No match in Negativ-DB.",
    )


# ---------------------------------------------------------------------------
# Gate 3: DUPLIKAT
# ---------------------------------------------------------------------------


def _load_all_done_ideas() -> set[str]:
    """Lädt alle Ideen aus done/ Dateien."""
    ideas: set[str] = set()
    if not DONE_DIR.exists():
        return ideas

    for f in DONE_DIR.glob("*.txt"):
        try:
            for line in f.read_text(encoding="utf-8").split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    ideas.add(line.lower())
        except Exception as e:
            log.warning("Could not read %s: %s", f.name, e)

    return ideas


def _load_db_ideas() -> set[str]:
    """Lädt Ideen aus den strategies.db-Dateien."""
    ideas: set[str] = set()

    for db_dir in OUTPUT_DIRS:
        db_path = db_dir / "strategies.db"
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Versuche verschiedene Tabellen-Namen
            for table in ["strategies", "runs", "ideas", "queue_items"]:
                try:
                    cursor.execute(f"PRAGMA table_info({table})")
                    cols = [row["name"] for row in cursor.fetchall() if row]
                    if not cols:
                        continue

                    # Suche nach einem Feld das die Idee enthält
                    id_col = None
                    for c in ["raw", "text", "idea", "idea_text", "params"]:
                        if c in cols:
                            id_col = c
                            break

                    if id_col:
                        cursor.execute(f"SELECT {id_col} FROM {table}")
                        for row in cursor.fetchall():
                            val = row[0]
                            if val and isinstance(val, str):
                                ideas.add(val.strip().lower())
                except sqlite3.OperationalError:
                    continue

            conn.close()
        except Exception as e:
            log.debug("DB read error (%s): %s", db_path.name, e)

    return ideas


def _check_duplicate(idea: str) -> GateResult:
    """Gate 3: Prüft ob die Idee bereits getestet wurde."""
    idea_lower = idea.strip().lower()

    done_ideas = _load_all_done_ideas()
    db_ideas = _load_db_ideas()

    if idea_lower in db_ideas:
        return GateResult(
            name="DUPLIKAT",
            passed=False,
            detail=f"Already in output DB (exact match).",
        )

    # done/ check only as warning — NEG-001: most done/ files were never processed
    if idea_lower in done_ideas:
        return GateResult(
            name="DUPLIKAT",
            passed=True,
            severity="info",
            detail=f"Found in ideas/queue/done/ (NOT a blocker — NEG-001: done/ files may be from lost batches).",
        )

    # Fuzzy: Prüfe ob gleiche Konzepte (selbe Wörter, andere Reihenfolge)
    idea_words = set(idea_lower.split())
    for existing in done_ideas | db_ideas:
        existing_words = set(existing.split())
        # Wenn die gleichen Wörter → wahrscheinlich Duplikat
        if len(idea_words) > 2 and len(existing_words) > 2:
            overlap = idea_words & existing_words
            # >80% Wort-Überlappung = Duplikat
            if len(overlap) >= min(len(idea_words), len(existing_words)) * 0.8:
                if abs(len(idea_words) - len(existing_words)) <= 2:
                    if existing in db_ideas:
                        return GateResult(
                            name="DUPLIKAT",
                            passed=False,
                            detail=f"Similar to existing idea in DB: '{existing}' ({len(overlap)}/{min(len(idea_words), len(existing_words))} words match)",
                        )
                    else:
                        # done/ only — not a blocker (NEG-001)
                        return GateResult(
                            name="DUPLIKAT",
                            passed=True,
                            severity="info",
                            detail=f"Similar to idea in done/: '{existing}' ({len(overlap)}/{min(len(idea_words), len(existing_words))} words match) — not a blocker (NEG-001)",
                        )

    return GateResult(
        name="DUPLIKAT",
        passed=True,
        detail="No duplicate found.",
    )


# ---------------------------------------------------------------------------
# Gate 4: BROKEN-ALGO
# ---------------------------------------------------------------------------


def _extract_concept_names(idea: str) -> list[str]:
    """Einfache Extraktion von Konzept-/Algo-Namen aus einer Idee."""
    text = idea.strip().lower()
    found: list[str] = []

    # Bekannte Konzept-Keywords aus parser.py
    # Zone-Konzepte
    zone_keywords = {
        "fvg standard": "fvg_standard",
        "fvg std": "fvg_standard",
        "ifvg 1woche": "ifvg_1woche",
        "ifvg woche": "ifvg_1woche",
        "ifvg sameday": "ifvg_sameday",
        "ifvg same": "ifvg_sameday",
        "fvg 2tage": "fvg_2tage",
        "fvg 2t": "fvg_2tage",
        "fvg 1-2wochen": "fvg_1_2wochen",
    }
    ob_keywords = {
        "ob chaos": "ob_chaos",
        "chaos ob": "ob_chaos",
        "ob tageshoch": "ob_tageshoch",
        "ob session": "ob_session",
    }
    context_keywords = {
        "ict_fibonacci": "ict_fibonacci_levels",
        "ict_fibonacci_levels": "ict_fibonacci_levels",
        "fibonacci": "ict_fibonacci_levels",
    }
    exit_keywords = {
        "ict_trailing": "ict_trailing_stop",
        "ict_trailing_stop": "ict_trailing_stop",
        "ict_turtle": "ict_turtle_soup_multi_tf",
        "ict_turtle_soup": "ict_turtle_soup_multi_tf",
        "ict_turtle_soup_multi_tf": "ict_turtle_soup_multi_tf",
        "turtle_soup": "ict_turtle_soup_multi_tf",
        "turtle soup": "ict_turtle_soup_multi_tf",
        "ict_partial": "ict_partial_close",
        "ict_partial_close": "ict_partial_close",
    }
    timing_keywords = {
        "macro short": "macro_short",
        "macro time short": "macro_short",
        "macro long": "macro_long",
        "macro time long": "macro_long",
    }

    all_maps = {
        **zone_keywords,
        **ob_keywords,
        **context_keywords,
        **exit_keywords,
        **timing_keywords,
    }

    # Suche nach längeren Keywords zuerst
    for keyword, algo_name in sorted(all_maps.items(), key=lambda x: -len(x[0])):
        if keyword in text:
            found.append(algo_name)

    # Extrahiere auch einzelne Wörter die auf Algos verweisen
    words = text.split()
    word_to_algo = {
        "fvg": "fvg_standard",
        "ifvg": "ifvg_1woche",
        "ob": "ob_chaos",
        "atr": "atr_standard",
        "natr": "natr",
        "breakeven": "exit_breakeven",
        "hurst": "hurst_exponent",
        "manip": "manip_liquidity_sweep",
        "bos": "bos",
        "choch": "choch",
    }
    for word in words:
        if word in word_to_algo:
            algo = word_to_algo[word]
            if algo not in found:
                found.append(algo)

    return found


def _load_broken_algos() -> set[str]:
    """Ermittelt welche Algos broken sind (Config-Only-Stubs ohne run())."""
    broken: set[str] = set()

    for folder, filename in BROKEN_ALGO_FILES:
        full_path = SB_DIR / "david_bibliothek" / folder / f"{filename}.py"
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8")
            # Prüfe ob run() fehlt — diese Files haben nur @dataclass Config
            if "def run(" not in content and "def compute_" not in content:
                broken.add(filename)
                broken.add(filename.replace("_", " "))

    return broken


def _check_broken_algo(idea: str) -> GateResult:
    """Gate 4: Prüft ob die Idee einen Config-Only-Stub nutzt."""
    idea_lower = idea.strip().lower()
    broken_algos = _load_broken_algos()

    # Extrahiere genutzte Konzepte
    concepts = _extract_concept_names(idea)

    # Prüfe jedes Keyword
    for keyword in BROKEN_ALGO_KEYWORDS:
        if keyword in idea_lower:
            return GateResult(
                name="BROKEN-ALGO",
                passed=False,
                detail=f"Idea references '{keyword}' which is a Config-Only stub (no run()).",
            )

    # Prüfe extrahierte Konzepte
    for concept in concepts:
        if concept in broken_algos:
            return GateResult(
                name="BROKEN-ALGO",
                passed=False,
                detail=f"Concept '{concept}' is a Config-Only stub (no run()).",
            )

    return GateResult(
        name="BROKEN-ALGO",
        passed=True,
        detail="No Config-Only stubs detected.",
    )


# ---------------------------------------------------------------------------
# Gate 5: LOCK-STATUS
# ---------------------------------------------------------------------------


def _find_locked_algos() -> set[str]:
    """Findet alle gelockten (chmod 444) Algo-Dateien in david_bibliothek."""
    locked: set[str] = set()

    for d in BIBLIOTHEK_ALGO_DIRS:
        if not d.exists():
            continue
        for f in d.glob("*.py"):
            mode = f.stat().st_mode & 0o777
            # chmod 444 = owner write-bit fehlt
            if not bool(mode & 0o200):
                # Extrahiere den Algo-Namen (Dateiname ohne .py)
                algo_name = f.stem
                locked.add(algo_name.lower())
                # Auch mit Leerzeichen statt Unterstrichen
                locked.add(algo_name.replace("_", " ").lower())

    return locked


def _check_lock_status(idea: str) -> GateResult:
    """Gate 5: Prüft ob mindestens ein benötigter Algo unlocked ist."""
    idea_lower = idea.strip().lower()
    locked_algos = _find_locked_algos()

    # Extrahiere Konzepte aus der Idee
    concepts = _extract_concept_names(idea)

    # Prüfe jedes extrahierte Konzept
    for concept in concepts:
        if concept in locked_algos:
            return GateResult(
                name="LOCK-STATUS",
                passed=False,
                detail=f"Algo '{concept}' is locked (chmod 444). Locked algos cannot be modified. Unlock with ./sb.py unlock first.",
            )

    # Prüfe auch jedes Wort der Idee
    for word in idea_lower.split():
        word_clean = word.strip(".,!?;:")
        if word_clean in locked_algos:
            return GateResult(
                name="LOCK-STATUS",
                passed=False,
                detail=f"Algo '{word_clean}' is locked (chmod 444).",
            )

    # Spezialfall: Prüfe ob der Pfad zum broken Algo locked ist
    # Die broken Algos sind per Definition locked
    for folder, filename in BROKEN_ALGO_FILES:
        if filename.lower() in idea_lower or filename.replace("_", " ").lower() in idea_lower:
            return GateResult(
                name="LOCK-STATUS",
                passed=False,
                detail=f"Algo '{filename}' is a locked Config-Only stub (chmod 444). Cannot be used.",
            )

    return GateResult(
        name="LOCK-STATUS",
        passed=True,
        detail="All referenced algos appear unlocked.",
    )


# ---------------------------------------------------------------------------
# Main Validierung
# ---------------------------------------------------------------------------


def validate_idea(idea: str) -> ValidationResult:
    """Führt alle 5 Gates aus und gibt das Ergebnis zurück."""
    result = ValidationResult(idea=idea.strip())

    # Gate 1: SYNTAX
    g1 = _check_syntax(idea)
    result.gates.append(g1)

    # Nur weiter prüfen wenn Syntax OK
    if g1.passed:
        # Gate 2: NEGATIV-DB
        g2 = _check_negativ_db(idea)
        result.gates.append(g2)

        # Gate 3: DUPLIKAT
        g3 = _check_duplicate(idea)
        result.gates.append(g3)

        # Gate 4: BROKEN-ALGO
        g4 = _check_broken_algo(idea)
        result.gates.append(g4)

        # Gate 5: LOCK-STATUS
        g5 = _check_lock_status(idea)
        result.gates.append(g5)
    else:
        # Syntax failed → restliche Gates mit "skipped" markieren
        for name in ["NEGATIV-DB", "DUPLIKAT", "BROKEN-ALGO", "LOCK-STATUS"]:
            result.gates.append(
                GateResult(
                    name=name,
                    passed=False,
                    detail="Skipped — SYNTAX gate failed first.",
                    severity="warning",
                )
            )

    return result


# ---------------------------------------------------------------------------
# Output-Formatierung
# ---------------------------------------------------------------------------


def _format_table(result: ValidationResult) -> str:
    """Formatiert als einfache Tabelle."""
    lines: list[str] = []
    lines.append(f"Idea: {result.idea}")
    lines.append(f"{'─' * 60}")
    lines.append(f"{'GATE':<25} {'RESULT':<10} DETAIL")
    lines.append(f"{'─' * 60}")

    for gate in result.gates:
        status = "✅ PASS" if gate.passed else "❌ FAIL"
        severity_tag = f" [{gate.severity}]" if gate.severity != "error" else ""
        lines.append(f"{gate.name:<25} {status:<10}{severity_tag} {gate.detail}")

    lines.append(f"{'─' * 60}")
    if result.all_passed:
        lines.append(">>>  ALLE GATES PASSED  <<<")
    else:
        lines.append(f">>>  {len(result.failed_gates)} GATE(S) FAILED  <<<")

    return "\n".join(lines)


def _format_json(result: ValidationResult) -> str:
    """Formatiert als JSON."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validierungs-Pipeline für Strategie-Ideen (5 Gates)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Beispiele:\n"
            "  python -m sb.validate_idea \"fvg standard bull NY trail\"\n"
            "  python -m sb.validate_idea --json \"ifvg 1woche ict_turtle_soup NY trail\"\n"
            "  python -m sb.validate_idea --batch bull_filter_batch.txt\n"
        ),
    )
    parser.add_argument(
        "input",
        nargs="*",
        help="Ideen-Text oder Datei mit Ideen (eine pro Zeile)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON (default: table)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        default=None,
        help="Datei mit mehreren Ideen (eine pro Zeile)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Pfad zum Report-File (default: /tmp/phase2b_validierung_report.txt)",
    )

    args = parser.parse_args()

    # Sammle alle zu prüfenden Ideen
    ideas_to_check: list[str] = []

    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            batch_path = SB_DIR / args.batch
        if not batch_path.exists():
            print(f"Error: Batch file not found: {args.batch}", file=sys.stderr)
            return 1
        ideas_to_check.extend(
            line.strip()
            for line in batch_path.read_text(encoding="utf-8").split("\n")
            if line.strip() and not line.strip().startswith("#")
        )

    if args.input:
        # Wenn mehrere Argumente, jede als separate Idee (explizite Trennung)
        if args.json:
            # For JSON, join multi-word ideas
            if len(args.input) == 1:
                ideas_to_check.append(args.input[0])
            else:
                ideas_to_check.append(" ".join(args.input))
        else:
            # For table output, treat each arg as separate idea
            # But also try joining as one (in case shell quoting failed)
            combined = " ".join(args.input)
            # Check if combined looks like a valid single idea
            # (has direction words like bull/bear/long/short)
            if any(w in combined.lower() for w in ["bull", "bear", "long", "short"]):
                ideas_to_check.extend(args.input)
            else:
                ideas_to_check.append(combined)

    if not ideas_to_check:
        print("Error: No idea provided. Usage: python -m sb.validate_idea \"<idea>\"", file=sys.stderr)
        return 1

    # Validiere jede Idee
    results: list[ValidationResult] = []
    for idea in ideas_to_check:
        result = validate_idea(idea)
        results.append(result)

    # Output
    report_path = args.report or "/tmp/phase2b_validierung_report.txt"

    all_passed = True
    output_lines: list[str] = []

    for i, result in enumerate(results):
        if args.json:
            out = _format_json(result)
        else:
            out = _format_table(result)

        print(f"\n{'=' * 60}")
        print(out)

        output_lines.append(f"{'=' * 60}")
        output_lines.append(out)
        output_lines.append("")

        if not result.all_passed:
            all_passed = False

    # Zusammenfassung
    passed_count = sum(1 for r in results if r.all_passed)
    failed_count = sum(1 for r in results if not r.all_passed)

    summary = (
        f"\n{'=' * 60}\n"
        f"SUMMARY: {passed_count} passed, {failed_count} failed "
        f"(total: {len(results)} ideas)\n"
        f"{'=' * 60}"
    )
    print(summary)
    output_lines.append(summary)

    # Report schreiben
    write_to = Path(report_path)
    write_to.parent.mkdir(parents=True, exist_ok=True)
    write_to.write_text("\n".join(output_lines), encoding="utf-8")
    log.info("Report written to %s", write_to)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
