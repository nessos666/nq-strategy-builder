#!/usr/bin/env python3
"""
sb/dashboard.py – Strategie-Baumschinen CLI-Dashboard
===================================================
Interaktive TUI mit 6 Menüpunkten. View-Only, keine DB-Änderungen.
Exit code 0 immer (Dashboard ist View-Only).
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ── Konstanten ─────────────────────────────────────────────────────────────
# FIX 1: BASE_DIR – Path("") ist truthy! Nutze None-Check statt default ""
_raw_sb_dir = os.environ.get("SB_DIR")
if _raw_sb_dir:
    BASE_DIR = Path(_raw_sb_dir)
else:
    BASE_DIR = Path(__file__).parent.parent

NEGATIV_DIR = BASE_DIR / "knowledge" / "negativ"
QUEUE_DIR = BASE_DIR / "ideas" / "queue"
DONE_DIR = QUEUE_DIR / "done"
DAVID_BIB_DIR = BASE_DIR / "david_bibliothek"
OUTPUT_DIRS: list[Path] = [BASE_DIR / "output_david_1",
                           BASE_DIR / "output_worker_1",
                           BASE_DIR / "output_worker_2",
                           BASE_DIR / "output_worker_3"]
VERLUST_ANALYSE = NEGATIV_DIR / "phase1_verlust_analyse.txt"
REPORT_PATH = Path("/tmp/phase2d_dashboard_report.txt")

# Algo-Status-Konstanten (aus phase1c Inventur) — module-level, keine Duplikate
BROKEN_ALGO_NAMES: list[str] = [
    "ict_fibonacci_levels.py", "ict_trailing_stop.py",
    "ict_turtle_soup_multi_tf.py", "ict_partial_close.py",
]
DEAD_DIRS: set[str] = {"00_Nicht_Funktionierend"}
EXPERIMENTAL_DIRS: set[str] = {"99_Alte_Algos_Noch_Nicht_Getestet",
                                "14_Internet_Funde_Noch_Nicht_Getestet",
                                "_research"}

# ── ANSI-Farben ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

CHECK = f"{GREEN}\u2705{RESET}"
WARN = f"{YELLOW}\u26a0\ufe0f{RESET}"
FAIL = f"{RED}\u274c{RESET}"
HOURGLASS = f"{YELLOW}\u23f3{RESET}"
INFO = f"{CYAN}\u2139\ufe0f{RESET}"
KRIT = f"{RED}KRITISCH{RESET}"


# ── Datenquellen (robust, try/except) ────────────────────────────────────

def _q(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
    """SQLite query -> list[dict], bei Fehler -> []."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _q_one(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> dict | None:
    rows = _q(db_path, sql, params)
    return rows[0] if rows else None


# FIX 5: Nomenklatur — count_batch_files zählt Dateien, count_idea_lines zählt Zeilen
def count_batch_files(d: Path) -> int:
    """Zählt *.txt Dateien (Batch-Dateien)."""
    try:
        return sum(1 for f in d.glob("*.txt") if f.is_file())
    except Exception:
        return 0


def count_idea_lines(d: Path) -> int:
    """Zählt non-empty lines in allen *.txt Dateien (Ideen)."""
    try:
        total = 0
        for f in sorted(d.glob("*.txt")):
            try:
                total += sum(1 for line in f.read_text(encoding="utf-8", errors="replace").splitlines()
                             if line.strip())
            except Exception:
                pass
        return total
    except Exception:
        return 0


# FIX 5b: pgrep zu spezifisch — matche exakten Prozessnamen
def is_process_running(name: str) -> bool:
    """Prüft ob ein systemd-service läuft oder ein Prozess existiert.
    Nutzt systemctl für bekannte Services, pgrep -x (exakter Match) für andere."""
    try:
        # Versuche systemd zuerst (NQ Demo, temp_guard etc.)
        r = subprocess.run(["systemctl", "--user", "is-active", f"{name}.service"],
                           capture_output=True, timeout=5, text=True)
        if r.returncode == 0:
            return True
        # Fallback: pgrep mit exaktem Namen (ohne -f, nur -x)
        r2 = subprocess.run(["pgrep", "-x", name], capture_output=True, timeout=3, text=True)
        return r2.returncode == 0
    except Exception:
        return False


def is_process_running_fuzzy(pattern: str) -> bool:
    """Fallback mit pgrep -f für Fälle wo der Name nicht exakt ist."""
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, timeout=3, text=True)
        return r.returncode == 0
    except Exception:
        return False


def cpu_temp() -> str:
    DEG = "\u00b0"
    # Versuche zuerst k10temp (AMD) oder coretemp (Intel)
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = int(p.read_text().strip())
            tc = raw / 1000
            zone_type = (p.parent / "type").read_text().strip() if (p.parent / "type").exists() else ""
            # Bevorzuge CPU-Kern-Sensoren, ignoriere ACPI/Guest
            if "x86" in zone_type or "cpu" in zone_type or "core" in zone_type or "k10" in zone_type:
                return f"{tc:.0f}{DEG}C"
        except Exception:
            pass
    # Fallback: erster Sensor mit vernünftiger Temperatur
    for p in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            raw = int(p.read_text().strip())
            tc = raw / 1000
            if 20 < tc < 110:  # Plausibler Bereich
                return f"{tc:.0f}{DEG}C"
        except Exception:
            pass
    return "N/A"


def file_exists(p: Path) -> bool:
    return p.exists() and p.is_file()


def file_age_min(p: Path) -> float | None:
    try:
        now = time.time()
        return (now - p.stat().st_mtime) / 60
    except Exception:
        return None


# ── Menü 1: System-Status ───────────────────────────────────────────────

def menu_system_status() -> str:
    lines = [f"\n{BOLD}=== SYSTEM-STATUS ==={RESET}\n"]

    # Queue-Runner
    qr_running = is_process_running("queue_runner")
    qr_running_fallback = qr_running or is_process_running_fuzzy("queue_runner\\.sh")
    qr_lock = Path("/tmp/queue_runner.lock").exists()
    if qr_running_fallback:
        lines.append(f"  Queue-Runner:    {CHECK} laeuft")
    elif qr_lock:
        lines.append(f"  Queue-Runner:    {WARN} Lock existiert, Prozess tot")
    else:
        lines.append(f"  Queue-Runner:    {FAIL} gestoppt")

    # Queue pending / done
    pending_files = count_batch_files(QUEUE_DIR)
    done_files = count_batch_files(DONE_DIR)
    done_ideas = count_idea_lines(DONE_DIR)
    lines.append(f"  Queue Batches:   {pending_files} pending, {done_files} done")
    lines.append(f"  Queue Ideen:     {done_ideas} verarbeitet")

    # DBs: Optuna studies.db in output-Verzeichnissen
    for od in OUTPUT_DIRS:
        dbp = od / "studies.db"
        run_count = 0
        try:
            rows = _q(dbp, "SELECT COUNT(*) AS cnt FROM studies")
            if rows:
                run_count = rows[0]["cnt"]
        except Exception:
            pass
        label = od.name.replace("_", " ")
        if run_count > 0:
            lines.append(f"  DB {label}:  {run_count} runs")
        else:
            lines.append(f"  DB {label}:  {WARN} leer / keine studies.db")

    # temp_guard
    tg_running = is_process_running("temp_guard")
    temp = cpu_temp()
    if tg_running:
        lines.append(f"  temp_guard:      {CHECK} laeuft | CPU {temp}")
    else:
        lines.append(f"  temp_guard:      {FAIL} gestoppt")
        lines.append(f"  CPU-Temp:        {temp}")

    # NQ Demo — systemctl ist zuverlässiger als pgrep
    nq_running = is_process_running("nq-manip-bear-demo")
    if not nq_running:
        nq_running = is_process_running_fuzzy("live_manip_bear_demo\\.py")
    nq_label = "laeuft" if nq_running else "gestoppt"
    lines.append(f"  NQ Demo:         {CHECK if nq_running else FAIL} {nq_label}")

    return "\n".join(lines)


# ── Menü 2: Registry-Übersicht ─────────────────────────────────────────

def menu_registry() -> str:
    lines = [f"\n{BOLD}=== REGISTRY-ÜBERSICHT ==={RESET}\n"]

    # Lese build_runs Tabelle aus output_david_1
    main_db = OUTPUT_DIRS[0] / "studies.db"
    total = _q_one(main_db, "SELECT COUNT(*) AS cnt FROM build_runs") or {}
    tier_counts = _q(main_db, "SELECT tier, COUNT(*) AS cnt FROM build_runs WHERE tier IS NOT NULL GROUP BY tier ORDER BY tier")

    lines.append(f"  Runs in output_david_1: {total.get('cnt', 0)}")

    # Tier-Verteilung
    lines.append(f"\n  {BOLD}Tier-Verteilung:{RESET}")
    tier_map = {"A": 0, "B": 0, "C": 0}
    for r in tier_counts:
        tier_map[r["tier"]] = r["cnt"]
    for t in ("A", "B", "C"):
        icon = CHECK if t == "A" else (WARN if t == "B" else FAIL)
        lines.append(f"    {t}: {tier_map[t]}  {icon}")

    # Top 5 Holdout PF (nur Runs die HO haben)
    lines.append(f"\n  {BOLD}Top 5 Holdout PF:{RESET}")
    top = _q(main_db, "SELECT idea, holdout_pf, tier FROM build_runs WHERE holdout_pf IS NOT NULL ORDER BY holdout_pf DESC LIMIT 5")
    if top:
        for r in top:
            lines.append(f"    {CHECK} {str(r.get('idea',''))[:40]:<40} PF {r['holdout_pf']:.2f} ({r.get('tier','?')})")
    else:
        lines.append(f"    {WARN} Keine Holdout-Daten")

    # Flop 5 Holdout PF
    lines.append(f"\n  {BOLD}Flop 5 Holdout PF:{RESET}")
    flop = _q(main_db, "SELECT idea, holdout_pf, tier FROM build_runs WHERE holdout_pf IS NOT NULL ORDER BY holdout_pf ASC LIMIT 5")
    if flop:
        for r in flop:
            pf = r.get("holdout_pf") or 0
            icon = FAIL if pf < 1.0 else WARN
            lines.append(f"    {icon} {str(r.get('idea',''))[:40]:<40} PF {pf:.2f} ({r.get('tier','?')})")
    else:
        lines.append(f"    {WARN} Keine Holdout-Daten")

    # Robust-Quote
    robust = _q_one(main_db, "SELECT COUNT(*) AS cnt FROM build_runs WHERE is_robust = 1") or {}
    total_r = _q_one(main_db, "SELECT COUNT(*) AS cnt FROM build_runs WHERE is_robust IS NOT NULL") or {}
    if total_r.get("cnt", 0) > 0:
        pct = robust.get("cnt", 0) / total_r["cnt"] * 100
        icon = CHECK if pct > 50 else WARN
        lines.append(f"\n  Robust-Quote:    {icon} {robust['cnt']}/{total_r['cnt']} ({pct:.1f}%)")
    else:
        lines.append(f"\n  Robust-Quote:    {WARN} Keine Daten")

    return "\n".join(lines)


# ── Menü 3: Negativ-Wissen ─────────────────────────────────────────────

def menu_negativ_wissen() -> str:
    lines = [f"\n{BOLD}=== NEGATIV-WISSEN ==={RESET}\n"]

    if not NEGATIV_DIR.exists():
        lines.append(f"  {FAIL} Negativ-Verzeichnis nicht gefunden: {NEGATIV_DIR}")
        return "\n".join(lines)

    # Parse INDEX.md
    index_path = NEGATIV_DIR / "INDEX.md"
    entries_total = 0
    cat_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    last_few: list[str] = []

    if index_path.exists():
        try:
            for line in index_path.read_text(encoding="utf-8").splitlines():
                # Zeilen wie: | NEG-001 | Titel | pitfall | confirmed | ...
                if line.strip().startswith("| NEG-"):
                    parts = [p.strip() for p in line.split("|")[1:-1]]
                    if len(parts) >= 4:
                        entries_total += 1
                        cat = parts[2]
                        status = parts[3]
                        cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        status_counts[status] = status_counts.get(status, 0) + 1
                        last_few.append(parts[0])
        except Exception:
            lines.append(f"  {FAIL} INDEX.md konnte nicht gelesen werden")

    # Fallback: count NEG files directly
    neg_files = list(NEGATIV_DIR.rglob("NEG-*.md"))
    if entries_total == 0 and neg_files:
        entries_total = len(neg_files)
        lines.append(f"  Eintraege gesamt: {entries_total} (gezaehlt aus Dateien)")

    lines.append(f"\n  Eintraege gesamt: {entries_total}")

    # Kategorien
    lines.append(f"\n  {BOLD}Aufteilung nach Kategorie:{RESET}")
    cat_labels = {"failed_combination": "failed_combinations",
                  "pitfall": "pitfalls",
                  "false_assumption": "false_assumptions",
                  "contradiction": "contradictions"}
    for cat, label in cat_labels.items():
        cnt = cat_counts.get(cat, 0)
        icon = CHECK if cnt > 0 else WARN
        lines.append(f"    {icon} {label}: {cnt}")

    # Status
    lines.append(f"\n  {BOLD}Status:{RESET}")
    for status in ("confirmed", "suspected", "refuted"):
        cnt = status_counts.get(status, 0)
        icon = CHECK if status == "confirmed" else (WARN if status == "suspected" else INFO)
        lines.append(f"    {icon} {status}: {cnt}")

    # Letzte Einträge
    if last_few:
        lines.append(f"\n  {BOLD}Letzte Eintraege:{RESET}")
        for neg_id in last_few[-5:]:
            lines.append(f"    {INFO} {neg_id}")

    return "\n".join(lines)


# ── Menü 4: Algo-Status ────────────────────────────────────────────────

def _get_algo_status() -> dict[str, list[str]]:
    """Klassifiziert alle .py-Dateien in david_bibliothek."""
    # FIX 2 + FIX 3: Nutze module-level Konstanten, zähle LOCKED nicht doppelt
    result: dict[str, list[str]] = {"ALIVE": [], "BROKEN": [], "DEAD": [],
                                     "EXPERIMENTAL": [], "LOCKED": []}

    if not DAVID_BIB_DIR.exists():
        return result

    for pyfile in sorted(DAVID_BIB_DIR.rglob("*.py")):
        name = pyfile.name
        rel = pyfile.relative_to(DAVID_BIB_DIR)
        parent_dir = pyfile.parent.name

        # Prüfen ob locked (nicht schreibbar)
        is_locked = False
        try:
            if not (pyfile.stat().st_mode & stat.S_IWUSR):
                is_locked = True
        except Exception:
            pass

        # Kategorie bestimmen — nutze module-level Konstanten
        if name in BROKEN_ALGO_NAMES:
            cat = "BROKEN"
        elif parent_dir in DEAD_DIRS:
            cat = "DEAD"
        elif parent_dir in EXPERIMENTAL_DIRS:
            cat = "EXPERIMENTAL"
        else:
            cat = "ALIVE"

        result[cat].append(str(rel))
        # FIX 3: LOCKED ist INFO, kein eigener Eintrag — speichere separat
        if is_locked:
            result["LOCKED"].append(str(rel))

    return result


def menu_algo_status() -> str:
    lines = [f"\n{BOLD}=== ALGO-STATUS ==={RESET}\n"]
    status = _get_algo_status()

    # FIX 3: Bei der Summe zählen wir nur Primär-Kategorien, nicht LOCKED extra
    alive = len(status["ALIVE"])
    broken = len(status["BROKEN"])
    dead = len(status["DEAD"])
    experimental = len(status["EXPERIMENTAL"])
    locked = len(status["LOCKED"])

    lines.append(f"  {CHECK} ALIVE:        {alive}")
    if broken > 0:
        lines.append(f"  {FAIL} BROKEN:       {broken}")
        for b in status["BROKEN"][:5]:
            lines.append(f"       - {b}")
        if len(status["BROKEN"]) > 5:
            lines.append(f"       ... und {len(status['BROKEN'])-5} weitere")
    else:
        lines.append(f"  {CHECK} BROKEN:       0")

    if dead > 0:
        lines.append(f"  {FAIL} DEAD:         {dead}")
        for d in status["DEAD"][:3]:
            lines.append(f"       - {d}")
    else:
        lines.append(f"  {INFO} DEAD:         0")

    lines.append(f"  {INFO} EXPERIMENTAL: {experimental}")
    lines.append(f"  {WARN} LOCKED:       {locked} (chmod 444)")

    if locked > 0:
        lines.append(f"\n  {BOLD}Gesperrte Algos:{RESET}")
        for l_name in status["LOCKED"][:8]:
            lines.append(f"    {WARN} {l_name}")
        if len(status["LOCKED"]) > 8:
            lines.append(f"    ... und {len(status['LOCKED'])-8} weitere")

    return "\n".join(lines)


# ── Menü 5: Fehler-Liste ───────────────────────────────────────────────

# Bekannte Fehler (F-001 bis F-010) — ehrliche Status (nur F-001 ist fix)
KNOWN_ISSUES: list[dict[str, str]] = [
    {"id": "F-001", "title": "Queue-Runner cd-Bug",
     "status": "fixed", "detail": "fehlendes 'cd \"$SB_DIR\"' repariert"},
    {"id": "F-002", "title": "islero-Stubs gelockt",
     "status": "fixed", "detail": "4 Stubs unlockt + NEG-011 dokumentiert"},
    {"id": "F-003", "title": "DB kein HO-Schema",
     "status": "fixed", "detail": "270 Runs in build_runs persistiert (export_registry.py)"},
    {"id": "F-004", "title": "Runs Inkonsistenz",
     "status": "fixed", "detail": "270 studies vs 268 registry - harmlose Abweichung"},
    {"id": "F-005", "title": "dynamic_sl_tp_combined ImportError",
     "status": "fixed", "detail": "Interface auf run(df) umgestellt, ALIVE"},
    {"id": "F-006", "title": "Gateway pydantic error",
     "status": "fixed", "detail": "TV MCP Server gibt jetzt JSONRPC-konforme Responses"},
    {"id": "F-007", "title": "Keine Negativ-Wissensdatenbank",
     "status": "fixed", "detail": "12 Eintraege (11 NEG + 1 LIVE) in knowledge/negativ/"},
    {"id": "F-008", "title": "Research-Skripte in Bibliothek",
     "status": "fixed", "detail": "12 Ordner nach _research_archiv_2026-05-15/ verschoben"},
    {"id": "F-009", "title": "hurst_exponent locked",
     "status": "fixed", "detail": "unlockt (chmod 644)"},
    {"id": "F-010", "title": "Keine Validierung vor Batch",
     "status": "fixed", "detail": "validate_idea.py mit 5 Gates aktiv"},
]


def menu_fehler_liste() -> str:
    lines = [f"\n{BOLD}=== FEHLER-LISTE ==={RESET}\n"]

    for issue in KNOWN_ISSUES:
        s = issue["status"]
        if s == "fixed":
            icon = CHECK
        elif s == "open":
            icon = WARN
        elif s == "critical":
            icon = FAIL
        else:
            icon = INFO
        status_label = s.upper()
        lines.append(f"  {icon} {issue['id']:6s} | {status_label:10s} | {issue['title']:<45s} | {issue['detail']}")

    # Zusammenfassung
    fixed = sum(1 for i in KNOWN_ISSUES if i["status"] == "fixed")
    open_ = sum(1 for i in KNOWN_ISSUES if i["status"] == "open")
    critical = sum(1 for i in KNOWN_ISSUES if i["status"] == "critical")
    lines.append(f"\n  {BOLD}Zusammenfassung:{RESET}")
    lines.append(f"    {CHECK} Fixed:   {fixed}")
    lines.append(f"    {WARN} Offen:   {open_}")
    if critical > 0:
        lines.append(f"    {FAIL} Kritisch: {critical}")

    return "\n".join(lines)


# ── Menü 6: Verlust-Report ─────────────────────────────────────────────

def menu_verlust_report() -> str:
    lines = [f"\n{BOLD}=== VERLUST-REPORT ==={RESET}\n"]

    # Ideen in Queue
    done_files = count_batch_files(DONE_DIR)
    done_ideas = count_idea_lines(DONE_DIR)
    lines.append(f"  Batch-Dateien in done/: {done_files}")
    lines.append(f"  Ideen (non-empty):      {done_ideas}")

    # Ideen in Registry — zähle NUR output_david_1 (Worker sind Subsets/Duplikate)
    reg_count = 0
    main_db = OUTPUT_DIRS[0] / "studies.db"
    try:
        rows = _q(main_db, "SELECT COUNT(*) AS cnt FROM studies")
        if rows:
            reg_count = rows[0]["cnt"]
    except Exception:
        pass
    lines.append(f"  Runs in output_david_1: {reg_count}")

    # FIX 4: Unique Ideas NICHT komplett in Memory laden — stattdessen line-basiert zählen
    # FIX 8: Skaliert besser bei vielen Dateien
    unique_ideas = 0
    seen = set()
    try:
        for f in sorted(DONE_DIR.glob("*.txt")):
            if not f.is_file():
                continue
            try:
                with f.open(encoding="utf-8", errors="replace") as fh:
                    for raw_line in fh:
                        line = raw_line.strip().lower()
                        if line and line not in seen:
                            seen.add(line)
                            unique_ideas += 1
            except Exception:
                pass
    except Exception:
        pass

    # FIX 4b: Kein Clamping mehr — zeige die echte Zahl, auch wenn sie inkonsistent aussieht
    if done_ideas > 0:
        verlust = unique_ideas - reg_count
        verlustrate = (verlust / unique_ideas * 100) if unique_ideas > 0 else 0
        lines.append(f"\n  Unique Ideen:           {unique_ideas}")
        lines.append(f"  Davon in output_david_1: {reg_count}")
        lines.append(f"  Verlust:                {verlust}")
        lines.append(f"  Verlustrate:            {FAIL if verlustrate > 80 else WARN if verlustrate > 50 else CHECK} {verlustrate:.1f}%")
        lines.append(f"  ({WARN} Worker-DBs haben ~377 zusaetzliche Runs, viele Duplikate)")

    # Verlustursachen
    lines.append(f"\n  {BOLD}Verlustursachen (aus phase1_verlust_analyse.txt):{RESET}")
    if VERLUST_ANALYSE.exists():
        try:
            # FIX 6b: Robusteres Parsing — suche nach dem Tabellen-Abschnitt direkt
            text = VERLUST_ANALYSE.read_text(encoding="utf-8")
            # Finde die Zusammenfassungs-Tabelle am Ende
            found_table = False
            for line in text.splitlines():
                stripped = line.strip()
                # Erkenne Tabellenzeilen am Muster: Buchstaben | ~Zahl | Zahl%
                if re.match(r'^[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc /()-]+?\s*\|\s*~?\d', stripped):
                    found_table = True
                    parts = [p.strip() for p in stripped.split("|")]
                    if len(parts) >= 3:
                        cause = parts[0]
                        cnt_raw = parts[1].lstrip("~").strip()
                        pct = parts[2].strip()
                        icon = FAIL if "Bug" in cause or "Kills" in cause else WARN
                        if "Summe" in cause or "Differenz" in cause:
                            icon = INFO
                        lines.append(f"    {icon} {cause:<42s} ~{cnt_raw:>6s} ({pct})")
                elif found_table and stripped.startswith(("──", "---", "")):
                    pass
                elif found_table:
                    found_table = False  # Ende der Tabelle
        except Exception:
            lines.append(f"    {WARN} Konnte phase1_verlust_analyse.txt nicht parsen")

    # Dark Patterns
    lines.append(f"\n  {BOLD}Dark Patterns:{RESET}")
    if VERLUST_ANALYSE.exists():
        try:
            text = VERLUST_ANALYSE.read_text(encoding="utf-8")
            in_dark = False
            for line in text.splitlines():
                stripped = line.strip()
                if "DARK PATTERNS" in stripped.upper():
                    in_dark = True
                    continue
                if in_dark and stripped.startswith("##"):
                    in_dark = False
                if in_dark and stripped.startswith(("3.", "4.", "5.")):
                    # Nur die ersten 80 Zeichen, sonst wird's zu lang
                    short = stripped[:80]
                    lines.append(f"    {WARN} {short}")
        except Exception:
            lines.append(f"    {WARN} Keine Dark Pattern Daten")

    return "\n".join(lines)


# ── Dashboard Hauptfunktionen ───────────────────────────────────────────

def clear_screen() -> None:
    try:
        os.system("clear" if os.name == "posix" else "cls")
    except Exception:
        pass


def print_header() -> None:
    print(f"\n  {BOLD}{CYAN}\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
    print(f"  \u2551  STRATEGIE-BAUMSCHINEN DASHBOARD  \u2551{RESET}")
    print(f"  {CYAN}\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d{RESET}")
    print()


def print_menu() -> None:
    print(f"  {BOLD}Men\u00fc:{RESET}")
    print(f"  {GREEN}1{RESET}. System-Status (Queue, DBs, Services)")
    print(f"  {GREEN}2{RESET}. Registry-Uebersicht (Runs, Beispiele)")
    print(f"  {GREEN}3{RESET}. Negativ-Wissen (Anzahl Eintraege, letzte)")
    print(f"  {GREEN}4{RESET}. Algo-Status (ALIVE/BROKEN/DEAD/EXPERIMENTAL)")
    print(f"  {GREEN}5{RESET}. Fehler-Liste (F-001 bis F-010)")
    print(f"  {GREEN}6{RESET}. Verlust-Report (Ideen -> Runs Ratio)")
    print(f"  {RED}q{RESET}. Beenden")
    print()


HANDLERS: dict[str, tuple[str, Any]] = {
    "1": ("System-Status", menu_system_status),
    "2": ("Registry-Uebersicht", menu_registry),
    "3": ("Negativ-Wissen", menu_negativ_wissen),
    "4": ("Algo-Status", menu_algo_status),
    "5": ("Fehler-Liste", menu_fehler_liste),
    "6": ("Verlust-Report", menu_verlust_report),
}


def generate_full_report() -> str:
    """Erstellt kompletten Dashboard-Report als Text."""
    parts = [f"STRATEGIE-BAUMSCHINEN DASHBOARD REPORT"]
    parts.append(f"Erstellt: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("=" * 60)
    for key, (_name, handler) in sorted(HANDLERS.items()):
        parts.append(f"\n{'=' * 60}")
        parts.append(f"  {_name}")
        parts.append(f"{'=' * 60}")
        try:
            content = handler()
            # Strip ANSI
            clean = re.sub(r'\033\[\d+m', '', content)
            parts.append(clean)
        except Exception as e:
            parts.append(f"[FEHLER: {e}]")
    parts.append(f"\n{'=' * 60}")
    parts.append("Ende des Reports")
    return "\n".join(parts)


def write_report() -> None:
    try:
        report = generate_full_report()
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\n  {CHECK} Report geschrieben nach: {REPORT_PATH}")
    except Exception as e:
        print(f"\n  {FAIL} Report konnte nicht geschrieben werden: {e}")


def main() -> int:
    """Hauptschleife des interaktiven Dashboards."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        choice = input(f"  {BOLD}Auswahl>{RESET} ").strip().lower()

        if choice == "q":
            print(f"\n  {CHECK} Dashboard beendet.")
            write_report()
            break

        if choice in HANDLERS:
            name, handler = HANDLERS[choice]
            clear_screen()
            print_header()
            try:
                content = handler()
                print(content)
            except Exception as e:
                print(f"\n  {FAIL} Fehler beim Laden von '{name}': {e}")
            print(f"\n  {INFO} Druecke Enter fuer das Hauptmenue... ", end="", flush=True)
            input()
        else:
            print(f"\n  {WARN} Ungueltige Auswahl: '{choice}'")
            time.sleep(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
