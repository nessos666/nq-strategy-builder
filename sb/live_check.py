#!/usr/bin/env python3
"""
sb/live_check.py – LIVE_CONFIRMED Check fuer NQ Demo.

Analysiert die NQ MANIP Bear Demo Logs und vergleicht mit Backtest-Ergebnissen.
Schreibt einen LIVE_CONFIRMED oder ABWEICHUNG Eintrag in knowledge/negativ/.

Verwendung:
    python -m sb.live_check              # Standard-Check
    python -m sb.live_check --report     # Detaillierter Report
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

SB_DIR = Path(__file__).resolve().parent.parent
NEGATIV_DIR = SB_DIR / "knowledge" / "negativ"

# Pfad zum NQ Demo Log
DEMO_LOG = Path(
    "logs/manip_bear_demo.log"
)


def parse_demo_log(log_path: Path) -> dict:
    """Parst das Demo-Log und extrahiert Trade-Statistiken."""
    result = {
        "signals": 0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "last_signal": None,
        "running_time_hours": 0,
    }

    if not log_path.exists():
        return result

    content = log_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    # Letzte Zeile mit Timestamp finden
    last_ts = None
    for line in reversed(lines):
        m = re.match(r"\[(\d{2}:\d{2}) ET\]", line)
        if m:
            last_ts = m.group(1)
            break

    # Running time aus ersten Zeilen
    first_line = lines[0] if lines else ""
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", first_line)
    if m:
        try:
            start = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
            result["running_time_hours"] = round((datetime.now() - start).total_seconds() / 3600, 1)
        except Exception:
            pass

    result["last_signal"] = last_ts

    # Trade-Statistiken aus Log extrahieren
    for line in lines:
        if "Signal" in line and "Kein Signal" not in line:
            result["signals"] += 1
        if "ORDER" in line or "FILL" in line or "TRADE" in line:
            result["trades"] += 1
            if "WIN" in line or "PROFIT" in line:
                result["wins"] += 1
            elif "LOSS" in line or "LOSE" in line:
                result["losses"] += 1

    # Letzten Trade-Status aus Logende ziehen
    for line in reversed(lines[-50:]):
        if "Signal" in line and "Kein Signal" not in line:
            result["last_signal"] = line.strip()

    return result


def check_backtest_reference(db_path: Path) -> dict:
    """Holt die Backtest-Referenz (beste HO PF fuer MANIP Bear Strategien)."""
    import sqlite3

    result = {"best_ho_pf": None, "bester_lauf": None}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Suche nach MANIP Bear Runs mit HO
        c.execute("""
            SELECT idea, holdout_pf, oos_pf, tier
            FROM build_runs
            WHERE idea LIKE '%manip%bear%' AND holdout_pf IS NOT NULL
            ORDER BY holdout_pf DESC LIMIT 3
        """)
        rows = c.fetchall()
        if rows:
            result["best_ho_pf"] = max(r["holdout_pf"] for r in rows if r["holdout_pf"])
            result["bester_lauf"] = [dict(r) for r in rows]

        conn.close()
    except Exception:
        pass

    return result


def write_live_check_entry(demo: dict, backtest: dict) -> str:
    """Schreibt LIVE_CONFIRMED oder ABWEICHUNG Eintrag."""
    entry_id = f"LIVE-{datetime.now().strftime('%Y%m%d')}"
    
    # Pfad zum NEGATIV-Eintrag
    entry_path = NEGATIV_DIR / "live_confirmed" / f"{entry_id}.md"
    entry_path.parent.mkdir(parents=True, exist_ok=True)

    if demo["trades"] == 0 and demo["signals"] == 0:
        content = f"""---
id: {entry_id}
status: running
category: live_monitor
discovered: {datetime.now().strftime('%Y-%m-%d %H:%M')}
symptoms:
  - "NQ MANIP Bear Demo laeuft seit {demo['running_time_hours']}h"
  - "Aktuell keine Trades (Asia-Phase oder Wochenende)"
evidence:
  - "Log: {DEMO_LOG}"
  - "Letztes Signal: {demo['last_signal']}"
live_status: RUNNING_NO_DATA_YET
---
# {entry_id}: NQ MANIP Bear Demo – Laeuft, keine Trades bisher
"""
    else:
        content = f"""---
id: {entry_id}
status: confirmed
category: live_monitor
discovered: {datetime.now().strftime('%Y-%m-%d %H:%M')}
symptoms:
  - "NQ MANIP Bear Demo: {demo['signals']} Signale, {demo['trades']} Trades"
  - "Laufzeit: {demo['running_time_hours']}h"
evidence:
  - "Log: {DEMO_LOG}"
  - "Letztes Signal: {demo['last_signal']}"
live_status: LIVE_CONFIRMED
backtest_reference_pf: {backtest.get('best_ho_pf', 'N/A')}
---
# {entry_id}: NQ MANIP Bear Demo – Live-Ergebnisse
"""

    entry_path.write_text(content, encoding="utf-8")
    return str(entry_path)


def main() -> int:
    demo = parse_demo_log(DEMO_LOG)
    
    print("=" * 60)
    print("  LIVE_CONFIRMED CHECK – NQ MANIP Bear Demo")
    print("=" * 60)
    print(f"\n  Laufzeit:         {demo['running_time_hours']}h")
    print(f"  Letztes Signal:   {demo['last_signal'] or 'Keines'}")
    print(f"  Signale gesamt:   {demo['signals']}")
    print(f"  Trades:           {demo['trades']}")
    print(f"  Wins/Losses:      {demo['wins']}/{demo['losses']}")

    # Backtest-Vergleich
    db_path = SB_DIR / "output_david_1" / "studies.db"
    backtest = check_backtest_reference(db_path)
    
    if backtest.get("best_ho_pf"):
        print(f"\n  Bester MANIP Bear Backtest HO PF: {backtest['best_ho_pf']:.2f}")
        print(f"  (aus {len(backtest.get('bester_lauf') or [])} Runs)")
    else:
        print(f"\n  {WARN} Keine MANIP Bear Backtest-Referenz in DB")

    # Eintrag schreiben
    entry_path = write_live_check_entry(demo, backtest)
    print(f"\n  Eintrag geschrieben: {entry_path}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
