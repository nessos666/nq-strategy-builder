"""
import_reports.py
=================
Parst alle report_*.md im output-Verzeichnis und traegt fehlende
Eintraege in die builder.db nach.

Abgleich: Timestamp aus Dateiname (Minuten-genau) vs. created_at in DB.
Wenn kein DB-Eintrag fuer diese Minute existiert → neuer Eintrag.

Tier-Logik (identisch mit BuilderDB.compute_and_save_tier):
    A: avg_oos_pf >= 2.0
    B: avg_oos_pf >= 1.5
    C: sonst

Aufruf:
    .venv/bin/python scripts/import_reports.py
    .venv/bin/python scripts/import_reports.py --output output --dry-run
"""

from __future__ import annotations

import re
import sqlite3
import shutil
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Report-Parser
# ---------------------------------------------------------------------------
def parse_report(path: Path) -> dict | None:
    """Extrahiert alle relevanten Felder aus einem report_*.md."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Kann Report nicht lesen: {} – {}", path, e)
        return None

    # --- Timestamp aus Dateiname ---
    ts_match = re.search(r"report_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})\.md", path.name)
    if not ts_match:
        logger.warning("Unbekanntes Dateiformat: {}", path.name)
        return None
    created_at = f"{ts_match.group(1)} {ts_match.group(2).replace('-', ':')}:00"

    # --- Idee ---
    idea_match = re.search(r"\*\*Idee:\*\*\s*(.+)", content)
    if not idea_match:
        logger.warning("Keine Idee in {}", path.name)
        return None
    idea = idea_match.group(1).strip()

    # --- Session ---
    session_match = re.search(r"\*\*Session:\*\*\s*(\S+)", content)
    session = session_match.group(1).strip() if session_match else None

    # --- Robustheit ---
    is_robust = 1 if "✅ ROBUST" in content and "NICHT ROBUST" not in content else 0

    # --- OOS PF Gesamt (aus der Gesamt-Zeile der Walk-Forward-Tabelle) ---
    pf_match = re.search(
        r"\|\s*\*\*Gesamt\*\*\s*\|\s*[–\-]\s*\|\s*\*\*([0-9.]+)\*\*", content
    )
    avg_oos_pf = float(pf_match.group(1)) if pf_match else None

    # --- Tier ---
    if avg_oos_pf is None:
        tier = "C"
    elif avg_oos_pf >= 2.0:
        tier = "A"
    elif avg_oos_pf >= 1.5:
        tier = "B"
    else:
        tier = "C"

    return {
        "idea": idea,
        "session": session,
        "is_robust": is_robust,
        "avg_oos_pf": avg_oos_pf,
        "tier": tier,
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------
@app.command()
def main(
    output: str = typer.Option("output_david_1", "--output", "-o", help="Output-Verzeichnis"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Nur simulieren, nicht schreiben"
    ),
) -> None:
    """Fehlende Reports in builder.db nachtraeglich eintragen."""
    output_dir = Path(output)
    db_path = output_dir / "builder.db"

    if not db_path.exists():
        typer.echo(f"Keine DB gefunden: {db_path}")
        raise typer.Exit(1)

    # --- Backup vor Import ---
    if not dry_run:
        backup_path = db_path.with_suffix(".db.pre_import_backup")
        shutil.copy2(db_path, backup_path)
        logger.info("DB-Backup erstellt: {}", backup_path)

    # --- Alle vorhandenen Timestamps aus DB laden ---
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Minuten-genaue Timestamps (erste 16 Zeichen: "2026-04-22 08:02")
    cur.execute("SELECT substr(created_at, 1, 16) as ts_minute FROM build_runs")
    existing_minutes: set[str] = {row["ts_minute"] for row in cur.fetchall()}
    logger.info("{} vorhandene Timestamps in DB", len(existing_minutes))

    # --- Reports parsen ---
    reports = sorted(output_dir.glob("report_*.md"))
    logger.info("{} Report-Dateien gefunden", len(reports))

    inserted = 0
    skipped = 0
    errors = 0

    for report_path in reports:
        data = parse_report(report_path)
        if data is None:
            errors += 1
            continue

        ts_minute = data["created_at"][:16]

        if ts_minute in existing_minutes:
            skipped += 1
            continue

        # Neuen Eintrag anlegen
        if not dry_run:
            cur.execute(
                """
                INSERT INTO build_runs
                    (idea, trials, created_at, avg_oos_pf, tier, session, is_robust)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["idea"],
                    50,  # Standard-Trials
                    data["created_at"],
                    data["avg_oos_pf"],
                    data["tier"],
                    data["session"],
                    data["is_robust"],
                ),
            )
            existing_minutes.add(ts_minute)

        logger.info(
            "[{}] {!r} | PF={} | Tier={}",
            "DRY" if dry_run else "INS",
            data["idea"],
            data["avg_oos_pf"],
            data["tier"],
        )
        inserted += 1

    if not dry_run:
        conn.commit()
    conn.close()

    print(
        f"\nFertig: {inserted} eingefuegt | {skipped} bereits vorhanden | {errors} Fehler"
    )
    if dry_run:
        print("(Dry-Run – keine Aenderungen geschrieben)")


if __name__ == "__main__":
    app()
