"""
backfill_params.py
==================
Kopiert fehlende results-Zeilen von einer Quell-DB in die Ziel-DB.

Einmalige Migration: Strategien die mit --output output_v2 gebaut wurden
haben ihre params in output_v2/builder.db aber NICHT in output/builder.db.

Aufruf:
    .venv/bin/python scripts/backfill_params.py
    .venv/bin/python scripts/backfill_params.py --dry-run
    .venv/bin/python scripts/backfill_params.py \\
        --target output/builder.db \\
        --source output_v2/builder.db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

_TARGET_DEFAULT = Path(__file__).parent.parent / "output" / "builder.db"
_SOURCE_DEFAULT = Path(__file__).parent.parent / "output_v2" / "builder.db"


def backfill(target_db: Path, source_db: Path) -> int:
    """
    Kopiert results-Zeilen fuer alle Runs die in target_db keine results haben.

    Matching: idea-String (exakt) zwischen target und source.
    run_id wird auf die target run_id remappt.

    Returns:
        Anzahl der kopierten Run-Ergebnisse (nicht Zeilen).
    """
    target = sqlite3.connect(target_db)
    source = sqlite3.connect(source_db)
    try:
        missing = target.execute("""
            SELECT id, idea FROM build_runs
            WHERE id NOT IN (SELECT DISTINCT run_id FROM results)
        """).fetchall()

        copied = 0
        try:
            for target_run_id, idea in missing:
                src_run = source.execute(
                    "SELECT id FROM build_runs WHERE idea = ? ORDER BY created_at DESC LIMIT 1",
                    (idea,),
                ).fetchone()
                if not src_run:
                    logger.debug("Keine Quell-Run fuer Idee: {}", idea)
                    continue
                src_run_id = src_run[0]

                src_results = source.execute(
                    "SELECT params, pf, winrate, num_trades, score, rank, warnings, created_at "
                    "FROM results WHERE run_id = ? ORDER BY rank",
                    (src_run_id,),
                ).fetchall()
                if not src_results:
                    logger.debug("Keine results in Quelle fuer run_id={}", src_run_id)
                    continue

                target.executemany(
                    "INSERT INTO results "
                    "(run_id, params, pf, winrate, num_trades, score, rank, warnings, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(target_run_id, *row) for row in src_results],
                )
                copied += 1
                logger.info(
                    "Kopiert: '{}' ({} Zeilen, src_run={} -> target_run={})",
                    idea,
                    len(src_results),
                    src_run_id,
                    target_run_id,
                )
            target.commit()
        except Exception:
            target.rollback()
            raise

        return copied
    finally:
        target.close()
        source.close()


@app.command()
def main(
    target: Path = typer.Option(_TARGET_DEFAULT, help="Ziel-DB (output/builder.db)"),
    source: Path = typer.Option(
        _SOURCE_DEFAULT, help="Quell-DB (output_v2/builder.db)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Nur anzeigen, nichts schreiben"
    ),
) -> None:
    if not target.exists():
        console.print(f"[red]Ziel-DB nicht gefunden: {target}[/red]")
        raise typer.Exit(1)
    if not source.exists():
        console.print(f"[red]Quell-DB nicht gefunden: {source}[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]DRY-RUN - keine Aenderungen[/yellow]")
        target_conn = sqlite3.connect(target)
        missing = target_conn.execute("""
            SELECT id, idea FROM build_runs
            WHERE id NOT IN (SELECT DISTINCT run_id FROM results)
        """).fetchall()
        target_conn.close()
        console.print(f"[cyan]{len(missing)} Runs ohne Params:[/cyan]")
        for rid, idea in missing[:30]:
            console.print(f"  ID {rid}: {idea}")
        return

    n = backfill(target_db=target, source_db=source)
    console.print(f"\n[green]Fertig: {n} Runs backgefuellt.[/green]")


if __name__ == "__main__":
    app()
