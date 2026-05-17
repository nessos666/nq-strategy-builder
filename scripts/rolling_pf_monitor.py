"""
rolling_pf_monitor.py
=====================
Zeigt PF-Trend pro Strategie-Idee über alle Build-Runs.
Alarm (rot) wenn PF von >= 2.0 auf < 1.5 gefallen ist.

Aufruf:
    .venv/bin/python scripts/rolling_pf_monitor.py
    .venv/bin/python scripts/rolling_pf_monitor.py --db output/builder.db
    .venv/bin/python scripts/rolling_pf_monitor.py --tier A
    .venv/bin/python scripts/rolling_pf_monitor.py --decay
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

DB_DEFAULT = Path(__file__).parent.parent / "output" / "builder.db"
ARTIFACT_PF = 500.0  # PF >= 500 = Artefakt (z.B. gross_loss=0)
DECAY_FROM = 2.0
DECAY_TO = 1.5
TREND_UP = 0.1
TREND_DOWN = -0.1


def load_pf_trends(db_path: Path) -> list[dict[str, Any]]:
    """Liest build_runs, gruppiert nach idea, berechnet PF-Trend."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT idea, avg_oos_pf, tier, created_at, session
            FROM build_runs
            WHERE avg_oos_pf < ?
            ORDER BY idea, created_at ASC
            """,
            (ARTIFACT_PF,),
        ).fetchall()
    finally:
        conn.close()

    groups: dict[str, list[tuple]] = {}
    for idea, pf, tier, created_at, session in rows:
        groups.setdefault(idea, []).append((pf, tier, created_at, session))

    trends = []
    for idea, entries in groups.items():
        pfs = [e[0] for e in entries]
        first_pf = pfs[0]
        last_pf = pfs[-1]
        delta = last_pf - first_pf
        runs = len(entries)
        last_tier = entries[-1][1]
        session = entries[-1][3] or ""

        if delta > TREND_UP:
            trend = "↑"
        elif delta < TREND_DOWN:
            trend = "↓"
        else:
            trend = "→"

        decay_alarm = first_pf >= DECAY_FROM and last_pf < DECAY_TO

        trends.append(
            {
                "idea": idea,
                "runs": runs,
                "first_pf": round(first_pf, 3),
                "last_pf": round(last_pf, 3),
                "delta": round(delta, 3),
                "trend": trend,
                "last_tier": last_tier,
                "session": session,
                "decay_alarm": decay_alarm,
            }
        )

    return sorted(trends, key=lambda x: x["last_pf"], reverse=True)


@app.command()
def main(
    db: Path = typer.Option(DB_DEFAULT, help="Pfad zur builder.db"),
    tier: str = typer.Option("", help="Filter nach Tier (A/B/C)"),
    decay_only: bool = typer.Option(False, "--decay", help="Nur Decay-Alarme zeigen"),
) -> None:
    if not db.exists():
        logger.error(f"DB nicht gefunden: {db}")
        raise typer.Exit(1)

    trends = load_pf_trends(db)

    if tier:
        trends = [t for t in trends if t["last_tier"] == tier.upper()]
    if decay_only:
        trends = [t for t in trends if t["decay_alarm"]]

    table = Table(
        title=f"Rolling PF Monitor – {len(trends)} Strategien", show_lines=False
    )
    table.add_column("Idea", style="cyan", max_width=40)
    table.add_column("Session", style="dim")
    table.add_column("Runs", justify="right")
    table.add_column("Tier", justify="center")
    table.add_column("Erster PF", justify="right")
    table.add_column("Letzter PF", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Trend")
    table.add_column("Alarm")

    for t in trends:
        alarm_str = "🔴 DECAY" if t["decay_alarm"] else ""
        delta_str = f"{t['delta']:+.3f}"
        tier_color = {"A": "green", "B": "yellow", "C": "red"}.get(
            t["last_tier"], "white"
        )
        table.add_row(
            t["idea"],
            t["session"],
            str(t["runs"]),
            f"[{tier_color}]{t['last_tier']}[/{tier_color}]",
            f"{t['first_pf']:.3f}",
            f"{t['last_pf']:.3f}",
            delta_str,
            t["trend"],
            alarm_str,
        )

    console.print(table)
    decay_count = sum(1 for t in trends if t["decay_alarm"])
    if decay_count:
        console.print(f"\n[red]⚠ {decay_count} Strategie(n) mit Decay-Alarm![/red]")


if __name__ == "__main__":
    app()
