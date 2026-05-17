"""
param_sensitivity.py
====================
Analysiert Parameter-Sensitivitaet fuer eine gegebene run_id.

Zeigt fuer jeden numerischen Parameter:
  - PF pro Werte-Bucket (5 Buckets)
  - Stability Score (hoch = robust, niedrig = noise-sensitiv)

Aufruf:
    # Zuerst eine run_id mit Trials finden:
    .venv/bin/python -c "
    import sqlite3
    conn = sqlite3.connect('output/builder.db')
    rows = conn.execute('SELECT run_id, COUNT(*) FROM results GROUP BY run_id HAVING COUNT(*) > 10 ORDER BY COUNT(*) DESC LIMIT 5').fetchall()
    for r in rows: print(r)
    conn.close()
    "

    .venv/bin/python scripts/param_sensitivity.py --run-id <run_id>
    .venv/bin/python scripts/param_sensitivity.py --run-id 310 --db output/builder.db
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

DB_DEFAULT = Path(__file__).parent.parent / "output" / "builder.db"
N_BUCKETS = 5
STABILITY_WARN = 2.0  # Stability Score unter diesem Wert = Warnung


def analyze_sensitivity(db_path: Path, run_id: int) -> list[dict[str, Any]]:
    """
    Laedt alle results fuer run_id, analysiert Parameter-Sensitivitaet.

    Returns:
        Liste von dicts: param, buckets, stability_score, is_sensitive,
                         min_val, max_val, n_trials
        Sortiert nach stability_score aufsteigend (sensitiv zuerst).
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT params, pf FROM results WHERE run_id = ? AND pf IS NOT NULL AND pf < 500.0",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    parsed: list[tuple[dict, float]] = []
    for params_str, pf in rows:
        try:
            params = json.loads(params_str) if params_str else {}
        except (json.JSONDecodeError, TypeError):
            continue
        parsed.append((params, pf))

    if not parsed:
        return []

    # Numerische Parameter identifizieren
    numeric_keys: set[str] = set()
    for params, _ in parsed:
        for k, v in params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_keys.add(k)

    # Werte + PF pro Parameter sammeln
    param_data: dict[str, list[tuple[float, float]]] = {k: [] for k in numeric_keys}
    for params, pf in parsed:
        for k in numeric_keys:
            if (
                k in params
                and isinstance(params[k], (int, float))
                and not isinstance(params[k], bool)
            ):
                param_data[k].append((float(params[k]), pf))

    results = []
    for param_name, value_pf_pairs in param_data.items():
        if len(value_pf_pairs) < N_BUCKETS:
            continue

        values = np.array([v for v, _ in value_pf_pairs])
        # NEU: Skip wenn kein Variationsbereich
        if values.min() == values.max():
            continue
        pfs = np.array([p for _, p in value_pf_pairs])

        edges = np.linspace(values.min(), values.max(), N_BUCKETS + 1)
        bucket_pfs: list[float] = []
        bucket_labels: list[str] = []

        for i in range(N_BUCKETS):
            lo, hi = edges[i], edges[i + 1]
            if i < N_BUCKETS - 1:
                mask = (values >= lo) & (values < hi)
            else:
                mask = (values >= lo) & (values <= hi)
            if mask.sum() == 0:
                continue
            avg_pf = float(pfs[mask].mean())
            bucket_pfs.append(avg_pf)
            bucket_labels.append(f"{lo:.2f}-{hi:.2f}")

        if not bucket_pfs:
            continue

        std = float(np.std(bucket_pfs))
        stability_score = round(1.0 / (std + 0.001), 2)
        is_sensitive = stability_score < STABILITY_WARN

        results.append(
            {
                "param": param_name,
                "buckets": list(zip(bucket_labels, [round(p, 3) for p in bucket_pfs])),
                "stability_score": stability_score,
                "is_sensitive": is_sensitive,
                "min_val": round(float(values.min()), 3),
                "max_val": round(float(values.max()), 3),
                "n_trials": len(value_pf_pairs),
            }
        )

    return sorted(results, key=lambda x: x["stability_score"])


@app.command()
def main(
    run_id: int = typer.Option(..., "--run-id", help="run_id aus build_runs"),
    db: Path = typer.Option(DB_DEFAULT, help="Pfad zur builder.db"),
) -> None:
    if not db.exists():
        logger.error(f"DB nicht gefunden: {db}")
        raise typer.Exit(1)

    results = analyze_sensitivity(db, run_id)

    if not results:
        console.print(f"[yellow]Keine Trials fuer run_id={run_id} gefunden.[/yellow]")
        raise typer.Exit(0)

    n_trials = results[0]["n_trials"]
    console.print(
        f"\n[bold]Parameter Sensitivity - run_id={run_id}[/bold] ({n_trials} Trials)\n"
    )

    for r in results:
        color = "red" if r["is_sensitive"] else "green"
        label = "NOISE-SENSITIV" if r["is_sensitive"] else "STABIL"
        console.print(
            f"[{color}]{label}[/{color}]  [cyan]{r['param']}[/cyan]  "
            f"(Stability: {r['stability_score']:.1f})  "
            f"Range: {r['min_val']} - {r['max_val']}"
        )

        table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
        table.add_column("Wertebereich")
        table.add_column("Avg PF", justify="right")
        for label_str, avg_pf in r["buckets"]:
            pf_color = (
                "green" if avg_pf >= 1.5 else ("yellow" if avg_pf >= 1.0 else "red")
            )
            table.add_row(label_str, f"[{pf_color}]{avg_pf:.3f}[/{pf_color}]")
        console.print(table)
        console.print()

    sensitive_count = sum(1 for r in results if r["is_sensitive"])
    if sensitive_count:
        console.print(
            f"[red]{sensitive_count} noise-sensitive Parameter - "
            f"erwaege diese zu fixieren oder zu entfernen.[/red]"
        )


if __name__ == "__main__":
    app()
