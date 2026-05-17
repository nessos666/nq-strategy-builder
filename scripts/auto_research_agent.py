"""auto_research_agent.py – Karpathy-Style Auto Research fuer Baustein-Ideen.

Laeuft autom in einem Loop:
    Baseline-Idee -> sb.py bauen -> PF messen
    -> Baustein variieren -> neu bauen -> Keep/Discard

Nutzung:
    PYTHONPATH=. .venv/bin/python scripts/auto_research_agent.py run \\
        "INSIDE_DAY + MB NY" --experiments 20

Ergebnis:
    output_v3/ara_experiments.tsv – alle Experimente
    Beste Idee am Ende angezeigt (nicht automatisch uebernommen!)
"""

from __future__ import annotations

import random
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.table import Table

from scripts.baustein_analyse import extract_bausteine


# -- Konfiguration -------------------------------------------------------------

DEFAULT_DB = Path(__file__).parent.parent / "output_v3" / "builder.db"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "output_v3"
DEFAULT_RESULTS = Path(__file__).parent.parent / "output_v3" / "ara_experiments.tsv"

BAUSTEIN_POOL = [
    # Struktur
    "BOS",
    "OB",
    "FVG",
    "SWEEP",
    "BPR",
    "AMD",
    "JUDAS",
    "DISPLACEMENT",
    "EQH",
    "EQL",
    "MANIP",
    "IMBALANCE",
    "BB",
    "MMXM",
    "INSIDE_DAY",
    "MB",
    "CHOCH",
    # Zeit
    "ASIA_SWEEP",
    "LONDON_SWEEP",
    "KILLZONE",
    "NDOG",
    "NWOG",
    # Momentum
    "HURST",
    "CBDR",
    "DEALING_RANGE",
]
SESSIONS = ["NY", "LONDON", "ASIA", "RTH"]

app = typer.Typer(help="Auto-Research Agent: Karpathy-Style Strategie-Optimierung")
console = Console()


# -- Dataclass -----------------------------------------------------------------


@dataclass
class ExperimentResult:
    experiment_id: int
    idea: str
    pf: float
    status: str  # "baseline", "keep", "discard"


# -- Evaluation ----------------------------------------------------------------


def _run_sb_batch(idea: str, trials: int, output_dir: Path) -> None:
    """Fuehrt ./sb.py batch fuer eine Idee aus. Separat fuer Mocking."""
    sb_path = Path(__file__).parent.parent / "sb.py"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="ara_", delete=False
    ) as f:
        tmp = Path(f.name)
        f.write(idea + "\n")
    try:
        subprocess.run(
            [
                str(sb_path),
                "batch",
                str(tmp),
                "--trials",
                str(trials),
                "--output",
                str(output_dir),
                "--force",
            ],
            check=True,
            cwd=str(Path(__file__).parent.parent),
        )
    finally:
        tmp.unlink(missing_ok=True)


def evaluate_idea(
    idea: str,
    trials: int,
    db_path: Path,
    output_dir: Path,
) -> float:
    """Fuehrt sb.py batch aus und liest avg_oos_pf aus build_runs.

    Gibt 0.0 zurueck wenn Idee nicht in DB oder PF None.
    """
    _run_sb_batch(idea, trials, output_dir)
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT avg_oos_pf FROM build_runs WHERE idea=? ORDER BY created_at DESC LIMIT 1",
            (idea,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        return 0.0
    return float(row[0])


# -- Mutation ------------------------------------------------------------------


def _extract_session(idea: str) -> str:
    """Extrahiert Session-Token aus einer Idee. Default: NY.

    Erkennt nur SESSIONS (NY/LONDON/ASIA/RTH).
    ETH/PREMARKET werden ignoriert – der Agent generiert diese nicht.
    """
    for token in idea.upper().split():
        if token in SESSIONS:
            return token
    return "NY"


def _build_idea(bausteine: list[str], session: str) -> str:
    """Baut Ideen-String aus Baustein-Liste und Session."""
    return " + ".join(bausteine) + f" {session}"


def mutate_idea(idea: str, rng: Optional[random.Random] = None) -> str:
    """Variiert eine Idee: Baustein add/remove oder Session-Wechsel.

    Regeln:
    - max 4 Bausteine (add wird uebersprungen wenn bereits 4)
    - min 1 Baustein (remove wird uebersprungen wenn nur 1)
    - Fallback: swap_session
    """
    if rng is None:
        rng = random.Random()

    bausteine = sorted(extract_bausteine(idea))  # deterministisch sortiert
    session = _extract_session(idea)
    op = rng.choice(["add", "remove", "swap_session"])

    if op == "add" and len(bausteine) < 4:
        available = [b for b in BAUSTEIN_POOL if b not in bausteine]
        if available:
            new_b = rng.choice(available)
            return _build_idea(sorted(bausteine + [new_b]), session)

    elif op == "remove" and len(bausteine) > 1:
        remove_b = rng.choice(bausteine)
        remaining = [b for b in bausteine if b != remove_b]
        return _build_idea(remaining, session)

    # swap_session (auch als Fallback)
    other_sessions = [s for s in SESSIONS if s != session]
    if not other_sessions:
        return _build_idea(bausteine, session)
    new_session = rng.choice(other_sessions)
    return _build_idea(bausteine, new_session)


# -- Agent Loop ---------------------------------------------------------------


def run_agent(
    base_idea: str,
    n_experiments: int = 20,
    trials: int = 30,
    db_path: Path = DEFAULT_DB,
    output_dir: Path = DEFAULT_OUTPUT,
    results_file: Path = DEFAULT_RESULTS,
    _evaluate: Callable = evaluate_idea,
) -> tuple[str, list[ExperimentResult]]:
    """Karpathy-Style Loop: Baseline bauen -> Variieren -> Keep/Discard.

    Gibt (beste_idee, alle_experimente) zurueck.
    `_evaluate` ist inject-bar fuer Tests.
    """
    # Basis-Idee normalisieren: gleiche kanonische Form wie mutate_idea-Output
    base_idea = _build_idea(
        sorted(extract_bausteine(base_idea)), _extract_session(base_idea)
    )

    console.print("\n[bold green]AUTO RESEARCH AGENT[/bold green]")
    console.print(f"Base-Idee: [cyan]{base_idea}[/cyan]")
    console.print(f"Experimente: {n_experiments} | Trials: {trials}\n")

    results: list[ExperimentResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline
    console.print("[dim]Messe Baseline...[/dim]")
    baseline_pf = _evaluate(base_idea, trials, db_path, output_dir)
    console.print(f"Baseline: PF={baseline_pf:.3f}")

    results.append(
        ExperimentResult(
            experiment_id=0, idea=base_idea, pf=baseline_pf, status="baseline"
        )
    )

    best_idea = base_idea
    best_pf = baseline_pf
    rng = random.Random()

    # Experiment-Loop
    for exp_id in range(1, n_experiments + 1):
        new_idea = mutate_idea(best_idea, rng=rng)
        pf = _evaluate(new_idea, trials, db_path, output_dir)

        if pf > best_pf:
            status = "keep"
            best_idea = new_idea
            best_pf = pf
            marker = "[green]KEEP[/green]"
        else:
            status = "discard"
            marker = "[dim]discard[/dim]"

        results.append(
            ExperimentResult(experiment_id=exp_id, idea=new_idea, pf=pf, status=status)
        )
        console.print(
            f"  [{exp_id:3d}/{n_experiments}] PF={pf:.3f} {marker}  {new_idea}"
        )

    # TSV speichern
    rows = ["id\tidea\tpf\tstatus"]
    for r in results:
        rows.append(f"{r.experiment_id}\t{r.idea}\t{r.pf:.4f}\t{r.status}")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text("\n".join(rows) + "\n")

    # Summary
    kept = [r for r in results if r.status == "keep"]
    console.print("\n[bold]ERGEBNIS[/bold]")
    console.print(f"Behalten: {len(kept)} | Verworfen: {n_experiments - len(kept)}")
    console.print(f"Baseline PF: {baseline_pf:.3f}")
    console.print(f"Beste PF:    {best_pf:.3f} ({best_pf - baseline_pf:+.3f})")
    console.print(f"Beste Idee:  [cyan]{best_idea}[/cyan]")
    console.print(f"\n-> Gespeichert in {results_file}")
    console.print(
        "[yellow]-> NICHT automatisch uebernommen! David entscheidet.[/yellow]"
    )

    return best_idea, results


# -- CLI -----------------------------------------------------------------------


@app.command("run")
def run_cmd(
    base_idea: str = typer.Argument(help='Base-Idee, z.B. "INSIDE_DAY + MB NY"'),
    experiments: int = typer.Option(20, help="Anzahl Experimente"),
    trials: int = typer.Option(30, help="Trials pro sb.py-Build"),
    db: str = typer.Option(str(DEFAULT_DB), help="Pfad zur builder.db"),
    output: str = typer.Option(str(DEFAULT_OUTPUT), help="Output-Verzeichnis"),
    results_file: str = typer.Option(str(DEFAULT_RESULTS), help="TSV-Ergebnis-Datei"),
) -> None:
    """Startet den Auto-Research Agent fuer eine Basis-Idee."""
    try:
        run_agent(
            base_idea=base_idea,
            n_experiments=experiments,
            trials=trials,
            db_path=Path(db),
            output_dir=Path(output),
            results_file=Path(results_file),
        )
    except Exception as exc:
        console.print(f"[red]Fehler: {exc}[/red]")
        raise typer.Exit(1)


@app.command("show")
def show_cmd(
    results_file: str = typer.Option(str(DEFAULT_RESULTS), help="TSV-Ergebnis-Datei"),
) -> None:
    """Zeigt die letzten Experiment-Ergebnisse als Tabelle."""
    path = Path(results_file)
    if not path.exists():
        console.print(f"[red]Datei nicht gefunden: {path}[/red]")
        raise typer.Exit(1)
    lines = path.read_text().strip().split("\n")
    table = Table(title=f"ARA-Experimente ({path.name})", show_lines=True)
    table.add_column("ID", justify="right")
    table.add_column("Idee", style="cyan")
    table.add_column("PF", justify="right", style="green")
    table.add_column("Status", justify="center")
    for line in lines[1:]:  # Header ueberspringen
        parts = line.split("\t")
        if len(parts) == 4:
            color = (
                "green"
                if parts[3] == "keep"
                else ("yellow" if parts[3] == "baseline" else "dim")
            )
            table.add_row(
                parts[0], parts[1], parts[2], f"[{color}]{parts[3]}[/{color}]"
            )
    console.print(table)


if __name__ == "__main__":
    app()
