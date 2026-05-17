"""strategie_pipeline.py – Verbindet Baustein-Analyse, TV Vorfilter und sb.py Queue."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from scripts.baustein_analyse import load_build_runs, compute_matrix, DEFAULT_DB

app = typer.Typer(help="Strategie-Pipeline: scan → filter → queue")
console = Console()

QUEUE_DIR = Path(__file__).parent.parent / "ideas" / "queue"

_IDEA_TO_STRATEGY: dict[str, str] = {
    "SWEEP": "s1_8_reborn",
    "OB": "s1_8_reborn",
}


def _idea_to_strategy(idea: str) -> Optional[str]:
    """Mappt eine Idee auf ein TV-Template. Gibt None wenn kein Template gefunden."""
    idea_upper = idea.upper()
    for keyword, strategy in _IDEA_TO_STRATEGY.items():
        if keyword in idea_upper:
            return strategy
    return None


def _run_quick_check(strategy: str, n_trials: int, min_trades: int) -> Optional[dict]:
    """Führt quick_check() gegen echtes TradingView aus. Separat für Mocking."""
    from tv_kombinatorik.runner import KombinatorikRunner
    from tv_kombinatorik.mcp_bridge import MCPBridge
    from tv_kombinatorik.tv_client import TVClient

    bridge = MCPBridge()
    client = TVClient(bridge=bridge)
    runner = KombinatorikRunner(
        strategy=strategy,
        tv_client=client,
        result_dir=Path("tv_kombinatorik_results"),
    )
    return runner.quick_check(n_trials=n_trials, min_trades=min_trades)


@app.command("scan")
def scan(
    db: str = typer.Option(str(DEFAULT_DB), help="Pfad zur builder.db"),
    min_delta: float = typer.Option(0.05, help="Minimales PF-Delta für Anzeige"),
    min_count: int = typer.Option(3, help="Mindestanzahl Strategien pro Baustein"),
) -> None:
    """Zeigt starke Bausteine aus der Baustein-Analyse."""
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]FEHLER: DB nicht gefunden: {db_path}[/red]")
        raise typer.Exit(1)

    df = load_build_runs(db_path)
    matrix = compute_matrix(df)

    table = Table(title="Starke Bausteine (nach Delta PF)", show_lines=True)
    table.add_column("Baustein", style="cyan")
    table.add_column("n MIT", justify="right")
    table.add_column("Ø PF MIT", justify="right", style="green")
    table.add_column("Delta PF", justify="right")
    table.add_column("Empfehlung", justify="center")

    shown = 0
    for baustein, m in sorted(
        matrix.items(), key=lambda x: x[1]["delta"], reverse=True
    ):
        if m["count_mit"] < min_count:
            continue
        if m["delta"] < min_delta:
            continue
        delta_str = f"+{m['delta']:.4f}"
        empfehlung = "NUTZEN" if m["delta"] > 0.1 else "testen"
        table.add_row(
            baustein, str(m["count_mit"]), f"{m['pf_mit']:.4f}", delta_str, empfehlung
        )
        shown += 1

    console.print(table)
    console.print(f"\nAngezeigt: {shown} Bausteine mit delta >= {min_delta}")


@app.command("filter")
def filter_idea(
    idea: str = typer.Argument(help='Idee, z.B. "SWEEP + OB NY"'),
    n_trials: int = typer.Option(20, help="Anzahl Quick-Check Trials"),
    min_pf: float = typer.Option(1.2, help="Mindest-PF für PASS"),
    min_trades: int = typer.Option(30, help="Mindest-Trades für PASS"),
) -> None:
    """Schneller TV-Vorfilter: 20 Trials in TradingView → PASS oder FAIL."""
    strategy = _idea_to_strategy(idea)
    if strategy is None:
        console.print(f"[yellow]Kein TV-Template für '{idea}' gefunden.[/yellow]")
        console.print(
            "Verfügbare Templates: " + ", ".join(set(_IDEA_TO_STRATEGY.values()))
        )
        raise typer.Exit(1)

    console.print(
        f"Starte Quick-Check: {idea} → Template '{strategy}' | {n_trials} Trials..."
    )
    result = _run_quick_check(strategy, n_trials, min_trades)

    if result is None:
        console.print(
            f"FAIL – Kein gültiges Ergebnis (alle Trials < {min_trades} Trades)"
        )
        raise typer.Exit(0)

    pf = result["pf"]
    verdict = "PASS" if pf >= min_pf else "FAIL"
    console.print(
        f"\n{verdict} | PF={pf:.3f} | WR={result['wr']:.1f}% | Trades={result['trades']}"
    )
    console.print(f"Best Params: {result['params']}")

    if pf >= min_pf:
        console.print(f'\nNächster Schritt: pipeline queue "{idea}"')


def _next_pipeline_batch_num(queue_dir: Path) -> int:
    """Findet die nächste freie Batch-Nummer für pipeline-Dateien."""
    existing = list(queue_dir.glob("batch_pipeline_*.txt"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


@app.command("queue")
def add_to_queue(
    idea: str = typer.Argument(help='Hauptidee, z.B. "INSIDE_DAY + MB NY"'),
    extra: list[str] = typer.Option([], help="Weitere Ideen hinzufügen"),
) -> None:
    """Legt eine Idee als Batch-Datei in die sb.py Queue."""
    all_ideas = [idea] + list(extra)

    num = _next_pipeline_batch_num(QUEUE_DIR)
    filename = QUEUE_DIR / f"batch_pipeline_{num:03d}.txt"

    content = "\n".join(all_ideas) + "\n"
    filename.write_text(content)

    console.print(f"Queue-Datei erstellt: {filename.name}")
    for i in all_ideas:
        console.print(f"   {i}")
    console.print("\nqueue_runner holt die Datei automatisch ab.")


if __name__ == "__main__":
    app()
