"""Zeit/ATR analysis commands for Strategy Builder."""
from pathlib import Path

import typer


def register(app) -> None:
    """Register all zeit/atr commands on the Typer app."""
    from sb.cli import console, _require_non_empty_text

    # ── zeit-performance ───────────────────────────────────────────

    @app.command(name="zeit-performance")
    def zeit_performance(
        algo_name_fragment_parts: list[str] = typer.Argument(
            ..., help="Namensfragment des Zeit-Algos"
        ),
        db: str = typer.Option(
            "output_v3/builder.db", "--db", help="Pfad zu builder.db"
        ),
        save: bool = typer.Option(
            False, "--save", help="Ergebnisse in _research/ speichern"
        ),
    ) -> None:
        from sb.zeit_analysis import analyze_zeit_performance

        algo_name_fragment = _require_non_empty_text(
            " ".join(algo_name_fragment_parts), "Algo-Name"
        )
        db_path = Path(db).expanduser().resolve()

        try:
            result = analyze_zeit_performance(algo_name_fragment, str(db_path))
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(
                f"[red]Fehler beim Lesen der Performance-Daten: {exc}[/red]"
            )
            raise typer.Exit(1) from exc

        console.print(
            f"\n[bold cyan]Zeit-Performance: [white]{algo_name_fragment}[/white][/bold cyan]"
        )
        console.print(f"Datenbank: {db_path}\n")

        if not result or not hasattr(result, "rows") or not result.rows:
            console.print("[yellow]Keine Performance-Daten gefunden.[/yellow]")
            return

        from rich.table import Table

        t = Table(title="Performance nach Zeitfenster", show_header=True)
        t.add_column("Fenster", style="cyan")
        t.add_column("Trades", justify="right")
        t.add_column("Win%", justify="right")
        t.add_column("PF", justify="right")
        t.add_column("Ø R", justify="right")

        for row in result.rows[:30]:
            t.add_row(
                row.get("window", "?"),
                str(row.get("trades", 0)),
                f"{row.get('winrate', 0):.0%}",
                f"{row.get('pf', 0):.2f}",
                f"{row.get('avg_r', 0):.2f}",
            )
        console.print(t)

        if save and hasattr(result, "to_csv"):
            import os

            os.makedirs("_research", exist_ok=True)
            csv_path = f"_research/zeit_perf_{algo_name_fragment.replace(' ', '_')}.csv"
            result.to_csv(csv_path)
            console.print(f"\n[dim]Gespeichert: {csv_path}[/dim]")

    # ── zeit-phasen ────────────────────────────────────────────────

    @app.command(name="zeit-phasen")
    def zeit_phasen(
        algo_name_fragment_parts: list[str] = typer.Argument(
            ..., help="Namensfragment des Zeit-Algos"
        ),
        db: str = typer.Option(
            "output_v3/builder.db", "--db", help="Pfad zu builder.db"
        ),
    ) -> None:
        from sb.zeit_analysis import analyze_zeit_phasen

        algo_name_fragment = _require_non_empty_text(
            " ".join(algo_name_fragment_parts), "Algo-Name"
        )
        db_path = Path(db).expanduser().resolve()

        try:
            result = analyze_zeit_phasen(algo_name_fragment, str(db_path))
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(
            f"\n[bold cyan]Zeit-Phasen: [white]{algo_name_fragment}[/white][/bold cyan]"
        )
        console.print(f"Datenbank: {db_path}\n")

        if result is None:
            console.print("[yellow]Keine Phasen-Daten gefunden.[/yellow]")
            return

        if isinstance(result, dict) and "phases" in result:
            from rich.table import Table

            t = Table(title="Tagesphasen-Analyse", show_header=True)
            t.add_column("Phase", style="cyan")
            t.add_column("Trades", justify="right")
            t.add_column("Win%", justify="right")
            t.add_column("PF", justify="right")

            for phase in result.get("phases", [])[:20]:
                t.add_row(
                    phase.get("name", "?"),
                    str(phase.get("trades", 0)),
                    f"{phase.get('winrate', 0):.0%}",
                    f"{phase.get('pf', 0):.2f}",
                )
            console.print(t)
        else:
            console.print(str(result)[:500])

    # ── zeit-fenster ───────────────────────────────────────────────

    @app.command(name="zeit-fenster")
    def zeit_fenster(
        algo_name_fragment_parts: list[str] = typer.Argument(
            ..., help="Namensfragment des Zeit-Algos"
        ),
        db: str = typer.Option(
            "output_v3/builder.db", "--db", help="Pfad zu builder.db"
        ),
        window: int = typer.Option(
            30, "--window", "-w", help="Fenstergröße in Minuten"
        ),
    ) -> None:
        from sb.zeit_analysis import analyze_zeit_fenster

        algo_name_fragment = _require_non_empty_text(
            " ".join(algo_name_fragment_parts), "Algo-Name"
        )
        db_path = Path(db).expanduser().resolve()

        try:
            result = analyze_zeit_fenster(algo_name_fragment, str(db_path), window)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(
            f"\n[bold cyan]Zeit-Fenster: [white]{algo_name_fragment}[/white] "
            f"({window}min)[/bold cyan]"
        )

        if result is None:
            console.print("[yellow]Keine Fenster-Daten.[/yellow]")
            return

        if isinstance(result, dict):
            from rich.table import Table

            for section_name, rows in result.items():
                if not rows:
                    continue
                console.print(f"\n[bold]{section_name}[/bold]")
                t = Table(show_header=True)
                if rows and isinstance(rows[0], dict):
                    for key in rows[0]:
                        t.add_column(key, justify="right" if key != "window" else "left")
                    for row in rows[:25]:
                        t.add_row(*[str(row.get(k, "")) for k in rows[0]])
                console.print(t)
        else:
            console.print(str(result)[:500])

    # ── atr-stats ──────────────────────────────────────────────────

    @app.command(name="atr-stats")
    def atr_stats(
        db: str = typer.Option(
            "output_v3/builder.db", "--db", help="Pfad zu builder.db"
        ),
        percentile: float = typer.Option(
            50.0, "--pct", "-p", help="Perzentil für ATR (default: 50)"
        ),
    ) -> None:
        from sb.atr_analysis import analyze_atr_stats

        db_path = Path(db).expanduser().resolve()

        try:
            result = analyze_atr_stats(str(db_path), percentile)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(
            f"\n[bold cyan]ATR-Statistik (P{percentile:.0f})[/bold cyan]"
        )
        console.print(f"Datenbank: {db_path}\n")

        if result is None:
            console.print("[yellow]Keine ATR-Daten.[/yellow]")
            return

        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float):
                    console.print(f"  {key}: {value:.2f}")
                else:
                    console.print(f"  {key}: {value}")
        else:
            console.print(str(result)[:500])

    # ── regime-test ────────────────────────────────────────────────

    @app.command(name="regime-test")
    def regime_test_cmd(
        algo_name: str = typer.Argument(..., help="Name des Algos"),
    ) -> None:
        from sb.regime_test import run_regime_test

        try:
            result = run_regime_test(algo_name)
            console.print(f"\n[bold cyan]Regime-Test: {algo_name}[/bold cyan]")
            console.print(str(result)[:1000])
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc
