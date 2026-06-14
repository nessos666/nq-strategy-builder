"""Maintenance commands for Strategy Builder: maintain, inspect, lock/unlock."""
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table


def register(app) -> None:
    """Register all maintenance commands on the Typer app."""
    from sb.cli import (
        console,
        _backup_db,
        _cleanup_old_studies,
        _print_strategie_baumaschinen_lager,
        _resolve_backtest_data_path,
        _DEFAULT_SOURCES,
        DEFAULT_SIGNAL_ALGO_DIRS,
        resolve_pda_library_dirs,
    )
    from sb.inspect import find_algo_file
    from sb.memory.db import BuilderDB

    # ── strategie-baumaschinen-lager ──────────────────────────────

    @app.command(name="strategie-baumaschinen-lager")
    def strategie_baumaschinen_lager() -> None:
        """Zeigt den aktuellen Stand aller output-Ordner (Strategien pro Tier)."""
        _print_strategie_baumaschinen_lager()

    # ── maintain ──────────────────────────────────────────────────

    @app.command()
    def maintain(
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
    ) -> None:
        """DB-Wartung: Metadaten rückfüllen, studies.db aufräumen, DB-Backup erstellen."""
        db_path = Path(output) / "builder.db"
        if not db_path.exists():
            console.print(f"[yellow]Keine DB gefunden: {db_path}[/yellow]")
            raise typer.Exit(code=0)

        console.print("[bold]Strategie Builder Wartung[/bold]\n")

        db = BuilderDB(db_path=db_path)
        try:
            updated = db.backfill_missing_metadata()
            console.print(f"✅ {updated} Runs mit session/is_robust rückgefüllt")
        finally:
            db.close()

        studies_path = Path(output) / "studies.db"
        deleted = _cleanup_old_studies(studies_path, max_age_days=30)
        if deleted > 0:
            console.print(f"✅ {deleted} alte Optuna-Studies gelöscht (> 30 Tage)")
        else:
            console.print("✅ studies.db – nichts zu bereinigen")

        bak = _backup_db(Path(output) / "builder.db")
        if bak:
            console.print(f"✅ DB-Backup erstellt: {bak.name}")

    # ── inspect ───────────────────────────────────────────────────

    @app.command()
    def inspect(
        algo_name: str = typer.Argument(
            ..., help="Name oder Teilstring des Bausteins (z.B. 'FVG 2Tage')"
        ),
        window: int = typer.Option(
            30, "--window", "-w", help="Fenstergröße in Minuten (default: 30)"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option(
            "", "--dir", "-d", help="Zusätzliches Verzeichnis für Algo-Suche"
        ),
        heatmap: bool = typer.Option(
            False, "--heatmap", help="Zeige Heatmap: Zeit × Wochentag"
        ),
        events_only: bool = typer.Option(
            False, "--events-only", help="Nur 0→1 Übergänge zählen (kein ffill-Rauschen)"
        ),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten in _research/ speichern"
        ),
    ) -> None:
        """Baustein-Inspektor: Zeigt wann ein Algo Signale erzeugt (nach NY-Zeitfenster)."""
        from sb.inspect import inspect_algo

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        console.print(
            f"\n[bold cyan]Baustein-Inspektor: [white]{algo_name}[/white][/bold cyan]"
        )
        console.print(f"Fenstergröße: {window} min | {len(algo_dirs)} Verzeichnis(se)\n")

        try:
            result = inspect_algo(
                name=algo_name,
                algo_dirs=algo_dirs,
                data_path=data_path,
                window_minutes=window,
                with_heatmap=heatmap,
                events_only=events_only,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen: {exc}[/red]")
            raise typer.Exit(1)

        modus = (
            "[yellow]Events-Only (0→1)[/yellow]"
            if result.events_only
            else "Alle aktiven Bars"
        )
        console.print(f"Algo:   [green]{result.algo_file.name}[/green]")
        console.print(f"Bars:   {result.total_bars:,} über {result.total_days} Handelstage")
        console.print(f"Modus:  {modus}")
        console.print(f"Signale: {', '.join(result.signal_cols) or '–'}\n")

        if not result.windows:
            console.print("[yellow]Keine Signale im Datensatz gefunden.[/yellow]")
            return

        max_total = max(w["total"] for w in result.windows)

        t = Table(
            title=f"Signal-Verteilung – NY-Zeit ({window}-min-Fenster)", show_header=True
        )
        t.add_column("Fenster (NY)", style="cyan", width=12)
        t.add_column("Bull ↑", justify="right", style="green")
        t.add_column("Bear ↓", justify="right", style="red")
        t.add_column("Other", justify="right")
        t.add_column("Total", justify="right", style="bold")
        t.add_column("Ø/Tag", justify="right", style="yellow")
        t.add_column("Balken", width=28)

        for w in result.windows:
            bar_len = int(w["total"] / max_total * 26) if max_total > 0 else 0
            t.add_row(
                w["window"],
                str(w["bull"]),
                str(w["bear"]),
                str(w["other"]),
                str(w["total"]),
                f"{w['rate_per_day']:.2f}",
                f"[blue]{'█' * bar_len}[/blue]",
            )

        console.print(t)
        grand_total = sum(w["total"] for w in result.windows)
        console.print(
            f"\nGesamt: [bold]{grand_total:,}[/bold] Signale | "
            f"Ø {grand_total / result.total_days:.1f}/Tag | "
            f"{len(result.windows)} aktive Fenster"
        )

        if heatmap and hasattr(result, "heatmap") and result.heatmap is not None:
            console.print("\n[bold cyan]Heatmap (Zeit × Wochentag):[/bold cyan]")
            ht = result.heatmap
            days = ["Mo", "Di", "Mi", "Do", "Fr"]
            hm_table = Table(show_header=True)
            hm_table.add_column("Zeit", style="cyan")
            for d in days:
                hm_table.add_column(d, justify="right")
            for i, window_name in enumerate(ht.get("windows", [])):
                row = [window_name]
                for j in range(5):
                    val = ht.get("data", [])[i][j] if i < len(ht.get("data", [])) else 0
                    row.append(str(val) if val > 0 else "–")
                hm_table.add_row(*row)
            console.print(hm_table)

    # ── lock / unlock / lock-status ────────────────────────────────

    @app.command(name="lock")
    def lock_algo(
        algo_name: str = typer.Argument(..., help="Name des Algo (Teilstring reicht)"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
    ) -> None:
        """Algo-Datei auf read-only setzen (chmod 444) – schützt verifizierte Bausteine."""
        import os

        sources_path = Path(sources).expanduser() if sources else None
        try:
            _, cfg, _ = _resolve_backtest_data_path(sources)
            algo_dirs = resolve_pda_library_dirs(
                cfg, (sources_path or _DEFAULT_SOURCES).parent
            )
        except RuntimeError:
            algo_dirs = []
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        os.chmod(algo_file, 0o444)
        console.print(f"[green]Gesperrt:[/green] {algo_file.name}")
        console.print(f"[dim]  Pfad: {algo_file}[/dim]")
        console.print(f'[dim]  Entsperren: ./sb.py unlock "{algo_name}"[/dim]')

    @app.command(name="unlock")
    def unlock_algo(
        algo_name: str = typer.Argument(..., help="Name des Algo (Teilstring reicht)"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
    ) -> None:
        """Algo-Datei entsperren (chmod 644) – für temporäre Bearbeitung."""
        import os

        sources_path = Path(sources).expanduser() if sources else None
        try:
            _, cfg, _ = _resolve_backtest_data_path(sources)
            algo_dirs = resolve_pda_library_dirs(
                cfg, (sources_path or _DEFAULT_SOURCES).parent
            )
        except RuntimeError:
            algo_dirs = []
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        os.chmod(algo_file, 0o644)
        console.print(f"[yellow]Entsperrt:[/yellow] {algo_file.name}")
        console.print(
            f'[yellow]  ACHTUNG: Danach wieder sperren mit ./sb.py lock "{algo_name}"[/yellow]'
        )

    @app.command(name="lock-status")
    def lock_status(
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
    ) -> None:
        """Zeigt Lock-Status aller Algo-Dateien in der david_bibliothek."""
        sources_path = Path(sources).expanduser() if sources else None
        try:
            _, cfg, _ = _resolve_backtest_data_path(sources)
            algo_dirs = resolve_pda_library_dirs(
                cfg, (sources_path or _DEFAULT_SOURCES).parent
            )
        except RuntimeError:
            algo_dirs = []
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)

        console.print("\n[bold cyan]Algo Lock-Status[/bold cyan]")

        locked_count = 0
        unlocked_count = 0
        rows = []

        for d in algo_dirs:
            d = Path(d)
            if not d.exists():
                continue
            for f in sorted(d.glob("*.py")):
                mode = f.stat().st_mode & 0o777
                is_locked = not bool(mode & 0o200)
                rows.append((is_locked, f.name, d.name))
                if is_locked:
                    locked_count += 1
                else:
                    unlocked_count += 1

        if not rows:
            console.print("[yellow]Keine Algo-Dateien gefunden.[/yellow]")
            return

        t = Table(show_header=True)
        t.add_column("Status", width=12)
        t.add_column("Datei", min_width=40)
        t.add_column("Ordner", style="dim")

        for is_locked, name, folder in rows:
            status = "[green]gesperrt[/green]" if is_locked else "[yellow]offen[/yellow]"
            t.add_row(status, name, folder)

        console.print(t)
        console.print(f"\n[dim]{locked_count} gesperrt | {unlocked_count} entsperrt[/dim]")
