"""Core commands for Strategy Builder."""
import typer


def _read_ideas(path: "Path") -> "list[str]":
    """Liest Ideen aus Textdatei. Kommentare (#) und leere Zeilen werden ignoriert."""
    from pathlib import Path
    ideas: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ideas.append(line)
    return ideas


def register(app) -> None:
    """Register all core commands on the Typer app."""
    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path
    import json, shutil
    from rich.table import Table

    from sb.cli import console, _DEFAULT_SOURCES, DEFAULT_SIGNAL_ALGO_DIRS, resolve_pda_library_dirs
    from sb._helpers import (
        _require_non_empty_text, _require_positive_int, _require_non_negative_int,
        _resolve_backtest_data_path,
    )
    from sb.cache.signal_cache import SignalCache, SignalCacheConfig
    from sb.engine.knowledge import load_knowledge
    from sb.engine.nautilus_bridge import NautilusBridge
    from sb.engine.parser import parse_idea
    from sb.engine.walk_forward import WalkForwardEngine
    from sb.memory.db import BuilderDB
    from sb.report import generate_wf_report

    @app.command()
    def build(
        idee: str = typer.Argument(
            ..., help='Strategie-Idee, z.B. "BOS + FVG London Open"'
        ),
        trials: int = typer.Option(50, "--trials", "-t", help="Anzahl Optuna-Versuche"),
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        sources: str = typer.Option(
            "", "--sources", "-s", help="Pfad zu sources.yaml (optional)"
        ),
        min_trades: int = typer.Option(
            30, "--min-trades", help="Minimum Trades für Wertung"
        ),
    ) -> None:
        """Strategie-Idee via Parameter-Suche verfeinern."""
        try:
            idee = _require_non_empty_text(idee, "Idee")
            trials = _require_positive_int(trials, "Trials")
            min_trades = _require_non_negative_int(min_trades, "Min-Trades")
            output_dir = Path(output)
            db_path = output_dir / "builder.db"

            console.print("\n[bold green]Strategie Builder v0.1[/bold green]")
            console.print(f"Idee: [cyan]{idee}[/cyan]")
            console.print(f"Trials: {trials} | Min-Trades: {min_trades}\n")

            # --- Buchhalter: Duplikat-Check ---
            _buch_db = BuilderDB(db_path=db_path)
            try:
                prior_runs = _buch_db.find_runs_by_idea(idee)
                if prior_runs:
                    console.print(
                        f"\n[yellow]Buchhalter:[/yellow] Diese Idee wurde bereits "
                        f"[bold]{len(prior_runs)}×[/bold] gebaut."
                    )
                    best_prior = _buch_db.get_best_result_for_idea(idee)
                    if best_prior:
                        console.print(
                            f"   Bestes bisheriges Ergebnis: PF=[cyan]{best_prior['pf']:.3f}[/cyan]  "
                            f"Winrate=[cyan]{best_prior['winrate']:.1%}[/cyan]  "
                            f"Trades=[cyan]{best_prior['num_trades']}[/cyan]"
                        )
                    if not typer.confirm("Trotzdem neu bauen?", default=False):
                        console.print("[dim]Abgebrochen.[/dim]")
                        raise typer.Exit(code=0)
            finally:
                _buch_db.close()

            console.print("[dim]1/5 Parser läuft...[/dim]")
            parsed = parse_idea(idee)
            console.print(f"   Konzepte: {parsed.concepts} | Session: {parsed.session}")

            console.print("[dim]2/5 Wissen wird geladen...[/dim]")
            sources_path = Path(sources).expanduser() if sources else None
            ctx = load_knowledge(sources_path=sources_path)
            console.print(
                f"   PDA-Algos: {len(ctx.pda_algos)} | Fehler: {len(ctx.known_errors)} | Ideen: {len(ctx.ideas)}"
            )

            console.print("[dim]3/5 Signal-Cache + NautilusBridge initialisieren...[/dim]")
            data_path, cfg, holdout_start = _resolve_backtest_data_path(sources)

            cache_path = data_path.parent / "signal_cache.parquet"
            console.print(f"   Cache-Pfad: [dim]{cache_path}[/dim]")
            algo_dirs = resolve_pda_library_dirs(
                cfg, (sources_path or _DEFAULT_SOURCES).parent
            )
            if not algo_dirs:
                algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
            sc_cfg = SignalCacheConfig(
                bars_path=data_path,
                cache_path=cache_path,
                algo_dirs=algo_dirs,
            )
            sc = SignalCache(sc_cfg)
            console.print("   Signal-Cache wird gebaut (einmalig, kann dauern)...")
            sc.build(concepts=parsed.concepts if parsed.concepts else None)
            console.print("   Signal-Cache bereit.")

            bridge = NautilusBridge(
                data_path=data_path,
                cache_path=cache_path,
                algo_dirs=algo_dirs,
                max_date=holdout_start,
                concepts=parsed.concepts if parsed.concepts else None,
            )
            if holdout_start:
                console.print(f"   [yellow]Holdout gesperrt ab: {holdout_start}[/yellow]")

            console.print(
                f"[dim]4/5 Walk-Forward (3 Fenster) + Optuna {trials} Trials/Fenster...[/dim]"
            )
            console.print(
                "[dim]   (dauert ca. 3× länger als früher – dafür echte OOS-Validierung)[/dim]"
            )
            studies_path = output_dir / "studies.db"
            wfe = WalkForwardEngine(bridge=bridge, storage=f"sqlite:///{studies_path}")
            wf_result = wfe.run(parsed, trials)
            if not wf_result.windows:
                raise RuntimeError("Walk-Forward lieferte keine Ergebnisse.")

            console.print("[dim]5/5 Ergebnisse werden gespeichert...[/dim]")
            db = BuilderDB(db_path=db_path)
            try:
                run_id = db.save_run(
                    idea=idee,
                    trials=trials,
                    session=parsed.session,
                    is_robust=wf_result.is_robust,
                )
                for i, w in enumerate(wf_result.windows):
                    db.save_result(
                        run_id=run_id,
                        result=w.oos,
                        score=w.oos.profit_factor,
                        rank=i + 1,
                        warnings=[],
                    )
                tier = db.compute_and_save_tier(run_id)
            finally:
                db.close()
            tier_color = {"A": "green", "B": "yellow", "C": "red"}.get(tier, "white")
            console.print(
                f"\n[bold]Registry:[/bold] Tier [{tier_color}]{tier}[/{tier_color}] gespeichert (run_id={run_id})"
            )

            generate_wf_report(idea=parsed, wf_result=wf_result, output_dir=output_dir)

            # suppress unused-variable warning for min_trades (kept for CLI compatibility)
            _ = min_trades
            # suppress unused-variable warning for ctx (kept for future use)
            _ = ctx
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    @dataclass
    class _BatchRow:
        idee: str
        status: str = ""
        pf: float | None = None
        winrate: float | None = None
        trades: int | None = None
        error: str = ""


    def _setup_cache(
        sources: str,
        output: str,
        concepts: list[str] | None = None,
    ) -> tuple[Path, Path, NautilusBridge]:
        """Baut Signal-Cache einmalig und gibt (cache_path, db_path, bridge) zurück."""
        sources_path = Path(sources).expanduser() if sources else None
        data_path, cfg, holdout_start = _resolve_backtest_data_path(sources)

        cache_path = data_path.parent / "signal_cache.parquet"
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        sc_cfg = SignalCacheConfig(
            bars_path=data_path,
            cache_path=cache_path,
            algo_dirs=algo_dirs,
        )
        sc = SignalCache(sc_cfg)
        console.print("   Signal-Cache wird gebaut (einmalig)...")
        sc.build()
        if concepts:
            console.print(
                f"   [dim]Baue Science-Shards für {len(concepts)} Konzepte...[/dim]"
            )
            sc.build(concepts=concepts)
        console.print("   Signal-Cache bereit.")

        bridge = NautilusBridge(
            data_path=data_path,
            cache_path=cache_path,
            algo_dirs=algo_dirs,
            max_date=holdout_start,
            concepts=concepts if concepts else None,
        )
        if holdout_start:
            console.print(f"   [yellow]Holdout gesperrt ab: {holdout_start}[/yellow]")
        db_path = Path(output) / "builder.db"
        return cache_path, db_path, bridge


    @app.command()
    def batch(
        ideen_datei: Path = typer.Argument(..., help="Textdatei mit einer Idee pro Zeile"),
        trials: int = typer.Option(
            50, "--trials", "-t", help="Anzahl Optuna-Versuche pro Idee"
        ),
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        sources: str = typer.Option(
            "", "--sources", "-s", help="Pfad zu sources.yaml (optional)"
        ),
        min_trades: int = typer.Option(
            30, "--min-trades", help="Minimum Trades für Wertung"
        ),
        force: bool = typer.Option(
            False, "--force", help="Duplikate trotzdem laufen lassen"
        ),
    ) -> None:
        """Mehrere Strategie-Ideen aus einer Textdatei sequenziell abarbeiten."""
        try:
            trials = _require_positive_int(trials, "Trials")
            min_trades = _require_non_negative_int(min_trades, "Min-Trades")
            console.print("\n[bold green]Strategie Builder – Batch-Modus[/bold green]")

            if not ideen_datei.exists():
                console.print(
                    f"[bold red]FEHLER: Datei nicht gefunden: {ideen_datei}[/bold red]"
                )
                raise typer.Exit(code=1)

            try:
                ideas = _read_ideas(ideen_datei)
            except OSError as exc:
                raise RuntimeError(
                    f"Ideen-Datei konnte nicht gelesen werden: {ideen_datei}"
                ) from exc
            if not ideas:
                console.print("[yellow]Keine Ideen in der Datei gefunden.[/yellow]")
                raise typer.Exit(code=0)

            console.print(
                f"[cyan]{len(ideas)} Ideen geladen[/cyan] | Trials: {trials} | Force: {force}\n"
            )

            # Alle Konzepte aus den Ideen sammeln (für Science-Shards)
            from sb.engine.parser import parse_idea as _parse_idea

            _all_concepts: list[str] = []
            for _idea in ideas:
                _all_concepts.extend(_parse_idea(_idea).concepts)
            _batch_concepts = list(dict.fromkeys(_all_concepts))

            # Signal-Cache einmalig bauen (inkl. Science-Shards)
            console.print("[dim]Signal-Cache + NautilusBridge vorbereiten...[/dim]")
            _, db_path, bridge = _setup_cache(sources, output, concepts=_batch_concepts)
            studies_path = Path(output) / "studies.db"
            _ = min_trades  # für zukünftige Nutzung

            rows: list[_BatchRow] = []

            for i, idee in enumerate(ideas, 1):
                console.print(
                    f"\n[bold]── Idee {i}/{len(ideas)}: [cyan]{idee}[/cyan][/bold]"
                )
                row = _BatchRow(idee=idee)

                # Duplikat-Check
                buch_db = BuilderDB(db_path=db_path)
                try:
                    prior = buch_db.find_runs_by_idea(idee)
                finally:
                    buch_db.close()
                if prior and not force:
                    console.print(
                        f"   [yellow]Buchhalter:[/yellow] {len(prior)}× bereits gebaut – übersprungen (--force zum Erzwingen)"
                    )
                    row.status = "⏭ Skip"
                    rows.append(row)
                    continue

                try:
                    parsed = parse_idea(idee)
                    console.print(
                        f"   Konzepte: {parsed.concepts} | Session: {parsed.session}"
                    )

                    wfe = WalkForwardEngine(
                        bridge=bridge, storage=f"sqlite:///{studies_path}"
                    )
                    wf_result = wfe.run(parsed, trials)
                    if not wf_result.windows:
                        raise RuntimeError("Walk-Forward lieferte keine Ergebnisse.")

                    db = BuilderDB(db_path=db_path)
                    try:
                        run_id = db.save_run(
                            idea=idee,
                            trials=trials,
                            session=parsed.session,
                            is_robust=wf_result.is_robust,
                        )
                        for j, w in enumerate(wf_result.windows):
                            db.save_result(
                                run_id=run_id,
                                result=w.oos,
                                score=w.oos.profit_factor,
                                rank=j + 1,
                                warnings=[],
                            )
                        tier = db.compute_and_save_tier(run_id)
                    finally:
                        db.close()
                    tier_color = {"A": "green", "B": "yellow", "C": "red"}.get(
                        tier, "white"
                    )
                    console.print(
                        f"   [bold]Tier [{tier_color}]{tier}[/{tier_color}][/bold] gespeichert"
                    )

                    generate_wf_report(
                        idea=parsed, wf_result=wf_result, output_dir=Path(output)
                    )

                    # Bestes OOS-Fenster für die Tabelle
                    best = max(wf_result.windows, key=lambda w: w.oos.profit_factor)
                    row.status = "✅ OK"
                    row.pf = best.oos.profit_factor
                    row.winrate = best.oos.winrate
                    row.trades = best.oos.num_trades
                    console.print(
                        f"   PF={best.oos.profit_factor:.3f} | Winrate={best.oos.winrate:.1%} | Trades={best.oos.num_trades}"
                    )

                except Exception as exc:
                    row.status = "❌ Fehler"
                    row.error = str(exc)
                    console.print(f"   [red]Fehler:[/red] {exc}")

                rows.append(row)

            # Zusammenfassungs-Tabelle
            console.print("\n")
            table = Table(title="Batch-Ergebnis")
            table.add_column("Idee", style="cyan", no_wrap=False)
            table.add_column("Status")
            table.add_column("PF")
            table.add_column("Winrate")
            table.add_column("Trades")

            for row in rows:
                pf_str = f"{row.pf:.3f}" if row.pf is not None else "-"
                wr_str = f"{row.winrate:.1%}" if row.winrate is not None else "-"
                tr_str = str(row.trades) if row.trades is not None else "-"
                table.add_row(row.idee, row.status, pf_str, wr_str, tr_str)

            console.print(table)

            ok = sum(1 for r in rows if r.status == "✅ OK")
            skip = sum(1 for r in rows if r.status == "⏭ Skip")
            err = sum(1 for r in rows if r.status == "❌ Fehler")
            console.print(f"\n[bold]Gesamt: {ok} OK | {skip} Skip | {err} Fehler[/bold]")
            _print_strategie_baumaschinen_lager()
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    @app.command()
    def registry(
        tier: str = typer.Option("", "--tier", "-t", help="Filter: A, B oder C"),
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        recalc: bool = typer.Option(
            False, "--recalc", help="Tier für alle bestehenden Runs neu berechnen"
        ),
    ) -> None:
        """Strategie-Registry anzeigen – alle gespeicherten Ergebnisse mit Tier."""
        try:
            db_path = Path(output) / "builder.db"
            normalized_tier = tier.strip().upper() if tier.strip() else ""
            if not db_path.exists():
                console.print(f"[yellow]Keine DB gefunden: {db_path}[/yellow]")
                raise typer.Exit(code=0)

            db = BuilderDB(db_path=db_path)
            try:
                if recalc:
                    runs = db.conn.execute("SELECT id FROM build_runs").fetchall()
                    for (run_id,) in runs:
                        db.compute_and_save_tier(run_id)
                    console.print(
                        f"[green]Tier für {len(runs)} Runs neu berechnet.[/green]\n"
                    )

                try:
                    rows = db.get_registry(tier=normalized_tier or None)
                    registry_counts = db.get_registry_counts()
                except ValueError as exc:
                    console.print(f"[bold red]FEHLER:[/bold red] {exc}")
                    raise typer.Exit(code=1) from exc
            finally:
                db.close()

            if not rows:
                console.print("[yellow]Keine Einträge gefunden.[/yellow]")
                raise typer.Exit(code=0)

            table = Table(
                title=f"Strategy Registry{' – Tier ' + normalized_tier if normalized_tier else ''}",
                padding=(0, 1),
            )
            table.add_column("ID", style="dim", width=3, no_wrap=True)
            table.add_column("Tier", width=4, no_wrap=True)
            table.add_column("Ø OOS PF", width=7, no_wrap=True)
            table.add_column("Tr", width=3, no_wrap=True)
            table.add_column("Robust", width=5, no_wrap=True)
            table.add_column("Session", width=6, no_wrap=True)
            table.add_column("Idee", style="cyan", min_width=6)
            table.add_column("Datum", style="dim", width=7, no_wrap=True)
            table.add_column("Holdout PF", width=8, no_wrap=True)

            tier_colors = {"A": "green", "B": "yellow", "C": "red"}
            for r in rows:
                t = r.get("tier") or "?"
                color = tier_colors.get(t, "white")
                pf_str = (
                    f"{r['avg_oos_pf']:.3f}" if r.get("avg_oos_pf") is not None else "?"
                )
                avg_trades = r.get("avg_trades")
                trades_str = str(int(avg_trades)) if avg_trades is not None else "?"
                robust_val = r.get("is_robust")
                robust_str = "[green]✅[/green]" if robust_val else "[red]❌[/red]"
                session_str = r.get("session") or "?"
                holdout_pf_val = r.get("holdout_pf")
                holdout_validated = r.get("holdout_validated")
                if holdout_validated and holdout_pf_val is not None:
                    h_color = "green" if holdout_pf_val >= 1.3 else "red"
                    holdout_str = f"[{h_color}]{holdout_pf_val:.3f}[/{h_color}] ✅"
                else:
                    holdout_str = "[dim]—[/dim]"
                table.add_row(
                    str(r["id"]),
                    f"[{color}]{t}[/{color}]",
                    pf_str,
                    trades_str,
                    robust_str,
                    session_str,
                    r["idea"],
                    r.get("created_at", "")[:16],
                    holdout_str,
                )

            console.print(table)
            shown = len(rows)
            total = registry_counts["total"]
            note = f" (zeige {shown} von {total})" if shown < total else ""
            console.print(
                f"\n[bold]Gesamt: {total}{note} | "
                f"[green]A: {registry_counts['A']}[/green] | "
                f"[yellow]B: {registry_counts['B']}[/yellow] | "
                f"[red]C: {registry_counts['C']}[/red][/bold]"
            )
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    @app.command()
    def export(
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
    ) -> None:
        """Exportiert alle Registry-Daten als CSV + JSON nach output/registry/."""
        import csv as csv_mod

        try:
            db_path = Path(output) / "builder.db"
            if not db_path.exists():
                console.print(f"[yellow]Keine DB gefunden: {db_path}[/yellow]")
                raise typer.Exit(code=0)

            registry_dir = Path(output) / "registry"
            details_dir = registry_dir / "details"
            registry_dir.mkdir(parents=True, exist_ok=True)
            details_dir.mkdir(parents=True, exist_ok=True)

            db = BuilderDB(db_path=db_path)
            try:
                all_runs = db.get_registry()
                counts = db.get_registry_counts()
            finally:
                db.close()

            if not all_runs:
                console.print("[yellow]Keine Runs in DB.[/yellow]")
                raise typer.Exit(code=0)

            csv_fields = [
                "id",
                "idea",
                "tier",
                "avg_oos_pf",
                "avg_trades",
                "session",
                "is_robust",
                "holdout_pf",
                "holdout_validated",
                "pbo_score",
                "mc_pct_profitable",
                "created_at",
            ]

            def write_csv(path: Path, rows: list[dict]) -> None:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv_mod.DictWriter(
                        f, fieldnames=csv_fields, extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(rows)

            write_csv(registry_dir / "alle_runs.csv", all_runs)

            for tier_val in ("A", "B", "C"):
                tier_runs = [r for r in all_runs if r.get("tier") == tier_val]
                write_csv(registry_dir / f"tier_{tier_val}.csv", tier_runs)

            sessions = {r.get("session") for r in all_runs if r.get("session")}
            for session in sessions:
                session_runs = [r for r in all_runs if r.get("session") == session]
                safe_name = str(session).lower().replace(" ", "_")
                write_csv(registry_dir / f"session_{safe_name}.csv", session_runs)

            db2 = BuilderDB(db_path=db_path)
            try:
                for run in all_runs:
                    run_id = run["id"]
                    windows = db2.conn.execute(
                        "SELECT pf, winrate, num_trades, score, rank FROM results "
                        "WHERE run_id = ? ORDER BY rank ASC",
                        (run_id,),
                    ).fetchall()
                    detail = dict(run)
                    detail["windows"] = [dict(w) for w in windows]
                    safe_idea = (
                        str(run["idea"])
                        .replace(" ", "_")
                        .replace("+", "")
                        .replace("/", "_")[:40]
                    )
                    fname = f"run_{run_id:03d}_{safe_idea}.json"
                    with open(details_dir / fname, "w", encoding="utf-8") as f:
                        json.dump(detail, f, ensure_ascii=False, indent=2, default=str)
            finally:
                db2.close()

            console.print(
                f"\n[bold green]Export abgeschlossen:[/bold green] {registry_dir}"
            )
            console.print(f"  alle_runs.csv     → {len(all_runs)} Runs")
            console.print(f"  tier_A.csv        → {counts['A']} Runs")
            console.print(f"  tier_B.csv        → {counts['B']} Runs")
            console.print(f"  tier_C.csv        → {counts['C']} Runs")
            console.print(f"  session_*.csv     → {len(sessions)} Sessions")
            console.print(f"  details/          → {len(all_runs)} JSON-Dateien")

        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    @app.command()
    def validate(
        idee: str = typer.Argument(
            default="",
            help='Strategie-Idee, z.B. "BOS + FVG NY". Leer wenn --tier gesetzt.',
        ),
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        tier: str = typer.Option(
            "",
            "--tier",
            help="Alle Strategien mit diesem Tier validieren (z.B. B)",
        ),
    ) -> None:
        """Holdout-Validierung: testet gespeicherte Best-Params auf ungesehenen Daten."""
        try:
            import json as _json

            output_dir = Path(output)
            db_path = output_dir / "builder.db"

            data_path, cfg, holdout_start = _resolve_backtest_data_path(sources)
            if not holdout_start:
                raise RuntimeError(
                    "sources.yaml enthält kein 'backtest_data.holdout_start'.\n"
                    "Eintragen: holdout_start: '2026-01-01'"
                )

            console.print(
                "\n[bold green]Strategie Builder – Holdout-Validierung[/bold green]"
            )
            console.print(f"Holdout ab: [yellow]{holdout_start}[/yellow]\n")

            # --- Welche Runs validieren? ---
            db = BuilderDB(db_path=db_path)
            try:
                if idee.strip():
                    runs_to_validate = db.find_runs_by_idea(idee)
                    if not runs_to_validate:
                        raise RuntimeError(f"Keine Runs für Idee '{idee}' in DB gefunden.")
                    # Neuesten Run (erster nach ORDER BY created_at DESC)
                    runs_to_validate = [runs_to_validate[0]]
                elif tier.strip():
                    tier_upper = tier.strip().upper()
                    all_runs = db.get_registry()
                    runs_to_validate = [
                        r
                        for r in all_runs
                        if r.get("tier") == tier_upper and not r.get("holdout_validated")
                    ]
                    if not runs_to_validate:
                        console.print(
                            f"[green]Alle Tier-{tier_upper} Strategien bereits validiert.[/green]"
                        )
                        return
                else:
                    raise RuntimeError("Idee oder --tier angeben.")
            finally:
                db.close()

            # --- Holdout-Bridge aufbauen ---
            sources_path = Path(sources).expanduser() if sources else None
            algo_dirs = resolve_pda_library_dirs(
                cfg, (sources_path or _DEFAULT_SOURCES).parent
            )
            if not algo_dirs:
                algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)

            cache_path = data_path.parent / "signal_cache.parquet"

            # Alle Konzepte aus allen zu validierenden Runs sammeln
            from sb.engine.parser import parse_idea as _parse_idea

            _all_val_concepts: list[str] = []
            for _run in runs_to_validate:
                _all_val_concepts.extend(_parse_idea(_run["idea"]).concepts)
            _val_concepts = list(dict.fromkeys(_all_val_concepts))

            # Science-Shards für Holdout sicherstellen
            if _val_concepts:
                _sc_val = SignalCache(
                    SignalCacheConfig(
                        bars_path=data_path,
                        cache_path=cache_path,
                        algo_dirs=algo_dirs,
                    )
                )
                _sc_val.build(concepts=_val_concepts)

            holdout_bridge = NautilusBridge(
                data_path=data_path,
                cache_path=cache_path,
                algo_dirs=algo_dirs,
                min_date=holdout_start,
                concepts=_val_concepts if _val_concepts else None,
            )
            n_holdout_bars = len(holdout_bridge._bars)
            console.print(f"Holdout-Bars geladen: [cyan]{n_holdout_bars}[/cyan]\n")

            if n_holdout_bars < 100:
                raise RuntimeError(
                    f"Zu wenige Holdout-Bars ({n_holdout_bars}). "
                    "holdout_start möglicherweise zu spät oder Daten fehlen."
                )

            # --- Jeden Run validieren ---
            import pandas as _pd
            from rich.table import Table as _RichTable

            # Datumsbereiche für Tabellenheader ermitteln
            _bars_df = _pd.read_parquet(data_path)
            _data_start = str(_bars_df.index.min())[:10]  # type: ignore[union-attr]
            _data_end = str(_bars_df.index.max())[:10]  # type: ignore[union-attr]
            _train_period = f"{_data_start} → {holdout_start} (Training)"
            _holdout_period = f"{holdout_start} → {_data_end} (Holdout)"

            result_table = _RichTable(title="Holdout-Ergebnisse", show_lines=True)
            result_table.add_column("ID", style="dim")
            result_table.add_column("Idee")
            result_table.add_column(f"Ø OOS PF\n{_train_period}", justify="right")
            result_table.add_column(f"Holdout PF\n{_holdout_period}", justify="right")
            result_table.add_column("Holdout\nTrades", justify="right")
            result_table.add_column("Holdout\nWR", justify="right")

            db = BuilderDB(db_path=db_path)
            try:
                for run in runs_to_validate:
                    run_id = run["id"]
                    run_idea = run["idea"]
                    console.print(f"Validiere: [cyan]{run_idea}[/cyan] (run_id={run_id})")

                    best_row = db.conn.execute(
                        "SELECT params FROM results WHERE run_id = ? ORDER BY score DESC LIMIT 1",
                        (run_id,),
                    ).fetchone()

                    if not best_row or not best_row[0]:
                        console.print(
                            "  [yellow]Keine Params gefunden – übersprungen.[/yellow]"
                        )
                        continue

                    raw_params = best_row[0]
                    best_params = (
                        _json.loads(raw_params)
                        if isinstance(raw_params, str)
                        else raw_params
                    )

                    holdout_result = holdout_bridge.run(best_params)

                    pf = (
                        holdout_result.gross_profit / holdout_result.gross_loss
                        if holdout_result.gross_loss > 0
                        else 0.0
                    )

                    db.save_holdout_result(
                        run_id,
                        holdout_pf=pf,
                        holdout_trades=holdout_result.num_trades,
                    )

                    pf_color = "green" if pf >= 1.3 else "red"
                    wr = (
                        holdout_result.num_wins / holdout_result.num_trades
                        if holdout_result.num_trades > 0
                        else 0.0
                    )
                    console.print(
                        f"  Holdout PF=[{pf_color}]{pf:.3f}[/{pf_color}]  "
                        f"Trades=[cyan]{holdout_result.num_trades}[/cyan]  "
                        f"WR=[cyan]{wr:.1%}[/cyan]"
                    )
                    oos_pf = run.get("avg_oos_pf") or 0.0
                    oos_color = "green" if oos_pf >= 1.5 else "yellow"
                    result_table.add_row(
                        str(run_id),
                        run_idea,
                        f"[{oos_color}]{oos_pf:.3f}[/{oos_color}]",
                        f"[{pf_color}]{pf:.3f}[/{pf_color}]",
                        str(holdout_result.num_trades),
                        f"{wr:.1%}",
                    )
            finally:
                db.close()

            console.print()
            console.print(result_table)
            console.print("\n[bold green]Holdout-Validierung abgeschlossen.[/bold green]")

        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

