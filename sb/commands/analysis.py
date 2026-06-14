"""Analysis commands for Strategy Builder."""
import typer


def register(app) -> None:
    """Register all analysis commands on the Typer app."""
    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path
    from rich.table import Table

    from sb.cli import (
        console,
        _require_non_empty_text,
        _require_positive_int,
        _require_non_negative_int,
        _resolve_backtest_data_path,
        _DEFAULT_SOURCES,
        DEFAULT_SIGNAL_ALGO_DIRS,
        resolve_pda_library_dirs,
    )
    from sb.memory.db import BuilderDB
    from sb.engine.parser import parse_idea
    from sb.engine.walk_forward import WalkForwardEngine
    from sb.report import generate_wf_report

    @app.command(name="export-trades")
    def export_trades(
        tier: str = typer.Option("", "--tier", help="Tier exportieren (A oder B)"),
        run_id: int = typer.Option(-1, "--run-id", help="Spezifische run_id exportieren"),
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
    ) -> None:
        """Trade-Export: Baut Tier-A/B Strategien neu und speichert Trade-Parquets."""
        import json as _json

        import pandas as pd
        from sb.engine.parser import parse_idea as _parse_idea
        from sb.memory.db import BuilderDB
        from sb.trade_export import enrich_and_save

        output_dir = Path(output)
        db_path = output_dir / "builder.db"
        if not db_path.exists():
            console.print(f"[red]DB nicht gefunden: {db_path}[/red]")
            raise typer.Exit(1)

        data_path, cfg, _ = _resolve_backtest_data_path(sources)
        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        cache_path = data_path.parent / "signal_cache.parquet"

        db = BuilderDB(db_path=db_path)
        try:
            if run_id >= 0:
                row = db.conn.execute(
                    "SELECT id, idea, tier FROM build_runs WHERE id = ?", (run_id,)
                ).fetchone()
                runs = [{"id": row[0], "idea": row[1], "tier": row[2]}] if row else []
            elif tier.strip():
                runs = [
                    r for r in db.get_registry() if r.get("tier") == tier.strip().upper()
                ]
            else:
                console.print("[red]--tier oder --run-id angeben.[/red]")
                raise typer.Exit(1)
        finally:
            db.close()

        if not runs:
            console.print("[yellow]Keine Runs gefunden.[/yellow]")
            return

        ohlcv_df = pd.read_parquet(data_path)
        ohlcv_df.columns = [c.lower() for c in ohlcv_df.columns]

        exported = 0
        for run in runs:
            rid = run["id"]
            idea = run["idea"]
            console.print(f"Exportiere: [cyan]{idea}[/cyan] (run_id={rid})")

            db2 = BuilderDB(db_path=db_path)
            try:
                best_row = db2.conn.execute(
                    "SELECT params FROM results WHERE run_id = ? ORDER BY score DESC LIMIT 1",
                    (rid,),
                ).fetchone()
            finally:
                db2.close()

            if not best_row:
                console.print("  [yellow]Keine Params – übersprungen.[/yellow]")
                continue

            params = _json.loads(best_row[0])
            parsed = _parse_idea(idea)
            params.setdefault("concepts", parsed.concepts)
            params.setdefault("session", parsed.session)

            bridge = NautilusBridge(
                data_path=data_path,
                cache_path=cache_path,
                algo_dirs=algo_dirs,
                concepts=parsed.concepts,
            )
            result = bridge.run(params)

            out_path = enrich_and_save(
                result.raw_trades,
                run_id=rid,
                ohlcv_df=ohlcv_df,
                output_dir=output_dir,
            )
            console.print(f"  → {len(result.raw_trades)} Trades → {out_path}")
            exported += 1

        console.print(f"\n[green]{exported} Trade-Exports gespeichert.[/green]")


    @app.command()
    def diagnose(
        tier: str = typer.Option("A", "--tier", help="Tier analysieren (A oder B)"),
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
    ) -> None:
        """Diagnose: Analysiert warum Strategien den Holdout nicht bestehen."""
        import pandas as _pd
        from rich.table import Table as _Table

        from sb.diagnose import (
            analyse_bausteine,
            analyse_overfitting,
            analyse_regime_shift,
            analyse_trade_distribution,
        )

        output_dir = Path(output)
        db_path = output_dir / "builder.db"
        if not db_path.exists():
            console.print(f"[red]DB nicht gefunden: {db_path}[/red]")
            raise typer.Exit(1)

        data_path, _cfg, _ = _resolve_backtest_data_path(sources)
        cache_path = data_path.parent / "signal_cache.parquet"

        console.print(f"\n[bold cyan]═══ DIAGNOSE – Tier {tier.upper()} ═══[/bold cyan]\n")

        # Modul 1: Regime-Shift
        console.print("[bold]Modul 1: Regime-Shift[/bold]")
        ohlcv = _pd.read_parquet(data_path)
        regime = analyse_regime_shift(ohlcv)
        t1 = _Table(show_header=True)
        t1.add_column("Quartal")
        t1.add_column("ATR Ø", justify="right")
        t1.add_column("ATR σ", justify="right")
        t1.add_column("ADX Ø", justify="right")
        for q in regime["quarters"]:
            t1.add_row(
                q["period"], str(q["atr_mean"]), str(q["atr_std"]), str(q["adx_mean"])
            )
        console.print(t1)
        shift_color = "red" if regime["shift_detected"] else "green"
        console.print(
            f"Shift erkannt: [{shift_color}]{regime['shift_detected']}[/{shift_color}]"
            f" – {regime['shift_details']}\n"
        )

        # Modul 2: Overfitting
        console.print("[bold]Modul 2: Overfitting-Diagnose[/bold]")
        ov = analyse_overfitting(db_path, tier=tier)
        t2 = _Table(show_header=True)
        t2.add_column("ID", style="dim")
        t2.add_column("Idee")
        t2.add_column("OOS PF", justify="right")
        t2.add_column("HO PF", justify="right")
        t2.add_column("Degrad.", justify="right")
        t2.add_column("Klasse")
        for r in ov:
            cls_color = (
                "green"
                if r["classification"] == "Echte Edge"
                else ("yellow" if r["classification"] == "Partial" else "red")
            )
            t2.add_row(
                str(r["run_id"]),
                str(r["idea"]),
                str(r["avg_oos_pf"]),
                str(r["holdout_pf"]),
                f"{r['degradation']:.1%}",
                f"[{cls_color}]{r['classification']}[/{cls_color}]",
            )
        console.print(t2)
        console.print()

        # Modul 3: Bausteine
        console.print("[bold]Modul 3: Baustein-Qualität[/bold]")
        if cache_path.exists():
            bq = analyse_bausteine(cache_path, db_path)
            console.print(
                f"  Aktive Bausteine: [green]{len(bq['active_bausteine'])}[/green]"
            )
            console.print(
                f"  Tote Bausteine (fire_rate=0): "
                f"[red]{', '.join(bq['dead_bausteine']) or 'keine'}[/red]"
            )
            t3 = _Table(show_header=True)
            t3.add_column("Baustein")
            t3.add_column("Status")
            t3.add_column("Fire-Rate", justify="right")
            t3.add_column("Ø OOS PF", justify="right")
            for b in sorted(bq["bausteine"], key=lambda x: -(x["fire_rate"] or 0))[:15]:
                color = "green" if b["status"] == "aktiv" else "red"
                t3.add_row(
                    b["name"],
                    f"[{color}]{b['status']}[/{color}]",
                    f"{b['fire_rate']:.1%}",
                    str(b["avg_oos_pf"] or "-"),
                )
            console.print(t3)
        else:
            console.print(f"  [yellow]Signal-Cache nicht gefunden: {cache_path}[/yellow]")
        console.print()

        # Modul 4: Trade-Verteilung
        console.print("[bold]Modul 4: Trade-Verteilung[/bold]")
        td = analyse_trade_distribution(output_dir)
        if td["total_trades"] == 0:
            console.print(
                "  [yellow]Keine Trade-Parquets gefunden. "
                "Erst `./sb.py export-trades --tier A` ausführen.[/yellow]"
            )
        else:
            console.print(f"  Trades gesamt: [cyan]{td['total_trades']}[/cyan]")
            t4 = _Table(show_header=True, title="PnL nach Session")
            t4.add_column("Session")
            t4.add_column("Ø PnL (Pts)", justify="right")
            t4.add_column("Win-Rate", justify="right")
            t4.add_column("Trades", justify="right")
            for sess, stats in td["by_session"].items():
                t4.add_row(
                    sess,
                    str(stats["avg_pnl_pts"]),
                    f"{stats['win_rate']:.1%}",
                    str(stats["n"]),
                )
            console.print(t4)
            t5 = _Table(show_header=True, title="PnL nach Regime")
            t5.add_column("Regime")
            t5.add_column("Ø PnL (Pts)", justify="right")
            t5.add_column("Win-Rate", justify="right")
            t5.add_column("Trades", justify="right")
            for reg, stats in td["by_regime"].items():
                t5.add_row(
                    reg,
                    str(stats["avg_pnl_pts"]),
                    f"{stats['win_rate']:.1%}",
                    str(stats["n"]),
                )
            console.print(t5)

        console.print("\n[green]Diagnose abgeschlossen.[/green]")


    @app.command()
    def suggest(
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        n: int = typer.Option(10, "--n", help="Anzahl vorgeschlagener Ideen"),
        min_runs: int = typer.Option(20, "--min-runs", help="Minimum Runs fuer Training"),
        explore_ratio: float = typer.Option(
            0.3, "--explore-ratio", help="Anteil explorativer Ideen (0.0-1.0)"
        ),
        model_path: str = typer.Option(
            "",
            "--model-path",
            help="Pfad zum gespeicherten Modell (leer = output/meta_learner.pkl)",
        ),
    ) -> None:
        """Meta-Learner: analysiert Registry, bewertet neue Ideen und speichert sie zur Review.

        Ergebnis landet in der suggestions-Tabelle der DB.
        Anschliessend: ./sb.py suggest-review
        """
        try:
            from sb.engine.meta_learner import MetaLearner

            db_path = Path(output) / "builder.db"
            resolved_model_path = (
                Path(model_path) if model_path else Path(output) / "meta_learner.pkl"
            )

            db = BuilderDB(db_path=db_path)
            try:
                registry = db.get_registry()
            finally:
                db.close()

            console.print(
                f"\n[bold green]Meta-Learner – {len(registry)} Runs in Registry[/bold green]"
            )

            if len(registry) < min_runs:
                console.print(
                    f"[yellow]Zu wenige Runs ({len(registry)} < {min_runs}). "
                    f"Weitere Batches laufen lassen.[/yellow]"
                )
                raise typer.Exit(code=1)

            learner = MetaLearner()
            if resolved_model_path.exists():
                try:
                    learner = MetaLearner.load(resolved_model_path)
                    if len(registry) >= len(learner._existing_ideas) + 10:
                        console.print(
                            "[dim]Registry gewachsen – Modell wird neu trainiert...[/dim]"
                        )
                        learner = MetaLearner()
                        learner.fit(registry)
                        learner.save(resolved_model_path)
                    else:
                        console.print(f"[dim]Modell geladen: {resolved_model_path}[/dim]")
                except Exception as exc:
                    console.print(
                        f"[yellow]Modell-Ladefehler ({exc}) – neu trainieren...[/yellow]"
                    )
                    learner = MetaLearner()
                    learner.fit(registry)
                    learner.save(resolved_model_path)
            else:
                console.print("[dim]Kein gespeichertes Modell – trainiere neu...[/dim]")
                learner.fit(registry)
                learner.save(resolved_model_path)

            console.print("\n[bold]Top Konzepte laut Modell:[/bold]")
            for feat, imp in learner.feature_importance(top_n=5):
                console.print(f"  {feat}: {imp:.3f}")

            suggestions = learner.suggest(n=n, explore_ratio=explore_ratio)

            db = BuilderDB(db_path=db_path)
            try:
                saved = 0
                for s in suggestions:
                    db.save_suggestion(
                        idea=s.idea,
                        model_runs=len(registry),
                        prob_ab=s.prob_ab,
                        uncertainty=s.uncertainty,
                        novelty=s.novelty,
                        band=s.band,
                    )
                    saved += 1
            finally:
                db.close()

            table = Table(title=f"Meta-Learner – {saved} Vorschlaege gespeichert")
            table.add_column("Band", style="bold", min_width=14)
            table.add_column("Idee", min_width=30)
            table.add_column("P(B)", justify="right")
            table.add_column("Unsich.", justify="right")
            table.add_column("Treiber")

            band_colors = {
                "auto_queue": "green",
                "human_review": "yellow",
                "auto_reject": "red",
            }
            for s in sorted(suggestions, key=lambda x: x.prob_ab, reverse=True):
                color = band_colors.get(s.band, "white")
                table.add_row(
                    f"[{color}]{s.band}[/{color}]",
                    s.idea,
                    f"{s.prob_ab:.2f}",
                    f"{s.uncertainty:.2f}",
                    s.top_feature,
                )
            console.print(table)
            console.print(
                f"\n[dim]{saved} Vorschlaege gespeichert. Weiter: .venv/bin/python sb.py suggest-review[/dim]\n"
            )

        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    @app.command(name="suggest-review")
    def suggest_review(
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        approved_dir: str = typer.Option(
            "",
            "--approved-dir",
            help="Zielordner fuer genehmigte Batch-Dateien (Standard: ideas/approved relativ zu CWD)",
        ),
    ) -> None:
        """Zeigt ausstehende Meta-Learner Vorschlaege zur manuellen Genehmigung.

        [a]pprove – Idee wird in Batch-Datei aufgenommen
        [r]eject  – Idee wird abgelehnt (mit optionalem Grund)
        [s]kip    – Idee bleibt pending fuer spaeter

        Abgelehnte Ideen fliessen NICHT als Trainings-Negative zurueck.
        """
        try:
            db_path = Path(output) / "builder.db"
            db = BuilderDB(db_path=db_path)
            try:
                pending = db.get_pending_suggestions()
            finally:
                db.close()

            if not pending:
                console.print("[green]Keine ausstehenden Vorschlaege.[/green]")
                return

            band_colors = {
                "auto_queue": "green",
                "human_review": "yellow",
                "auto_reject": "red",
            }

            # Uebersichts-Tabelle
            table = Table(title=f"{len(pending)} ausstehende Vorschlaege")
            table.add_column("#", style="dim", width=4)
            table.add_column("Band", min_width=14)
            table.add_column("Idee", min_width=30)
            table.add_column("P(B)", justify="right")
            table.add_column("Unsich.", justify="right")
            table.add_column("Novelty", justify="right")
            for i, s in enumerate(pending, 1):
                color = band_colors.get(s["band"], "white")
                table.add_row(
                    str(i),
                    f"[{color}]{s['band']}[/{color}]",
                    s["idea"],
                    f"{s['prob_ab']:.2f}",
                    f"{s['uncertainty']:.2f}",
                    f"{s['novelty']:.2f}",
                )
            console.print(table)

            approved_ideas: list[str] = []
            db = BuilderDB(db_path=db_path)
            try:
                for s in pending:
                    color = band_colors.get(s["band"], "white")
                    console.print(
                        f"\n[{color}][{s['band']}][/{color}]  "
                        f"[bold]{s['idea']}[/bold]  "
                        f"P={s['prob_ab']:.2f}  Unsich={s['uncertainty']:.2f}  "
                        f"Novelty={s['novelty']:.2f}"
                    )
                    raw = typer.prompt("[a]pprove / [r]eject / [s]kip", default="s")
                    choice = raw.strip().lower()

                    if choice == "a":
                        db.update_suggestion_status(s["id"], "approved")
                        approved_ideas.append(s["idea"])
                        console.print("[green]  Genehmigt.[/green]")
                    elif choice == "r":
                        reason = typer.prompt(
                            "  Grund (optional, Enter = leer lassen)", default=""
                        )
                        db.update_suggestion_status(
                            s["id"], "rejected", reason=reason.strip()
                        )
                        console.print("[red]  Abgelehnt.[/red]")
                    else:
                        console.print("[dim]  Uebersprungen (bleibt pending).[/dim]")
            finally:
                db.close()

            if approved_ideas:
                # approved_dir relativ zu CWD, nicht zu output
                approved_path = (
                    Path(approved_dir) if approved_dir else Path("ideas") / "approved"
                )
                approved_path.mkdir(parents=True, exist_ok=True)
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_file = approved_path / f"batch_ml_{date_str}.txt"
                batch_file.write_text("\n".join(approved_ideas) + "\n")
                console.print(
                    f"\n[bold green]{len(approved_ideas)} Idee(n) gespeichert:[/bold green] {batch_file}"
                )
                console.print(
                    f"[dim]Starten mit: .venv/bin/python sb.py batch {batch_file}[/dim]\n"
                )
            else:
                console.print("\n[yellow]Keine Ideen genehmigt.[/yellow]\n")

        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    @app.command()
    def analyse(
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
    ) -> None:
        """Strategie-Testmaschine: Analysiert alle Strategien und berechnet welcher Baustein Edge bringt.

        6 Phasen: Ablation, LASSO, ANOVA-Interaktionen, SHAP, Mutual Information, Overfitting-Check.
        """
        from sb.analyse import run_analyse
        from sb.memory.db import BuilderDB

        try:
            output_dir = Path(output)
            db = BuilderDB(output_dir / "builder.db")
            registry = db.get_registry()
            db.close()

            if len(registry) < 20:
                console.print(
                    "[red]Zu wenig Daten – mindestens 20 Strategien noetig.[/red]"
                )
                raise typer.Exit(1)

            result = run_analyse(registry, output_dir)

            if result.report_path:
                console.print(f"\nReport: [green]{result.report_path}[/green]")

        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    @app.command()
    def parallel(
        workers: int = typer.Option(2, "--workers", "-w", help="Anzahl paralleler Worker"),
        trials: int = typer.Option(50, "--trials", "-t", help="Optuna-Trials pro Idee"),
        output: str = typer.Option(
            "output_david_1", "--output", "-o", help="Output-Verzeichnis"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        min_trades: int = typer.Option(
            30, "--min-trades", help="Minimum Trades für Wertung"
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run", help="Nur anzeigen, nicht ausführen"
        ),
    ) -> None:
        """Alle Kombinationen automatisch generieren und parallel abarbeiten."""
        import concurrent.futures

        from sb.combinator import generate_ideas
        from sb.engine.worker import WorkerConfig, run_worker_task

        try:
            trials = _require_positive_int(trials, "Trials")
            workers = _require_positive_int(workers, "Workers")
            min_trades = _require_non_negative_int(min_trades, "Min-Trades")

            console.print("\n[bold green]Strategie Builder – Parallel-Modus[/bold green]")

            all_ideas = generate_ideas()
            console.print(
                f"[cyan]{len(all_ideas)} Ideen[/cyan] generiert | Workers: {workers} | Trials: {trials}\n"
            )

            if dry_run:
                for idea in all_ideas:
                    console.print(f"  [dim]{idea}[/dim]")
                raise typer.Exit(code=0)

            from sb.cache.signal_cache import SignalCache, SignalCacheConfig
            from sb.engine.parser import parse_idea

            data_path, cfg_yaml, holdout_start = _resolve_backtest_data_path(sources)
            sources_path = Path(sources).expanduser() if sources else None
            algo_dirs = resolve_pda_library_dirs(
                cfg_yaml, (sources_path or _DEFAULT_SOURCES).parent
            )
            if not algo_dirs:
                algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)

            cache_path = data_path.parent / "signal_cache.parquet"
            db_path = Path(output) / "builder.db"
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Alle Konzepte aus allen pending Ideen sammeln
            # Bereits getestete rausfiltern
            buch_db = BuilderDB(db_path=db_path)
            try:
                pending = [
                    idea for idea in all_ideas if not buch_db.find_runs_by_idea(idea)
                ]
            finally:
                buch_db.close()

            skipped = len(all_ideas) - len(pending)
            if skipped:
                console.print(
                    f"[yellow]{skipped} Ideen bereits in DB – übersprungen.[/yellow]"
                )

            if not pending:
                console.print("[green]Alle Ideen bereits getestet.[/green]")
                raise typer.Exit(code=0)

            # Alle benötigten Konzepte sammeln und Shards vorab bauen
            all_concepts: list[str] = []
            for idea in pending:
                parsed = parse_idea(idea)
                all_concepts.extend(parsed.concepts)
            unique_concepts = list(
                dict.fromkeys(all_concepts)
            )  # dedupliziert, Reihenfolge erhalten

            console.print(
                f"[dim]Baue Shards für {len(unique_concepts)} Konzepte ({len(pending)} Ideen)...[/dim]"
            )
            sc = SignalCache(
                SignalCacheConfig(
                    bars_path=data_path,
                    cache_path=cache_path,
                    algo_dirs=algo_dirs,
                )
            )
            sc.build(concepts=unique_concepts)
            console.print("   Signal-Shards bereit.")

            console.print(f"[cyan]{len(pending)} Ideen werden getestet...[/cyan]\n")

            configs = [
                WorkerConfig(
                    idea=idea,
                    trials=trials,
                    data_path=data_path,
                    cache_path=cache_path,
                    db_path=db_path,
                    output_dir=output_dir,
                    min_trades=min_trades,
                    algo_dirs=algo_dirs,
                    max_date=holdout_start,
                )
                for idea in pending
            ]

            done = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_worker_task, cfg): cfg.idea for cfg in configs}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    done += 1
                    idea = result["idea"]
                    status = result["status"]
                    if status == "ok":
                        tier = result.get("tier", "?")
                        pf = result.get("avg_pf") or 0.0
                        tier_color = {"A": "green", "B": "yellow", "C": "red"}.get(
                            tier, "white"
                        )
                        console.print(
                            f"[{done}/{len(pending)}] \u2705 {idea} \u2192 "
                            f"Tier [{tier_color}]{tier}[/{tier_color}] | PF {pf:.2f}"
                        )
                    elif status == "skip":
                        console.print(
                            f"[{done}/{len(pending)}] \u23ed  {idea} \u2192 bereits vorhanden"
                        )
                    else:
                        err = result.get("error", "unbekannt")
                        console.print(
                            f"[{done}/{len(pending)}] \u274c {idea} \u2192 FEHLER: {err}"
                        )

            console.print(f"\n[bold green]Fertig. {done} Ideen verarbeitet.[/bold green]")

            # Auto-Backup nach jedem Batch
            bak = _backup_db(Path(output) / "builder.db")
            if bak:
                console.print(f"[dim]DB-Backup: {bak.name}[/dim]")

        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]FEHLER:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


    def _print_strategie_baumaschinen_lager() -> None:
        """Gibt den aktuellen Stand aller output-Ordner aus (build_runs Tabelle)."""
        import sqlite3

        ordner = [
            "output_david_1",
            "output_david_2",
            "output_david_3",
        ]
        # Worker-Ordner automatisch erkennen
        for p in sorted(Path(".").glob("output_v3_worker*")):
            if (p / "builder.db").exists():
                ordner.append(p.name)
        console.print("\n[bold cyan]=== STRATEGIE-BAUMASCHINEN-LAGER ===[/bold cyan]")
        for ordner_name in ordner:
            db_path = Path(ordner_name) / "builder.db"
            if not db_path.exists():
                continue
            try:
                con = sqlite3.connect(db_path)
                total = con.execute("SELECT COUNT(*) FROM build_runs").fetchone()[0]
                holdout_ok = con.execute(
                    "SELECT COUNT(*) FROM build_runs WHERE holdout_validated=1 AND holdout_pf IS NOT NULL"
                ).fetchone()[0]
                robust_only = con.execute(
                    "SELECT COUNT(*) FROM build_runs WHERE is_robust=1 AND (holdout_validated IS NULL OR holdout_validated=0)"
                ).fetchone()[0]
                sonstige = total - holdout_ok - robust_only
                con.close()
                console.print(
                    f"  [cyan]{ordner_name}/[/cyan]  {total:4d} Strategien  |  "
                    f"[bold green]HOLDOUT OK: {holdout_ok}[/bold green]  "
                    f"[yellow]Robust (kein HO): {robust_only}[/yellow]  "
                    f"[dim]Sonstige: {sonstige}[/dim]"
                )
            except Exception:
                console.print(f"  [dim]{ordner_name}/ – DB-Fehler[/dim]")
        console.print("[bold cyan]=====================================[/bold cyan]\n")


    @app.command(name="strategie-baumaschinen-lager")
    def strategie_baumaschinen_lager() -> None:
        """Zeigt den aktuellen Stand aller output-Ordner (Strategien pro Tier)."""
        _print_strategie_baumaschinen_lager()


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

        # 1. Metadaten rückfüllen
        db = BuilderDB(db_path=db_path)
        try:
            updated = db.backfill_missing_metadata()
            console.print(f"✅ {updated} Runs mit session/is_robust rückgefüllt")
        finally:
            db.close()

        # 2. studies.db aufräumen
        studies_path = Path(output) / "studies.db"
        deleted = _cleanup_old_studies(studies_path, max_age_days=30)
        if deleted > 0:
            console.print(f"✅ {deleted} alte Optuna-Studies gelöscht (> 30 Tage)")
        else:
            console.print("✅ studies.db – nichts zu bereinigen")

        # 3. DB-Backup  (Reports werden NICHT gelöscht – Daten bleiben immer erhalten)
        bak = _backup_db(Path(output) / "builder.db")
        if bak:
            console.print(f"✅ DB-Backup erstellt: {bak.name}")


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
        from sb.inspect import inspect_algo  # noqa: PLC0415

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

        # ── Haupt-Tabelle ──────────────────────────────────────────────────────────
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

        # ── Heatmap Zeit × Wochentag ───────────────────────────────────────────────
        if heatmap and result.heatmap:
            console.print()
            dow_order = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
            present_dows = [
                d for d in dow_order if any(d in v for v in result.heatmap.values())
            ]

            h_table = Table(title="Heatmap: Zeit × Wochentag", show_header=True)
            h_table.add_column("Fenster", style="cyan", width=10)
            for dow in present_dows:
                h_table.add_column(dow, justify="right", width=6)

            max_cell = max(
                (v for row in result.heatmap.values() for v in row.values()), default=1
            )
            for slot in sorted(result.heatmap):
                row_vals = result.heatmap[slot]
                cells = []
                for dow in present_dows:
                    cnt = row_vals.get(dow, 0)
                    if cnt == 0:
                        cells.append("[dim]·[/dim]")
                    elif cnt >= max_cell * 0.75:
                        cells.append(f"[bold red]{cnt}[/bold red]")
                    elif cnt >= max_cell * 0.4:
                        cells.append(f"[yellow]{cnt}[/yellow]")
                    else:
                        cells.append(str(cnt))
                h_table.add_row(slot, *cells)

            console.print(h_table)

        # ── In _research/ speichern ────────────────────────────────────────────────
        if save:
            import json as _json
            from datetime import datetime as _dt

            grand_total = sum(w["total"] for w in result.windows)
            modus_str = "events-only" if result.events_only else "alle-bars"
            date_str = _dt.now().strftime("%Y-%m-%d")
            stem = result.algo_file.stem
            research_dir = result.algo_file.parent / "_research"
            research_dir.mkdir(exist_ok=True)
            base_name = f"{stem}_inspect_{modus_str}_{date_str}"

            # JSON
            payload = {
                "algo": result.algo_file.name,
                "generated": date_str,
                "modus": modus_str,
                "window_minutes": window,
                "total_bars": result.total_bars,
                "total_days": result.total_days,
                "signal_cols": result.signal_cols,
                "grand_total": grand_total,
                "avg_per_day": round(grand_total / result.total_days, 1),
                "windows": result.windows,
            }
            json_path = research_dir / f"{base_name}.json"
            json_path.write_text(
                _json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Markdown
            md_lines = [
                f"# Inspect: {result.algo_file.name}",
                "",
                f"**Generiert:** {date_str}  ",
                f"**Modus:** {modus_str}  ",
                f"**Fenstergröße:** {window} min  ",
                f"**Bars:** {result.total_bars:,} über {result.total_days} Handelstage  ",
                f"**Signale gesamt:** {grand_total:,} | Ø {grand_total / result.total_days:.1f}/Tag  ",
                f"**Signal-Spalten:** {', '.join(result.signal_cols)}  ",
                "",
                f"## Signal-Verteilung (NY-Zeit, {window}-min-Fenster)",
                "",
                "| Fenster (NY) | Bull ↑ | Bear ↓ | Other | Total | Ø/Tag |",
                "|---|---|---|---|---|---|",
            ]
            for w in result.windows:
                md_lines.append(
                    f"| {w['window']} | {w['bull']} | {w['bear']} | {w['other']} | {w['total']} | {w['rate_per_day']:.2f} |"
                )
            md_path = research_dir / f"{base_name}.md"
            md_path.write_text("\n".join(md_lines), encoding="utf-8")

            console.print(f"\nForschungsdaten gespeichert: [green]{research_dir}[/green]")

