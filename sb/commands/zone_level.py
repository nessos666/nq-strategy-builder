"""Zone/Level/FVG analysis commands."""
import typer


def register(app) -> None:
    """Register all zone/level/FVG commands."""
    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path
    import json
    from rich.table import Table

    from sb.cli import console, _DEFAULT_SOURCES, DEFAULT_SIGNAL_ALGO_DIRS, resolve_pda_library_dirs
    from sb._helpers import (
        _require_non_empty_text, _require_positive_int, _require_non_negative_int,
        _resolve_backtest_data_path,
    )

    @app.command(name="zone-stats")
    def zone_stats(
        algo_name: str = typer.Argument(..., help="Name des Zonen-Bausteins"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten in _research/ speichern"
        ),
        data: str = typer.Option(
            "", "--data", help="Alternativer Datenpfad (z.B. nq_5m_1year.parquet)"
        ),
    ) -> None:
        """Zonen-Trefferquote: Bounce vs. Durch für alle Zonen-Bausteine (FVG, iFVG, OB...).

        Erkennt Zone-Prefixe automatisch (fvg, ifvg, ob, ...) und analysiert
        für jeden: Bounce-Rate, Durch-Rate, Penetrations-Tiefe.
        """
        import pandas as _pd

        from sb.inspect import (
            analyze_zone_outcomes,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
            save_zone_research,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Zone-Stats: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print("[dim]Lade Daten und führe Algo aus...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)

        try:
            result_df = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(result_df)
        if not prefixes:
            console.print(
                "[red]Keine Zone-Prefixe erkannt. Algo gibt keine Zonen-Spalten aus.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[dim]Erkannte Prefixe: {', '.join(prefixes)}[/dim]")

        all_stats: dict = {}
        for prefix in prefixes:
            try:
                stats = analyze_zone_outcomes(result_df, prefix)
            except ValueError as exc:
                console.print(f"[yellow]Prefix '{prefix}' übersprungen: {exc}[/yellow]")
                continue
            all_stats[prefix] = stats

            # ── Haupttabelle: Bounce/Durch ────────────────────────────────────────
            for side_label, direction, s in [
                (f"{prefix.upper()} Bull", "Preis fällt von oben in Zone", stats["bull"]),
                (f"{prefix.upper()} Bear", "Preis steigt von unten in Zone", stats["bear"]),
            ]:
                total = s["total_zones"]
                no_touch = s["no_touch"]
                touches = s["touches"]
                bounce = s["bounce"]
                through = s["through"]
                bounce_pct = s["bounce_pct"]
                through_pct = s["through_pct"]

                console.print(
                    f"\n[bold cyan]{side_label}[/bold cyan]  [dim]{direction}[/dim]"
                )
                console.print(
                    f"  {total:,} Zonen entstanden  │  "
                    f"[dim]{no_touch:,} nie berührt ({round(no_touch / total * 100) if total else 0}%)[/dim]  │  "
                    f"[bold]{touches:,} berührt[/bold]"
                )
                if touches > 0:
                    console.print(
                        f"  Von {touches:,} Berührungen:\n"
                        f"    [green]↩  {bounce:,}× gedreht      → Zone hat gehalten   "
                        f"({bounce_pct}%)[/green]\n"
                        f"    [red]→  {through:,}× durchgegangen → Zone war kein Hindernis "
                        f"({through_pct}%)[/red]"
                    )

                d = s.get("depth", {})
                if d:
                    pts_med = d["depth_pts_median"]
                    pct_med = d["depth_pct_median"]
                    pct_lt25 = d["pct_lt25"]
                    pct_2550 = d["pct_25_50"]
                    pct_5075 = d["pct_50_75"]
                    pct_75100 = d["pct_75_100"]
                    pct_gte100 = d["pct_gte100"]
                    console.print(
                        f"\n  [bold]Wie tief dringt der Preis in die Zone ein?[/bold]  "
                        f"[dim](vor dem Drehen oder Durchgehen)[/dim]\n"
                        f"    Median: [yellow]{pts_med} Punkte[/yellow] tief  "
                        f"= [yellow]{pct_med}% der Zonengröße[/yellow]\n"
                        f"\n"
                        f"    Aufteilung aller Berührungen:\n"
                        f"    [dim]  < 25% Zone (nur reingeschaut)[/dim]  {pct_lt25}%\n"
                        f"    [dim]  25–50%  (bis zur Mitte)[/dim]        {pct_2550}%\n"
                        f"    [dim]  50–75%  (CE-Bereich)[/dim]           {pct_5075}%\n"
                        f"    [dim]  75–100% (fast durch)[/dim]           {pct_75100}%\n"
                        f"    [red]  ≥ 100%  (komplett durchbrochen)[/red] {pct_gte100}%"
                    )

        if not all_stats:
            console.print("[red]Keine validen Zonen-Prefixe analysiert.[/red]")
            raise typer.Exit(1)

        # ── Daten-Info ────────────────────────────────────────────────────────────
        bars = len(df)
        idx = df.index
        data_info = {
            "bars": bars,
            "from": str(idx[0])[:10],
            "to": str(idx[-1])[:10],
            "days": len(set(idx.normalize())) if hasattr(idx, "normalize") else "?",  # type: ignore[union-attr]
        }
        console.print(
            f"\n[dim]Daten: {bars:,} Bars | {data_info['from']} → {data_info['to']} | Algo: {algo_file.name}[/dim]"
        )

        if save:
            research_path = save_zone_research(all_stats, algo_file, data_info)
            console.print(f"[green]Forschungsdaten gespeichert: {research_path}[/green]")


    @app.command(name="fvg-stats")
    def fvg_stats(
        algo_name: str = typer.Argument(
            "FVG Standard", help="Name des FVG-Bausteins (default: 'FVG Standard')"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten speichern"
        ),
    ) -> None:
        """Alias für zone-stats mit FVG Standard als Default."""
        zone_stats(algo_name=algo_name, sources=sources, extra_dir=extra_dir, save=save)


    @app.command(name="level-stats")
    def level_stats(
        algo_name: str = typer.Argument(
            ..., help="Name des Level-Bausteins (z.B. 'Genauer Hoch')"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
    ) -> None:
        """Level-Trefferquote: Bounce vs. Durch für einzelne Preis-Level (PDH, PDL, PWH...).

        Erkennt Level-Spalten automatisch (_high/_low) und analysiert
        für jede: Bounce-Rate, Durch-Rate, Eindringstiefe in Punkten.
        """
        import pandas as _pd

        from sb.inspect import (
            analyze_level_outcomes,
            detect_level_columns,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Level-Stats: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print("[dim]Lade Daten und führe Algo aus...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)

        try:
            result_df = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen: {exc}[/red]")
            raise typer.Exit(1)

        level_cols = detect_level_columns(result_df)
        if not level_cols:
            console.print(
                "[red]Keine Level-Spalten erkannt (_high/_low). Algo gibt keine Level aus.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[dim]Erkannte Level: {', '.join(sorted(level_cols))}[/dim]\n")

        from rich.table import Table

        table = Table(title="Level-Statistik", show_lines=True)
        table.add_column("Level", style="cyan")
        table.add_column("Touches", justify="right")
        table.add_column("Return", justify="right", style="green")
        table.add_column("Durch", justify="right", style="red")
        table.add_column("Median Tiefe (Pt)", justify="right")

        for level_col in sorted(level_cols):
            try:
                stats = analyze_level_outcomes(result_df, level_col)
            except ValueError as exc:
                console.print(f"[yellow]{level_col} übersprungen: {exc}[/yellow]")
                continue

            table.add_row(
                level_col,
                str(stats["touches"]),
                f"{stats['bounce_pct']}%",
                f"{stats['through_pct']}%",
                f"{stats['depth_pts_median']}",
            )

        console.print(table)

        n_bars = len(result_df)
        t_start = str(result_df.index[0])[:10]
        t_end = str(result_df.index[-1])[:10]
        console.print(
            f"\n[dim]Daten: {n_bars:,} Bars | {t_start} → {t_end} | Algo: {algo_file.name}[/dim]"
        )


    def _load_level_analysis_input(
        algo_name: str,
        sources: str,
        data: str,
    ) -> tuple[Path, "object", list[str]]:
        import pandas as _pd

        from sb.inspect import detect_level_columns, find_algo_file, run_algo

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc

        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)

        try:
            result_df = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen: {exc}[/red]")
            raise typer.Exit(1) from exc

        level_cols = [
            col for col in detect_level_columns(result_df) if col.startswith("prev_")
        ]
        if not level_cols:
            console.print("[red]Keine prev_* Level-Spalten erkannt.[/red]")
            raise typer.Exit(1)

        return algo_file, result_df, sorted(level_cols)


    def _render_level_analysis_table(
        title: str,
        df: "object",
        group_label: str,
        show_avg_distance: bool = False,
    ) -> None:
        table = Table(title=title, show_lines=True)
        table.add_column("Level", style="cyan")
        table.add_column(group_label, style="white")
        table.add_column("N", justify="right")
        table.add_column("Return", justify="right", style="green")
        table.add_column("Durch", justify="right", style="red")
        if show_avg_distance:
            table.add_column("Ø Abstand", justify="right")

        if df.empty:
            console.print("[yellow]Keine Touches gefunden.[/yellow]")
            return

        for _, row in df.iterrows():
            cells = [
                str(row["level_col"]),
                str(row["group"]),
                str(int(row["touches"])),
                f"{row['bounce_pct']}%",
                f"{row['through_pct']}%",
            ]
            if show_avg_distance:
                avg_distance = row.get("avg_distance")
                cells.append("-" if avg_distance != avg_distance else str(avg_distance))
            table.add_row(*cells)

        console.print(table)


    def _save_level_analysis_markdown(
        algo_file: Path,
        slug: str,
        title: str,
        result_df: "object",
        group_label: str,
        bars_df: "object",
        show_avg_distance: bool = False,
    ) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        research_dir = algo_file.parent / "_research" / algo_file.stem
        research_dir.mkdir(parents=True, exist_ok=True)
        md_path = research_dir / f"{slug}_{date_str}.md"

        md_lines = [
            f"# {title}: {algo_file.name}",
            "",
            f"**Generiert:** {date_str}  ",
            f"**Bars:** {len(bars_df):,} | {str(bars_df.index[0])[:10]} → {str(bars_df.index[-1])[:10]}  ",
            "",
            f"| Level | {group_label} | N | Return | Durch |"
            + (" Ø Abstand |" if show_avg_distance else ""),
            "|---|---:|---:|---:|---:|" + ("---:|" if show_avg_distance else ""),
        ]
        for _, row in result_df.iterrows():
            line = (
                f"| {row['level_col']} | {row['group']} | {int(row['touches'])} | "
                f"{row['bounce_pct']}% | {row['through_pct']}% |"
            )
            if show_avg_distance:
                avg_distance = row.get("avg_distance")
                distance_text = "-" if avg_distance != avg_distance else str(avg_distance)
                line = line[:-1] + f" {distance_text} |"
            md_lines.append(line)

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        return md_path


    @app.command(name="level-multi-touch")
    def level_multi_touch(
        algo_name: str = typer.Argument(..., help="Name des Level-Bausteins"),
        threshold: float = typer.Option(
            3.0, "--threshold", help="Touch-Toleranz in Punkten"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten speichern"
        ),
    ) -> None:
        from sb.level_analysis import analyze_level_multi_touch

        algo_file, result_df, level_cols = _load_level_analysis_input(
            algo_name, sources, data
        )
        console.print(
            f"\n[bold cyan]Level Multi-Touch: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(
            f"[dim]Threshold: {threshold} Pkt | Level: {', '.join(level_cols)}[/dim]\n"
        )

        analysis_df = analyze_level_multi_touch(result_df, level_cols, threshold=threshold)
        _render_level_analysis_table("Level Multi-Touch", analysis_df, "Touch")
        console.print(
            f"\n[dim]Daten: {len(result_df):,} Bars | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}[/dim]"
        )
        if save:
            md_path = _save_level_analysis_markdown(
                algo_file,
                "multi_touch",
                "Level Multi-Touch",
                analysis_df,
                "Touch",
                result_df,
            )
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="level-weekday")
    def level_weekday(
        algo_name: str = typer.Argument(..., help="Name des Level-Bausteins"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten speichern"
        ),
    ) -> None:
        from sb.level_analysis import analyze_level_weekday

        algo_file, result_df, level_cols = _load_level_analysis_input(
            algo_name, sources, data
        )
        console.print(
            f"\n[bold cyan]Level Weekday: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(f"[dim]Level: {', '.join(level_cols)} | NY-Zeit[/dim]\n")

        analysis_df = analyze_level_weekday(result_df, level_cols)
        _render_level_analysis_table("Level Weekday", analysis_df, "Wochentag")
        console.print(
            f"\n[dim]Daten: {len(result_df):,} Bars | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}[/dim]"
        )
        if save:
            md_path = _save_level_analysis_markdown(
                algo_file,
                "weekday",
                "Level Weekday",
                analysis_df,
                "Wochentag",
                result_df,
            )
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="level-session")
    def level_session(
        algo_name: str = typer.Argument(..., help="Name des Level-Bausteins"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten speichern"
        ),
    ) -> None:
        from sb.level_analysis import analyze_level_session

        algo_file, result_df, level_cols = _load_level_analysis_input(
            algo_name, sources, data
        )
        console.print(
            f"\n[bold cyan]Level Session: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(f"[dim]Level: {', '.join(level_cols)} | NY-Zeit[/dim]\n")

        analysis_df = analyze_level_session(result_df, level_cols)
        _render_level_analysis_table("Level Session", analysis_df, "Session")
        console.print(
            f"\n[dim]Daten: {len(result_df):,} Bars | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}[/dim]"
        )
        if save:
            md_path = _save_level_analysis_markdown(
                algo_file,
                "session",
                "Level Session",
                analysis_df,
                "Session",
                result_df,
            )
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="level-confluence")
    def level_confluence(
        algo_name: str = typer.Argument(..., help="Name des Level-Bausteins"),
        proximity: float = typer.Option(
            10.0, "--proximity", help="Maximaler Abstand in Punkten"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten speichern"
        ),
    ) -> None:
        from sb.level_analysis import analyze_level_confluence

        algo_file, result_df, level_cols = _load_level_analysis_input(
            algo_name, sources, data
        )
        console.print(
            f"\n[bold cyan]Level Confluence: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(
            f"[dim]Proximity: {proximity} Pkt | Level: {', '.join(level_cols)}[/dim]\n"
        )

        analysis_df = analyze_level_confluence(result_df, level_cols, proximity=proximity)
        _render_level_analysis_table(
            "Level Confluence", analysis_df, "Konfluenz", show_avg_distance=True
        )
        console.print(
            f"\n[dim]Daten: {len(result_df):,} Bars | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}[/dim]"
        )
        if save:
            md_path = _save_level_analysis_markdown(
                algo_file,
                "confluence",
                "Level Confluence",
                analysis_df,
                "Konfluenz",
                result_df,
                show_avg_distance=True,
            )
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="level-direction")
    def level_direction(
        algo_name: str = typer.Argument(..., help="Name des Level-Bausteins"),
        lookback: int = typer.Option(10, "--lookback", help="Bars für Richtungs-Vergleich"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten speichern"
        ),
    ) -> None:
        from sb.level_analysis import analyze_level_direction

        algo_file, result_df, level_cols = _load_level_analysis_input(
            algo_name, sources, data
        )
        console.print(
            f"\n[bold cyan]Level Direction: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(
            f"[dim]Lookback: {lookback} Bars | Level: {', '.join(level_cols)}[/dim]\n"
        )

        try:
            analysis_df = analyze_level_direction(result_df, level_cols, lookback=lookback)
        except ValueError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc
        _render_level_analysis_table("Level Direction", analysis_df, "Richtung")
        console.print(
            f"\n[dim]Daten: {len(result_df):,} Bars | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}[/dim]"
        )
        if save:
            md_path = _save_level_analysis_markdown(
                algo_file,
                "direction",
                "Level Direction",
                analysis_df,
                "Richtung",
                result_df,
            )
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="level-inspect")
    def level_inspect(
        algo_name: str = typer.Argument(
            ..., help="Name des Level-Bausteins (z.B. 'Genauer Hoch')"
        ),
        threshold: float = typer.Option(
            3.0, "--threshold", "-t", help="Touch-Toleranz in Punkten"
        ),
        window: int = typer.Option(30, "--window", "-w", help="Fenstergröße in Minuten"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten in _research/ speichern"
        ),
    ) -> None:
        """Level Touch-Heatmap: wann im Tagesverlauf kommt Preis an PDH/PDL/PWH/PWL?

        Zeigt für jeden erkannten Preis-Level eine Zeit × Wochentag Heatmap
        mit Touch-Häufigkeiten (Rising-Edge, 1 Touch pro Episode).
        """
        import pandas as _pd

        from sb.inspect import (
            compute_level_touch_heatmap,
            detect_level_columns,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Level-Inspect: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(
            f"[dim]Threshold: {threshold} Pkt | Fenster: {window} min | Rising-Edge[/dim]"
        )
        console.print("[dim]Lade Daten und führe Algo aus...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)

        try:
            result_df = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen: {exc}[/red]")
            raise typer.Exit(1)

        level_cols = detect_level_columns(result_df)
        if not level_cols:
            console.print(
                "[red]Keine Level-Spalten erkannt. Algo gibt keine Level aus.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[dim]Erkannte Level: {', '.join(sorted(level_cols))}[/dim]\n")

        from rich.table import Table as _Table

        dow_order = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
        all_heatmaps: dict[str, dict[str, dict[str, int]]] = {}

        for level_col in sorted(level_cols):
            heatmap = compute_level_touch_heatmap(
                result_df, level_col, threshold=threshold, window_minutes=window
            )
            all_heatmaps[level_col] = heatmap
            if not heatmap:
                console.print(f"[yellow]{level_col}: keine Touches gefunden[/yellow]")
                continue

            total_touches = sum(sum(v.values()) for v in heatmap.values())
            present_dows = [d for d in dow_order if any(d in v for v in heatmap.values())]
            max_cell = max((v for row in heatmap.values() for v in row.values()), default=1)

            h_table = _Table(
                title=f"{level_col}  ({total_touches} Touches gesamt)", show_header=True
            )
            h_table.add_column("Fenster (NY)", style="cyan", width=10)
            for dow in present_dows:
                h_table.add_column(dow, justify="right", width=6)

            for slot in sorted(heatmap):
                row_vals = heatmap[slot]
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
            console.print()

        n_bars = len(result_df)
        t_start = str(result_df.index[0])[:10]
        t_end = str(result_df.index[-1])[:10]
        console.print(f"[dim]Daten: {n_bars:,} Bars | {t_start} → {t_end}[/dim]")

        if save:
            from datetime import datetime as _dt

            date_str = _dt.now().strftime("%Y-%m-%d")
            research_dir = algo_file.parent / "_research"
            research_dir.mkdir(exist_ok=True)
            base_name = f"{algo_file.stem}_level_inspect_{date_str}"

            md_lines = [
                f"# Level-Inspect: {algo_file.name}",
                "",
                f"**Generiert:** {date_str}  ",
                f"**Threshold:** {threshold} Punkte  ",
                f"**Fenster:** {window} min  ",
                f"**Bars:** {n_bars:,} | {t_start} → {t_end}  ",
                "",
            ]
            for level_col, heatmap in all_heatmaps.items():
                if not heatmap:
                    continue
                total = sum(sum(v.values()) for v in heatmap.values())
                present_dows = [
                    d for d in dow_order if any(d in v for v in heatmap.values())
                ]
                md_lines += [f"## {level_col} ({total} Touches)", ""]
                header = "| Fenster | " + " | ".join(present_dows) + " |"
                sep = "|---|" + "|".join(["---"] * len(present_dows)) + "|"
                md_lines += [header, sep]
                for slot in sorted(heatmap):
                    row_vals = heatmap[slot]
                    vals = " | ".join(str(row_vals.get(d, 0)) for d in present_dows)
                    md_lines.append(f"| {slot} | {vals} |")
                md_lines.append("")

            md_path = research_dir / f"{base_name}.md"
            md_path.write_text("\n".join(md_lines), encoding="utf-8")
            console.print(f"\nForschungsdaten gespeichert: [green]{research_dir}[/green]")


    @app.command(name="zone-near-level")
    def zone_near_level(
        zone_algo: str = typer.Argument(
            ..., help="Algo mit Zone-Spalten (z.B. 'FVG Standard')"
        ),
        level_algo: str = typer.Argument(
            ..., help="Algo mit Level-Spalten (z.B. 'Genauer Hoch und Tief')"
        ),
        proximity: float = typer.Option(
            20.0, "--proximity", "-p", help="Max. Distanz zone_mid ↔ level in Punkten"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        save: bool = typer.Option(
            False, "--save", help="Ergebnisse in _research/ speichern"
        ),
    ) -> None:
        """Proximity-Analyse: Bounce-Rate von Zonen nahe vs. fern von Preis-Leveln."""
        import pandas as _pd

        from sb.inspect import (
            analyze_zone_near_level,
            detect_level_columns,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        zone_algo_file = find_algo_file(zone_algo, algo_dirs)
        if zone_algo_file is None:
            console.print(f"[red]Kein Algo gefunden für Zone-Algo '{zone_algo}'[/red]")
            raise typer.Exit(1)

        level_algo_file = find_algo_file(level_algo, algo_dirs)
        if level_algo_file is None:
            console.print(f"[red]Kein Algo gefunden für Level-Algo '{level_algo}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Zone-Near-Level:[/bold cyan] "
            f"[white]{zone_algo_file.name}[/white] × [white]{level_algo_file.name}[/white]"
            f"  [dim](proximity ≤ {proximity} Pt)[/dim]"
        )
        console.print("[dim]Lade Daten und führe Algos aus...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)

        try:
            df_zones = run_algo(zone_algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen von Zone-Algo: {exc}[/red]")
            raise typer.Exit(1)

        try:
            df_combined = run_algo(level_algo_file, df_zones)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen von Level-Algo: {exc}[/red]")
            raise typer.Exit(1)

        zone_prefixes = detect_zone_prefixes(df_combined)
        if not zone_prefixes:
            console.print(
                "[red]Keine Zone-Spalten erkannt. Zone-Algo gibt keine Zonen-Spalten aus.[/red]"
            )
            raise typer.Exit(1)

        all_level_cols = detect_level_columns(df_combined)
        level_cols = [
            c
            for c in all_level_cols
            if not c.startswith("run_") and not c.startswith("session_")
        ]
        if not level_cols:
            console.print(
                "[red]Keine Level-Spalten erkannt (nach Filter run_*/session_*). "
                "Level-Algo gibt keine Level-Spalten aus.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[dim]Erkannte Zonen-Prefixe: {', '.join(zone_prefixes)}[/dim]")
        console.print(
            f"[dim]Erkannte Level-Spalten: {', '.join(sorted(level_cols))}[/dim]\n"
        )

        from rich.table import Table

        table = Table(
            title=f"Zone-Near-Level Analyse: {zone_algo_file.name} × {level_algo_file.name}"
            f"  (proximity ≤ {proximity} Pt)",
            show_lines=True,
        )
        table.add_column("Zone", style="cyan")
        table.add_column("Level", style="magenta")
        table.add_column("Nahe Level\nN / Return%", justify="right")
        table.add_column("Nicht nahe Level\nN / Return%", justify="right")
        table.add_column("Delta Return", justify="right")

        rows = []
        for zone_prefix in zone_prefixes:
            for level_col in sorted(level_cols):
                try:
                    stats = analyze_zone_near_level(
                        df_combined, zone_prefix, level_col, proximity_pts=proximity
                    )
                except ValueError as exc:
                    console.print(
                        f"[yellow]{zone_prefix} × {level_col} übersprungen: {exc}[/yellow]"
                    )
                    continue

                near = stats["near"]
                far = stats["far"]
                delta = round(near["bounce_pct"] - far["bounce_pct"], 1)
                rows.append((zone_prefix, level_col, near, far, delta))

        # Sort by delta descending
        rows.sort(key=lambda r: r[4], reverse=True)

        for zone_prefix, level_col, near, far, delta in rows:
            near_str = f"{near['touches']:,} / {near['bounce_pct']}%"
            far_str = f"{far['touches']:,} / {far['bounce_pct']}%"
            delta_color = "green" if delta >= 0 else "red"
            delta_str = f"[{delta_color}]{delta:+.1f}%[/{delta_color}]"

            table.add_row(
                zone_prefix,
                level_col,
                near_str,
                far_str,
                delta_str,
            )

        console.print(table)

        n_bars = len(df_combined)
        t_start = str(df_combined.index[0])[:10]
        t_end = str(df_combined.index[-1])[:10]
        console.print(
            f"\n[dim]Daten: {n_bars:,} Bars | {t_start} → {t_end} | "
            f"Zone: {zone_algo_file.name} | Level: {level_algo_file.name}[/dim]"
        )

        if save and rows:
            import datetime as _dt

            date_str = _dt.date.today().isoformat()
            res_dir = zone_algo_file.parent / "_research"
            res_dir.mkdir(exist_ok=True)
            z_slug = zone_algo_file.stem.replace(" ", "_").replace(".", "")
            l_slug = level_algo_file.stem.replace(" ", "_").replace(".", "")
            out_path = (
                res_dir
                / f"zone_near_level_{z_slug}_x_{l_slug}_p{int(proximity)}_{date_str}.md"
            )
            lines = [
                f"# Zone-Near-Level: {zone_algo_file.name} × {level_algo_file.name}",
                f"Proximity ≤ {proximity} Pt | {t_start} → {t_end} | {n_bars:,} Bars",
                "",
                "| Zone | Level | Nahe N | Nahe Return% | Fern N | Fern Return% | Delta |",
                "|------|-------|--------|--------------|--------|--------------|-------|",
            ]
            for zone_prefix, level_col, near, far, delta in rows:
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"| {zone_prefix} | {level_col} | {near['touches']:,} | {near['bounce_pct']}% "
                    f"| {far['touches']:,} | {far['bounce_pct']}% | {sign}{delta:.1f}% |"
                )
            out_path.write_text("\n".join(lines) + "\n")
            console.print(f"[green]Gespeichert:[/green] {out_path}")


    @app.command(name="zone-return")
    def zone_return(
        algo_name: str = typer.Argument(
            ..., help="Algo mit Zone-Spalten (z.B. '7d. Session Hoch-Tief Orderblock')"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        windows: str = typer.Option(
            "15,30,60,120",
            "--windows",
            "-w",
            help="Return-Fenster in Bars (Komma-getrennt)",
        ),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Forschungsdaten speichern"
        ),
    ) -> None:
        """Return-Rate nach Durch: Wie oft kommt Preis nach Durchbruch zurück? (Fake-Out vs. echter Breakout)"""
        import pandas as _pd

        from sb.inspect import (
            analyze_zone_return_after_through,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        return_windows = [int(x.strip()) for x in windows.split(",")]

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Zone-Return-Analyse:[/bold cyan] [white]{algo_file.name}[/white]"
        )
        console.print(f"[dim]Return-Fenster: {return_windows} Bars | Lade Daten...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)

        try:
            df_result = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen des Algos: {exc}[/red]")
            raise typer.Exit(1)

        zone_prefixes = detect_zone_prefixes(df_result)
        if not zone_prefixes:
            console.print("[red]Keine Zone-Spalten erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"[dim]Erkannte Zonen-Prefixe: {', '.join(zone_prefixes)}[/dim]\n")

        win_headers = [f"{w}min" for w in return_windows]
        table = Table(
            title=f"Return-Rate nach Durch: {algo_file.name}",
            show_lines=True,
        )
        table.add_column("Zone", style="cyan")
        table.add_column("Seite", style="white")
        table.add_column("Durch\n(N)", justify="right")
        for wh in win_headers:
            table.add_column(f"Return\n{wh}", justify="right", style="green")

        all_results: dict = {}
        for prefix in zone_prefixes:
            try:
                res = analyze_zone_return_after_through(df_result, prefix, return_windows)
            except ValueError as exc:
                console.print(f"[yellow]Prefix '{prefix}' übersprungen: {exc}[/yellow]")
                continue
            all_results[prefix] = res

            for side in ("bull", "bear"):
                s = res[side]
                n_through = s["n_through"]
                cells = [str(n_through)]
                for w in return_windows:
                    pct = s.get(f"return_pct_{w}", 0.0)
                    n_ret = s.get(f"return_{w}", 0)
                    color = "green" if pct >= 50 else ("yellow" if pct >= 30 else "red")
                    cells.append(f"[{color}]{n_ret} / {pct}%[/{color}]")
                side_label = "🟢 Bull" if side == "bull" else "🔴 Bear"
                table.add_row(prefix, side_label, *cells)

        console.print(table)

        bar_count = len(df)
        date_range = ""
        if hasattr(df.index, "min"):
            try:
                date_range = f"{df.index.min().date()} → {df.index.max().date()}"
            except Exception:
                pass
        console.print(f"\n[dim]Daten: {bar_count:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            if research_dir:
                import json as _json

                today = datetime.now().strftime("%Y-%m-%d")
                fname = f"return_after_through_{today}"
                # JSON
                (research_dir / f"{fname}.json").write_text(
                    _json.dumps(all_results, indent=2, default=str)
                )
                # MD
                lines = [
                    f"# Zone-Return nach Durch: {algo_file.name}",
                    f"\n**Generiert:** {today}",
                    f"**Daten:** {date_range}",
                    f"**Return-Fenster:** {return_windows} Bars (Minuten auf 1min-Daten)",
                    "\n## Interpretation",
                    "- **Hohe Return-Rate** (>50%): Zone wird oft nur kurz durchstochen (Fake-Out / Liquidity Sweep)",
                    "- **Niedrige Return-Rate** (<30%): Zone wird wirklich gebrochen (echter Breakout)",
                    "",
                ]
                for prefix, res in all_results.items():
                    lines.append(f"## Prefix: `{prefix}`")
                    for side in ("bull", "bear"):
                        s = res[side]
                        lines.append(
                            f"\n### {'Bull (von oben)' if side == 'bull' else 'Bear (von unten)'}"
                        )
                        lines.append("| Kennzahl | Wert |")
                        lines.append("|---|---|")
                        lines.append(f"| Durch gesamt | {s['n_through']} |")
                        for w in return_windows:
                            lines.append(
                                f"| Return in {w}min | {s.get(f'return_{w}', 0)} ({s.get(f'return_pct_{w}', 0)}%) |"
                            )
                    lines.append("")
                (research_dir / f"{fname}.md").write_text("\n".join(lines))
                console.print(f"\n[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-break")
    def zone_break(
        algo_name: str = typer.Argument(..., help="Algo-Name (ohne .py)"),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        window: int = typer.Option(
            60, "--window", "-w", help="Bars ohne Return = echter Break"
        ),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Break-Profil: Hold / Fake-Out / Echter Ausbruch – mit Heatmap nach NY-Stunde."""
        from sb.inspect import (
            analyze_zone_break_profile,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)

        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen des Algos: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nZone-Break-Profil: [bold]{algo_file.name}[/bold]")
        console.print(f"Echter Break = kein Return in {window} Bars | Lade Daten...")
        console.print(f"Erkannte Zonen-Prefixe: {', '.join(prefixes)}\n")

        all_results: dict = {}

        for prefix in prefixes:
            res = analyze_zone_break_profile(df, prefix, no_return_window=window)
            all_results[prefix] = res

            for side in ("bull", "bear"):
                s = res[side]
                label = "🟢 Bull" if side == "bull" else "🔴 Bear"
                total = s["total"]
                console.print(f"[bold]{prefix} {label}[/bold] — {total} Zonen gesamt")
                console.print(
                    f"  Hold (Return):      {s['hold']:>5}  ({s['hold_pct']:>5.1f}%)"
                )
                console.print(
                    f"  Fake-Out:           {s['fake_out']:>5}  ({s['fake_out_pct']:>5.1f}%)"
                )
                console.print(
                    f"  [red]Echter Break[/red]:       {s['real_break']:>5}  ({s['real_break_pct']:>5.1f}%)"
                )

                # Heatmap echter Break nach NY-Stunde
                bh = s["break_by_hour"]
                if bh:
                    top_hours = sorted(bh.items(), key=lambda x: -x[1])[:5]
                    top_str = "  ".join(f"{h:02d}h={c}" for h, c in top_hours)
                    console.print(f"  Break-Hotspot (NY):  {top_str}")
                console.print()

        # Zusammenfassung als Tabelle
        from rich import box as rich_box

        table = Table(title=f"Break-Profil: {algo_file.name}", box=rich_box.SIMPLE_HEAVY)
        table.add_column("Zone", style="dim")
        table.add_column("Seite")
        table.add_column("Gesamt", justify="right")
        table.add_column("Hold", justify="right")
        table.add_column("Fake-Out", justify="right")
        table.add_column("Echter Break", justify="right", style="red")
        table.add_column("Break-Hotspot (NY)", style="dim")

        for prefix, res in all_results.items():
            for side in ("bull", "bear"):
                s = res[side]
                label = "🟢 Bull" if side == "bull" else "🔴 Bear"
                bh = s["break_by_hour"]
                if bh:
                    top = sorted(bh.items(), key=lambda x: -x[1])[:3]
                    hotspot = "  ".join(f"{h:02d}h={c}" for h, c in top)
                else:
                    hotspot = "–"
                table.add_row(
                    prefix,
                    label,
                    str(s["total"]),
                    f"{s['hold']} ({s['hold_pct']}%)",
                    f"{s['fake_out']} ({s['fake_out_pct']}%)",
                    f"{s['real_break']} ({s['real_break_pct']}%)",
                    hotspot,
                )

        console.print(table)
        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"break_profile_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [
                f"# Break-Profil: {algo_file.name}",
                f"Datum: {date.today()} | Echter Break = kein Return in {window} Bars",
                f"Daten: {n_bars:,} Bars | {date_range}",
                "",
            ]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    label = "Bull" if side == "bull" else "Bear"
                    lines.append(f"## {prefix} {label}")
                    lines.append(f"- Gesamt: {s['total']}")
                    lines.append(f"- Hold (Return): {s['hold']} ({s['hold_pct']}%)")
                    lines.append(f"- Fake-Out: {s['fake_out']} ({s['fake_out_pct']}%)")
                    lines.append(
                        f"- Echter Break: {s['real_break']} ({s['real_break_pct']}%)"
                    )
                    bh = s["break_by_hour"]
                    if bh:
                        top = sorted(bh.items(), key=lambda x: -x[1])[:5]
                        lines.append(
                            "- Break-Hotspot: "
                            + ", ".join(f"{h:02d}:00 NY ({c}x)" for h, c in top)
                        )
                    lines.append("")
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-break-dur")
    def zone_break_dur(
        algo_name: str = typer.Argument(..., help="Algo-Name (ohne .py)"),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        max_window: int = typer.Option(
            120, "--max-window", "-m", help="Max Bars Beobachtung"
        ),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Wie lange ist der Preis AUSSERHALB einer Zone nach Ausbruch?

        Misst die Draußen-Dauer in Bars (= Minuten auf 1min-Daten).
        Buckets: 1-5, 6-10, 11-15, 16-20, 21-30, 31-60, 61-120, 120+
        """
        from sb.inspect import (
            analyze_zone_break_duration,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen des Algos: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nZone-Break-Dauer: [bold]{algo_file.name}[/bold]")
        console.print(f"Max-Fenster: {max_window} Bars | Buckets in Minuten (1min-Daten)\n")

        BUCKETS = [5, 10, 15, 20, 30, 60, 120]
        all_results: dict = {}

        for prefix in prefixes:
            res = analyze_zone_break_duration(df, prefix, max_window=max_window)
            all_results[prefix] = res

            for side in ("bull", "bear"):
                s = res[side]
                if s["n"] == 0:
                    continue
                label = "🟢 Bull" if side == "bull" else "🔴 Bear"
                console.print(f"[bold]{prefix} {label}[/bold] — {s['n']} Ausbrüche")
                console.print(
                    f"  Median: {s['median_bars']} Bars  |  Ø {s['mean_bars']} Bars  |  Permanent (>{max_window}min): {s['n_permanent']} ({s['permanent_pct']}%)"
                )

                # Noch-draußen-Kurve
                still = s["still_out_pct"]
                line = "  Noch draußen: " + "  ".join(
                    f">{t}min={still[t]}%" for t in BUCKETS
                )
                console.print(line)

                # Bucket-Verteilung als Mini-Bar
                bc = s["bucket_counts"]
                total = s["n"]
                bar_parts = []
                for key, cnt in bc.items():
                    pct = cnt / total * 100
                    bar_parts.append(f"{key}min: {cnt} ({pct:.0f}%)")
                console.print("  Verteilung:  " + "  |  ".join(bar_parts))
                console.print()

        # Zusammenfassungs-Tabelle
        table = Table(title=f"Break-Dauer: {algo_file.name}")
        table.add_column("Zone", style="dim")
        table.add_column("Seite")
        table.add_column("Ausbrüche", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Ø Bars", justify="right")
        table.add_column(">5min", justify="right")
        table.add_column(">10min", justify="right")
        table.add_column(">15min", justify="right")
        table.add_column(">20min", justify="right")
        table.add_column(f">{max_window}min", justify="right", style="red")

        for prefix, res in all_results.items():
            for side in ("bull", "bear"):
                s = res[side]
                if s["n"] == 0:
                    continue
                label = "🟢 Bull" if side == "bull" else "🔴 Bear"
                so = s["still_out_pct"]
                table.add_row(
                    prefix,
                    label,
                    str(s["n"]),
                    f"{s['median_bars']}min",
                    f"{s['mean_bars']}min",
                    f"{so.get(5, 0)}%",
                    f"{so.get(10, 0)}%",
                    f"{so.get(15, 0)}%",
                    f"{so.get(20, 0)}%",
                    f"{s['permanent_pct']}%",
                )

        console.print(table)
        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"break_duration_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [
                f"# Break-Dauer: {algo_file.name}",
                f"Datum: {date.today()} | Max-Fenster: {max_window} Bars",
                f"Daten: {n_bars:,} Bars | {date_range}",
                "",
            ]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    if s["n"] == 0:
                        continue
                    label = "Bull" if side == "bull" else "Bear"
                    so = s["still_out_pct"]
                    lines.append(f"## {prefix} {label}")
                    lines.append(f"- Ausbrüche: {s['n']}")
                    lines.append(f"- Median: {s['median_bars']} Bars/min")
                    lines.append(f"- Ø: {s['mean_bars']} Bars/min")
                    lines.append(
                        f"- Permanent (>{max_window}min): {s['n_permanent']} ({s['permanent_pct']}%)"
                    )
                    lines.append(
                        f"- Noch draußen: >5min={so.get(5)}%  >10min={so.get(10)}%  >15min={so.get(15)}%  >20min={so.get(20)}%  >30min={so.get(30)}%"
                    )
                    bc = s["bucket_counts"]
                    lines.append(
                        "- Verteilung: "
                        + "  |  ".join(f"{k}min={v}" for k, v in bc.items())
                    )
                    lines.append("")
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-bounce-depth")
    def zone_bounce_depth(
        algo_name: str = typer.Argument(...),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Penetrations-Tiefe bei Bounces: wie tief geht Preis in Hold-Zonen?"""
        from sb.inspect import (
            analyze_zone_bounce_depth,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nReturn-Tiefe: [bold]{algo_file.name}[/bold]\n")
        all_results: dict = {}

        table = Table(title=f"Return-Tiefe: {algo_file.name}")
        table.add_column("Zone", style="dim")
        table.add_column("Seite")
        table.add_column("Hold-N", justify="right")
        table.add_column("Median %", justify="right")
        table.add_column("Ø %", justify="right")
        table.add_column("Median Pts", justify="right")
        table.add_column("0-10%", justify="right")
        table.add_column("10-25%", justify="right")
        table.add_column("25-50%", justify="right")
        table.add_column("50-75%", justify="right")
        table.add_column("75-100%", justify="right")

        for prefix in prefixes:
            res = analyze_zone_bounce_depth(df, prefix)
            all_results[prefix] = res
            for side in ("bull", "bear"):
                s = res[side]
                if s.get("n", 0) == 0:
                    continue
                label = "Bull" if side == "bull" else "Bear"
                bc = s["bucket_counts"]
                table.add_row(
                    prefix,
                    label,
                    str(s["n"]),
                    f"{s['median_pct']}%",
                    f"{s['mean_pct']}%",
                    f"{s['median_pts']}pts",
                    str(bc.get("0-10%", 0)),
                    str(bc.get("10-25%", 0)),
                    str(bc.get("25-50%", 0)),
                    str(bc.get("50-75%", 0)),
                    str(bc.get("75-100%", 0)),
                )

        console.print(table)
        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"bounce_depth_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [
                f"# Return-Tiefe: {algo_file.name}",
                f"Datum: {date.today()}",
                f"Daten: {n_bars:,} Bars | {date_range}",
                "",
            ]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    if s.get("n", 0) == 0:
                        continue
                    label = "Bull" if side == "bull" else "Bear"
                    lines += [
                        f"## {prefix} {label}",
                        f"- Hold-Zonen: {s['n']}",
                        f"- Median Tiefe: {s['median_pct']}% ({s['median_pts']}pts)",
                        f"- Ø Tiefe: {s['mean_pct']}% ({s['mean_pts']}pts)",
                        "- Buckets: "
                        + "  ".join(f"{k}={v}" for k, v in s["bucket_counts"].items()),
                        "",
                    ]
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-session-profile")
    def zone_session_profile(
        algo_name: str = typer.Argument(...),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        window: int = typer.Option(60, "--window", "-w"),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Hold/Break-Rate nach Session-Entstehung (Asia/London/AM/PM)."""
        from sb.inspect import (
            analyze_zone_by_session,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nSession-Profil: [bold]{algo_file.name}[/bold]\n")
        SESSIONS = ["Asia", "London", "PreM", "AM", "Lunch", "PM", "AH"]
        all_results: dict = {}

        for prefix in prefixes:
            res = analyze_zone_by_session(df, prefix, no_return_window=window)
            all_results[prefix] = res

            for side in ("bull", "bear"):
                s = res[side]
                label = "Bull" if side == "bull" else "Bear"
                table = Table(title=f"{prefix} {label} – Session-Profil")
                table.add_column("Session")
                table.add_column("N", justify="right")
                table.add_column("Hold", justify="right")
                table.add_column("Fake-Out", justify="right")
                table.add_column("Echter Break", justify="right", style="red")
                for sess in SESSIONS:
                    d = s.get(sess, {})
                    if d.get("total", 0) == 0:
                        continue
                    table.add_row(
                        sess,
                        str(d["total"]),
                        f"{d['hold_pct']}%",
                        f"{d['fake_out_pct']}%",
                        f"{d['real_break_pct']}%",
                    )
                console.print(table)

        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"session_profile_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [f"# Session-Profil: {algo_file.name}", f"Datum: {date.today()}", ""]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    label = "Bull" if side == "bull" else "Bear"
                    lines.append(f"## {prefix} {label}")
                    for sess in SESSIONS:
                        d = s.get(sess, {})
                        if d.get("total", 0) == 0:
                            continue
                        lines.append(
                            f"- {sess}: N={d['total']} Hold={d['hold_pct']}% Break={d['real_break_pct']}%"
                        )
                    lines.append("")
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-context-filter")
    def zone_context_filter(
        algo_name: str = typer.Argument(..., help="OB-Algo Name"),
        context_algo: str = typer.Option(
            ..., "--context", "-c", help="Context-Algo Name (z.B. 'manip_liquidity_sweep')"
        ),
        context_col: str = typer.Option(
            ..., "--col", help="Context-Spalte (z.B. 'manip_bear')"
        ),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        window: int = typer.Option(60, "--window", "-w"),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Vergleicht Hold/Break-Rate einer Zone mit vs. ohne Context-Signal."""
        from sb.inspect import (
            analyze_zone_with_context,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        ctx_file = find_algo_file(context_algo, algo_dirs)
        if algo_file is None or ctx_file is None:
            console.print(
                f"[red]Algo nicht gefunden: {algo_name} oder {context_algo}[/red]"
            )
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)

        try:
            df_zone = run_algo(algo_file, raw_df)
            df_ctx = run_algo(ctx_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        if context_col not in df_ctx.columns:
            console.print(
                f"[red]Spalte '{context_col}' nicht in Context-Algo. Verfügbar: {list(df_ctx.columns)}[/red]"
            )
            raise typer.Exit(1)

        df_zone[context_col] = df_ctx[context_col]
        prefixes = detect_zone_prefixes(df_zone)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(
            f"\nContext-Filter: [bold]{algo_file.name}[/bold] × [bold]{context_col}[/bold]\n"
        )
        all_results: dict = {}

        for prefix in prefixes:
            res = analyze_zone_with_context(
                df_zone, prefix, context_col, no_return_window=window
            )
            all_results[prefix] = res

            for side in ("bull", "bear"):
                s = res[side]
                label = "Bull" if side == "bull" else "Bear"
                table = Table(title=f"{prefix} {label} – mit vs. ohne {context_col}")
                table.add_column("Kontext")
                table.add_column("N", justify="right")
                table.add_column("Hold", justify="right")
                table.add_column("Fake-Out", justify="right")
                table.add_column("Echter Break", justify="right", style="red")
                for ctx_val, ctx_label in [
                    (True, f"{context_col}=aktiv"),
                    (False, f"{context_col}=inaktiv"),
                ]:
                    d = s.get(ctx_val, {})
                    if d.get("total", 0) == 0:
                        continue
                    table.add_row(
                        ctx_label,
                        str(d["total"]),
                        f"{d['hold_pct']}%",
                        f"{d['fake_out_pct']}%",
                        f"{d['real_break_pct']}%",
                    )
                console.print(table)

        n_bars = len(df_zone)
        date_range = f"{df_zone.index[0].date()} → {df_zone.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"context_{context_col}_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [
                f"# Context-Filter: {algo_file.name} × {context_col}",
                f"Datum: {date.today()}",
                "",
            ]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    label = "Bull" if side == "bull" else "Bear"
                    lines.append(f"## {prefix} {label}")
                    for ctx_val in (True, False):
                        d = s.get(ctx_val, {})
                        ctx_str = "MIT Kontext" if ctx_val else "OHNE Kontext"
                        lines.append(
                            f"- {ctx_str}: N={d.get('total', 0)} Hold={d.get('hold_pct', 0)}% Break={d.get('real_break_pct', 0)}%"
                        )
                    lines.append("")
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-visits")
    def zone_visits(
        algo_name: str = typer.Argument(...),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        max_visits: int = typer.Option(3, "--max-visits", "-v"),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """First vs. Second vs. Third Visit: Wird die Zone mit jedem Besuch schwächer?"""
        from sb.inspect import (
            analyze_zone_visit_sequence,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nZone-Besuche: [bold]{algo_file.name}[/bold]\n")
        all_results: dict = {}

        for prefix in prefixes:
            res = analyze_zone_visit_sequence(df, prefix, max_visits=max_visits)
            all_results[prefix] = res

            for side in ("bull", "bear"):
                s = res[side]
                label = "Bull" if side == "bull" else "Bear"
                table = Table(title=f"{prefix} {label} – Visit-Sequenz")
                table.add_column("Besuch")
                table.add_column("N", justify="right")
                table.add_column("Ø Dauer", justify="right")
                table.add_column("Return ↑", justify="right", style="green")
                table.add_column("Break ↓", justify="right", style="red")
                table.add_column("Unclear", justify="right", style="dim")
                for v in range(1, max_visits + 1):
                    d = s.get(f"visit_{v}", {})
                    if d.get("n", 0) == 0:
                        continue
                    table.add_row(
                        f"#{v}",
                        str(d["n"]),
                        f"{d['avg_duration']}min",
                        f"{d['bounce_pct']}%",
                        f"{d['break_pct']}%",
                        f"{d['unclear_pct']}%",
                    )
                console.print(table)

        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"visit_sequence_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [f"# Visit-Sequenz: {algo_file.name}", f"Datum: {date.today()}", ""]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    label = "Bull" if side == "bull" else "Bear"
                    lines.append(f"## {prefix} {label}")
                    for v in range(1, max_visits + 1):
                        d = s.get(f"visit_{v}", {})
                        if d.get("n", 0) == 0:
                            continue
                        lines.append(
                            f"- Besuch #{v}: N={d['n']} Ø={d['avg_duration']}min Return={d['bounce_pct']}% Break={d['break_pct']}%"
                        )
                    lines.append("")
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-visit-dur")
    def zone_visit_dur(
        algo_name: str = typer.Argument(...),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        max_visits: int = typer.Option(4, "--max-visits", "-v"),
        post_death: int = typer.Option(120, "--post-death", "-p"),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Dauer je Besuch (drin/draußen) + Post-Death-Analyse nach finalem Break."""
        from sb.inspect import (
            analyze_zone_visit_duration,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nVisit-Dauer: [bold]{algo_file.name}[/bold]\n")
        all_results: dict = {}

        for prefix in prefixes:
            res = analyze_zone_visit_duration(
                df, prefix, max_visits=max_visits, post_death_window=post_death
            )
            all_results[prefix] = res

            for side in ("bull", "bear"):
                s = res[side]
                label = "Bull" if side == "bull" else "Bear"
                table = Table(title=f"{prefix} {label} – Visit-Dauer")
                table.add_column("Besuch")
                table.add_column("N", justify="right")
                table.add_column("Drin (med)", justify="right")
                table.add_column("Außen vor (med)", justify="right")
                table.add_column("Return ↑", justify="right", style="green")
                table.add_column("Break ↓", justify="right", style="red")
                table.add_column("Unclear", justify="right", style="dim")
                for v in range(1, max_visits + 1):
                    d = s.get(f"visit_{v}", {})
                    if d.get("n", 0) == 0:
                        continue
                    outside = f"{d['median_outside_before']}min" if v > 1 else "–"
                    table.add_row(
                        f"#{v}",
                        str(d["n"]),
                        f"{d['median_duration_inside']}min",
                        outside,
                        f"{d['exit_return_pct']}%",
                        f"{d['exit_break_pct']}%",
                        f"{d['exit_unclear_pct']}%",
                    )
                console.print(table)

                pd_stats = s.get("post_death", {})
                if pd_stats.get("n", 0) > 0:
                    console.print(
                        f"  Post-Death: N={pd_stats['n']} | "
                        f"Permanent={pd_stats['permanent_pct']}% | "
                        f"Median Re-Touch={pd_stats['median_bars_to_retouch']}min"
                    )

        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"visit_duration_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [f"# Visit-Dauer: {algo_file.name}", f"Datum: {date.today()}", ""]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    label = "Bull" if side == "bull" else "Bear"
                    lines.append(f"## {prefix} {label}")
                    for v in range(1, max_visits + 1):
                        d = s.get(f"visit_{v}", {})
                        if d.get("n", 0) == 0:
                            continue
                        lines.append(
                            f"- Besuch #{v}: N={d['n']} Drin={d['median_duration_inside']}min "
                            f"AußenVor={d['median_outside_before']}min "
                            f"Return={d['exit_return_pct']}% Break={d['exit_break_pct']}%"
                        )
                    pd_s = s.get("post_death", {})
                    if pd_s.get("n", 0) > 0:
                        lines.append(
                            f"- Post-Death: N={pd_s['n']} Permanent={pd_s['permanent_pct']}% "
                            f"ReTouch-Median={pd_s['median_bars_to_retouch']}min"
                        )
                    lines.append("")
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-return-excursion")
    def zone_return_excursion(
        algo_name: str = typer.Argument(...),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        window: int = typer.Option(120, "--window", "-w"),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Fake-Out SL-Analyse: Wie weit geht Preis nach Durchbruch bevor Rückkehr?"""
        from sb.inspect import (
            analyze_zone_return_excursion,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nFake-Out Gegenlauf: [bold]{algo_file.name}[/bold]\n")
        all_results: dict = {}

        table = Table(title=f"Fake-Out SL-Bereich: {algo_file.name}")
        table.add_column("Zone", style="dim")
        table.add_column("Seite")
        table.add_column("N Fake-Outs", justify="right")
        table.add_column("p50", justify="right", style="green")
        table.add_column("p75", justify="right")
        table.add_column("p80", justify="right")
        table.add_column("p90", justify="right", style="red")
        table.add_column("Ø", justify="right", style="dim")

        for prefix in prefixes:
            res = analyze_zone_return_excursion(df, prefix, return_window=window)
            all_results[prefix] = res
            for side in ("bull", "bear"):
                s = res[side]
                if s.get("n_fake_outs", 0) == 0:
                    continue
                label = "Bull" if side == "bull" else "Bear"
                table.add_row(
                    prefix,
                    label,
                    str(s["n_fake_outs"]),
                    f"{s['p50']}pts",
                    f"{s['p75']}pts",
                    f"{s['p80']}pts",
                    f"{s['p90']}pts",
                    f"{s['mean']}pts",
                )

        console.print(table)
        console.print(
            f"[dim]Return-Fenster: {window} Bars | "
            f"Daten: {len(df):,} Bars | {df.index[0].date()} → {df.index[-1].date()}[/dim]"
        )

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"return_excursion_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [
                f"# Fake-Out SL-Bereich: {algo_file.name}",
                f"Datum: {date.today()}",
                "",
            ]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    s = res[side]
                    if s.get("n_fake_outs", 0) == 0:
                        continue
                    label = "Bull" if side == "bull" else "Bear"
                    lines.append(
                        f"## {prefix} {label}: N={s['n_fake_outs']} "
                        f"p50={s['p50']}pts p75={s['p75']}pts p80={s['p80']}pts p90={s['p90']}pts"
                    )
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="manip-day-stats")
    def manip_day_stats(
        algo_name: str = typer.Argument(
            ..., help="MANIP-Algo Name (z.B. 'manip_liquidity_sweep')"
        ),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        signal: str = typer.Option(
            "manip_bear", "--signal", help="Signal-Spalte im Algo-Output"
        ),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Tages-Richtung an MANIP-Tagen vs. normalen Tagen: Warum wirkt MANIP Bear?"""
        from sb.inspect import (
            analyze_manip_day_bias,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            sig_df = run_algo(algo_file, raw_df)
            # MANIP algo nur gibt Signal-Spalten zurück → mit OHLCV joinen
            df = raw_df.join(sig_df, how="left")
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        if signal not in df.columns:
            console.print(
                f"[red]Signal-Spalte '{signal}' nicht im Algo-Output. Verfügbar: {list(sig_df.columns[:10])}[/red]"
            )
            raise typer.Exit(1)

        res = analyze_manip_day_bias(df, signal_col=signal)

        console.print(
            f"\n[bold]MANIP-Tages-Bias: {algo_file.name} | Signal: {signal}[/bold]\n"
        )

        table = Table(title=f"Tages-Richtung: {signal}")
        table.add_column("Gruppe")
        table.add_column("N Tage", justify="right")
        table.add_column("Ø Delta", justify="right")
        table.add_column("Median Delta", justify="right")
        table.add_column("% Bearish", justify="right", style="red")
        table.add_column("% Bullish", justify="right", style="green")
        table.add_column("Std", justify="right", style="dim")

        for key, label in [
            ("manip_active", f"{signal} AKTIV"),
            ("no_manip", "Kein Signal"),
        ]:
            s = res[key]
            if s["n"] == 0:
                continue
            color = "red" if s["avg_delta"] < 0 else "green"
            table.add_row(
                label,
                str(s["n"]),
                f"[{color}]{s['avg_delta']:+.1f}pts[/{color}]",
                f"{s['median_delta']:+.1f}pts",
                f"{s['pct_bearish']}%",
                f"{s['pct_bullish']}%",
                f"{s.get('std_delta', 0):.1f}",
            )

        console.print(table)

        m = res["manip_active"]
        n = res["no_manip"]
        if m["n"] > 0 and n["n"] > 0:
            diff = m["avg_delta"] - n["avg_delta"]
            color = "red" if diff < 0 else "green"
            console.print(
                f"\n[bold]Delta-Differenz (MANIP − Kein Signal): [{color}]{diff:+.1f}pts[/{color}][/bold]"
            )

        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"manip_day_bias_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(res, indent=2, default=str)
            )
            m_s = res["manip_active"]
            n_s = res["no_manip"]
            lines = [
                f"# MANIP Tages-Bias: {signal}",
                f"Datum: {date.today()}",
                "",
                f"## {signal} AKTIV (N={m_s['n']})",
                f"- Ø Delta: {m_s['avg_delta']:+.1f}pts",
                f"- % Bearish: {m_s['pct_bearish']}%",
                "",
                f"## Kein Signal (N={n_s['n']})",
                f"- Ø Delta: {n_s['avg_delta']:+.1f}pts",
                f"- % Bearish: {n_s['pct_bearish']}%",
            ]
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="zone-session-return")
    def zone_session_return(
        algo_name: str = typer.Argument(...),
        sources: str = typer.Option("", "--sources", "-s"),
        extra_dir: str = typer.Option("", "--dir", "-d"),
        data: str = typer.Option("", "--data"),
        save: bool = typer.Option(True, "--save/--no-save"),
    ) -> None:
        """Session-Break-Analyse: London/AM Breaks – kehren sie zurück oder permanent?"""
        from sb.inspect import (
            analyze_zone_session_break_return,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        import pandas as _pd

        raw_df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in raw_df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        raw_df = raw_df.rename(columns=rename_map)
        try:
            df = run_algo(algo_file, raw_df)
        except Exception as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        prefixes = detect_zone_prefixes(df)
        if not prefixes:
            console.print("[red]Keine Zone-Prefixe erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"\nSession Break→Return: [bold]{algo_file.name}[/bold]\n")
        all_results: dict = {}

        SESSIONS = ["Asia", "London", "PreM", "AM", "Lunch", "PM", "AH"]
        WINDOWS = [15, 30, 60, 120]

        for prefix in prefixes:
            res = analyze_zone_session_break_return(df, prefix, return_windows=WINDOWS)
            all_results[prefix] = res

            for side in ("bull", "bear"):
                label = "Bull" if side == "bull" else "Bear"
                side_data = res[side]

                table = Table(title=f"{prefix} {label} – Session Break → Return %")
                table.add_column("Session")
                table.add_column("N Break", justify="right")
                table.add_column("15min%", justify="right")
                table.add_column("30min%", justify="right")
                table.add_column("60min%", justify="right")
                table.add_column("120min%", justify="right", style="green")

                for sess in SESSIONS:
                    s = side_data.get(sess, {})
                    n = s.get("n_through", 0)
                    if n == 0:
                        continue
                    table.add_row(
                        sess,
                        str(n),
                        f"{s.get('return_pct_15', 0)}%",
                        f"{s.get('return_pct_30', 0)}%",
                        f"{s.get('return_pct_60', 0)}%",
                        f"{s.get('return_pct_120', 0)}%",
                    )
                console.print(table)

        n_bars = len(df)
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        console.print(f"[dim]Daten: {n_bars:,} Bars | {date_range}[/dim]")

        if save and all_results:
            research_dir = algo_file.parent.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            import json
            from datetime import date

            fname = f"session_break_return_{date.today()}"
            (research_dir / f"{fname}.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            lines = [
                f"# Session Break-Return: {algo_file.name}",
                f"Datum: {date.today()}",
                "",
            ]
            for prefix, res in all_results.items():
                for side in ("bull", "bear"):
                    label = "Bull" if side == "bull" else "Bear"
                    lines.append(f"## {prefix} {label}")
                    for sess in SESSIONS:
                        s = res[side].get(sess, {})
                        n = s.get("n_through", 0)
                        if n == 0:
                            continue
                        lines.append(
                            f"- {sess}: N={n} | 15min={s.get('return_pct_15', 0)}% "
                            f"30min={s.get('return_pct_30', 0)}% "
                            f"60min={s.get('return_pct_60', 0)}% "
                            f"120min={s.get('return_pct_120', 0)}%"
                        )
                    lines.append("")
            (research_dir / f"{fname}.md").write_text("\n".join(lines))
            console.print(f"[dim]Gespeichert: {research_dir}[/dim]")


    @app.command(name="fvg-overlap")
    def fvg_overlap(
        algo_name: str = typer.Argument(
            ..., help="Algo mit Zone-Spalten (z.B. 'FVG Standard')"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
    ) -> None:
        """Overlap-Analyse: Bounce-Rate wenn FVG-Zone mit anderer aktiver Zone überlappt (Bull×Bear)."""
        import pandas as _pd

        from sb.inspect import (
            analyze_zone_overlap_outcomes,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]FVG-Overlap-Analyse:[/bold cyan] [white]{algo_file.name}[/white]"
        )
        console.print("[dim]Lade Daten und führe Algo aus...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)

        try:
            df_result = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen des Algos: {exc}[/red]")
            raise typer.Exit(1)

        zone_prefixes = detect_zone_prefixes(df_result)
        if not zone_prefixes:
            console.print(
                "[red]Keine Zone-Spalten erkannt. Algo gibt keine Zonen-Spalten aus.[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[dim]Erkannte Zonen-Prefixe: {', '.join(zone_prefixes)}[/dim]\n")

        table = Table(
            title=f"FVG-Overlap-Analyse: {algo_file.name}",
            show_lines=True,
        )
        table.add_column("Zone", style="cyan")
        table.add_column("Richtung", style="white")
        table.add_column("Single\nN / Return%", justify="right")
        table.add_column("Double (Bull×Bear)\nN / Return%", justify="right")
        table.add_column("Double%\n(Anteil)", justify="right")
        table.add_column("Delta Return", justify="right")

        for prefix in zone_prefixes:
            try:
                res = analyze_zone_overlap_outcomes(df_result, prefix)
            except ValueError as exc:
                console.print(f"[yellow]{prefix} übersprungen: {exc}[/yellow]")
                continue

            for direction, label in (("bull", "Bull"), ("bear", "Bear")):
                d = res[direction]
                single = d["single"]
                double = d["double"]

                total = single["touches"] + double["touches"]
                double_pct = round(double["touches"] / total * 100, 1) if total else 0.0
                delta = round(double["bounce_pct"] - single["bounce_pct"], 1)

                single_str = f"{single['touches']:,} / {single['bounce_pct']}%"
                double_str = f"{double['touches']:,} / {double['bounce_pct']}%"
                double_pct_str = f"{double_pct}%"

                delta_color = "green" if delta >= 0 else "red"
                delta_str = f"[{delta_color}]{delta:+.1f}%[/{delta_color}]"

                table.add_row(
                    prefix, label, single_str, double_str, double_pct_str, delta_str
                )

        console.print(table)

        n_bars = len(df_result)
        t_start = str(df_result.index[0])[:10]
        t_end = str(df_result.index[-1])[:10]
        console.print(
            f"\n[dim]Daten: {n_bars:,} Bars | {t_start} → {t_end} | Algo: {algo_file.name}[/dim]"
        )
        console.print(
            "[dim]Double = Bull-Zone überlappt mit aktiver Bear-Zone (oder umgekehrt) bei Touch-Bar[/dim]"
        )


    @app.command(name="fvg-nest")
    def fvg_nest(
        algo_name: str = typer.Argument(
            ..., help="Algo mit Zone-Spalten (z.B. 'FVG Standard')"
        ),
        htf_data: str = typer.Option("", "--htf", help="HTF Datenpfad (z.B. 5min Parquet)"),
        ltf_data: str = typer.Option("", "--ltf", help="LTF Datenpfad (z.B. 1min Parquet)"),
        nesting: str = typer.Option(
            "contained", "--nesting", "-n", help="'contained' oder 'overlap'"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
    ) -> None:
        """Multi-TF Nesting: Bounce-Rate wenn HTF-Zone eine aktive LTF-Zone enthält."""
        import pandas as _pd

        from sb.inspect import (
            analyze_zone_mtf_nesting,
            detect_zone_prefixes,
            find_algo_file,
            run_algo,
        )

        try:
            _, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        if not htf_data or not ltf_data:
            console.print(
                "[red]Fehler: --htf und --ltf Datenpfade sind erforderlich.[/red]"
            )
            console.print(
                "[dim]Beispiel: sb.py fvg-nest 'FVG Standard' --htf nq_5m.parquet --ltf nq_1m.parquet[/dim]"
            )
            raise typer.Exit(1)

        htf_path = Path(htf_data).expanduser().resolve()
        ltf_path = Path(ltf_data).expanduser().resolve()
        for p in (htf_path, ltf_path):
            if not p.exists():
                console.print(f"[red]Datei nicht gefunden: {p}[/red]")
                raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]FVG Multi-TF Nesting:[/bold cyan] [white]{algo_file.name}[/white]"
            f"  [dim]({htf_path.name} → {ltf_path.name}, nesting={nesting})[/dim]"
        )
        console.print("[dim]Lade Daten und führe Algos aus...[/dim]")

        def _load_and_run(path: Path) -> "_pd.DataFrame":
            df = _pd.read_parquet(path)
            rename_map = {
                c: c.title()
                for c in df.columns
                if c.lower() in {"open", "high", "low", "close", "volume"}
            }
            df = df.rename(columns=rename_map)
            return run_algo(algo_file, df)

        try:
            df_htf = _load_and_run(htf_path)
            df_ltf = _load_and_run(ltf_path)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen des Algos: {exc}[/red]")
            raise typer.Exit(1)

        zone_prefixes = detect_zone_prefixes(df_htf)
        if not zone_prefixes:
            console.print("[red]Keine Zone-Spalten erkannt.[/red]")
            raise typer.Exit(1)

        console.print(f"[dim]Erkannte Zonen-Prefixe: {', '.join(zone_prefixes)}[/dim]\n")

        table = Table(
            title=f"MTF Nesting: {algo_file.name}  ({htf_path.name} → {ltf_path.name})",
            show_lines=True,
        )
        table.add_column("Zone", style="cyan")
        table.add_column("Richtung", style="white")
        table.add_column("Ohne LTF-Zone\nN / Return%", justify="right")
        table.add_column("Mit LTF-Zone (nested)\nN / Return%", justify="right")
        table.add_column("Nested%\n(Anteil)", justify="right")
        table.add_column("Delta Return", justify="right")

        for prefix in zone_prefixes:
            try:
                res = analyze_zone_mtf_nesting(df_htf, df_ltf, prefix, nesting=nesting)
            except ValueError as exc:
                console.print(f"[yellow]{prefix} übersprungen: {exc}[/yellow]")
                continue

            for direction, label in (("bull", "Bull"), ("bear", "Bear")):
                d = res[direction]
                single = d["single"]
                nested = d["nested"]

                total = single["touches"] + nested["touches"]
                nested_pct = round(nested["touches"] / total * 100, 1) if total else 0.0
                delta = round(nested["bounce_pct"] - single["bounce_pct"], 1)

                single_str = f"{single['touches']:,} / {single['bounce_pct']}%"
                nested_str = f"{nested['touches']:,} / {nested['bounce_pct']}%"
                nested_pct_str = f"{nested_pct}%"

                delta_color = "green" if delta >= 0 else "red"
                delta_str = f"[{delta_color}]{delta:+.1f}%[/{delta_color}]"

                table.add_row(
                    prefix, label, single_str, nested_str, nested_pct_str, delta_str
                )

        console.print(table)

        console.print(
            f"\n[dim]HTF: {len(df_htf):,} Bars | LTF: {len(df_ltf):,} Bars | "
            f"nesting={nesting} | Algo: {algo_file.name}[/dim]"
        )
        console.print(
            "[dim]Nested = LTF-Zone gleicher Richtung aktiv und innerhalb der HTF-Zone beim Touch[/dim]"
        )


    @app.command(name="lock")
    def lock_algo(
        algo_name: str = typer.Argument(..., help="Name des Algo (Teilstring reicht)"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
    ) -> None:
        """Algo-Datei auf read-only setzen (chmod 444) – schützt verifizierte Bausteine.

        Verhindert versehentliche Änderungen. Zum Entsperren: sb.py unlock <name>
        """
        import os

        from sb.inspect import find_algo_file

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
        """Algo-Datei entsperren (chmod 644) – für temporäre Bearbeitung.

        ACHTUNG: Danach sofort wieder sperren mit sb.py lock <name>!
        """
        import os

        from sb.inspect import find_algo_file

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
                is_locked = not bool(mode & 0o200)  # owner write-bit fehlt → locked
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


    @app.command(name="entry-stats")
    def entry_stats_cmd(
        algo_name: str = typer.Argument(
            ..., help="Entry-Algo Name (z.B. 'entry_first_touch')"
        ),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusätzliches Verzeichnis"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        forward: int = typer.Option(
            200, "--forward", "-f", help="Bars voraus für MAE/MFE (default 200)"
        ),
        tp: str = typer.Option(
            "5,10,20,50",
            "--tp",
            help="TP-Levels in Punkten, kommagetrennt (default '5,10,20,50')",
        ),
        session: str = typer.Option(
            "", "--session", help="Session-Filter: 'ny', 'london', 'asia' (optional)"
        ),
    ) -> None:
        """MAE/MFE-Analyse für Entry-Algo-Signale → SL- und TP-Sizing.

        Für jeden Entry-Signal-Kanal wird gemessen:
          MAE (Max Adverse Excursion) → SL-Empfehlung
          MFE (Max Favorable Excursion) → TP-Potential
          Win-Rate bei 1:1 RR für verschiedene TP-Levels

        Beispiel:
          ./sb.py entry-stats entry_first_touch
          ./sb.py entry-stats entry_second_touch_50 --tp 5,10,20,50 --session ny
        """
        import pandas as _pd
        from rich.panel import Panel

        from sb.inspect import find_algo_file, run_algo
        from sb.research.entry_stats import analyze_entry_stats

        # TP-Levels parsen
        try:
            tp_levels = [float(x.strip()) for x in tp.split(",")]
        except ValueError:
            console.print("[red]Ungültige TP-Levels. Beispiel: --tp 5,10,20,50[/red]")
            raise typer.Exit(1)

        # Daten laden
        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Entry-Stats: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print("[dim]Lade Daten und führe Algo aus...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        if rename_map:
            df = df.rename(columns=rename_map)

        # Session-Filter
        if session:
            try:
                idx_et = _pd.DatetimeIndex(df.index).tz_convert("America/New_York")
            except Exception:
                idx_et = _pd.DatetimeIndex(df.index)
            hours = idx_et.hour
            if session.lower() == "ny":
                mask = (hours >= 9) & (hours < 16)
            elif session.lower() in ("london", "ldn"):
                mask = (hours >= 2) & (hours < 8)
            elif session.lower() == "asia":
                mask = (hours >= 18) | (hours < 1)
            else:
                mask = _pd.Series([True] * len(df)).values
            df = df[mask]
            console.print(f"[dim]Session-Filter: {session.upper()} → {len(df)} Bars[/dim]")

        result_df = run_algo(algo_file, df)
        if result_df is None:
            console.print("[red]Algo konnte nicht ausgeführt werden.[/red]")
            raise typer.Exit(1)

        console.print(
            f"[dim]Analysiere {len(tp_levels)} TP-Levels, {forward} Bars voraus...[/dim]\n"
        )

        merged = df.join(result_df)
        results = analyze_entry_stats(merged, forward_bars=forward, tp_levels=tp_levels)

        if not results:
            console.print("[yellow]Keine Entry-Signal-Spalten gefunden.[/yellow]")
            raise typer.Exit(1)

        # Ausgabe
        for r in results:
            direction_icon = "🟢 LONG" if r.direction == "bull" else "🔴 SHORT"
            sl_empfehlung = r.mae_p80

            lines = []
            lines.append(
                f"[bold]{r.col}[/bold]  {direction_icon}  |  "
                f"[cyan]{r.n_signals} Signale[/cyan]  Ø {r.signals_per_day:.1f}/Tag"
            )
            lines.append("")
            lines.append("[yellow]MAE – Max Adverse Excursion (SL-Sizing):[/yellow]")
            lines.append(
                f"  50%: [white]{r.mae_p50:.1f} Pkt[/white]  |  "
                f"80%: [bold yellow]{r.mae_p80:.1f} Pkt[/bold yellow]  ← SL-Empfehlung  |  "
                f"90%: {r.mae_p90:.1f} Pkt  |  95%: {r.mae_p95:.1f} Pkt"
            )
            lines.append("")
            lines.append("[green]MFE – Max Favorable Excursion (TP-Potential):[/green]")
            lines.append(
                f"  50%: [white]{r.mfe_p50:.1f} Pkt[/white]  |  "
                f"75%: [bold green]{r.mfe_p75:.1f} Pkt[/bold green]  |  "
                f"90%: {r.mfe_p90:.1f} Pkt"
            )
            lines.append("")
            win_parts = []
            for tp_level, wr in sorted(r.win_rates.items()):
                color = "green" if wr >= 0.5 else ("yellow" if wr >= 0.4 else "red")
                win_parts.append(f"TP{int(tp_level)}: [{color}]{wr:.0%}[/{color}]")
            lines.append("[bold]Win-Rate (1:1 RR):[/bold]  " + "  |  ".join(win_parts))
            lines.append("")
            lines.append(
                f"[dim]SL-Empfehlung (MAE 80%): {sl_empfehlung:.1f} Pkt  |  "
                f"RR bei MFE75/SL80: {r.mfe_p75 / sl_empfehlung:.1f}:1[/dim]"
            )

            console.print(
                Panel(
                    "\n".join(lines),
                    border_style="cyan",
                    padding=(0, 1),
                )
            )


    @app.command(name="zeit-performance")
    def zeit_performance(
        algo_name_fragment_parts: list[str] = typer.Argument(
            ..., help="Namensfragment des Zeit-Algos"
        ),
        db: str = typer.Option("output_v3/builder.db", "--db", help="Pfad zu builder.db"),
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
            console.print(f"[red]Fehler beim Lesen der Performance-Daten: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(
            f"\n[bold cyan]Zeit-Performance: [white]{algo_name_fragment}[/white][/bold cyan]"
        )
        console.print(f"[dim]DB: {db_path}[/dim]\n")

        metric_labels = [
            ("n", "N"),
            ("avg_oos_pf", "Ø OOS PF"),
            ("avg_holdout_pf", "Ø Holdout PF"),
            ("pct_tier_a", "% Tier A"),
            ("pct_tier_b", "% Tier B"),
            ("pct_robust", "% Robust"),
        ]
        table = Table(title="MIT vs OHNE", show_lines=True)
        table.add_column("Metrik", style="cyan")
        table.add_column("MIT", justify="right", style="green")
        table.add_column("OHNE", justify="right", style="yellow")

        for key, label in metric_labels:
            mit_value = result["mit"][key]
            ohne_value = result["ohne"][key]
            if key == "n":
                mit_text = str(mit_value)
                ohne_text = str(ohne_value)
            elif key.startswith("pct_"):
                mit_text = f"{mit_value:.1f}%"
                ohne_text = f"{ohne_value:.1f}%"
            else:
                mit_text = f"{mit_value:.3f}"
                ohne_text = f"{ohne_value:.3f}"
            table.add_row(label, mit_text, ohne_text)

        console.print(table)
        if result["mit"]["n"] < 10 or result["ohne"]["n"] < 10:
            console.print("[yellow]Stichprobe klein, Ergebnis orientierend[/yellow]")

        if save:
            date_str = datetime.now().strftime("%Y-%m-%d")
            research_dir = Path("_research") / algo_name_fragment
            research_dir.mkdir(parents=True, exist_ok=True)
            md_path = research_dir / f"performance_{date_str}.md"
            md_lines = [
                f"# Zeit-Performance: {algo_name_fragment}",
                "",
                f"**Generiert:** {date_str}  ",
                f"**DB:** {db_path}  ",
                "",
                "| Metrik | MIT | OHNE |",
                "|---|---:|---:|",
            ]
            for key, label in metric_labels:
                mit_value = result["mit"][key]
                ohne_value = result["ohne"][key]
                if key == "n":
                    mit_text = str(mit_value)
                    ohne_text = str(ohne_value)
                elif key.startswith("pct_"):
                    mit_text = f"{mit_value:.1f}%"
                    ohne_text = f"{ohne_value:.1f}%"
                else:
                    mit_text = f"{mit_value:.3f}"
                    ohne_text = f"{ohne_value:.3f}"
                md_lines.append(f"| {label} | {mit_text} | {ohne_text} |")
            if result["mit"]["n"] < 10 or result["ohne"]["n"] < 10:
                md_lines.extend(["", "_Stichprobe klein, Ergebnis orientierend_"])
            md_path.write_text("\n".join(md_lines), encoding="utf-8")
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="zeit-phasen")
    def zeit_phasen(
        algo_name_parts: list[str] = typer.Argument(..., help="Name des Zeit-Algos"),
        window: int = typer.Option(30, "--window", help="Bars vor/nach dem Signal"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        save: bool = typer.Option(
            False, "--save", help="Ergebnisse in _research/ speichern"
        ),
    ) -> None:
        import pandas as _pd

        from sb.inspect import find_algo_file, run_algo
        from sb.zeit_analysis import analyze_zeit_phasen

        algo_name = _require_non_empty_text(" ".join(algo_name_parts), "Algo-Name")
        if window <= 0:
            console.print("[red]Fehler: window muss größer als 0 sein[/red]")
            raise typer.Exit(1)

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc

        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        try:
            df = _pd.read_parquet(data_path)
        except Exception as exc:
            console.print(f"[red]Fehler beim Laden der Daten: {exc}[/red]")
            raise typer.Exit(1) from exc

        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)
        original_cols = set(df.columns)

        try:
            result_df = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen: {exc}[/red]")
            raise typer.Exit(1) from exc

        try:
            analysis_df = analyze_zeit_phasen(
                result_df,
                algo_file.stem,
                window=window,
                original_cols=original_cols,
            )
        except ValueError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc

        console.print(
            f"\n[bold cyan]Zeit-Phasen: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(f"[dim]Window: {window} Bars | Daten: {data_path}[/dim]\n")

        table = Table(title="Phase × Metrik", show_lines=True)
        table.add_column("Phase", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Ø Range", justify="right")
        table.add_column("Up %", justify="right", style="green")
        table.add_column("Down %", justify="right", style="red")
        for _, row in analysis_df.iterrows():
            table.add_row(
                str(row["phase"]),
                str(int(row["count"])),
                f"{row['avg_range']:.3f}",
                f"{row['up_pct']:.1f}%",
                f"{row['down_pct']:.1f}%",
            )
        console.print(table)
        console.print(
            f"\n[dim]Bars: {len(result_df):,} | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}[/dim]"
        )

        if save:
            date_str = datetime.now().strftime("%Y-%m-%d")
            research_dir = Path("_research") / algo_name
            research_dir.mkdir(parents=True, exist_ok=True)
            md_path = research_dir / f"phasen_{date_str}.md"
            md_lines = [
                f"# Zeit-Phasen: {algo_file.name}",
                "",
                f"**Generiert:** {date_str}  ",
                f"**Window:** {window} Bars  ",
                f"**Bars:** {len(result_df):,} | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}  ",
                "",
                "| Phase | Count | Avg Range | Up % | Down % |",
                "|---|---:|---:|---:|---:|",
            ]
            for _, row in analysis_df.iterrows():
                md_lines.append(
                    f"| {row['phase']} | {int(row['count'])} | {row['avg_range']:.3f} | {row['up_pct']:.1f}% | {row['down_pct']:.1f}% |"
                )
            md_path.write_text("\n".join(md_lines), encoding="utf-8")
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="zeit-fenster")
    def zeit_fenster(
        algo_name_parts: list[str] = typer.Argument(..., help="Name des Zeit-Algos"),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad"),
        sources: str = typer.Option("", "--sources", "-s", help="Pfad zu sources.yaml"),
        save: bool = typer.Option(
            False, "--save", help="Ergebnisse in _research/ speichern"
        ),
    ) -> None:
        import pandas as _pd

        from sb.inspect import find_algo_file, run_algo
        from sb.zeit_analysis import _detect_signal_columns, analyze_zeit_fenster

        algo_name = _require_non_empty_text(" ".join(algo_name_parts), "Algo-Name")

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1) from exc

        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden für '{algo_name}'[/red]")
            raise typer.Exit(1)

        try:
            df = _pd.read_parquet(data_path)
        except Exception as exc:
            console.print(f"[red]Fehler beim Laden der Daten: {exc}[/red]")
            raise typer.Exit(1) from exc

        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        df = df.rename(columns=rename_map)
        original_cols = set(df.columns)

        try:
            result_df = run_algo(algo_file, df)
        except Exception as exc:
            console.print(f"[red]Fehler beim Ausführen: {exc}[/red]")
            raise typer.Exit(1) from exc

        signal_cols = _detect_signal_columns(result_df, original_cols=original_cols)
        if not signal_cols:
            console.print("[red]Keine Bool-/0-1-Signalspalten erkannt.[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Zeit-Fenster: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print(
            f"[dim]Daten: {data_path} | Signal-Spalten: {', '.join(signal_cols)}[/dim]\n"
        )

        analyses: list[tuple[str, _pd.DataFrame]] = []
        for signal_col in signal_cols:
            try:
                analysis_df = analyze_zeit_fenster(result_df, signal_col)
            except ValueError as exc:
                console.print(f"[yellow]{signal_col} übersprungen: {exc}[/yellow]")
                continue
            analyses.append((signal_col, analysis_df))

            table = Table(title=f"Fenster-Analyse: {signal_col}", show_lines=True)
            table.add_column("Fenster", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Ø Range", justify="right")
            table.add_column("Up%", justify="right", style="green")
            table.add_column("Down%", justify="right", style="red")
            table.add_column("Ø Net Move", justify="right", style="yellow")
            for _, row in analysis_df.iterrows():
                table.add_row(
                    str(row["hour_label"]),
                    str(int(row["count"])),
                    f"{row['avg_range']:.3f}",
                    f"{row['up_pct']:.1f}%",
                    f"{row['down_pct']:.1f}%",
                    f"{row['avg_net_move']:.3f}",
                )
            console.print(table)

        if not analyses:
            console.print("[red]Keine validen Zeitfenster-Analysen erzeugt.[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[dim]Bars: {len(result_df):,} | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}[/dim]"
        )

        if save:
            date_str = datetime.now().strftime("%Y-%m-%d")
            research_dir = algo_file.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            md_path = research_dir / f"fenster_{date_str}.md"
            md_lines = [
                f"# Zeit-Fenster: {algo_file.name}",
                "",
                f"**Generiert:** {date_str}  ",
                f"**Bars:** {len(result_df):,} | {str(result_df.index[0])[:10]} → {str(result_df.index[-1])[:10]}  ",
                f"**Signal-Spalten:** {', '.join(signal_cols)}  ",
            ]
            for signal_col, analysis_df in analyses:
                md_lines.extend(
                    [
                        "",
                        f"## {signal_col}",
                        "",
                        "| Fenster | Count | Ø Range | Up% | Down% | Ø Net Move |",
                        "|---|---:|---:|---:|---:|---:|",
                    ]
                )
                for _, row in analysis_df.iterrows():
                    md_lines.append(
                        f"| {row['hour_label']} | {int(row['count'])} | {row['avg_range']:.3f} | {row['up_pct']:.1f}% | {row['down_pct']:.1f}% | {row['avg_net_move']:.3f} |"
                    )
            md_path.write_text("\n".join(md_lines), encoding="utf-8")
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="atr-stats")
    def atr_stats(
        algo_name: str = typer.Argument(
            ..., help="Algo-Name (Teilstring), z.B. 'ATR Standard'"
        ),
        data: str = typer.Option("", "--data", help="Alternativer Datenpfad (.parquet)"),
        sources: str = typer.Option(
            "", "--sources", "-s", help="Alternativer Pfad zu sources.yaml"
        ),
        extra_dir: str = typer.Option("", "--dir", "-d", help="Zusaetzliches Verzeichnis"),
        save: bool = typer.Option(
            True, "--save/--no-save", help="Ergebnisse als Markdown speichern"
        ),
    ) -> None:
        """Descriptive Stats + Stunden-Profil fuer ATR/NATR Float-Algos (05_Stoploss_TakeProfit)."""
        import pandas as _pd

        from sb.atr_analysis import analyze_atr_stats
        from sb.inspect import find_algo_file, run_algo

        try:
            data_path, cfg, _ = _resolve_backtest_data_path(sources)
        except RuntimeError as exc:
            console.print(f"[red]Fehler: {exc}[/red]")
            raise typer.Exit(1)
        if data:
            data_path = Path(data).expanduser().resolve()
            if not data_path.exists():
                console.print(f"[red]Datei nicht gefunden: {data_path}[/red]")
                raise typer.Exit(1)

        sources_path = Path(sources).expanduser() if sources else None
        algo_dirs = resolve_pda_library_dirs(cfg, (sources_path or _DEFAULT_SOURCES).parent)
        if not algo_dirs:
            algo_dirs = list(DEFAULT_SIGNAL_ALGO_DIRS)
        if extra_dir:
            algo_dirs.insert(0, Path(extra_dir).expanduser().resolve())

        algo_file = find_algo_file(algo_name, algo_dirs)
        if algo_file is None:
            console.print(f"[red]Kein Algo gefunden fuer '{algo_name}'[/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]ATR-Stats: [white]{algo_file.name}[/white][/bold cyan]"
        )
        console.print("[dim]Lade Daten und fuehre Algo aus...[/dim]")

        df = _pd.read_parquet(data_path)
        rename_map = {
            c: c.title()
            for c in df.columns
            if c.lower() in {"open", "high", "low", "close", "volume"}
        }
        if rename_map:
            df = df.rename(columns=rename_map)

        original_cols = set(df.columns)
        result_df = run_algo(algo_file, df)
        new_cols = [c for c in result_df.columns if c not in original_cols]

        try:
            result = analyze_atr_stats(
                result_df, algo_name=algo_file.stem, cols=new_cols if new_cols else None
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

        cols = result["columns"]

        # ── Descriptive Stats ───────────────────────────────────────────────────────
        stats_df = result["stats"]
        t_stats = Table(
            title=f"Descriptive Stats – {algo_file.name}",
            show_header=True,
            header_style="bold cyan",
        )
        t_stats.add_column("Spalte", style="bold")
        t_stats.add_column("N", justify="right")
        t_stats.add_column("Mean", justify="right")
        t_stats.add_column("Std", justify="right")
        t_stats.add_column("p25", justify="right")
        t_stats.add_column("p50 (Median)", justify="right")
        t_stats.add_column("p75", justify="right")
        t_stats.add_column("p90", justify="right")
        t_stats.add_column("p95", justify="right")
        for col in stats_df.index:
            row = stats_df.loc[col]
            t_stats.add_row(
                col,
                f"{int(row['N']):,}",
                f"{row['Mean']:.3f}",
                f"{row['Std']:.3f}",
                f"{row['p25']:.3f}",
                f"{row['p50']:.3f}",
                f"{row['p75']:.3f}",
                f"{row['p90']:.3f}",
                f"{row['p95']:.3f}",
            )
        console.print(t_stats)

        # ── Stunden-Profil ──────────────────────────────────────────────────────────
        hourly_df = result["hourly"]
        if not hourly_df.empty:
            t_hourly = Table(
                title="Intraday-Profil (NY-Zeit, 30-min Fenster)",
                show_header=True,
                header_style="bold cyan",
            )
            t_hourly.add_column("Zeit (NY)", style="dim")
            for col in cols:
                if col in hourly_df.columns:
                    t_hourly.add_column(col, justify="right")
            for lbl, row in hourly_df.iterrows():
                t_hourly.add_row(
                    str(lbl),
                    *[
                        f"{row[col]:.3f}" if col in row.index else "-"
                        for col in cols
                        if col in hourly_df.columns
                    ],
                )
            console.print(t_hourly)

        console.print(
            f"\n[dim]Bars: {result['n_bars']:,} | {result['date_from']} → {result['date_to']}[/dim]"
        )

        if save:
            from datetime import datetime as _dt

            date_str = _dt.now().strftime("%Y-%m-%d")
            research_dir = algo_file.parent / "_research" / algo_file.stem
            research_dir.mkdir(parents=True, exist_ok=True)
            md_path = research_dir / f"atr_stats_{date_str}.md"
            lines = [
                f"# ATR-Stats: {algo_file.name}",
                "",
                f"**Generiert:** {date_str}  ",
                f"**Bars:** {result['n_bars']:,} | {result['date_from']} → {result['date_to']}  ",
                "",
                "## Descriptive Stats",
                "",
                "| Spalte | N | Mean | Std | p25 | p50 | p75 | p90 | p95 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for col in stats_df.index:
                row = stats_df.loc[col]
                lines.append(
                    f"| {col} | {int(row['N']):,} | {row['Mean']:.3f} | {row['Std']:.3f}"
                    f" | {row['p25']:.3f} | {row['p50']:.3f} | {row['p75']:.3f}"
                    f" | {row['p90']:.3f} | {row['p95']:.3f} |"
                )
            if not hourly_df.empty:
                hourly_cols = [c for c in cols if c in hourly_df.columns]
                lines += [
                    "",
                    "## Intraday-Profil (NY-Zeit, 30-min Fenster)",
                    "",
                    "| Zeit (NY) | " + " | ".join(hourly_cols) + " |",
                    "|---|" + "|".join(["---:"] * len(hourly_cols)) + "|",
                ]
                for lbl, row in hourly_df.iterrows():
                    vals = " | ".join(
                        f"{row[c]:.3f}" if c in row.index else "-" for c in hourly_cols
                    )
                    lines.append(f"| {lbl} | {vals} |")
            md_path.write_text("\n".join(lines), encoding="utf-8")
            console.print(f"[green]Gespeichert: {md_path}[/green]")


    @app.command(name="regime-test")
    def regime_test_cmd(
        algo: str = typer.Option("petar", "--algo", "-a", help="Algo: petar oder internet"),
        sample: float = typer.Option(0.1, "--sample", "-s", help="Sampling-Rate (0.1=10%)"),
        tf: str = typer.Option("15m", "--tf", "-t", help="Timeframe: 1m oder 15m"),
    ) -> None:
        """Forward-Looking Regime-Validierung: stimmt LRL/HRL mit Zukunft ueberein?"""
        from sb.regime_test import validate_regime

        validate_regime(algo_name=algo, sample_rate=sample, timeframe=tf)


    def main() -> None:
        app()
