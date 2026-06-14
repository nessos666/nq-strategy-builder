from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import typer
import yaml
from yaml import YAMLError
from rich.console import Console
from rich.table import Table

from sb.algo_paths import DEFAULT_SIGNAL_ALGO_DIRS
from sb.cache.signal_cache import SignalCache, SignalCacheConfig
from sb.engine.knowledge import load_knowledge, resolve_pda_library_dirs
from sb.engine.nautilus_bridge import NautilusBridge
from sb.engine.parser import parse_idea
from sb.engine.walk_forward import WalkForwardEngine
from sb.memory.db import BuilderDB
from sb.report import generate_wf_report

app = typer.Typer(help="Strategie Builder – standalone, kein LLM, kein Claude")
console = Console()

__all__ = ["app", "main", "_cleanup_old_studies", "_rotate_reports", "_backup_db"]

_DEFAULT_SOURCES = Path(__file__).parent.parent / "knowledge_sources" / "sources.yaml"


def _backup_db(db_path: Path, max_backups: int = 7) -> Path | None:
    """Sichert builder.db in output/backups/ mit Timestamp.

    Behaelt die letzten *max_backups* Sicherungen und loescht aeltere.
    Gibt den Backup-Pfad zurueck, oder None wenn die DB nicht existiert.
    """
    if not db_path.exists():
        return None
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    backup_path = backup_dir / f"builder_{ts}.db"
    shutil.copy2(db_path, backup_path)
    # Aeltere Backups loeschen – behalte nur die letzten max_backups
    old_backups = sorted(backup_dir.glob("builder_*.db"))[:-max_backups]
    for old in old_backups:
        old.unlink(missing_ok=True)
    return backup_path


def _resolve_cfg_path(path_str: str, base_dir: Path) -> Path:
    if not path_str:
        return Path()
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _require_non_empty_text(value: str, field_name: str) -> str:
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} darf nicht leer sein.")
    return normalized_value


def _require_positive_int(value: int, field_name: str) -> int:
    try:
        normalized_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} muss eine ganze Zahl sein.") from exc
    if normalized_value <= 0:
        raise ValueError(f"{field_name} muss größer als 0 sein.")
    return normalized_value


def _require_non_negative_int(value: int, field_name: str) -> int:
    try:
        normalized_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} muss eine ganze Zahl sein.") from exc
    if normalized_value < 0:
        raise ValueError(f"{field_name} darf nicht negativ sein.")
    return normalized_value


def _load_sources_cfg(sp: Path) -> dict:
    if not sp.exists():
        return {}
    try:
        with sp.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, YAMLError) as exc:
        raise RuntimeError(f"sources.yaml konnte nicht geladen werden: {sp}") from exc
    if not isinstance(cfg, dict):
        raise RuntimeError(f"sources.yaml hat ein ungültiges Format: {sp}")
    return cfg


def _resolve_backtest_data_path(sources: str) -> tuple[Path, dict, str | None]:
    sources_path = Path(sources).expanduser() if sources else None
    sp = sources_path or _DEFAULT_SOURCES
    cfg = _load_sources_cfg(sp)
    data_path_str = cfg.get("backtest_data", {}).get("path", "")
    data_path = _resolve_cfg_path(data_path_str, sp.parent)
    if not data_path_str:
        raise RuntimeError(
            "sources.yaml enthält keinen gültigen 'backtest_data.path'-Eintrag."
        )
    if not data_path.exists() or not data_path.is_file():
        raise RuntimeError(f"Backtest-Daten nicht gefunden: {data_path}")
    holdout_start: str | None = (
        cfg.get("backtest_data", {}).get("holdout_start") or None
    )
    return data_path, cfg, holdout_start


def _cleanup_old_studies(studies_path: Path, max_age_days: int = 30) -> int:
    """Loescht Optuna-Studies deren aeltester Trial aelter als max_age_days ist.

    In Optuna 4.x hat die studies-Tabelle keine datetime-Spalte. Das Alter wird
    ueber MIN(trials.datetime_start) pro study_id bestimmt. Studies ohne Trials
    werden nicht geloescht (sie haben kein messbares Alter).

    Gibt die Anzahl geloeschter Studies zurueck.
    """
    if not studies_path.exists():
        return 0
    import sqlite3 as _sqlite3

    con = _sqlite3.connect(str(studies_path))
    try:
        rows = con.execute(
            "SELECT s.study_id, s.study_name "
            "FROM studies s "
            "JOIN trials t ON t.study_id = s.study_id "
            "GROUP BY s.study_id "
            "HAVING MIN(t.datetime_start) < datetime('now', ? || ' days')",
            (f"-{max_age_days}",),
        ).fetchall()
        for study_id, _ in rows:
            for table in (
                "trial_params",
                "trial_values",
                "trial_user_attributes",
                "trial_system_attributes",
                "trial_intermediate_values",
                "trial_heartbeats",
            ):
                try:
                    con.execute(
                        f"DELETE FROM {table} WHERE trial_id IN "
                        "(SELECT trial_id FROM trials WHERE study_id = ?)",
                        (study_id,),
                    )
                except _sqlite3.OperationalError:
                    pass  # Tabelle existiert nicht in dieser Optuna-Version
            con.execute("DELETE FROM trials WHERE study_id = ?", (study_id,))
            con.execute("DELETE FROM study_directions WHERE study_id = ?", (study_id,))
            con.execute(
                "DELETE FROM study_user_attributes WHERE study_id = ?", (study_id,)
            )
            con.execute(
                "DELETE FROM study_system_attributes WHERE study_id = ?", (study_id,)
            )
            con.execute("DELETE FROM studies WHERE study_id = ?", (study_id,))
        con.commit()
        con.execute("VACUUM")
        con.commit()
        return len(rows)
    finally:
        con.close()


def _rotate_reports(output_dir: Path, keep: int = 50) -> int:
    """Löscht älteste report_*.md Dateien, behält die neuesten `keep` Stück."""
    reports = sorted(output_dir.glob("report_*.md"))
    to_delete = reports[: max(0, len(reports) - keep)]
    for f in to_delete:
        f.unlink()
    return len(to_delete)



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




from sb.commands.core import register as _register_core
from sb.commands.analysis import register as _register_analysis
from sb.commands.maintenance import register as _register_maintenance
from sb.commands.zone_level import register as _register_zone_level
from sb.commands.zeit_atr import register as _register_zeit_atr

_register_core(app)
_register_analysis(app)
_register_maintenance(app)
_register_zone_level(app)
_register_zeit_atr(app)
