"""Shared helper functions extracted from cli.py to avoid circular imports."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml
from yaml import YAMLError

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

