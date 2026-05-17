from sb.memory.db import BuilderDB


def _insert_run_with_results(db: BuilderDB, idea: str, pf_values: list[float]) -> int:
    run_id = db.save_run(idea=idea, trials=len(pf_values))
    from sb.models import BacktestResult

    for i, pf in enumerate(pf_values):
        result = BacktestResult(
            params={},
            gross_profit=pf * 100,
            gross_loss=100,
            num_trades=10,
            num_wins=5,
        )
        db.save_result(run_id=run_id, result=result, score=pf, rank=i + 1, warnings=[])
    db.compute_and_save_tier(run_id)
    return run_id


def test_backfill_sets_session_from_idea_name(tmp_path):
    db = BuilderDB(db_path=tmp_path / "builder.db")
    run_id = _insert_run_with_results(db, "BOS + FVG London Open", [1.3, 1.2, 1.1])
    db.backfill_missing_metadata()
    row = db.conn.execute(
        "SELECT session FROM build_runs WHERE id = ?", (run_id,)
    ).fetchone()
    db.close()
    assert row["session"] == "london"


def test_backfill_sets_is_robust_from_window_pfs(tmp_path):
    db = BuilderDB(db_path=tmp_path / "builder.db")
    run_id = _insert_run_with_results(db, "SWEEP + OB NY", [1.5, 1.3, 1.1])
    db.backfill_missing_metadata()
    row = db.conn.execute(
        "SELECT is_robust FROM build_runs WHERE id = ?", (run_id,)
    ).fetchone()
    db.close()
    assert row["is_robust"] == 1


def test_backfill_sets_not_robust_when_window_below_one(tmp_path):
    db = BuilderDB(db_path=tmp_path / "builder.db")
    run_id = _insert_run_with_results(db, "Judas Swing Asia", [1.5, 0.8, 1.1])
    db.backfill_missing_metadata()
    row = db.conn.execute(
        "SELECT is_robust FROM build_runs WHERE id = ?", (run_id,)
    ).fetchone()
    db.close()
    assert row["is_robust"] == 0


def test_backfill_skips_runs_that_already_have_session(tmp_path):
    db = BuilderDB(db_path=tmp_path / "builder.db")
    run_id = db.save_run(idea="BOS + FVG NY", trials=3, session="ny", is_robust=True)
    db.backfill_missing_metadata()
    row = db.conn.execute(
        "SELECT session FROM build_runs WHERE id = ?", (run_id,)
    ).fetchone()
    db.close()
    assert row["session"] == "ny"  # unveraendert


def test_maintain_cleans_old_studies(tmp_path):
    """maintain soll alte Studies aus studies.db entfernen."""
    import optuna

    studies_path = tmp_path / "studies.db"
    storage = f"sqlite:///{studies_path}"

    # 3 Studies anlegen
    optuna.create_study(study_name="old_study_1", storage=storage)
    optuna.create_study(study_name="old_study_2", storage=storage)
    optuna.create_study(study_name="new_study", storage=storage)

    # old_study_1 und old_study_2 als alt markieren (> 30 Tage):
    # In Optuna 4.x liegt datetime_start in der trials-Tabelle, nicht in studies.
    # Wir fuegen je einen Trial ein und setzen dessen datetime_start zurueck.
    import sqlite3 as _sqlite3

    con = _sqlite3.connect(str(studies_path))
    for study_name in ("old_study_1", "old_study_2"):
        study_id = con.execute(
            "SELECT study_id FROM studies WHERE study_name = ?", (study_name,)
        ).fetchone()[0]
        con.execute(
            "INSERT INTO trials (study_id, state, datetime_start, datetime_complete) "
            "VALUES (?, 'COMPLETE', datetime('now', '-31 days'), datetime('now', '-31 days'))",
            (study_id,),
        )
    con.commit()
    con.close()

    from sb.cli import _cleanup_old_studies

    deleted = _cleanup_old_studies(studies_path, max_age_days=30)
    assert deleted == 2

    # new_study muss noch da sein
    remaining = optuna.get_all_study_names(storage=storage)
    assert "new_study" in remaining
    assert "old_study_1" not in remaining


def test_rotate_reports_keeps_newest(tmp_path):
    """Report-Rotation löscht alte Reports, behält die neuesten N."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # 60 Reports anlegen mit verschiedenen Zeitstempeln
    for i in range(60):
        (output_dir / f"report_2026-01-{i + 1:02d}_10-00.md").write_text(f"Report {i}")

    from sb.cli import _rotate_reports

    deleted = _rotate_reports(output_dir, keep=50)
    assert deleted == 10
    remaining = list(output_dir.glob("report_*.md"))
    assert len(remaining) == 50


def test_rotate_reports_noop_when_below_limit(tmp_path):
    """Keine Löschung wenn weniger als keep Reports vorhanden."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    for i in range(10):
        (output_dir / f"report_2026-01-{i + 1:02d}_10-00.md").write_text(f"Report {i}")

    from sb.cli import _rotate_reports

    deleted = _rotate_reports(output_dir, keep=50)
    assert deleted == 0
