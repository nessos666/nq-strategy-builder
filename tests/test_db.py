from __future__ import annotations

import importlib
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from sb.memory.db import BuilderDB
from sb.models import BacktestResult


def _insert_result_with_pf(
    db: BuilderDB, run_id: int, pf: float | None, num_trades: int = 20
) -> None:
    db.conn.execute(
        """
        INSERT INTO results (run_id, params, pf, winrate, num_trades, score, rank, warnings)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, "{}", pf, 0.0, num_trades, 0.0, 1, "[]"),
    )
    db.conn.commit()


def _load_cli_module():
    original_bridge_module = sys.modules.get("sb.engine.nautilus_bridge")
    original_walk_forward_module = sys.modules.get("sb.engine.walk_forward")
    stub = types.ModuleType("sb.engine.nautilus_bridge")
    cast(Any, stub).NautilusBridge = type("NautilusBridge", (), {})
    sys.modules["sb.engine.nautilus_bridge"] = stub
    sys.modules.pop("sb.engine.walk_forward", None)
    sys.modules.pop("sb.cli", None)
    try:
        return importlib.import_module("sb.cli")
    finally:
        sys.modules.pop("sb.cli", None)
        sys.modules.pop("sb.engine.walk_forward", None)
        if original_walk_forward_module is not None:
            sys.modules["sb.engine.walk_forward"] = original_walk_forward_module
        if original_bridge_module is not None:
            sys.modules["sb.engine.nautilus_bridge"] = original_bridge_module
        else:
            sys.modules.pop("sb.engine.nautilus_bridge", None)


def _load_cli_app():
    return _load_cli_module().app


def test_db_creates_tables(tmp_db):
    db = BuilderDB(tmp_db)
    db.close()
    assert tmp_db.exists()


def test_db_sets_sqlite_pragmas(tmp_db):
    db = BuilderDB(tmp_db)
    journal_mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    busy_timeout = db.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    foreign_keys = db.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    db.close()

    assert journal_mode.lower() == "wal"
    assert busy_timeout == 30000
    assert foreign_keys == 1


def test_db_init_raises_clean_error_when_directory_creation_fails(
    tmp_path, monkeypatch
):
    original_mkdir = Path.mkdir

    def broken_mkdir(self, *args, **kwargs):
        if self == tmp_path / "blocked":
            raise PermissionError("blocked")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", broken_mkdir)

    with pytest.raises(
        RuntimeError, match=r"Unable to create database directory: .*blocked"
    ):
        BuilderDB(tmp_path / "blocked" / "builder.db")


def test_save_and_get_run(tmp_db):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="BOS + FVG London", trials=50)
    assert run_id > 0
    db.close()


@pytest.mark.parametrize(
    ("idea", "trials", "expected_error"),
    [
        ("", 5, "idea must not be empty"),
        ("   ", 5, "idea must not be empty"),
        ("Idea", 0, "trials must be greater than 0"),
        ("Idea", -1, "trials must be greater than 0"),
        ("Idea", "abc", "Invalid trials value: abc"),
    ],
)
def test_save_run_validates_inputs(tmp_db, idea, trials, expected_error):
    db = BuilderDB(tmp_db)
    with pytest.raises(ValueError, match=expected_error):
        db.save_run(idea=idea, trials=trials)
    db.close()


def test_save_result(tmp_db):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="FVG London", trials=20)
    result = BacktestResult(
        params={"sl": 10, "tp_mult": 2.5},
        gross_profit=800.0,
        gross_loss=300.0,
        num_trades=40,
        num_wins=26,
    )
    db.save_result(run_id=run_id, result=result, score=3.8, rank=1, warnings=[])
    best = db.get_best_results(limit=5)
    assert len(best) == 1
    assert best[0]["score"] == 3.8
    db.close()


def test_save_result_raises_for_missing_run(tmp_db):
    db = BuilderDB(tmp_db)
    result = BacktestResult(
        params={"sl": 10}, gross_profit=100, gross_loss=50, num_trades=10, num_wins=5
    )

    with pytest.raises(ValueError, match="Run 999 does not exist"):
        db.save_result(run_id=999, result=result, score=2.0, rank=1, warnings=[])
    db.close()


def test_save_result_rejects_non_serializable_payloads(tmp_db):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="Serialize", trials=5)
    result = BacktestResult(
        params={"sl": {1, 2, 3}},
        gross_profit=100,
        gross_loss=50,
        num_trades=10,
        num_wins=6,
    )

    with pytest.raises(ValueError, match="Result payload is not JSON serializable"):
        db.save_result(run_id=run_id, result=result, score=2.0, rank=1, warnings=[])
    db.close()


def test_get_best_results_ordered(tmp_db):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="Test", trials=10)
    for score in [1.5, 4.2, 2.8]:
        db.save_result(
            run_id=run_id,
            result=BacktestResult({"sl": 10}, 100, 50, 10, 6),
            score=score,
            rank=1,
            warnings=[],
        )
    best = db.get_best_results(limit=3)
    assert best[0]["score"] == 4.2
    assert best[1]["score"] == 2.8
    db.close()


def test_get_best_results_rejects_invalid_limit(tmp_db):
    db = BuilderDB(tmp_db)
    with pytest.raises(ValueError, match="Invalid limit: bad"):
        db.get_best_results(limit="bad")
    db.close()


def test_find_runs_by_idea_returns_matching(tmp_db):
    db = BuilderDB(tmp_db)
    db.save_run(idea="BOS + FVG London", trials=50)
    db.save_run(idea="BOS + FVG London", trials=50)
    db.save_run(idea="OB NY", trials=30)
    runs = db.find_runs_by_idea("BOS + FVG London")
    assert len(runs) == 2
    assert all(r["idea"] == "BOS + FVG London" for r in runs)
    db.close()


def test_find_runs_by_idea_is_case_insensitive_and_trims_input(tmp_db):
    db = BuilderDB(tmp_db)
    db.save_run(idea=" BOS + FVG London ", trials=50)
    db.save_run(idea="bos + fvg london", trials=40)

    runs = db.find_runs_by_idea("  BOS + FVG LONDON  ")
    db.close()

    assert len(runs) == 2
    assert {run["trials"] for run in runs} == {50, 40}


def test_find_runs_by_idea_empty_when_unknown(tmp_db):
    db = BuilderDB(tmp_db)
    runs = db.find_runs_by_idea("unbekannte Idee")
    assert runs == []
    db.close()


def test_find_runs_by_idea_returns_empty_for_blank_input(tmp_db):
    db = BuilderDB(tmp_db)
    db.save_run(idea="Known idea", trials=5)

    assert db.find_runs_by_idea("   ") == []
    db.close()


def test_get_best_result_for_idea_only_uses_matching_runs(tmp_db):
    db = BuilderDB(tmp_db)
    target_run_id = db.save_run(idea="Target", trials=5)
    other_run_id = db.save_run(idea="Other", trials=5)
    target_result = BacktestResult(
        params={"sl": 10},
        gross_profit=200,
        gross_loss=100,
        num_trades=10,
        num_wins=6,
    )
    other_result = BacktestResult(
        params={"sl": 12},
        gross_profit=900,
        gross_loss=100,
        num_trades=10,
        num_wins=8,
    )
    db.save_result(
        run_id=target_run_id, result=target_result, score=2.0, rank=1, warnings=[]
    )
    db.save_result(
        run_id=other_run_id, result=other_result, score=9.0, rank=1, warnings=[]
    )

    best = db.get_best_result_for_idea(" target ")
    db.close()

    assert best is not None
    assert best["score"] == 2.0
    assert best["pf"] == 2.0


def test_execute_with_retry_retries_locked_operational_errors(tmp_db, monkeypatch):
    db = BuilderDB(tmp_db)
    attempts = {"count": 0}
    monkeypatch.setattr("sb.memory.db.time.sleep", lambda _: None)

    def flaky_operation():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    assert db._execute_with_retry(flaky_operation) == "ok"
    assert attempts["count"] == 2
    db.close()


def test_migrate_schema_is_idempotent_for_legacy_db(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE build_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea TEXT NOT NULL,
            trials INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES build_runs(id),
            params TEXT NOT NULL,
            pf REAL,
            winrate REAL,
            num_trades INTEGER,
            score REAL,
            rank INTEGER,
            warnings TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()

    db = BuilderDB(db_path)
    db._migrate_schema()
    cols = {row[1] for row in db.conn.execute("PRAGMA table_info(build_runs)")}
    indexes = {row[1] for row in db.conn.execute("PRAGMA index_list(build_runs)")}
    db.close()

    assert "avg_oos_pf" in cols
    assert "tier" in cols
    assert "idx_runs_tier" in indexes


@pytest.mark.parametrize(
    ("pfs", "expected_tier", "expected_avg"),
    [
        ([2.0], "A", 2.0),
        ([1.5], "B", 1.5),
        ([1.49], "C", 1.49),
        ([-0.25], "C", None),
    ],
)
def test_compute_and_save_tier_thresholds(tmp_db, pfs, expected_tier, expected_avg):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="Tier-Test", trials=5)
    for pf in pfs:
        _insert_result_with_pf(db, run_id, pf)

    tier = db.compute_and_save_tier(run_id)
    stored = db.find_runs_by_idea("Tier-Test")[0]
    db.close()

    assert tier == expected_tier
    assert stored["tier"] == expected_tier
    assert stored["avg_oos_pf"] == expected_avg


def test_compute_and_save_tier_ignores_none_and_nan(tmp_db):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="Ignore invalid PF", trials=5)
    _insert_result_with_pf(db, run_id, None)
    _insert_result_with_pf(db, run_id, float("nan"))
    _insert_result_with_pf(db, run_id, 2.0)

    tier = db.compute_and_save_tier(run_id)
    stored = db.find_runs_by_idea("Ignore invalid PF")[0]
    db.close()

    assert tier == "A"
    assert stored["tier"] == "A"
    assert stored["avg_oos_pf"] == 2.0


def test_compute_and_save_tier_ignores_zero_and_negative_pf(tmp_db):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="Zero-Trade-Test", trials=5)
    _insert_result_with_pf(db, run_id, 0.0)  # 0 Trades → pf=0.0
    _insert_result_with_pf(db, run_id, -1.0)  # negativer PF
    _insert_result_with_pf(db, run_id, 1.8)  # echter Wert

    tier = db.compute_and_save_tier(run_id)
    stored = db.find_runs_by_idea("Zero-Trade-Test")[0]
    db.close()

    assert tier == "B"
    assert stored["avg_oos_pf"] == 1.8


def test_compute_and_save_tier_sets_c_and_null_average_when_no_valid_results(tmp_db):
    db = BuilderDB(tmp_db)
    empty_run_id = db.save_run(idea="No results", trials=1)
    invalid_run_id = db.save_run(idea="Invalid results", trials=1)
    _insert_result_with_pf(db, invalid_run_id, None)
    _insert_result_with_pf(db, invalid_run_id, float("nan"))

    empty_tier = db.compute_and_save_tier(empty_run_id)
    invalid_tier = db.compute_and_save_tier(invalid_run_id)
    runs = {run["idea"]: run for run in db.get_registry()}
    db.close()

    assert empty_tier == "C"
    assert runs["No results"]["tier"] == "C"
    assert runs["No results"]["avg_oos_pf"] is None
    assert invalid_tier == "C"
    assert runs["Invalid results"]["tier"] == "C"
    assert runs["Invalid results"]["avg_oos_pf"] is None


def test_compute_and_save_tier_raises_for_missing_run(tmp_db):
    db = BuilderDB(tmp_db)
    with pytest.raises(ValueError, match="Run 999 does not exist"):
        db.compute_and_save_tier(999)
    db.close()


def test_pbo_high_blocks_tier_a(tmp_db):
    """PBO >= 0.5 → Tier A wird zu Tier B herabgestuft, auch wenn PF >= 2.0."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="PBO-Block-Test", trials=5)
    _insert_result_with_pf(db, run_id, 2.5)
    _insert_result_with_pf(db, run_id, 2.1)

    tier = db.compute_and_save_tier(run_id, pbo_score=0.7)
    stored = db.find_runs_by_idea("PBO-Block-Test")[0]
    db.close()

    assert tier == "B", "Hoher PBO soll Tier A auf B herabstufen"
    assert stored["tier"] == "B"


def test_pbo_low_allows_tier_a(tmp_db):
    """PBO < 0.5 → Tier A bleibt Tier A."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="PBO-Allow-Test", trials=5)
    _insert_result_with_pf(db, run_id, 2.5)
    _insert_result_with_pf(db, run_id, 2.1)

    tier = db.compute_and_save_tier(run_id, pbo_score=0.3)
    stored = db.find_runs_by_idea("PBO-Allow-Test")[0]
    db.close()

    assert tier == "A", "Niedriger PBO soll Tier A erlauben"
    assert stored["tier"] == "A"


def test_pbo_nan_does_not_block_tier_a(tmp_db):
    """PBO=NaN (unbekannt) soll Tier A nicht blockieren."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="PBO-NaN-Test", trials=5)
    _insert_result_with_pf(db, run_id, 2.5)

    tier = db.compute_and_save_tier(run_id, pbo_score=float("nan"))
    stored = db.find_runs_by_idea("PBO-NaN-Test")[0]
    db.close()

    assert tier == "A", "NaN PBO soll Tier A nicht blockieren"
    assert stored["tier"] == "A"
    assert stored.get("pbo_score") is None  # NaN wird als NULL gespeichert


@pytest.mark.parametrize("pbo_score", [-0.1, 1.1])
def test_invalid_pbo_is_treated_as_unknown_and_stored_as_null(tmp_db, pbo_score):
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea=f"PBO-Invalid-{pbo_score}", trials=5)
    _insert_result_with_pf(db, run_id, 2.5)

    tier = db.compute_and_save_tier(run_id, pbo_score=pbo_score)
    stored = db.find_runs_by_idea(f"PBO-Invalid-{pbo_score}")[0]
    db.close()

    assert tier == "A"
    assert stored["tier"] == "A"
    assert stored["pbo_score"] is None


def test_get_registry_handles_blank_tier_and_filters_all_tiers(tmp_db):
    db = BuilderDB(tmp_db)
    runs = {
        "Alpha": ("A", 2.0),
        "Beta": ("B", 1.5),
        "Gamma": ("C", 1.0),
    }
    for idea, (tier, avg_pf) in runs.items():
        run_id = db.save_run(idea=idea, trials=3)
        db.conn.execute(
            "UPDATE build_runs SET avg_oos_pf = ?, tier = ? WHERE id = ?",
            (avg_pf, tier, run_id),
        )
    db.conn.commit()

    all_default = db.get_registry()
    all_none = db.get_registry(tier=None)
    all_empty = db.get_registry(tier="")
    all_blank = db.get_registry(tier="   ")
    only_a = db.get_registry(tier="a")
    only_b = db.get_registry(tier="B")
    only_c = db.get_registry(tier="c")
    db.close()

    assert [row["idea"] for row in all_default] == ["Alpha", "Beta", "Gamma"]
    assert [row["idea"] for row in all_none] == ["Alpha", "Beta", "Gamma"]
    assert [row["idea"] for row in all_empty] == ["Alpha", "Beta", "Gamma"]
    assert [row["idea"] for row in all_blank] == ["Alpha", "Beta", "Gamma"]
    assert [row["idea"] for row in only_a] == ["Alpha"]
    assert [row["idea"] for row in only_b] == ["Beta"]
    assert [row["idea"] for row in only_c] == ["Gamma"]


def test_get_registry_rejects_invalid_tier(tmp_db):
    db = BuilderDB(tmp_db)
    with pytest.raises(ValueError, match="Invalid tier filter"):
        db.get_registry(tier="X")
    db.close()


def test_registry_cli_empty_registry(tmp_path):
    app = _load_cli_app()
    db = BuilderDB(tmp_path / "builder.db")
    db.close()

    runner = CliRunner()
    result = runner.invoke(app, ["registry", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert "Keine Einträge gefunden." in result.output


def test_registry_cli_filters_and_formats_output(tmp_path):
    app = _load_cli_app()
    db = BuilderDB(tmp_path / "builder.db")
    for idea, tier, avg_pf in [
        ("Alpha", "A", 2.0),
        ("Beta", "B", 1.5),
        ("Gamma", "C", 1.2),
    ]:
        run_id = db.save_run(idea=idea, trials=3)
        db.conn.execute(
            "UPDATE build_runs SET avg_oos_pf = ?, tier = ? WHERE id = ?",
            (avg_pf, tier, run_id),
        )
    db.conn.commit()
    db.close()

    runner = CliRunner()
    result = runner.invoke(app, ["registry", "--output", str(tmp_path), "--tier", "b"])

    assert result.exit_code == 0
    assert "Strategy Registry" in result.output
    assert "Tier B" in result.output
    assert "Beta" in result.output
    assert "1.500" in result.output
    assert "Alpha" not in result.output
    assert "Gamma" not in result.output


def test_registry_cli_recalc_updates_existing_runs(tmp_path):
    app = _load_cli_app()
    db = BuilderDB(tmp_path / "builder.db")
    run_id = db.save_run(idea="Needs recalc", trials=3)
    _insert_result_with_pf(db, run_id, 2.0)
    db.close()

    runner = CliRunner()
    result = runner.invoke(app, ["registry", "--output", str(tmp_path), "--recalc"])

    assert result.exit_code == 0
    assert "Tier für 1 Runs neu berechnet." in result.output
    assert "Needs" in result.output  # Rich kann lange Namen umbrechen
    assert "2.000" in result.output
    assert "A: 1" in result.output


def test_registry_cli_reports_invalid_tier(tmp_path):
    app = _load_cli_app()
    db = BuilderDB(tmp_path / "builder.db")
    db.close()

    runner = CliRunner()
    result = runner.invoke(app, ["registry", "--output", str(tmp_path), "--tier", "D"])

    assert result.exit_code == 1
    assert "FEHLER:" in result.output
    assert "Invalid tier filter: D" in result.output


def test_build_cli_rejects_blank_idea_without_traceback():
    app = _load_cli_app()
    runner = CliRunner()

    result = runner.invoke(app, ["build", "   "])

    assert result.exit_code == 1
    assert "Idee darf nicht leer sein." in result.output
    assert "Traceback" not in result.output


def test_build_cli_rejects_invalid_trials_without_traceback():
    app = _load_cli_app()
    runner = CliRunner()

    result = runner.invoke(app, ["build", "Idea", "--trials", "0"])

    assert result.exit_code == 1
    assert "Trials muss größer als 0 sein." in result.output
    assert "Traceback" not in result.output


def test_batch_cli_rejects_invalid_trials_without_traceback(tmp_path):
    app = _load_cli_app()
    ideas_file = tmp_path / "ideas.txt"
    ideas_file.write_text("Idea\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["batch", str(ideas_file), "--trials", "-1"])

    assert result.exit_code == 1
    assert "Trials muss größer als 0 sein." in result.output
    assert "Traceback" not in result.output


def test_build_cli_duplicate_summary_uses_matching_idea_best_result(
    tmp_path, monkeypatch
):
    cli_module = _load_cli_module()
    app = cli_module.app
    db = BuilderDB(tmp_path / "builder.db")
    duplicate_run_id = db.save_run(idea="Target Idea", trials=5)
    other_run_id = db.save_run(idea="Other Idea", trials=5)
    db.save_result(
        run_id=duplicate_run_id,
        result=BacktestResult(
            params={"sl": 10},
            gross_profit=200,
            gross_loss=100,
            num_trades=20,
            num_wins=12,
        ),
        score=2.0,
        rank=1,
        warnings=[],
    )
    db.save_result(
        run_id=other_run_id,
        result=BacktestResult(
            params={"sl": 12},
            gross_profit=900,
            gross_loss=100,
            num_trades=20,
            num_wins=16,
        ),
        score=9.0,
        rank=1,
        warnings=[],
    )
    db.close()

    monkeypatch.setattr(cli_module.typer, "confirm", lambda *args, **kwargs: False)
    runner = CliRunner()
    result = runner.invoke(app, ["build", " target idea ", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert "Diese Idee wurde bereits" in result.output
    assert "PF=2.000" in result.output
    assert "PF=9.000" not in result.output


def test_resolve_backtest_data_path_rejects_missing_config_entry(tmp_path):
    cli_module = _load_cli_module()
    sources = tmp_path / "sources.yaml"
    sources.write_text("backtest_data: {}\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=r"sources.yaml enthält keinen gültigen 'backtest_data.path'-Eintrag.",
    ):
        cli_module._resolve_backtest_data_path(str(sources))


def test_builder_db_accepts_string_path(tmp_path):
    """BuilderDB soll String-Pfad akzeptieren ohne Crash."""
    db = BuilderDB(db_path=str(tmp_path / "test.db"))
    db.close()
    assert (tmp_path / "test.db").exists()


def test_builder_db_default_path_is_output():
    """Default-Pfad soll output/builder.db sein, nicht memory/builder.db."""
    from sb.memory.db import _DEFAULT_DB

    assert "output" in str(_DEFAULT_DB), (
        f"Expected 'output' in default path, got: {_DEFAULT_DB}"
    )


def test_db_has_holdout_columns(tmp_db):
    """build_runs hat holdout_pf, holdout_trades, holdout_validated Spalten."""
    db = BuilderDB(tmp_db)
    cols = {row[1] for row in db.conn.execute("PRAGMA table_info(build_runs)")}
    db.close()
    assert "holdout_pf" in cols
    assert "holdout_trades" in cols
    assert "holdout_validated" in cols


def test_save_holdout_result_updates_run(tmp_db):
    """save_holdout_result() setzt holdout_pf, holdout_trades, holdout_validated=1."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run("BOS + FVG NY", trials=50)
    db.save_holdout_result(run_id, holdout_pf=1.85, holdout_trades=312)
    row = db.conn.execute(
        "SELECT holdout_pf, holdout_trades, holdout_validated FROM build_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    db.close()
    assert abs(row[0] - 1.85) < 0.001
    assert row[1] == 312
    assert row[2] == 1


def test_save_holdout_result_invalid_run_raises(tmp_db):
    """save_holdout_result() wirft ValueError für nicht-existente run_id."""
    db = BuilderDB(tmp_db)
    with pytest.raises((ValueError, RuntimeError)):
        db.save_holdout_result(9999, holdout_pf=1.5, holdout_trades=100)
    db.close()


def test_get_registry_includes_holdout_columns(tmp_db):
    """get_registry() liefert holdout_pf und holdout_validated."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run("BOS + FVG NY", trials=50)
    db.save_holdout_result(run_id, holdout_pf=1.72, holdout_trades=200)
    rows = db.get_registry()
    db.close()
    assert len(rows) == 1
    assert "holdout_pf" in rows[0]
    assert "holdout_validated" in rows[0]
    assert abs(rows[0]["holdout_pf"] - 1.72) < 0.001


def test_db_has_mc_pct_profitable_column(tmp_path):
    db = BuilderDB(tmp_path / "t.db")
    run_id = db.save_run("mc test", 10)
    db.compute_and_save_tier(run_id, mc_pct_profitable=0.85)
    row = db.conn.execute(
        "SELECT mc_pct_profitable FROM build_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 0.85) < 1e-6
    db.close()


def test_get_registry_returns_all_runs_including_tier_c(tmp_db):
    """get_registry() ohne Argumente muss ALLE Runs zurückgeben, auch Tier C."""
    db = BuilderDB(tmp_db)
    for i in range(201):
        run_id = db.save_run(idea=f"IDEA_{i}", trials=10)
        if i == 0:
            for _ in range(3):
                db.save_result(
                    run_id=run_id,
                    result=BacktestResult(
                        params={},
                        gross_profit=250,
                        gross_loss=100,
                        num_trades=20,
                        num_wins=13,
                    ),
                    score=2.5,
                    rank=1,
                    warnings=[],
                )
        elif i == 1:
            for _ in range(3):
                db.save_result(
                    run_id=run_id,
                    result=BacktestResult(
                        params={},
                        gross_profit=160,
                        gross_loss=100,
                        num_trades=10,
                        num_wins=5,
                    ),
                    score=1.6,
                    rank=1,
                    warnings=[],
                )
        else:
            for _ in range(3):
                db.save_result(
                    run_id=run_id,
                    result=BacktestResult(
                        params={},
                        gross_profit=0,
                        gross_loss=100,
                        num_trades=0,
                        num_wins=0,
                    ),
                    score=0.0,
                    rank=1,
                    warnings=[],
                )
        db.compute_and_save_tier(run_id)

    all_runs = db.get_registry()
    assert len(all_runs) == 201, f"Erwartet 201, bekommen {len(all_runs)}"
    tier_c_runs = [r for r in all_runs if r.get("tier") == "C"]
    assert len(tier_c_runs) == 199, f"Erwartet 199 Tier-C, bekommen {len(tier_c_runs)}"
    db.close()


def test_get_registry_tier_c_not_cut_by_limit(tmp_db):
    """Tier-C-Filter muss alle C-Runs zurückgeben, kein Limit."""
    db = BuilderDB(tmp_db)
    for i in range(10):
        run_id = db.save_run(idea=f"ZERO_{i}", trials=10)
        for _ in range(3):
            db.save_result(
                run_id=run_id,
                result=BacktestResult(
                    params={}, gross_profit=0, gross_loss=100, num_trades=0, num_wins=0
                ),
                score=0.0,
                rank=1,
                warnings=[],
            )
        db.compute_and_save_tier(run_id)

    c_runs = db.get_registry(tier="C")
    assert len(c_runs) == 10
    db.close()


def test_get_registry_counts_returns_all_tiers(tmp_db):
    """get_registry_counts() zählt A/B/C korrekt – unabhängig von Anzahl."""
    db = BuilderDB(tmp_db)
    for gross_p, gross_l, label, n in [
        (250, 100, "A", 3),
        (160, 100, "B", 5),
        (0, 100, "C", 10),
    ]:
        for i in range(n):
            run_id = db.save_run(idea=f"{label}_{i}", trials=10)
            for _ in range(3):
                db.save_result(
                    run_id=run_id,
                    result=BacktestResult(
                        params={},
                        gross_profit=gross_p,
                        gross_loss=gross_l,
                        num_trades=25 if label == "A" else 5,
                        num_wins=3,
                    ),
                    score=gross_p / gross_l if gross_l else 0.0,
                    rank=1,
                    warnings=[],
                )
            db.compute_and_save_tier(run_id)

    counts = db.get_registry_counts()
    assert counts == {"A": 3, "B": 5, "C": 10, "total": 18}
    db.close()


def test_tier_a_requires_min_20_trades(tmp_db):
    """Tier A wird verweigert wenn avg_trades < 20, auch bei hohem PF."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="LowFreq-Test", trials=5)
    # PF=2.5 aber nur 5 Trades pro Fenster → avg_trades=5 < 20
    db.conn.execute(
        "INSERT INTO results (run_id, params, pf, winrate, num_trades, score, rank, warnings) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "{}", 2.5, 0.0, 5, 0.0, 1, "[]"),
    )
    db.conn.commit()
    tier = db.compute_and_save_tier(run_id)
    db.close()
    assert tier == "B", f"Erwartet Tier B (zu wenig Trades), bekommen {tier}"


def test_tier_a_granted_with_above_20_trades(tmp_db):
    """Tier A wird vergeben wenn avg_trades >= 20 und PF >= 2.0."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="HighFreq-Test", trials=5)
    db.conn.execute(
        "INSERT INTO results (run_id, params, pf, winrate, num_trades, score, rank, warnings) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "{}", 2.5, 0.0, 25, 0.0, 1, "[]"),
    )
    db.conn.commit()
    tier = db.compute_and_save_tier(run_id)
    db.close()
    assert tier == "A", f"Erwartet Tier A, bekommen {tier}"


def test_tier_a_exactly_20_trades(tmp_db):
    """Tier A bei genau 20 Trades (Grenzwert)."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="EdgeCase-20-Test", trials=5)
    db.conn.execute(
        "INSERT INTO results (run_id, params, pf, winrate, num_trades, score, rank, warnings) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "{}", 2.5, 0.0, 20, 0.0, 1, "[]"),
    )
    db.conn.commit()
    tier = db.compute_and_save_tier(run_id)
    db.close()
    assert tier == "A", "Genau 20 Trades soll Tier A ergeben"


def test_tier_b_not_affected_by_min_trades(tmp_db):
    """Tier B hat keinen Mindest-Trade-Constraint."""
    db = BuilderDB(tmp_db)
    run_id = db.save_run(idea="TierB-LowFreq", trials=5)
    db.conn.execute(
        "INSERT INTO results (run_id, params, pf, winrate, num_trades, score, rank, warnings) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "{}", 1.8, 0.0, 2, 0.0, 1, "[]"),
    )
    db.conn.commit()
    tier = db.compute_and_save_tier(run_id)
    db.close()
    assert tier == "B", "Tier B darf nicht durch Low-Trades blockiert werden"
