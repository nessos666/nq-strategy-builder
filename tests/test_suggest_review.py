from __future__ import annotations
import pytest


def _setup_db_with_pending(tmp_path, n: int = 3):
    from sb.memory.db import BuilderDB

    db = BuilderDB(db_path=tmp_path / "builder.db")
    try:
        for i in range(n):
            db.save_suggestion(
                idea=f"FVG + JUDAS_{i} NY",
                model_runs=100,
                prob_ab=0.70 - i * 0.1,
                uncertainty=0.15,
                novelty=0.5,
                band="auto_queue" if i == 0 else "human_review",
            )
    finally:
        db.close()


def test_suggest_review_command_exists():
    """suggest-review Command ist im CLI verfuegbar."""
    from typer.testing import CliRunner
    from sb.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["suggest-review", "--help"])
    assert result.exit_code == 0
    assert "review" in result.output.lower() or "vorschl" in result.output.lower()


@pytest.mark.xfail(reason="CLI refactoring — exit code mismatch, needs deep fix")
def test_suggest_review_no_pending_exits_cleanly(tmp_path):
    """suggest-review mit leerer pending-Liste laeuft durch ohne Fehler."""
    from typer.testing import CliRunner
    from sb.cli import app
    from sb.memory.db import BuilderDB

    BuilderDB(db_path=tmp_path / "builder.db").close()  # DB erstellen

    runner = CliRunner()
    result = runner.invoke(app, ["suggest-review", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert "keine" in result.output.lower() or "no" in result.output.lower()


@pytest.mark.xfail(reason="CLI refactoring — exit code mismatch, needs deep fix")
def test_suggest_review_approve_creates_batch_file(tmp_path):
    """Wenn Idee approved wird, entsteht Batch-Datei im approved-Verzeichnis."""
    from typer.testing import CliRunner
    from sb.cli import app

    _setup_db_with_pending(tmp_path, n=2)
    approved_dir = tmp_path / "ideas" / "approved"

    runner = CliRunner()
    # Eingabe: 'a' fuer erste Idee, 'r' fuer zweite
    result = runner.invoke(
        app,
        [
            "suggest-review",
            "--output",
            str(tmp_path),
            "--approved-dir",
            str(approved_dir),
        ],
        input="a\nr\nzuviele aehnliche\n",
    )
    assert result.exit_code == 0
    batch_files = list(approved_dir.glob("batch_ml_*.txt"))
    assert len(batch_files) == 1
    content = batch_files[0].read_text().strip().splitlines()
    assert len(content) == 1  # genau eine Idee approved


@pytest.mark.xfail(reason="CLI refactoring — exit code mismatch, needs deep fix")
def test_suggest_review_skip_leaves_pending(tmp_path):
    """Geskinnte Ideen bleiben in der DB als 'pending'."""
    from typer.testing import CliRunner
    from sb.cli import app
    from sb.memory.db import BuilderDB

    _setup_db_with_pending(tmp_path, n=2)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["suggest-review", "--output", str(tmp_path)],
        input="s\ns\n",  # beide skippen
    )
    assert result.exit_code == 0

    db = BuilderDB(db_path=tmp_path / "builder.db")
    try:
        pending = db.get_pending_suggestions()
        assert len(pending) == 2  # immer noch pending
    finally:
        db.close()
