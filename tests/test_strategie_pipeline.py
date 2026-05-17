import sys
import os
import sqlite3
import tempfile
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.strategie_pipeline import app


def _make_test_db(path: str) -> None:
    """Erstellt eine minimale Test-DB."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE build_runs (
            id INTEGER PRIMARY KEY, idea TEXT, avg_oos_pf REAL,
            tier TEXT, session TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY, run_id INTEGER,
            pf REAL, winrate REAL, num_trades INTEGER, rank INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO build_runs VALUES (?,?,?,?,?)",
        [
            (1, "INSIDE_DAY + MB NY", 1.5, "A", "ny"),
            (2, "INSIDE_DAY + OB NY", 1.4, "B", "ny"),
            (3, "INSIDE_DAY + FVG NY", 1.3, "B", "ny"),
            (4, "SWEEP + OB NY", 0.9, "C", "ny"),
            (5, "SWEEP + FVG NY", 0.8, "C", "ny"),
            (6, "SWEEP + BOS NY", 0.7, "C", "ny"),
        ],
    )
    conn.commit()
    conn.close()


def test_scan_zeigt_starke_bausteine():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_test_db(db_path)
        result = runner.invoke(app, ["scan", "--db", db_path, "--min-delta", "0.1"])
        assert result.exit_code == 0, (
            f"Exit code: {result.exit_code}\nOutput: {result.output}"
        )
        assert "INSIDE_DAY" in result.output
        # SWEEP hat negativen delta -> soll nicht erscheinen bei min_delta=0.1
        assert "SWEEP" not in result.output
    finally:
        os.unlink(db_path)


def test_filter_pass_mit_hohem_pf():
    from unittest.mock import patch

    runner = CliRunner()
    mock_result = {"pf": 1.5, "wr": 65.0, "trades": 55, "params": {"disp_mult": 2.5}}

    with patch("scripts.strategie_pipeline._run_quick_check", return_value=mock_result):
        result = runner.invoke(app, ["filter", "SWEEP + OB NY", "--min-pf", "1.2"])

    assert result.exit_code == 0, f"Exit: {result.exit_code}\n{result.output}"
    assert "PASS" in result.output
    assert "1.5" in result.output


def test_filter_fail_mit_niedrigem_pf():
    from unittest.mock import patch

    runner = CliRunner()
    mock_result = {"pf": 0.9, "wr": 40.0, "trades": 35, "params": {}}

    with patch("scripts.strategie_pipeline._run_quick_check", return_value=mock_result):
        result = runner.invoke(app, ["filter", "SWEEP + OB NY", "--min-pf", "1.2"])

    assert result.exit_code == 0, f"Exit: {result.exit_code}\n{result.output}"
    assert "FAIL" in result.output


def test_queue_erstellt_batch_datei():
    import tempfile
    from unittest.mock import patch

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_dir = Path(tmpdir) / "queue"
        queue_dir.mkdir()

        with patch("scripts.strategie_pipeline.QUEUE_DIR", queue_dir):
            result = runner.invoke(app, ["queue", "INSIDE_DAY + MB NY"])

        assert result.exit_code == 0, f"Exit: {result.exit_code}\n{result.output}"
        batch_files = list(queue_dir.glob("batch_pipeline_*.txt"))
        assert len(batch_files) == 1
        content = batch_files[0].read_text()
        assert "INSIDE_DAY + MB NY" in content


def test_queue_mehrere_ideen():
    import tempfile
    from unittest.mock import patch

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_dir = Path(tmpdir) / "queue"
        queue_dir.mkdir()

        with patch("scripts.strategie_pipeline.QUEUE_DIR", queue_dir):
            result = runner.invoke(
                app,
                ["queue", "INSIDE_DAY + MB NY", "--extra", "INSIDE_DAY + MB LONDON"],
            )

        assert result.exit_code == 0, f"Exit: {result.exit_code}\n{result.output}"
        batch_files = list(queue_dir.glob("batch_pipeline_*.txt"))
        content = batch_files[0].read_text()
        assert "INSIDE_DAY + MB NY" in content
        assert "INSIDE_DAY + MB LONDON" in content
