from __future__ import annotations

from typer.testing import CliRunner

from sb.cli import app

runner = CliRunner()


def test_diagnose_help():
    result = runner.invoke(app, ["diagnose", "--help"])
    assert result.exit_code == 0
    assert "diagnose" in result.output.lower() or "tier" in result.output.lower()


def test_diagnose_missing_db(tmp_path):
    result = runner.invoke(
        app,
        [
            "diagnose",
            "--output",
            str(tmp_path / "nonexistent"),
        ],
    )
    assert (
        result.exit_code != 0
        or "nicht gefunden" in result.output.lower()
        or "error" in result.output.lower()
    )
