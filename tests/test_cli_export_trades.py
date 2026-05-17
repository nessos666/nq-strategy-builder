from __future__ import annotations

from typer.testing import CliRunner
from sb.cli import app

runner = CliRunner()


def test_export_trades_help():
    result = runner.invoke(app, ["export-trades", "--help"])
    assert result.exit_code == 0
    assert "tier" in result.output.lower() or "export" in result.output.lower()


def test_export_trades_missing_db(tmp_path):
    result = runner.invoke(
        app,
        [
            "export-trades",
            "--tier",
            "A",
            "--output",
            str(tmp_path / "nonexistent"),
        ],
    )
    assert (
        result.exit_code != 0
        or "nicht gefunden" in result.output.lower()
        or "error" in result.output.lower()
        or "db" in result.output.lower()
    )
