from typer.testing import CliRunner

from sb.cli import app

runner = CliRunner()


def test_parallel_help():
    result = runner.invoke(app, ["parallel", "--help"])
    assert result.exit_code == 0
    assert "--workers" in result.output


def test_parallel_dry_run(tmp_path):
    """Mit --dry-run nur Kombinationen anzeigen, nichts ausführen."""
    result = runner.invoke(app, ["parallel", "--dry-run", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert "Ideen" in result.output
