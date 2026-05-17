from __future__ import annotations


from typer.testing import CliRunner

from sb.cli import app


def test_build_end_to_end(sample_parquet, tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        f"""
pda_library:
  path: {tmp_path / "pdas"}
  pattern: "algo_*.py"
backtest_data:
  path: {sample_parquet}
se_knowledge:
  path: {tmp_path / "k"}
  pattern: "*.md"
  max_files: 5
fehlerprotokoll:
  path: {tmp_path / "f.md"}
feedback_rules:
  path: {tmp_path / "fb.md"}
ideen_bibliothek:
  path: {tmp_path / "i"}
  pattern: "*.md"
  max_files: 5
"""
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build",
            "BOS + FVG London Open",
            "--trials",
            "5",
            "--output",
            str(tmp_path / "output"),
            "--sources",
            str(sources_yaml),
        ],
    )
    assert result.exit_code == 0, result.output
    report_files = list((tmp_path / "output").glob("report_*.md"))
    assert len(report_files) == 1


def test_build_fails_without_data(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        f"""
pda_library:
  path: {tmp_path}
  pattern: "algo_*.py"
backtest_data:
  path: {tmp_path / "nonexistent.parquet"}
se_knowledge:
  path: {tmp_path}
  pattern: "*.md"
  max_files: 5
fehlerprotokoll:
  path: {tmp_path / "f.md"}
feedback_rules:
  path: {tmp_path / "fb.md"}
ideen_bibliothek:
  path: {tmp_path}
  pattern: "*.md"
  max_files: 5
"""
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build",
            "BOS FVG",
            "--trials",
            "2",
            "--output",
            str(tmp_path / "out"),
            "--sources",
            str(sources_yaml),
        ],
    )
    assert result.exit_code == 1


def test_cli_buchhalter_warns_on_duplicate(tmp_path, sample_parquet):
    """CLI zeigt Buchhalter-Warnung wenn Idee bereits in DB ist."""
    from typer.testing import CliRunner

    from sb.cli import app
    from sb.memory.db import BuilderDB

    # Idee vorab in DB eintragen
    db_path = tmp_path / "builder.db"
    db = BuilderDB(db_path=db_path)
    db.save_run(idea="BOS + FVG London", trials=50)
    db.close()

    # sources.yaml mit korrektem data-path erstellen
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(
        f"backtest_data:\n  path: {sample_parquet}\n", encoding="utf-8"
    )

    runner = CliRunner()
    # Input "n" = User lehnt Rebuild ab → CLI soll mit Exit(0) beenden
    result = runner.invoke(
        app,
        [
            "build",
            "BOS + FVG London",
            "--output",
            str(tmp_path),
            "--sources",
            str(sources_path),
            "--trials",
            "2",
        ],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "Buchhalter" in result.output
