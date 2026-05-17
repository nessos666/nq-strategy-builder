from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from sb.cli import app
from sb.memory.db import BuilderDB


@pytest.fixture
def tmp_sources(tmp_path):
    """Minimal sources.yaml mit holdout_start."""
    dates = pd.DatetimeIndex(
        pd.date_range("2025-01-01", periods=500, freq="1min", tz="UTC").tolist()
        + pd.date_range("2026-01-01", periods=200, freq="1min", tz="UTC").tolist()
    )
    df = pd.DataFrame(
        {
            "open": [100.0] * 700,
            "high": [101.0] * 700,
            "low": [99.0] * 700,
            "close": [100.5] * 700,
            "volume": [100] * 700,
        },
        index=dates,
    )
    data_path = tmp_path / "bars.parquet"
    df.to_parquet(data_path)
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        f"backtest_data:\n  path: {data_path}\n  holdout_start: '2026-01-01'\n"
    )
    return sources_yaml, tmp_path


def _insert_run_with_params(db_path: Path, idea: str, params: dict) -> int:
    """Hilfsfunktion: Run + Ergebnis mit params in DB eintragen."""
    db = BuilderDB(db_path)
    run_id = db.save_run(idea, trials=50, session="ny", is_robust=True)
    from sb.models import BacktestResult

    result = BacktestResult(
        params=params, gross_profit=500.0, gross_loss=200.0, num_trades=50, num_wins=30
    )
    db.save_result(run_id, result, score=2.5, rank=1, warnings=[])
    db.compute_and_save_tier(run_id)
    db.close()
    return run_id


def test_validate_command_saves_holdout_result(tmp_sources):
    """validate-Command speichert holdout_pf und holdout_trades in DB."""
    sources_yaml, tmp_path = tmp_sources
    db_path = tmp_path / "output" / "builder.db"
    (tmp_path / "output").mkdir()

    params = {
        "sl_points": 10.0,
        "tp_mult": 2.0,
        "entry_bar_offset": 1,
        "session": "ny",
        "concepts": ["bos", "fvg"],
        "entry": ["bos"],
        "zone": ["fvg"],
        "context": [],
        "timing": [],
        "direction": 1,
    }
    run_id = _insert_run_with_params(db_path, "BOS + FVG NY", params)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate",
            "BOS + FVG NY",
            "--sources",
            str(sources_yaml),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0, result.output

    db = BuilderDB(db_path)
    row = db.conn.execute(
        "SELECT holdout_pf, holdout_trades, holdout_validated FROM build_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    db.close()

    assert row[2] == 1, "holdout_validated muss 1 sein"
    assert row[0] is not None, "holdout_pf muss gesetzt sein"
    assert row[1] is not None, "holdout_trades muss gesetzt sein"


def test_validate_command_fails_without_holdout_start(tmp_path):
    """validate gibt Fehler wenn holdout_start nicht in sources.yaml."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "volume": [100] * 100,
        },
        index=dates,
    )
    data_path = tmp_path / "bars.parquet"
    df.to_parquet(data_path)
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(f"backtest_data:\n  path: {data_path}\n")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate",
            "BOS + FVG NY",
            "--sources",
            str(sources_yaml),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    assert result.exit_code != 0


def test_registry_shows_holdout_column(tmp_path, monkeypatch):
    """registry-Ausgabe enthält 'Holdout' Spalte."""
    db_path = tmp_path / "output" / "builder.db"
    (tmp_path / "output").mkdir()

    params = {
        "sl_points": 10.0,
        "tp_mult": 2.0,
        "entry_bar_offset": 1,
        "session": "ny",
        "concepts": ["bos", "fvg"],
        "entry": ["bos"],
        "zone": ["fvg"],
        "context": [],
        "timing": [],
        "direction": 1,
    }
    run_id = _insert_run_with_params(db_path, "BOS + FVG NY", params)
    db = BuilderDB(db_path)
    db.save_holdout_result(run_id, holdout_pf=1.75, holdout_trades=200)
    db.close()

    import sb.cli as cli_mod
    from rich.console import Console

    monkeypatch.setattr(cli_mod, "console", Console(width=200))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "registry",
            "--output",
            str(tmp_path / "output"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Holdout" in result.output
    assert "1.750" in result.output
