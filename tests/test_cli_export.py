from __future__ import annotations

import csv
import json

import pytest
from typer.testing import CliRunner

from sb.cli import app
from sb.memory.db import BuilderDB
from sb.models import BacktestResult  # noqa: F401 – used in fixture


@pytest.fixture
def export_db(tmp_path):
    """DB mit je 2 Runs pro Tier (A/B/C) und 2 Sessions (ny/london)."""
    db_path = tmp_path / "output" / "builder.db"
    db_path.parent.mkdir(parents=True)
    db = BuilderDB(db_path=db_path)

    data = [
        ("JUDAS + OU NY", "ny", 2.5, 40),
        ("MANIP + ENTROPY NY", "ny", 2.1, 30),
        ("BOS + FVG London", "london", 1.7, 60),
        ("SWEEP + CBDR NY", "ny", 1.6, 80),
        ("MMXM + CHANGEPOINT NY", "ny", 0.0, 0),
        ("JUDAS + ZERO London", "london", 0.0, 0),
    ]
    for idea, session, pf, trades in data:
        run_id = db.save_run(
            idea=idea, trials=10, session=session, is_robust=(pf >= 2.0)
        )
        wins = max(1, int(trades * 0.4)) if trades > 0 else 0
        gross_p = pf * 100.0 if pf > 0 else 0.0
        for _ in range(3):
            db.save_result(
                run_id=run_id,
                result=BacktestResult(
                    params={},
                    gross_profit=gross_p,
                    gross_loss=100.0,
                    num_trades=trades,
                    num_wins=wins,
                ),
                score=pf,
                rank=1,
                warnings=[],
            )
        db.compute_and_save_tier(run_id)
    db.close()
    return tmp_path


def test_export_creates_all_files(export_db):
    runner = CliRunner()
    result = runner.invoke(app, ["export", "--output", str(export_db / "output")])
    assert result.exit_code == 0, result.output

    registry_dir = export_db / "output" / "registry"
    assert (registry_dir / "alle_runs.csv").exists()
    assert (registry_dir / "tier_A.csv").exists()
    assert (registry_dir / "tier_B.csv").exists()
    assert (registry_dir / "tier_C.csv").exists()
    assert (registry_dir / "session_ny.csv").exists()
    assert (registry_dir / "session_london.csv").exists()
    assert (registry_dir / "details").is_dir()


def test_export_alle_runs_contains_all(export_db):
    runner = CliRunner()
    runner.invoke(app, ["export", "--output", str(export_db / "output")])

    csv_path = export_db / "output" / "registry" / "alle_runs.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6


def test_export_tier_csv_correct_counts(export_db):
    runner = CliRunner()
    runner.invoke(app, ["export", "--output", str(export_db / "output")])

    registry_dir = export_db / "output" / "registry"
    for fname, expected in [("tier_A.csv", 2), ("tier_B.csv", 2), ("tier_C.csv", 2)]:
        with open(registry_dir / fname) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == expected, (
            f"{fname}: erwartet {expected}, bekommen {len(rows)}"
        )


def test_export_details_json_per_run(export_db):
    runner = CliRunner()
    runner.invoke(app, ["export", "--output", str(export_db / "output")])

    details_dir = export_db / "output" / "registry" / "details"
    json_files = list(details_dir.glob("run_*.json"))
    assert len(json_files) == 6

    with open(json_files[0]) as f:
        data = json.load(f)
    assert "id" in data
    assert "idea" in data
    assert "tier" in data
    assert "windows" in data
