"""Tests fuer auto_research_agent.py."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from scripts.auto_research_agent import (
    _build_idea,
    _extract_session,
    app,
    evaluate_idea,
    mutate_idea,
    run_agent,
)
from scripts.baustein_analyse import extract_bausteine


def _make_db(tmp_path: Path, idea: str, pf: float | None) -> Path:
    """Legt eine Test-DB mit einem build_runs-Eintrag an."""
    db = tmp_path / "builder.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE build_runs
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea TEXT, avg_oos_pf REAL, tier TEXT, session TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        "INSERT INTO build_runs (idea, avg_oos_pf, tier, session) VALUES (?,?,?,?)",
        (idea, pf, "B", "NY"),
    )
    conn.commit()
    conn.close()
    return db


def test_evaluate_idea_gibt_pf_zurueck(tmp_path):
    db = _make_db(tmp_path, "INSIDE_DAY + MB NY", 1.85)
    with patch("scripts.auto_research_agent._run_sb_batch") as mock_run:
        result = evaluate_idea(
            "INSIDE_DAY + MB NY",
            trials=5,
            db_path=db,
            output_dir=tmp_path,
        )
    mock_run.assert_called_once_with("INSIDE_DAY + MB NY", 5, tmp_path)
    assert result == pytest.approx(1.85)


def test_evaluate_idea_gibt_null_wenn_nicht_in_db(tmp_path):
    db = _make_db(tmp_path, "ANDERE_IDEE NY", 1.5)
    with patch("scripts.auto_research_agent._run_sb_batch"):
        result = evaluate_idea(
            "INSIDE_DAY + MB NY",
            trials=5,
            db_path=db,
            output_dir=tmp_path,
        )
    assert result == 0.0


def test_evaluate_idea_gibt_null_wenn_pf_none(tmp_path):
    db = _make_db(tmp_path, "INSIDE_DAY + MB NY", None)
    with patch("scripts.auto_research_agent._run_sb_batch"):
        result = evaluate_idea(
            "INSIDE_DAY + MB NY", trials=5, db_path=db, output_dir=tmp_path
        )
    assert result == 0.0


# -- Mutation Tests (Task 2) ---------------------------------------------------


def test_extract_session_findet_ny():

    assert _extract_session("INSIDE_DAY + MB NY") == "NY"


def test_extract_session_findet_london():

    assert _extract_session("SWEEP + FVG LONDON") == "LONDON"


def test_extract_session_default_ny():

    assert _extract_session("BOS + OB") == "NY"


def test_build_idea_formatiert_korrekt():

    result = _build_idea(["INSIDE_DAY", "MB"], "NY")
    assert result == "INSIDE_DAY + MB NY"


def test_mutate_idea_add_baustein():

    idea = "INSIDE_DAY + MB NY"
    results = {mutate_idea(idea, rng=random.Random(s)) for s in range(50)}
    added = [r for r in results if len(extract_bausteine(r)) > 2]
    assert len(added) > 0, "Keine add-Mutation gefunden in 50 Versuchen"


def test_mutate_idea_remove_baustein():

    idea = "INSIDE_DAY + MB + HURST NY"
    results = {mutate_idea(idea, rng=random.Random(s)) for s in range(50)}
    removed = [r for r in results if len(extract_bausteine(r)) < 3]
    assert len(removed) > 0, "Keine remove-Mutation gefunden in 50 Versuchen"


def test_mutate_idea_swap_session():

    idea = "INSIDE_DAY + MB NY"
    results = {mutate_idea(idea, rng=random.Random(s)) for s in range(50)}
    swapped = [r for r in results if "NY" not in r]
    assert len(swapped) > 0, "Keine swap_session-Mutation gefunden in 50 Versuchen"


def test_mutate_idea_max_4_bausteine():

    idea = "BOS + OB + FVG + SWEEP NY"
    for seed in range(20):
        result = mutate_idea(idea, rng=random.Random(seed))
        bausteine = extract_bausteine(result)
        assert len(bausteine) <= 4, f"Zu viele Bausteine: {result}"


def test_mutate_idea_min_1_baustein():

    idea = "INSIDE_DAY NY"
    for seed in range(20):
        result = mutate_idea(idea, rng=random.Random(seed))
        bausteine = extract_bausteine(result)
        assert len(bausteine) >= 1, f"Keine Bausteine: {result}"


# -- Agent Loop Tests (Task 3) ---------------------------------------------------


def _make_evaluate(pf_sequence: list[float]) -> Callable:
    """Gibt eine evaluate-Funktion zurueck die PF-Werte der Reihe nach liefert."""
    iterator = iter(pf_sequence)

    def _evaluate(idea: str, trials: int, db_path: Path, output_dir: Path) -> float:
        return next(iterator, 0.0)

    return _evaluate


def test_run_agent_keep_wenn_besser(tmp_path):
    evaluate_fn = _make_evaluate([1.0, 1.5])
    best_idea, results = run_agent(
        base_idea="INSIDE_DAY + MB NY",
        n_experiments=1,
        trials=5,
        db_path=tmp_path / "builder.db",
        output_dir=tmp_path,
        results_file=tmp_path / "ara.tsv",
        _evaluate=evaluate_fn,
    )
    assert results[1].status == "keep"
    assert results[1].pf == pytest.approx(1.5)


def test_run_agent_discard_wenn_schlechter(tmp_path):
    evaluate_fn = _make_evaluate([1.5, 1.0])
    best_idea, results = run_agent(
        base_idea="INSIDE_DAY + MB NY",
        n_experiments=1,
        trials=5,
        db_path=tmp_path / "builder.db",
        output_dir=tmp_path,
        results_file=tmp_path / "ara.tsv",
        _evaluate=evaluate_fn,
    )
    assert results[1].status == "discard"


def test_run_agent_baseline_ist_experiment_0(tmp_path):
    evaluate_fn = _make_evaluate([1.2, 1.3])
    best_idea, results = run_agent(
        base_idea="INSIDE_DAY + MB NY",
        n_experiments=1,
        trials=5,
        db_path=tmp_path / "builder.db",
        output_dir=tmp_path,
        results_file=tmp_path / "ara.tsv",
        _evaluate=evaluate_fn,
    )
    assert results[0].status == "baseline"
    assert results[0].experiment_id == 0


def test_run_agent_schreibt_tsv(tmp_path):
    evaluate_fn = _make_evaluate([1.0, 1.2, 0.9])
    tsv = tmp_path / "ara.tsv"
    run_agent(
        base_idea="INSIDE_DAY + MB NY",
        n_experiments=2,
        trials=5,
        db_path=tmp_path / "builder.db",
        output_dir=tmp_path,
        results_file=tsv,
        _evaluate=evaluate_fn,
    )
    assert tsv.exists()
    lines = tsv.read_text().strip().split("\n")
    assert len(lines) == 4  # Header + baseline + 2 Experimente
    assert lines[0] == "id\tidea\tpf\tstatus"


def test_run_agent_gibt_beste_idee_zurueck(tmp_path):
    # Sequenz nicht-monoton: exp 1 ist am besten (PF=2.0), exp 2 schlechter (PF=1.5)
    evaluate_fn = _make_evaluate([1.0, 2.0, 1.5])
    best_idea, results = run_agent(
        base_idea="INSIDE_DAY + MB NY",
        n_experiments=2,
        trials=5,
        db_path=tmp_path / "builder.db",
        output_dir=tmp_path,
        results_file=tmp_path / "ara.tsv",
        _evaluate=evaluate_fn,
    )
    assert results[1].status == "keep"  # exp 1: PF=2.0 > baseline 1.0 → KEEP
    assert results[2].status == "discard"  # exp 2: PF=1.5 < best 2.0 → DISCARD
    assert best_idea == results[1].idea  # beste Idee ist exp 1, nicht exp 2!


def test_run_agent_baseline_null_jeder_positive_wird_behalten(tmp_path):
    # Wenn Baseline 0.0 → jede positive PF wird gehalten
    evaluate_fn = _make_evaluate([0.0, 1.2])
    best_idea, results = run_agent(
        base_idea="INSIDE_DAY + MB NY",
        n_experiments=1,
        trials=5,
        db_path=tmp_path / "builder.db",
        output_dir=tmp_path,
        results_file=tmp_path / "ara.tsv",
        _evaluate=evaluate_fn,
    )
    assert results[0].pf == pytest.approx(0.0)
    assert results[1].status == "keep"
    assert results[1].pf == pytest.approx(1.2)


# -- CLI Tests (Task 4) -----------------------------------------------------------


def test_cli_run_zeigt_ergebnis(tmp_path):
    """run_agent läuft wirklich, evaluate_idea ist gemockt (nur _run_sb_batch)."""
    runner = CliRunner()

    db = tmp_path / "builder.db"
    call_count = [0]
    pf_values = [1.2, 1.5, 1.1]

    def mock_run_sb(idea: str, trials: int, output_dir: Path) -> None:
        """Simuliert sb.py: fügt Entry in DB mit PF-Wert ein."""
        pf = pf_values[call_count[0]] if call_count[0] < len(pf_values) else 0.0
        call_count[0] += 1

        conn = sqlite3.connect(db)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS build_runs
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                idea TEXT, avg_oos_pf REAL, tier TEXT, session TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
        )
        conn.execute(
            "INSERT INTO build_runs (idea, avg_oos_pf, tier, session) VALUES (?,?,?,?)",
            (idea, pf, "B", "NY"),
        )
        conn.commit()
        conn.close()

    with patch("scripts.auto_research_agent._run_sb_batch", side_effect=mock_run_sb):
        result = runner.invoke(
            app,
            [
                "run",
                "INSIDE_DAY + MB NY",
                "--experiments",
                "2",
                "--trials",
                "5",
                "--db",
                str(db),
                "--output",
                str(tmp_path),
                "--results-file",
                str(tmp_path / "ara.tsv"),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "ERGEBNIS" in result.output
    assert "Baseline" in result.output
    tsv = tmp_path / "ara.tsv"
    assert tsv.exists()
    lines = tsv.read_text().strip().split("\n")
    assert len(lines) == 4  # Header + baseline + 2 Experimente
