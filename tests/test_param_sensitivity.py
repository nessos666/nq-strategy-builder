from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            params TEXT,
            pf REAL,
            winrate REAL,
            num_trades INTEGER,
            score REAL,
            rank INTEGER,
            warnings TEXT,
            created_at TEXT
        )
    """)
    # sl_points stark korreliert mit PF (sensitiv)
    # tp_mult kaum korreliert (stabil)
    rows = []
    for i in range(50):
        sl = 5.0 + i * 0.5  # 5.0 .. 29.5
        tp = 2.0 + (i % 3) * 0.1  # 2.0 / 2.1 / 2.2 rotierend
        pf = 0.5 + sl * 0.08  # stark von sl abhaengig
        params = json.dumps({"sl_points": sl, "tp_mult": tp, "entry_bar_offset": i % 3})
        rows.append((i + 1, 99, params, pf, 0.5, 20, pf, i, "[]", "2026-04-28"))
    conn.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db_path


def test_sensitive_param_detected(tmp_path):
    from scripts.param_sensitivity import analyze_sensitivity

    db = _make_db(tmp_path)
    results = analyze_sensitivity(db, run_id=99)
    sl_result = next(r for r in results if r["param"] == "sl_points")
    tp_result = next(r for r in results if r["param"] == "tp_mult")
    # sl hat hohe Varianz -> niedrigerer stability score als tp
    assert sl_result["stability_score"] < tp_result["stability_score"]


def test_all_numeric_params_covered(tmp_path):
    from scripts.param_sensitivity import analyze_sensitivity

    db = _make_db(tmp_path)
    results = analyze_sensitivity(db, run_id=99)
    param_names = {r["param"] for r in results}
    assert "sl_points" in param_names
    assert "tp_mult" in param_names
    assert "entry_bar_offset" in param_names


def test_empty_run_returns_empty(tmp_path):
    from scripts.param_sensitivity import analyze_sensitivity

    db = _make_db(tmp_path)
    results = analyze_sensitivity(db, run_id=999)  # nicht vorhanden
    assert results == []


def test_single_value_param_skipped(tmp_path):
    from scripts.param_sensitivity import analyze_sensitivity

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY, run_id INTEGER, params TEXT, pf REAL,
            winrate REAL, num_trades INTEGER, score REAL, rank INTEGER,
            warnings TEXT, created_at TEXT
        )
    """)
    # sl_points hat immer denselben Wert 10.0 -> soll uebersprungen werden
    # tp_mult variiert -> soll im Ergebnis sein
    rows = []
    for i in range(20):
        params = json.dumps({"sl_points": 10.0, "tp_mult": 1.0 + i * 0.1})
        pf = 1.5 + i * 0.05
        rows.append((i + 1, 77, params, pf, 0.5, 20, pf, i, "[]", "2026-04-28"))
    conn.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    results = analyze_sensitivity(db_path, run_id=77)
    param_names = {r["param"] for r in results}
    assert "sl_points" not in param_names  # wurde uebersprungen
    assert "tp_mult" in param_names  # ist vorhanden


def test_bucket_boundary_no_overlap(tmp_path):
    from scripts.param_sensitivity import analyze_sensitivity

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY, run_id INTEGER, params TEXT, pf REAL,
            winrate REAL, num_trades INTEGER, score REAL, rank INTEGER,
            warnings TEXT, created_at TEXT
        )
    """)
    # Genau 10 Werte, gleichmaessig verteilt von 1.0 bis 10.0
    # Mit 5 Buckets muss jeder Bucket genau 2 Werte enthalten (kein Overlap)
    rows = []
    for i in range(10):
        val = 1.0 + i * 1.0  # 1..10
        params = json.dumps({"x": val})
        rows.append((i + 1, 88, params, 2.0, 0.5, 20, 2.0, i, "[]", "2026-04-28"))
    conn.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    results = analyze_sensitivity(db_path, run_id=88)
    assert len(results) == 1
    x_result = results[0]
    # Alle Buckets haben gleiche PF=2.0 -> std=0 -> stability_score = 1/(0+0.001)
    # Aber wichtig: kein Bucket wurde wegen leerem mask uebersprungen
    assert len(x_result["buckets"]) == 5  # alle 5 Buckets vorhanden
