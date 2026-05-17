from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _make_source_db(tmp_path: Path, db_name: str = "source.db") -> Path:
    """Quell-DB mit einer Strategie und 1 results-Zeile."""
    db_path = tmp_path / db_name
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE build_runs (
            id INTEGER PRIMARY KEY, idea TEXT, tier TEXT,
            avg_oos_pf REAL, created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY, run_id INTEGER, params TEXT,
            pf REAL, winrate REAL, num_trades INTEGER,
            score REAL, rank INTEGER, warnings TEXT, created_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO build_runs VALUES (?, ?, ?, ?, ?)",
        (42, "JUDAS + FVG NY", "A", 2.1, "2026-04-20 10:00:00"),
    )
    params = json.dumps(
        {
            "sl_points": 12.0,
            "tp_mult": 2.5,
            "entry_bar_offset": 1,
            "session": "ny",
            "concepts": ["JUDAS", "FVG"],
        }
    )
    conn.execute(
        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?)",
        (1, 42, params, 2.1, 0.62, 45, 2.1, 1, "[]", "2026-04-20 10:00:00"),
    )
    conn.commit()
    conn.close()
    return db_path


def _make_target_db(tmp_path: Path, db_name: str = "target.db") -> Path:
    """Ziel-DB mit gleicher Idee aber ohne results."""
    db_path = tmp_path / db_name
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE build_runs (
            id INTEGER PRIMARY KEY, idea TEXT, tier TEXT,
            avg_oos_pf REAL, created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE results (
            id INTEGER PRIMARY KEY, run_id INTEGER, params TEXT,
            pf REAL, winrate REAL, num_trades INTEGER,
            score REAL, rank INTEGER, warnings TEXT, created_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO build_runs VALUES (?, ?, ?, ?, ?)",
        (7, "JUDAS + FVG NY", "A", 2.1, "2026-04-20 10:00:00"),
    )
    conn.commit()
    conn.close()
    return db_path


def test_backfill_copies_results(tmp_path):
    from scripts.backfill_params import backfill

    source_db = _make_source_db(tmp_path)
    target_db = _make_target_db(tmp_path)

    n_copied = backfill(target_db=target_db, source_db=source_db)

    assert n_copied == 1
    conn = sqlite3.connect(target_db)
    rows = conn.execute("SELECT run_id, pf FROM results").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == 7  # run_id remappt auf target run_id
    assert abs(rows[0][1] - 2.1) < 0.01


def test_backfill_skips_already_present(tmp_path):
    from scripts.backfill_params import backfill

    source_db = _make_source_db(tmp_path)
    target_db = _make_target_db(tmp_path)

    backfill(target_db=target_db, source_db=source_db)
    n_copied = backfill(target_db=target_db, source_db=source_db)
    assert n_copied == 0


def test_backfill_unknown_idea_not_copied(tmp_path):
    from scripts.backfill_params import backfill

    source_db = _make_source_db(tmp_path)

    target_db = tmp_path / "target2.db"
    conn = sqlite3.connect(target_db)
    conn.execute(
        "CREATE TABLE build_runs (id INTEGER PRIMARY KEY, idea TEXT, tier TEXT, avg_oos_pf REAL, created_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE results (id INTEGER PRIMARY KEY, run_id INTEGER, params TEXT, pf REAL, winrate REAL, num_trades INTEGER, score REAL, rank INTEGER, warnings TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO build_runs VALUES (?,?,?,?,?)",
        (1, "MANIP + OB NY", "B", 1.7, "2026-04-20"),
    )
    conn.commit()
    conn.close()

    n_copied = backfill(target_db=target_db, source_db=source_db)
    assert n_copied == 0
