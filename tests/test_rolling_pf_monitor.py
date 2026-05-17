from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE build_runs (
            id INTEGER PRIMARY KEY,
            idea TEXT,
            avg_oos_pf REAL,
            tier TEXT,
            created_at TEXT,
            session TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO build_runs VALUES (?,?,?,?,?,?)",
        [
            (1, "JUDAS + FVG NY", 2.5, "A", "2026-04-20 10:00:00", "ny"),
            (2, "JUDAS + FVG NY", 1.8, "B", "2026-04-22 10:00:00", "ny"),
            (3, "JUDAS + FVG NY", 1.2, "C", "2026-04-25 10:00:00", "ny"),
            (4, "MANIP + OB NY", 2.1, "A", "2026-04-20 10:00:00", "ny"),
            (5, "MANIP + OB NY", 2.3, "A", "2026-04-25 10:00:00", "ny"),
            (6, "BOS + FVG London", 999.0, "A", "2026-04-20 10:00:00", "london"),
            (7, "LEGIT HIGH PF", 12.0, "A", "2026-04-20 10:00:00", "ny"),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_decay_detected(tmp_path):
    from scripts.rolling_pf_monitor import load_pf_trends

    db = _make_db(tmp_path)
    trends = load_pf_trends(db)
    decay = [t for t in trends if t["decay_alarm"]]
    assert len(decay) == 1
    assert decay[0]["idea"] == "JUDAS + FVG NY"


def test_stable_no_alarm(tmp_path):
    from scripts.rolling_pf_monitor import load_pf_trends

    db = _make_db(tmp_path)
    trends = load_pf_trends(db)
    stable = [t for t in trends if t["idea"] == "MANIP + OB NY"]
    assert len(stable) == 1
    assert stable[0]["decay_alarm"] is False
    assert stable[0]["trend"] == "↑"


def test_artifact_filtered(tmp_path):
    from scripts.rolling_pf_monitor import load_pf_trends

    db = _make_db(tmp_path)
    trends = load_pf_trends(db)
    ideas = [t["idea"] for t in trends]
    assert "BOS + FVG London" not in ideas


def test_trend_down(tmp_path):
    from scripts.rolling_pf_monitor import load_pf_trends

    db = _make_db(tmp_path)
    trends = load_pf_trends(db)
    judas = next(t for t in trends if t["idea"] == "JUDAS + FVG NY")
    assert judas["trend"] == "↓"
    assert judas["first_pf"] == pytest.approx(2.5)
    assert judas["last_pf"] == pytest.approx(1.2)


def test_legitimate_high_pf_not_filtered(tmp_path):
    from scripts.rolling_pf_monitor import load_pf_trends

    db = _make_db(tmp_path)
    trends = load_pf_trends(db)
    ideas = [t["idea"] for t in trends]
    assert "LEGIT HIGH PF" in ideas


def test_single_run_trend_neutral(tmp_path):
    from scripts.rolling_pf_monitor import load_pf_trends

    db_path = tmp_path / "single.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE build_runs (
            id INTEGER PRIMARY KEY, idea TEXT, avg_oos_pf REAL,
            tier TEXT, created_at TEXT, session TEXT
        )
    """)
    conn.execute(
        "INSERT INTO build_runs VALUES (1,'SOLO + FVG NY',1.8,'B','2026-04-20 10:00:00','ny')"
    )
    conn.commit()
    conn.close()
    trends = load_pf_trends(db_path)
    assert len(trends) == 1
    assert trends[0]["trend"] == "→"
    assert trends[0]["runs"] == 1
    assert trends[0]["decay_alarm"] is False
