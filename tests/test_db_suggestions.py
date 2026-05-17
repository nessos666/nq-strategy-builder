from __future__ import annotations


def test_save_suggestion_returns_id(tmp_path):
    from sb.memory.db import BuilderDB

    db = BuilderDB(db_path=tmp_path / "test.db")
    try:
        sid = db.save_suggestion(
            idea="FVG + JUDAS NY",
            model_runs=100,
            prob_ab=0.72,
            uncertainty=0.15,
            novelty=0.45,
            band="auto_queue",
        )
        assert isinstance(sid, int)
        assert sid > 0
    finally:
        db.close()


def test_get_pending_suggestions_returns_saved(tmp_path):
    from sb.memory.db import BuilderDB

    db = BuilderDB(db_path=tmp_path / "test.db")
    try:
        db.save_suggestion("FVG + JUDAS NY", 100, 0.72, 0.15, 0.45, "auto_queue")
        db.save_suggestion("BOS + CBDR LONDON", 100, 0.55, 0.20, 0.30, "human_review")
        pending = db.get_pending_suggestions()
        assert len(pending) == 2
        assert pending[0]["idea"] == "FVG + JUDAS NY"
        assert pending[0]["status"] == "pending"
        assert pending[0]["band"] == "auto_queue"
    finally:
        db.close()


def test_update_suggestion_status_approved(tmp_path):
    from sb.memory.db import BuilderDB

    db = BuilderDB(db_path=tmp_path / "test.db")
    try:
        sid = db.save_suggestion("FVG + JUDAS NY", 100, 0.72, 0.15, 0.45, "auto_queue")
        db.update_suggestion_status(sid, "approved")
        pending = db.get_pending_suggestions()
        assert len(pending) == 0
    finally:
        db.close()


def test_update_suggestion_status_rejected_with_reason(tmp_path):
    from sb.memory.db import BuilderDB

    db = BuilderDB(db_path=tmp_path / "test.db")
    try:
        sid = db.save_suggestion("BOS + OB NY", 100, 0.35, 0.30, 0.60, "auto_reject")
        db.update_suggestion_status(
            sid, "rejected", reason="zu viele aehnliche bereits getestet"
        )
        pending = db.get_pending_suggestions()
        assert len(pending) == 0
    finally:
        db.close()


def test_rejected_ideas_not_in_pending(tmp_path):
    from sb.memory.db import BuilderDB

    db = BuilderDB(db_path=tmp_path / "test.db")
    try:
        sid1 = db.save_suggestion("JUDAS + BPR NY", 100, 0.72, 0.15, 0.45, "auto_queue")
        sid2 = db.save_suggestion("BOS + OB NY", 100, 0.35, 0.30, 0.60, "auto_reject")
        db.update_suggestion_status(sid1, "approved")
        db.update_suggestion_status(sid2, "rejected")
        pending = db.get_pending_suggestions()
        assert len(pending) == 0
    finally:
        db.close()


def test_update_suggestion_status_invalid_raises(tmp_path):
    """update_suggestion_status wirft ValueError bei ungueltigem Status."""
    from sb.memory.db import BuilderDB
    import pytest

    db = BuilderDB(db_path=tmp_path / "test.db")
    try:
        sid = db.save_suggestion("FVG + OB NY", 100, 0.60, 0.20, 0.40, "human_review")
        with pytest.raises(ValueError, match="Ungueltiger Status"):
            db.update_suggestion_status(
                sid, "pending"
            )  # pending ist kein gueltiger Status
    finally:
        db.close()


def test_rejected_reason_stored_in_db(tmp_path):
    """Nach reject ist review_reason in der DB gespeichert (get_pending zeigt es nicht, aber Daten sind da)."""
    from sb.memory.db import BuilderDB

    db = BuilderDB(db_path=tmp_path / "test.db")
    try:
        sid = db.save_suggestion("BOS + FVG NY", 100, 0.38, 0.28, 0.55, "auto_reject")
        db.update_suggestion_status(sid, "rejected", reason="zu aehnlich zu ID 42")
        # Direkter DB-Check
        row = db._execute_with_retry(
            lambda: db.conn.execute(
                "SELECT review_reason FROM suggestions WHERE id = ?", (sid,)
            ).fetchone()
        )
        assert row is not None
        assert row[0] == "zu aehnlich zu ID 42"
    finally:
        db.close()
