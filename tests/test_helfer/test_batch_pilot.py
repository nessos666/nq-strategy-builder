"""Tests für batch_pilot – Batch-Vorflug-Check."""

import json
import pytest


@pytest.fixture
def mock_env(tmp_path):
    shards = tmp_path / "sb" / "cache" / "signal_shards"
    shards.mkdir(parents=True)
    meta = {"sources_hash": "abc123", "stale": False}
    (shards / ".cache_meta.json").write_text(json.dumps(meta))
    import pandas as pd

    df = pd.DataFrame({"a": [True]})
    df.to_parquet(shards / "algo_test.parquet")
    (tmp_path / "test_idea.txt").write_text("FVG Standard + MANIP Bear NY\n")
    return tmp_path


def test_preflight_ok_when_cache_fresh(mock_env):
    from helfer.batch_pilot import preflight_check

    result = preflight_check(mock_env, mock_env / "test_idea.txt")
    assert result.go is True


def test_preflight_nogo_when_cache_stale(mock_env):
    from helfer.batch_pilot import preflight_check

    meta_path = mock_env / "sb" / "cache" / "signal_shards" / ".cache_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["stale"] = True
    meta["stale_reason"] = "sources.yaml changed"
    meta_path.write_text(json.dumps(meta))
    result = preflight_check(mock_env, mock_env / "test_idea.txt")
    assert result.go is False
    assert "stale" in result.reason.lower()


def test_preflight_nogo_when_no_cache(mock_env):
    from helfer.batch_pilot import preflight_check

    meta_path = mock_env / "sb" / "cache" / "signal_shards" / ".cache_meta.json"
    meta_path.unlink()
    result = preflight_check(mock_env, mock_env / "test_idea.txt")
    assert result.go is False


def test_preflight_nogo_when_no_idea_file(mock_env):
    from helfer.batch_pilot import preflight_check

    result = preflight_check(mock_env, mock_env / "nonexistent.txt")
    assert result.go is False
