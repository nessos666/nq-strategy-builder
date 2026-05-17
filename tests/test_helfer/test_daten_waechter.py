"""Tests für daten_waechter – Parquet/Cache Datenqualitäts-Guardian."""

import json
import hashlib

import pandas as pd
import pytest


@pytest.fixture
def mock_env(tmp_path):
    sources = tmp_path / "knowledge_sources"
    sources.mkdir()
    (sources / "sources.yaml").write_text("pda_library:\n  paths:\n    - ../lib\n")
    shards = tmp_path / "sb" / "cache" / "signal_shards"
    shards.mkdir(parents=True)
    df = pd.DataFrame({"a": [True, False]})
    df.to_parquet(shards / "algo_fvg.parquet")
    meta = {
        "sources_hash": hashlib.md5(
            b"pda_library:\n  paths:\n    - ../lib\n"
        ).hexdigest(),
        "built_at": "2026-05-08T10:00:00",
    }
    (shards / ".cache_meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_check_cache_fresh_returns_ok(mock_env):
    from helfer.daten_waechter import check_cache_freshness

    result = check_cache_freshness(mock_env)
    assert result.is_fresh is True
    assert result.reason == "ok"


def test_check_cache_stale_after_sources_change(mock_env):
    from helfer.daten_waechter import check_cache_freshness

    (mock_env / "knowledge_sources" / "sources.yaml").write_text("changed: true\n")
    result = check_cache_freshness(mock_env)
    assert result.is_fresh is False
    assert "sources" in result.reason.lower()


def test_check_cache_missing_meta(mock_env):
    from helfer.daten_waechter import check_cache_freshness

    meta = mock_env / "sb" / "cache" / "signal_shards" / ".cache_meta.json"
    meta.unlink()
    result = check_cache_freshness(mock_env)
    assert result.is_fresh is False
    assert "meta" in result.reason.lower()


def test_check_parquet_integrity_ok(mock_env):
    from helfer.daten_waechter import check_parquet_integrity

    shards = mock_env / "sb" / "cache" / "signal_shards"
    results = check_parquet_integrity(shards)
    assert all(r.is_valid for r in results)


def test_check_parquet_integrity_corrupt(mock_env):
    from helfer.daten_waechter import check_parquet_integrity

    corrupt = mock_env / "sb" / "cache" / "signal_shards" / "algo_corrupt.parquet"
    corrupt.write_text("not a parquet file")
    shards = mock_env / "sb" / "cache" / "signal_shards"
    results = check_parquet_integrity(shards)
    bad = [r for r in results if not r.is_valid]
    assert len(bad) == 1
    assert "corrupt" in bad[0].file.name
