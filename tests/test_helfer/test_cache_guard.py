"""Tests für cache_guard – Echtzeit Cache-Invalidierung via watchdog."""

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_env(tmp_path):
    sources = tmp_path / "knowledge_sources"
    sources.mkdir()
    (sources / "sources.yaml").write_text("original: true\n")
    shards = tmp_path / "sb" / "cache" / "signal_shards"
    shards.mkdir(parents=True)
    meta = {"sources_hash": "abc123", "built_at": "2026-05-08"}
    (shards / ".cache_meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_on_sources_changed_writes_event(mock_env, tmp_path):
    from helfer.cache_guard import CacheGuardHandler

    helfer_base = tmp_path / "helfer"
    for d in ("events", "alerts", "status"):
        (helfer_base / d).mkdir(parents=True)
    handler = CacheGuardHandler(sb_path=mock_env, helfer_base=helfer_base)
    event = MagicMock()
    event.src_path = str(mock_env / "knowledge_sources" / "sources.yaml")
    event.is_directory = False
    handler.on_modified(event)
    events = list((helfer_base / "events").glob("*cache_invalidated*"))
    assert len(events) == 1


def test_on_algo_changed_writes_event(mock_env, tmp_path):
    from helfer.cache_guard import CacheGuardHandler

    helfer_base = tmp_path / "helfer"
    for d in ("events", "alerts", "status"):
        (helfer_base / d).mkdir(parents=True)
    handler = CacheGuardHandler(sb_path=mock_env, helfer_base=helfer_base)
    event = MagicMock()
    event.src_path = (
        "/home/boobi/HAUPTLAGER/david_bibliothek/02_FVG_Zonen/fvg_standard.py"
    )
    event.is_directory = False
    handler.on_modified(event)
    events = list((helfer_base / "events").glob("*algo_changed*"))
    assert len(events) == 1


def test_ignores_non_python_files(mock_env, tmp_path):
    from helfer.cache_guard import CacheGuardHandler

    helfer_base = tmp_path / "helfer"
    for d in ("events", "alerts", "status"):
        (helfer_base / d).mkdir(parents=True)
    handler = CacheGuardHandler(sb_path=mock_env, helfer_base=helfer_base)
    event = MagicMock()
    event.src_path = "/some/path/readme.md"
    event.is_directory = False
    handler.on_modified(event)
    events = list((helfer_base / "events").glob("*.json"))
    assert len(events) == 0
