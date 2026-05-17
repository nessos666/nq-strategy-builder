"""Tests für helfer/common.py – Shared Utilities."""

import json
import os
import time
from pathlib import Path

import pytest


@pytest.fixture
def helfer_tmp(tmp_path):
    """Erzeugt temporäre ~/.helfer/ Struktur."""
    (tmp_path / "events").mkdir()
    (tmp_path / "alerts").mkdir()
    (tmp_path / "status").mkdir()
    (tmp_path / "quality").mkdir()
    return tmp_path


def test_write_event_creates_json(helfer_tmp):
    from helfer.common import write_event

    write_event("cache_stale", {"reason": "sources changed"}, base=helfer_tmp)
    files = list((helfer_tmp / "events").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["type"] == "cache_stale"
    assert data["reason"] == "sources changed"
    assert "timestamp" in data


def test_write_alert_creates_markdown(helfer_tmp):
    from helfer.common import write_alert

    write_alert("Cache Drift erkannt", "sources.yaml Hash geändert", base=helfer_tmp)
    files = list((helfer_tmp / "alerts").glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "Cache Drift" in content
    assert "sources.yaml" in content


def test_write_status_overwrites(helfer_tmp):
    from helfer.common import write_status

    write_status("queue_runner", {"state": "idle", "jobs": 0}, base=helfer_tmp)
    write_status("queue_runner", {"state": "running", "jobs": 3}, base=helfer_tmp)
    path = helfer_tmp / "status" / "queue_runner.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["state"] == "running"
    assert data["jobs"] == 3


def test_read_events_returns_sorted(helfer_tmp):
    from helfer.common import write_event, read_events

    write_event("a_first", {"n": 1}, base=helfer_tmp)
    write_event("b_second", {"n": 2}, base=helfer_tmp)
    events = read_events(base=helfer_tmp)
    assert len(events) >= 2


def test_clear_old_events(helfer_tmp):
    from helfer.common import write_event, clear_old_events

    write_event("old_event", {"n": 1}, base=helfer_tmp)
    f = list((helfer_tmp / "events").glob("*.json"))[0]
    old_time = time.time() - 86400 * 8
    os.utime(f, (old_time, old_time))
    cleared = clear_old_events(max_age_days=7, base=helfer_tmp)
    assert cleared == 1
    assert len(list((helfer_tmp / "events").glob("*.json"))) == 0


def test_read_status_returns_data(helfer_tmp):
    from helfer.common import write_status, read_status

    write_status("test_service", {"state": "ok", "count": 42}, base=helfer_tmp)
    result = read_status("test_service", base=helfer_tmp)
    assert result is not None
    assert result["state"] == "ok"
    assert result["count"] == 42
    assert read_status("nonexistent", base=helfer_tmp) is None


def test_helfer_paths_from_yaml(tmp_path):
    from helfer.common import HelferConfig

    yaml_content = "strategy_builder: /tmp/strategy_builder\ntradingprojekt: /tmp/tradingprojekt\nhelfer_base: /tmp/test_helfer\n"
    config_file = tmp_path / "helfer.yaml"
    config_file.write_text(yaml_content)
    config = HelferConfig.from_yaml(config_file)
    assert config.strategy_builder == Path(
        "/tmp/strategy_builder"
    )
