"""Shared Utilities für alle Helfer: Events, Alerts, Status, Config."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel

HELFER_BASE = Path.home() / ".helfer"
HELFER_CONFIG = HELFER_BASE / "config" / "helfer.yaml"


class HelferConfig(BaseModel):
    strategy_builder: Path
    tradingprojekt: Path
    helfer_base: Path = HELFER_BASE
    check_interval_seconds: int = 600
    max_event_age_days: int = 7

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "HelferConfig":
        p = path or HELFER_CONFIG
        if not p.exists():
            logger.warning(f"Config nicht gefunden: {p} – nutze Defaults")
            return cls(
                strategy_builder=Path.home() / "HAUPTLAGER/24_Strategie_Builder",
                tradingprojekt=Path.home()
                / "HAUPTLAGER/05_Strategien_Entwicklung/TRADINGPROJEKT",
            )
        data = yaml.safe_load(p.read_text())
        return cls(**data)


def write_event(
    event_type: str, data: dict[str, Any], *, base: Path = HELFER_BASE
) -> Path:
    ts = datetime.now(timezone.utc)
    payload = {"type": event_type, "timestamp": ts.isoformat(), **data}
    filename = f"{ts.strftime('%Y%m%d_%H%M%S')}_{event_type}.json"
    path = base / "events" / filename
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Event: {event_type} → {path.name}")
    return path


def read_events(
    *, base: Path = HELFER_BASE, event_type: str | None = None
) -> list[dict]:
    events_dir = base / "events"
    if not events_dir.exists():
        return []
    events = []
    for f in sorted(events_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if event_type and data.get("type") != event_type:
            continue
        events.append(data)
    return events


def clear_old_events(*, max_age_days: int = 7, base: Path = HELFER_BASE) -> int:
    events_dir = base / "events"
    if not events_dir.exists():
        return 0
    cutoff = time.time() - (max_age_days * 86400)
    cleared = 0
    for f in events_dir.glob("*.json"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            cleared += 1
    if cleared:
        logger.info(f"{cleared} alte Events gelöscht (>{max_age_days}d)")
    return cleared


def write_alert(title: str, body: str, *, base: Path = HELFER_BASE) -> Path:
    ts = datetime.now()
    filename = f"{ts.strftime('%Y-%m-%d_%H%M')}_{title.replace(' ', '_')[:40]}.md"
    content = f"# {title}\n\n**{ts.strftime('%Y-%m-%d %H:%M')}**\n\n{body}\n"
    path = base / "alerts" / filename
    path.write_text(content)
    logger.warning(f"Alert: {title}")
    return path


def write_status(name: str, data: dict[str, Any], *, base: Path = HELFER_BASE) -> Path:
    payload = {"name": name, "updated": datetime.now(timezone.utc).isoformat(), **data}
    path = base / "status" / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_status(name: str, *, base: Path = HELFER_BASE) -> dict | None:
    path = base / "status" / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
