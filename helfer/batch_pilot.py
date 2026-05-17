"""Batch Pilot – Vorflug-Check vor jedem Batch-Run.

Wird von queue_runner.sh aufgerufen. Exit 0 = GO, Exit 1 = NOGO.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from loguru import logger

from helfer.common import HelferConfig, write_event, write_alert, write_status

app = typer.Typer(help="Batch Pilot: Vorflug-Check vor Batch-Runs")


@dataclass
class PreflightResult:
    go: bool
    reason: str
    checks: list[str]


def preflight_check(sb_path: Path, idea_file: Path) -> PreflightResult:
    checks = []
    issues = []

    if not idea_file.exists():
        return PreflightResult(
            go=False, reason=f"Idee-Datei fehlt: {idea_file}", checks=[]
        )
    checks.append(f"Idee: {idea_file.name} ✓")

    meta_path = sb_path / "sb" / "cache" / "signal_shards" / ".cache_meta.json"
    if not meta_path.exists():
        issues.append("Cache-Meta fehlt – Cache nie gebaut?")
    else:
        meta = json.loads(meta_path.read_text())
        if meta.get("stale", False):
            issues.append(f"Cache STALE: {meta.get('stale_reason', 'unbekannt')}")
        else:
            checks.append("Cache: frisch ✓")

    shards_dir = sb_path / "sb" / "cache" / "signal_shards"
    if shards_dir.exists():
        shard_count = len(list(shards_dir.glob("*.parquet")))
        if shard_count == 0:
            # Cache ist frisch aber leer → wird beim ersten Run gebaut
            checks.append("Shards: 0 (wird beim Run gebaut) ✓")
        else:
            checks.append(f"Shards: {shard_count} ✓")

    if issues:
        return PreflightResult(go=False, reason="; ".join(issues), checks=checks)
    return PreflightResult(go=True, reason="Alle Checks bestanden", checks=checks)


@app.command()
def check(idea: Path = typer.Argument(..., help="Pfad zur Idee-Datei")):
    logger.add("/tmp/helfer_batch_pilot.log", rotation="10 MB", retention="7 days")
    config = HelferConfig.from_yaml()
    result = preflight_check(config.strategy_builder, idea)
    if result.go:
        logger.info(f"GO: {result.reason}")
        for c in result.checks:
            logger.info(f"  {c}")
        write_status("batch_pilot", {"state": "go", "idea": idea.name})
        sys.exit(0)
    else:
        logger.warning(f"NOGO: {result.reason}")
        write_event("batch_nogo", {"idea": idea.name, "reason": result.reason})
        write_alert("Batch NOGO", f"Idee: {idea.name}\nGrund: {result.reason}")
        write_status(
            "batch_pilot", {"state": "nogo", "idea": idea.name, "reason": result.reason}
        )
        sys.exit(1)


if __name__ == "__main__":
    app()
