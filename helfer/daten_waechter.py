"""Daten-Wächter – Prüft Cache-Freshness und Parquet-Integrität.

Läuft als systemd Timer (alle 10 Minuten).
Bei Problemen: Event + Alert schreiben.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import typer
from loguru import logger

from helfer.common import HelferConfig, write_event, write_alert, write_status

app = typer.Typer(help="Daten-Wächter: Cache & Parquet Qualitäts-Check")


@dataclass
class CacheCheck:
    is_fresh: bool
    reason: str


@dataclass
class ParquetCheck:
    file: Path
    is_valid: bool
    reason: str


def _sources_hash(sb_path: Path) -> str:
    sources = sb_path / "knowledge_sources" / "sources.yaml"
    if not sources.exists():
        return "missing"
    return hashlib.md5(sources.read_bytes()).hexdigest()


def check_cache_freshness(sb_path: Path) -> CacheCheck:
    meta_path = sb_path / "sb" / "cache" / "signal_shards" / ".cache_meta.json"
    if not meta_path.exists():
        return CacheCheck(is_fresh=False, reason="Meta-Datei fehlt – Cache nie gebaut?")
    meta = json.loads(meta_path.read_text())
    cached_hash = meta.get("sources_hash", "")
    current_hash = _sources_hash(sb_path)
    if cached_hash != current_hash:
        return CacheCheck(
            is_fresh=False,
            reason=f"Sources-Hash geändert: cached={cached_hash[:8]} vs current={current_hash[:8]}",
        )
    return CacheCheck(is_fresh=True, reason="ok")


def check_parquet_integrity(shards_dir: Path) -> list[ParquetCheck]:
    results = []
    for pq_file in sorted(shards_dir.glob("*.parquet")):
        try:
            table = pq.read_table(pq_file)
            if table.num_rows == 0:
                results.append(ParquetCheck(pq_file, False, "Leer (0 Rows)"))
            else:
                results.append(ParquetCheck(pq_file, True, f"{table.num_rows} rows"))
        except Exception as e:
            results.append(ParquetCheck(pq_file, False, f"Korrupt: {e}"))
    return results


@app.command()
def run(once: bool = typer.Option(False, help="Nur einmal prüfen, nicht als Daemon")):
    logger.add("/tmp/helfer_daten_waechter.log", rotation="10 MB", retention="7 days")
    config = HelferConfig.from_yaml()
    sb = config.strategy_builder
    shards_dir = sb / "sb" / "cache" / "signal_shards"
    logger.info("Daten-Wächter Check gestartet")
    issues = []

    cache = check_cache_freshness(sb)
    if not cache.is_fresh:
        logger.warning(f"Cache STALE: {cache.reason}")
        write_event("cache_stale", {"reason": cache.reason}, base=config.helfer_base)
        issues.append(f"Cache: {cache.reason}")

    if shards_dir.exists():
        checks = check_parquet_integrity(shards_dir)
        bad = [c for c in checks if not c.is_valid]
        if bad:
            for b in bad:
                logger.warning(f"Parquet kaputt: {b.file.name} – {b.reason}")
            write_event(
                "parquet_corrupt",
                {"files": [{"name": b.file.name, "reason": b.reason} for b in bad]},
                base=config.helfer_base,
            )
            issues.append(f"Parquet: {len(bad)} korrupte Dateien")

    write_status(
        "daten_waechter",
        {
            "state": "issues" if issues else "ok",
            "issues": issues,
            "shards_count": len(list(shards_dir.glob("*.parquet")))
            if shards_dir.exists()
            else 0,
        },
        base=config.helfer_base,
    )

    if issues:
        write_alert(
            "Daten-Wächter: Probleme erkannt",
            "\n".join(f"- {i}" for i in issues),
            base=config.helfer_base,
        )
        logger.warning(f"Daten-Wächter: {len(issues)} Probleme")
    else:
        logger.info("Daten-Wächter: Alles OK")


if __name__ == "__main__":
    app()
