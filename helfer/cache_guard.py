"""Cache Guard – Echtzeit-Invalidierung via Filesystem-Watcher.

Überwacht sources.yaml und david_bibliothek/*.py.
Bei Änderung: Meta-Hash als stale markieren + Event schreiben.
"""

import json
from pathlib import Path

import typer
from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from helfer.common import (
    HelferConfig,
    write_event,
    write_alert,
    write_status,
    HELFER_BASE,
)

app = typer.Typer(help="Cache Guard: Echtzeit Cache-Invalidierung")

WATCH_EXTENSIONS = {".py", ".yaml", ".yml"}
SOURCES_NAMES = {"sources.yaml", "concept_algo_map.py", "cache_query.py"}


class CacheGuardHandler(FileSystemEventHandler):
    def __init__(self, sb_path: Path, helfer_base: Path = HELFER_BASE):
        self.sb_path = sb_path
        self.helfer_base = helfer_base

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in WATCH_EXTENSIONS:
            return
        if path.name in SOURCES_NAMES:
            logger.warning(f"Kritische Datei geändert: {path.name}")
            self._invalidate_cache(f"{path.name} geändert")
            write_event(
                "cache_invalidated",
                {
                    "trigger": path.name,
                    "path": str(path),
                },
                base=self.helfer_base,
            )
        elif path.suffix == ".py" and "bibliothek" in str(path).lower():
            logger.info(f"Algo geändert: {path.name}")
            write_event(
                "algo_changed",
                {
                    "algo": path.name,
                    "path": str(path),
                },
                base=self.helfer_base,
            )

    def _invalidate_cache(self, reason: str):
        meta_path = self.sb_path / "sb" / "cache" / "signal_shards" / ".cache_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["stale"] = True
            meta["stale_reason"] = reason
            meta_path.write_text(json.dumps(meta, indent=2))
            logger.warning(f"Cache als stale markiert: {reason}")
        write_alert("Cache invalidiert", f"Grund: {reason}", base=self.helfer_base)


@app.command()
def run():
    logger.add("/tmp/helfer_cache_guard.log", rotation="10 MB", retention="7 days")
    config = HelferConfig.from_yaml()
    sb = config.strategy_builder
    handler = CacheGuardHandler(sb_path=sb)
    observer = Observer()
    watch_paths = [sb / "knowledge_sources", sb / "sb" / "cache", sb / "sb"]
    bib = sb.parent / "david_bibliothek"
    if bib.exists():
        watch_paths.append(bib)
    for wp in watch_paths:
        if wp.exists():
            observer.schedule(handler, str(wp), recursive=True)
            logger.info(f"Überwache: {wp}")
    write_status("cache_guard", {"state": "running", "watching": len(watch_paths)})
    logger.info("Cache Guard gestartet")
    observer.start()
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    app()
