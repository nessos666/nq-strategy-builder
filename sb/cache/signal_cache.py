from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sb.cache.concept_algo_map import get_algo_files_for_concepts, get_all_algo_files

logger = logging.getLogger(__name__)


def _algo_worker(queue: object, algo_file: Path, df: "pd.DataFrame") -> None:
    """Modul-Level Worker für multiprocessing.Process (spawn-safe)."""
    import multiprocessing  # noqa: PLC0415

    q: multiprocessing.Queue = queue  # type: ignore[assignment]
    try:
        result = SignalCache._run_algo(algo_file, df)
        q.put(("ok", result))
    except Exception as exc:
        q.put(("error", str(exc)))


@dataclass
class SignalCacheConfig:
    bars_path: Path
    cache_path: Path
    algo_dirs: list[Path] = field(default_factory=list)
    algo_pattern: str = "*.py"


class SignalCache:
    """Per-Algo Shard Cache.

    Statt einem großen Parquet mit allen Signalen gibt es pro Algo-Datei
    einen eigenen Shard: signal_shards/{algo_stem}.parquet

    Vorteile:
    - Neuer Algo → nur dieser Shard wird gebaut, alles andere bleibt
    - Idee braucht nur FVG + BOS → nur diese Shards geladen
    - Rebuild eines hängenden Algos tötet nicht alles andere

    Interface:
        build(concepts=None, force=False)  – baut fehlende/veraltete Shards
        load(concepts=None) -> pd.DataFrame  – lädt und mergt benötigte Shards
    """

    def __init__(self, cfg: SignalCacheConfig) -> None:
        self.cfg = SignalCacheConfig(
            bars_path=Path(cfg.bars_path),
            cache_path=Path(cfg.cache_path),
            algo_dirs=[Path(d) for d in cfg.algo_dirs],
            algo_pattern=cfg.algo_pattern,
        )

    # ── Öffentliches Interface ────────────────────────────────────────────────

    def build(self, concepts: list[str] | None = None, force: bool = False) -> None:
        """Baut Shards für die angegebenen Konzepte (oder alle PDA+SMC wenn None).

        Science-Algos werden nur gebaut wenn sie explizit in concepts vorkommen.
        """
        bars = self._load_bars()
        if bars is None or bars.empty:
            logger.error("Keine Bar-Daten gefunden: %s", self.cfg.bars_path)
            return

        algo_files = self._select_algo_files(concepts)
        if not algo_files:
            logger.warning("Keine passenden Algo-Dateien für Konzepte: %s", concepts)
            # Schreibe leeres Manifest damit is_valid() stimmt
            self._write_manifest([])
            return

        shard_dir = self._shard_dir()
        shard_dir.mkdir(parents=True, exist_ok=True)

        built = 0
        skipped = 0
        for algo_file in algo_files:
            if not force and self._is_shard_valid(algo_file):
                skipped += 1
                continue
            self._build_shard(algo_file, bars)
            built += 1

        self._write_manifest([f.stem for f in algo_files])
        if built:
            logger.info("Shards gebaut: %d neu, %d bereits aktuell", built, skipped)
        else:
            logger.info("Alle %d Shards bereits aktuell.", skipped)

    def load(self, concepts: list[str] | None = None) -> pd.DataFrame:
        """Lädt Shards für die Konzepte und gibt einen kombinierten DataFrame zurück.

        Der DataFrame enthält Signal-Spalten (bool + float), keine Bar-OHLCV-Daten.
        Index = Timestamps (gleich wie Bar-Daten).
        Wenn keine Shards vorhanden, wird ein leerer DataFrame zurückgegeben.
        """
        algo_files = self._select_algo_files(concepts)
        shard_dir = self._shard_dir()

        frames: list[pd.DataFrame] = []
        for algo_file in algo_files:
            shard_path = shard_dir / f"{algo_file.stem}.parquet"
            if not shard_path.exists():
                logger.debug("Shard nicht vorhanden, übersprungen: %s", algo_file.stem)
                continue
            try:
                frames.append(pd.read_parquet(shard_path))
            except Exception as exc:
                logger.warning("Shard laden fehlgeschlagen %s: %s", algo_file.stem, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        # Doppelte Spalten entfernen (z.B. fvg_bullish aus mehreren FVG-Algos)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    def is_valid(self, concepts: list[str] | None = None) -> bool:
        """True wenn alle benötigten Shards aktuell sind."""
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return False
        algo_files = self._select_algo_files(concepts)
        return all(self._is_shard_valid(f) for f in algo_files)

    # ── Private Hilfsmethoden ────────────────────────────────────────────────

    def _shard_dir(self) -> Path:
        return self.cfg.cache_path.parent / "signal_shards"

    def _manifest_path(self) -> Path:
        return self.cfg.cache_path.parent / "signal_shards.manifest.json"

    def _shard_meta_path(self, algo_file: Path) -> Path:
        return self._shard_dir() / f"{algo_file.stem}.meta.json"

    def _is_shard_valid(self, algo_file: Path) -> bool:
        shard_path = self._shard_dir() / f"{algo_file.stem}.parquet"
        meta_path = self._shard_meta_path(algo_file)
        if not shard_path.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
            bars_ok = meta.get("bars_mtime_ns") == self.cfg.bars_path.stat().st_mtime_ns
            algo_ok = meta.get("algo_mtime_ns") == algo_file.stat().st_mtime_ns
            return bars_ok and algo_ok
        except Exception:
            return False

    _ALGO_TIMEOUT_SECONDS = (
        60  # 60s – Science-Algos (Hurst/GARCH) killen bevor sie hängen
    )

    def _build_shard(self, algo_file: Path, bars: pd.DataFrame) -> None:
        import multiprocessing as _mp

        ctx = _mp.get_context("fork")
        queue: _mp.Queue = ctx.Queue()  # type: ignore[type-arg]
        proc = ctx.Process(target=_algo_worker, args=(queue, algo_file, bars))
        try:
            proc.start()
            # Queue ZUERST drainieren, DANN joinen – verhindert Deadlock wenn
            # sig_df groß ist und die Pipe-Buffer des OS überläuft.
            try:
                status, data = queue.get(timeout=self._ALGO_TIMEOUT_SECONDS)
            except Exception:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=5)
                logger.warning(
                    "Algo %s überschritt %ds Timeout – leerer Shard gespeichert.",
                    algo_file.name,
                    self._ALGO_TIMEOUT_SECONDS,
                )
                # Leeren Shard + Meta schreiben damit nächster Run ihn überspringt
                shard_path = self._shard_dir() / f"{algo_file.stem}.parquet"
                pd.DataFrame(index=bars.index).to_parquet(shard_path)
                meta = {
                    "bars_mtime_ns": self.cfg.bars_path.stat().st_mtime_ns,
                    "algo_mtime_ns": algo_file.stat().st_mtime_ns,
                    "num_cols": 0,
                    "cols": [],
                    "timed_out": True,
                }
                self._shard_meta_path(algo_file).write_text(json.dumps(meta))
                return
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join()
            if status == "error":
                raise RuntimeError(data)
            sig_df = data
            _OHLCV = {"Open", "High", "Low", "Close", "Volume"}
            bool_cols = [
                c
                for c in sig_df.columns
                if c not in bars.columns and sig_df[c].dtype == bool
            ]
            float_cols = [
                c
                for c in sig_df.columns
                if c not in bars.columns
                and c not in _OHLCV
                and sig_df[c].dtype in ("float64", "float32")
            ]
            keep_cols = bool_cols + float_cols
            if not keep_cols:
                logger.debug(
                    "Algo %s lieferte keine Signal-Spalten – Shard leer.",
                    algo_file.name,
                )
                # Leeren Shard schreiben damit Meta gesetzt wird
                shard = pd.DataFrame(index=bars.index)
            else:
                shard = sig_df[keep_cols]

            shard_path = self._shard_dir() / f"{algo_file.stem}.parquet"
            shard.to_parquet(shard_path)

            meta = {
                "bars_mtime_ns": self.cfg.bars_path.stat().st_mtime_ns,
                "algo_mtime_ns": algo_file.stat().st_mtime_ns,
                "num_cols": len(keep_cols),
                "cols": keep_cols,
            }
            self._shard_meta_path(algo_file).write_text(json.dumps(meta))
            logger.debug(
                "Shard gebaut: %s (%d Spalten)", algo_file.name, len(bool_cols)
            )
        except Exception as exc:
            logger.warning("Algo %s fehlgeschlagen: %s", algo_file.name, exc)

    def _write_manifest(self, algo_stems: list[str]) -> None:
        manifest = {
            "bars_path": str(self.cfg.bars_path),
            "bars_mtime_ns": self.cfg.bars_path.stat().st_mtime_ns
            if self.cfg.bars_path.exists()
            else 0,
            "algo_stems": algo_stems,
        }
        self._manifest_path().write_text(json.dumps(manifest))
        # Manifest-Pfad auch als cache_path schreiben (Backward-Compat für Tests)
        cache_path = Path(self.cfg.cache_path)
        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"sharded": True}))

    def _select_algo_files(self, concepts: list[str] | None) -> list[Path]:
        """Wählt Algo-Dateien basierend auf Konzepten aus.

        concepts=None → alle PDA+SMC Algos (KEINE Science!)
        concepts=["FVG", "BOS"] → nur FVG + BOS Algos (kann Science enthalten)
        """
        if concepts is None:
            return get_all_algo_files(
                self.cfg.algo_dirs,
                self.cfg.algo_pattern,
                include_science=False,
            )
        return get_algo_files_for_concepts(
            concepts,
            self.cfg.algo_dirs,
            self.cfg.algo_pattern,
        )

    def _load_bars(self) -> pd.DataFrame | None:
        if not self.cfg.bars_path.exists():
            return None
        try:
            df = pd.read_parquet(self.cfg.bars_path)
            rename = {
                c: c.capitalize()
                for c in df.columns
                if c.lower() in {"open", "high", "low", "close", "volume"}
            }
            df = df.rename(columns=rename)
            return df
        except Exception as exc:
            logger.error("Bar-Daten konnten nicht geladen werden: %s", exc)
            return None

    @staticmethod
    def _run_algo(algo_file: Path, df: pd.DataFrame) -> pd.DataFrame:
        """Lädt Algo-Modul dynamisch und ruft run() / compute_*() auf."""
        import sys  # noqa: PLC0415

        algo_dir = str(algo_file.parent)
        nq_root = str(algo_file.parent.parent.parent)
        added: list[str] = []
        for p in (algo_dir, nq_root):
            if p not in sys.path:
                sys.path.insert(0, p)
                added.append(p)
        try:
            spec = importlib.util.spec_from_file_location(algo_file.stem, algo_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Kann Algo nicht laden: {algo_file}")
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = ""
            sys.modules[algo_file.stem] = mod
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            finally:
                sys.modules.pop(algo_file.stem, None)
            if hasattr(mod, "run"):
                return mod.run(df)  # type: ignore[no-any-return]
            for attr_name in sorted(dir(mod)):
                if attr_name.startswith("compute_"):
                    fn = getattr(mod, attr_name)
                    if callable(fn):
                        try:
                            return fn(df)  # type: ignore[no-any-return]
                        except TypeError:
                            continue
            raise AttributeError(f"Kein run() oder compute_*() in {algo_file.name}")
        finally:
            for p in added:
                try:
                    sys.path.remove(p)
                except ValueError:
                    pass
