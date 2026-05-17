from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from sb.algo_paths import DEFAULT_SIGNAL_ALGO_DIRS
from sb.engine.nautilus_bridge import NautilusBridge
from sb.engine.parser import parse_idea
from sb.engine.walk_forward import WalkForwardEngine
from sb.memory.db import BuilderDB
from sb.report import generate_wf_report

logger = logging.getLogger(__name__)


def _load_science_robustness() -> object | None:
    """Lädt science_robustness_tools_v2 dynamisch aus der Algo-Bibliothek."""
    for d in DEFAULT_SIGNAL_ALGO_DIRS:
        candidate = Path(d) / "science_robustness_tools_v2.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("_science_rt", candidate)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            try:
                import sys

                sys.modules[spec.name] = mod  # nötig für @dataclass in Python 3.12
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                return mod
            except Exception as exc:
                import sys

                sys.modules.pop(spec.name, None)
                logger.warning(
                    "Fehler beim Laden von science_robustness_tools_v2: %s", exc
                )
    return None


@dataclass
class WorkerConfig:
    idea: str
    trials: int
    data_path: Path
    cache_path: Path
    db_path: Path
    output_dir: Path
    min_trades: int
    algo_dirs: list[Path] | None = None
    max_date: str | None = None


def run_worker_task(cfg: WorkerConfig) -> dict[str, Any]:
    """
    Führt einen vollständigen Build-Run für eine Idee durch.
    Picklable – kann in ProcessPoolExecutor verwendet werden.
    Gibt dict zurück: {idea, status, tier, avg_pf, error}
    """
    # Duplikat-Check
    db = BuilderDB(db_path=cfg.db_path)
    try:
        prior = db.find_runs_by_idea(cfg.idea)
    finally:
        db.close()

    if prior:
        return {
            "idea": cfg.idea,
            "status": "skip",
            "tier": None,
            "avg_pf": None,
            "mc_pct_profitable": None,
            "error": None,
        }

    try:
        # Idee zuerst parsen → Concepts bestimmen → Bridge nur nötige Shards laden
        parsed = parse_idea(cfg.idea)
        bridge = NautilusBridge(
            data_path=cfg.data_path,
            cache_path=cfg.cache_path,
            concepts=parsed.concepts,
            algo_dirs=cfg.algo_dirs,
            max_date=cfg.max_date,
        )
        wfe = WalkForwardEngine(bridge=bridge)
        wf_result = wfe.run(parsed, cfg.trials)

        # Monte Carlo Robustheit
        mc_pct_profitable: float | None = None
        oos_pnl = wf_result.oos_pnl_series
        if oos_pnl:
            mc_mod = _load_science_robustness()
            if mc_mod is not None and hasattr(mc_mod, "compute_monte_carlo_trades"):
                try:
                    mc_summary = mc_mod.compute_monte_carlo_trades(  # type: ignore[union-attr]
                        np.asarray(oos_pnl, dtype=float)
                    )
                    raw_mc = getattr(mc_summary, "pct_profitable", None)
                    if raw_mc is not None:
                        mc_val = float(raw_mc)
                        if 0.0 <= mc_val <= 1.0:
                            mc_pct_profitable = mc_val
                except Exception as exc:
                    logger.warning("Monte Carlo fehlgeschlagen: %s", exc)

        db = BuilderDB(db_path=cfg.db_path)
        try:
            run_id = db.save_run(idea=cfg.idea, trials=cfg.trials)
            for i, w in enumerate(wf_result.windows):
                db.save_result(
                    run_id=run_id,
                    result=w.oos,
                    score=w.oos.profit_factor,
                    rank=i + 1,
                    warnings=[],
                )
            tier = db.compute_and_save_tier(
                run_id,
                pbo_score=wf_result.pbo_score,
                mc_pct_profitable=mc_pct_profitable,
            )
        finally:
            db.close()

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        generate_wf_report(
            idea=parsed,
            wf_result=wf_result,
            output_dir=cfg.output_dir,
            mc_pct_profitable=mc_pct_profitable,
        )

        return {
            "idea": cfg.idea,
            "status": "ok",
            "tier": tier,
            "avg_pf": wf_result.oos_pf,
            "mc_pct_profitable": mc_pct_profitable,
            "error": None,
        }

    except Exception as exc:
        return {
            "idea": cfg.idea,
            "status": "error",
            "tier": None,
            "avg_pf": None,
            "mc_pct_profitable": None,
            "error": str(exc),
        }
