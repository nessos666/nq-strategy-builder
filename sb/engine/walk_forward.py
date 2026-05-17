from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances

from sb.algo_paths import DEFAULT_SIGNAL_ALGO_DIRS
from sb.engine.kombinator import Kombinator
from sb.engine.nautilus_bridge import NautilusBridge
from sb.models import BacktestResult, ParsedIdea, WalkForwardResult, WindowResult

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

logger = logging.getLogger(__name__)

N_WINDOWS = 3
IN_SAMPLE_RATIO = 0.75


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


def _compute_window_pbo(trial_results: list[BacktestResult]) -> float:
    """PBO-Score für ein IS-Fenster aus Trial-Ergebnissen (0=kein Overfitting, 1=komplett)."""
    mod = _load_science_robustness()
    if mod is None:
        return float("nan")
    series = [
        np.asarray(r.pnl_series, dtype=float)
        for r in trial_results
        if len(r.pnl_series) >= 4
    ]
    if len(series) < 2:
        return float("nan")
    min_len = min(len(s) for s in series)
    truncated = [s[:min_len] for s in series]
    try:
        estimate = mod.compute_pbo(truncated)  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("PBO-Berechnung fehlgeschlagen: %s", exc)
        return float("nan")
    pbo = float(estimate.pbo)
    return pbo if math.isfinite(pbo) else float("nan")


def _average_importances(all_importances: list[dict[str, float]]) -> dict[str, float]:
    """Mittelt PED-ANOVA Wichtigkeiten über mehrere Fenster."""
    valid_importances = [d for d in all_importances if d]
    if not valid_importances:
        return {}
    keys: set[str] = set()
    for d in valid_importances:
        keys.update(d.keys())
    result: dict[str, float] = {}
    for k in keys:
        vals = [d.get(k, 0.0) for d in valid_importances]
        result[k] = round(sum(vals) / len(vals), 4)
    return result


def _idea_hash(idea: ParsedIdea) -> str:
    """Stabiler 8-Zeichen Hash der Idee für Study-Namen."""

    key = (
        f"{sorted(idea.concepts)}_{idea.session}_{idea.sl_hint_points}_{idea.use_trail}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _best_trial_params(study: object | None, idea: ParsedIdea) -> dict[str, object]:
    """Liest den besten Trial defensiv aus, auch wenn kein Trial abgeschlossen wurde."""
    if study is None:
        return {}
    try:
        best_trial = study.best_trial  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, ValueError):
        return {}
    params = getattr(best_trial, "params", None)
    if not isinstance(params, dict) or not params:
        return {}

    # Entry-Typ aus IS-Optimierung übernehmen (z.B. "ft"/"dp"/"st50" für OB-Konzepte).
    # Ohne diesen Schritt nutzt OOS immer idea.entry (z.B. OB_TAGESHOCH → ob_bull/ob_bear),
    # die nur bei UTC 00:00 feuern und nie in der NY-Session → 0 Trades → PF=None (B93).
    entry = list(idea.entry)
    entry_type = params.get("entry_type")
    if entry_type and isinstance(entry_type, str):
        from sb.engine.kombinator import _apply_entry_type, CONCEPT_PREFERRED_ENTRY  # noqa: PLC0415

        if any(c in CONCEPT_PREFERRED_ENTRY for c in entry):
            entry = _apply_entry_type(entry, entry_type)

    return {
        "sl_points": idea.sl_hint_points
        if idea.sl_hint_points is not None
        else params.get("sl_points", 10.0),
        "tp_mult": params.get("tp_mult", 2.5),
        "entry_bar_offset": params.get("entry_bar_offset", 0),
        "trail_activation": params.get("trail_activation", 0.0),
        "trail_distance": params.get("trail_distance", 0.0),
        "breakeven_rr": params.get("breakeven_rr", 0.0),
        "exit_mode": idea.exit_mode,
        "session": idea.session,
        "concepts": idea.concepts,
        "entry": entry,
        "zone": idea.zone,
        "context": idea.context,
        "timing": idea.timing,
        "direction": 1,
    }


class WalkForwardEngine:
    """Führt Rolling Walk-Forward mit N Fenstern durch.

    Für jedes Fenster:
    - Optimiert Params auf In-Sample-Bars (Optuna)
    - Evaluiert beste Params auf Out-of-Sample-Bars
    - Berechnet PED-ANOVA Parameter-Wichtigkeit

    Konstanten sind im Code fixiert – kein manuelles Justieren nötig.
    """

    def __init__(
        self,
        bridge: NautilusBridge,
        n_windows: int = N_WINDOWS,
        in_sample_ratio: float = IN_SAMPLE_RATIO,
        storage: str | None = None,
    ) -> None:
        if n_windows <= 0:
            raise ValueError("n_windows must be > 0")
        if not 0.0 < in_sample_ratio < 1.0:
            raise ValueError("in_sample_ratio must be between 0 and 1")
        self._bridge = bridge
        self._n_windows = n_windows
        self._in_sample_ratio = in_sample_ratio
        self._storage = storage

    def run(self, idea: ParsedIdea, n_trials: int) -> WalkForwardResult:
        """Führt Walk-Forward durch und gibt aggregiertes Ergebnis zurück."""
        all_bars: list[Bar] = self._bridge._bars  # type: ignore[assignment]
        cache_df: pd.DataFrame | None = self._bridge._cache_df
        instrument = self._bridge._instrument
        bar_type = self._bridge._bar_type
        venue = self._bridge._venue

        windows = self._split_windows(all_bars)
        if not windows:
            logger.warning(
                "Walk-Forward übersprungen: zu wenige Bars für valide Fenster"
            )
        window_results: list[WindowResult] = []
        all_importances: list[dict[str, float]] = []
        pbo_scores: list[float] = []

        for i, (is_bars, oos_bars) in enumerate(windows):
            logger.info(
                "Walk-Forward Fenster %d/%d (%d IS-Bars, %d OOS-Bars)",
                i + 1,
                len(windows),
                len(is_bars),
                len(oos_bars),
            )

            # In-Sample Bridge + Optimierung
            is_bridge = NautilusBridge.from_bars(
                is_bars, cache_df, instrument, bar_type, venue
            )
            idea_hash = _idea_hash(idea)
            study_name = f"{idea_hash}_w{i}"
            kom = Kombinator(
                is_bridge,
                n_trials,
                storage=self._storage,
                study_name=study_name,
                use_confopt=True,
            )
            kom.search(idea)

            # PBO für dieses IS-Fenster
            pbo_scores.append(_compute_window_pbo(kom.last_results))

            # PED-ANOVA für dieses Fenster
            importances: dict[str, float] = {}
            if kom.last_study is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ExperimentalWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        raw = get_param_importances(
                            kom.last_study,
                            evaluator=PedAnovaImportanceEvaluator(),
                        )
                    importances = dict(raw)
                except Exception as exc:
                    logger.warning("PED-ANOVA Fehler Fenster %d: %s", i, exc)
            all_importances.append(importances)

            # Beste Params aus Optuna-Study
            best_params = _best_trial_params(kom.last_study, idea)

            # In-Sample Ergebnis mit besten Params
            is_result: BacktestResult = (
                is_bridge.run(best_params)
                if best_params
                else _empty_result(best_params)
            )

            # Out-of-Sample Ergebnis
            oos_bridge = NautilusBridge.from_bars(
                oos_bars, cache_df, instrument, bar_type, venue
            )
            oos_result: BacktestResult = (
                oos_bridge.run(best_params)
                if best_params
                else _empty_result(best_params)
            )

            window_results.append(
                WindowResult(
                    window_idx=i,
                    in_sample=is_result,
                    oos=oos_result,
                    best_params=best_params,
                )
            )

        avg_importances = _average_importances(all_importances)
        valid_pbos = [p for p in pbo_scores if math.isfinite(p)]
        avg_pbo = sum(valid_pbos) / len(valid_pbos) if valid_pbos else float("nan")
        return WalkForwardResult(
            windows=window_results, importances=avg_importances, pbo_score=avg_pbo
        )

    def _split_windows(self, bars: list) -> list[tuple[list, list]]:
        """Teilt Bars in N rollende IS/OOS-Fenster auf.

        Beispiel mit 1000 Bars, N=3, ratio=0.75:
        - OOS-Größe: 1000 * 0.25 / 3 = 83 Bars pro Fenster
        - IS-Größe:  1000 * 0.75 = 750 Bars
        - Fenster 0: IS=[0:750],   OOS=[750:833]
        - Fenster 1: IS=[83:833],  OOS=[833:916]
        - Fenster 2: IS=[166:916], OOS=[916:999]
        """
        n = len(bars)
        oos_size = int(n * (1.0 - self._in_sample_ratio) / self._n_windows)
        is_size = int(n * self._in_sample_ratio)
        if oos_size <= 0 or is_size <= 0:
            return []
        windows: list[tuple[list, list]] = []
        for i in range(self._n_windows):
            start = i * oos_size
            is_end = start + is_size
            oos_end = is_end + oos_size
            if oos_end > n:
                break
            windows.append((bars[start:is_end], bars[is_end:oos_end]))
        return windows


def _empty_result(params: dict) -> BacktestResult:
    return BacktestResult(
        params=params, gross_profit=0.0, gross_loss=0.0, num_trades=0, num_wins=0
    )
