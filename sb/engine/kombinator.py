from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import optuna

from sb.cache.concept_algo_map import CONCEPT_PREFERRED_ENTRY
from sb.models import BacktestResult, ParsedIdea

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


@runtime_checkable
class BacktestBackend(Protocol):
    def run(self, params: dict) -> BacktestResult: ...


ATR_MULT_LOW = 1.0  # war 0.8 — empirisch: unter 1.0 zu eng (Chat 414)
ATR_MULT_HIGH = 2.0  # war 2.5 — empirisch: über 2.0 kein Vorteil

TRAIL_ACT_LOW = 5.0  # Punkte Profit bevor Trail aktiviert
TRAIL_ACT_HIGH = 25.0
TRAIL_DIST_LOW = 5.0  # Trail-Abstand in Punkten
TRAIL_DIST_HIGH = 20.0
BREAKEVEN_RR_LOW = 0.8
BREAKEVEN_RR_HIGH = 1.5


def _build_search_space(
    idea: ParsedIdea,
) -> dict[str, optuna.distributions.BaseDistribution]:
    """Baut den Suchraum als Distribution-Dict (für ConfOptSampler)."""
    space: dict[str, optuna.distributions.BaseDistribution] = {}
    if idea.sl_hint_points is None:
        space["atr_mult"] = optuna.distributions.FloatDistribution(
            ATR_MULT_LOW, ATR_MULT_HIGH
        )

    # Regime-bewusster TP: engerer Bereich wenn Regime-Filter aktiv (Chat 414)
    has_regime = any(
        c in ("HRL_LRL", "STAT_REGIME", "PDA_REGIME") for c in idea.context
    )
    if has_regime:
        space["tp_mult"] = optuna.distributions.FloatDistribution(1.5, 4.0)
    else:
        space["tp_mult"] = optuna.distributions.FloatDistribution(1.5, 5.0)

    space["entry_bar_offset"] = optuna.distributions.IntDistribution(0, 5)

    # Entry-Typ-Auswahl wenn OB-Konzepte mit Preferred Entry vorhanden (Chat 414)
    has_ob = any(c in CONCEPT_PREFERRED_ENTRY for c in (idea.entry + idea.zone))
    if has_ob:
        space["entry_type"] = optuna.distributions.CategoricalDistribution(
            ["ft", "dp", "st50"]
        )

    em = idea.exit_mode
    if em == "fixed" and idea.use_trail:
        # Aktueller Punkt-Trail (Backward-Kompatibilität)
        space["trail_activation"] = optuna.distributions.FloatDistribution(
            TRAIL_ACT_LOW, TRAIL_ACT_HIGH
        )
        space["trail_distance"] = optuna.distributions.FloatDistribution(
            TRAIL_DIST_LOW, TRAIL_DIST_HIGH
        )
    elif em == "breakeven":
        space["breakeven_rr"] = optuna.distributions.FloatDistribution(
            BREAKEVEN_RR_LOW, BREAKEVEN_RR_HIGH
        )
    elif em == "breakeven_trail":
        space["breakeven_rr"] = optuna.distributions.FloatDistribution(
            BREAKEVEN_RR_LOW, BREAKEVEN_RR_HIGH
        )
        space["trail_activation"] = optuna.distributions.FloatDistribution(
            TRAIL_ACT_LOW, TRAIL_ACT_HIGH
        )
        space["trail_distance"] = optuna.distributions.FloatDistribution(
            TRAIL_DIST_LOW, TRAIL_DIST_HIGH
        )
    # atr_trail, next_zone, session_level: kein Extra-Suchraum (Spalten aus Cache)

    return space


def _make_sampler(
    storage: str | None,
    study_name: str | None,
    search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
    use_confopt: bool = False,
) -> optuna.samplers.BaseSampler:
    """Erstellt Sampler mit Fallback-Kette: ConfOpt → MetaLearnTPE → TPE.

    Fehler werden nie als Exception propagiert.
    """
    # 1. ConfOptSampler wenn gewünscht + search_space vorhanden
    if use_confopt and search_space:
        try:
            import optunahub  # type: ignore[import-untyped]

            module = optunahub.load_module("samplers/confopt_sampler")
            logger.info("ConfOptSampler geladen für '%s'", study_name)
            return module.ConfOptSampler(search_space=search_space)
        except ImportError:
            logger.debug("confopt nicht installiert – versuche MetaLearnTPE")
        except Exception as exc:
            logger.warning("ConfOptSampler Fehler, Fallback: %s", exc)

    # 2. MetaLearnTPE wenn Storage vorhanden
    if storage is not None:
        try:
            import optunahub  # type: ignore[import-untyped]

            module = optunahub.load_module("samplers/meta_learn_tpe")
            summaries = optuna.get_all_study_summaries(storage=storage)
            source_studies = []
            for s in summaries:
                if s.study_name == study_name:
                    continue
                if s.best_trial is None:
                    continue
                try:
                    source_studies.append(
                        optuna.load_study(study_name=s.study_name, storage=storage)
                    )
                except Exception as exc:
                    logger.debug(
                        "Study '%s' konnte nicht geladen werden: %s",
                        s.study_name,
                        exc,
                    )
            if source_studies:
                logger.info(
                    "MetaLearnTPE: %d source studies geladen für '%s'",
                    len(source_studies),
                    study_name,
                )
                return module.MetaLearnTPESampler(source_studies=source_studies)
        except ImportError:
            logger.debug("optunahub nicht installiert – verwende Standard-TPE")
        except Exception as exc:
            logger.warning("MetaLearnTPE Fehler, Fallback auf TPE: %s", exc)

    # 3. Standard-TPE als letzter Fallback
    return optuna.samplers.TPESampler()


# Konzept-Name → Cache-Prefix (lowercase, wie in cache_query._CONCEPT_PREFIXES)
_ZONE_TO_CACHE_KEY: dict[str, str] = {
    "OB_CHAOS": "ob_chaos",
    "OB_TAGESHOCH": "ob_tagh",
    "OB_SESSION": "ob_sess",
    "OB_SESSION_HTIEF": "s_ob_sess",
    "FVG_STANDARD": "fvg_std",
    "IFVG_1WOCHE": "ifvg_1w",
    "IFVG_SAMEDAY": "ifvg_sd",
    "FVG_2TAGE": "fvg_2t",
    "FVG_1_2WOCHEN": "fvg_12w",
}


def _apply_entry_type(entry_concepts: list[str], entry_type: str) -> list[str]:
    """Wandelt Zone-Konzepte in spezifische Entry-Konzepte um.

    z.B. OB_CHAOS + entry_type="ft" → "ob_chaos_ft" (Cache-Key)
    Konzepte die bereits einen Suffix haben bleiben unverändert.
    """
    result = []
    for c in entry_concepts:
        cache_key = _ZONE_TO_CACHE_KEY.get(c)
        if cache_key and not c.lower().endswith(("_ft", "_dp", "_st50")):
            result.append(f"{cache_key}_{entry_type}")
        else:
            result.append(c)
    return result


class Kombinator:
    def __init__(
        self,
        bridge: BacktestBackend,
        n_trials: int = 50,
        storage: str | None = None,
        study_name: str | None = None,
        use_confopt: bool = False,
    ) -> None:
        if n_trials <= 0:
            raise ValueError("n_trials must be > 0")
        self.bridge = bridge
        self.n_trials = n_trials
        self._storage = storage or None
        self._study_name = study_name or None
        self._use_confopt = use_confopt
        self.last_study: optuna.Study | None = None
        self.last_results: list[BacktestResult] = []

    def search(self, idea: ParsedIdea) -> list[BacktestResult]:
        results: list[BacktestResult] = []

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, idea)
            result = self.bridge.run(params)
            results.append(result)
            if result.num_trades < 5:
                return 0.0
            dd_penalty = result.max_drawdown / 100.0
            return result.profit_factor * (1 + result.winrate) - dd_penalty

        search_space = _build_search_space(idea)
        sampler = _make_sampler(
            self._storage,
            self._study_name,
            search_space=search_space,
            use_confopt=self._use_confopt,
        )
        study = optuna.create_study(
            direction="maximize",
            storage=self._storage,
            study_name=self._study_name,
            load_if_exists=True,
            sampler=sampler,
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.last_study = study
        self.last_results = results
        return results

    def _suggest_params(self, trial: optuna.Trial, idea: ParsedIdea) -> dict:

        # TP-Range aus Suchraum (regime-bewusst, siehe _build_search_space)
        has_regime = any(
            c in ("HRL_LRL", "STAT_REGIME", "PDA_REGIME") for c in (idea.context or [])
        )
        tp_hi = 4.0 if has_regime else 5.0
        tp_mult = trial.suggest_float("tp_mult", 1.5, tp_hi)
        offset = trial.suggest_int("entry_bar_offset", 0, 5)

        if idea.sl_hint_points is not None:
            # Fixer SL aus Forschung – ATR-Modus deaktiviert
            atr_mult = None
            sl_points = idea.sl_hint_points
        else:
            # Legacy: ATR-Modus (Optuna sucht besten Multiplikator)
            atr_mult = trial.suggest_float("atr_mult", ATR_MULT_LOW, ATR_MULT_HIGH)
            sl_points = 10.0

        em = idea.exit_mode
        trail_activation = 0.0
        trail_distance = 0.0
        breakeven_rr = 0.0

        if em == "fixed" and idea.use_trail:
            trail_activation = trial.suggest_float(
                "trail_activation", TRAIL_ACT_LOW, TRAIL_ACT_HIGH
            )
            trail_distance = trial.suggest_float(
                "trail_distance", TRAIL_DIST_LOW, TRAIL_DIST_HIGH
            )
        elif em == "breakeven":
            breakeven_rr = trial.suggest_float(
                "breakeven_rr", BREAKEVEN_RR_LOW, BREAKEVEN_RR_HIGH
            )
        elif em == "breakeven_trail":
            breakeven_rr = trial.suggest_float(
                "breakeven_rr", BREAKEVEN_RR_LOW, BREAKEVEN_RR_HIGH
            )
            trail_activation = trial.suggest_float(
                "trail_activation", TRAIL_ACT_LOW, TRAIL_ACT_HIGH
            )
            trail_distance = trial.suggest_float(
                "trail_distance", TRAIL_DIST_LOW, TRAIL_DIST_HIGH
            )

        entry = list(idea.entry or [])
        zone = list(idea.zone or [])
        context = list(idea.context or [])
        timing = list(idea.timing or [])

        # Entry-Typ-Auswahl: OB-Konzepte → Optuna wählt ft/dp/st50 (Chat 414)
        has_ob = any(c in CONCEPT_PREFERRED_ENTRY for c in entry)
        if has_ob:
            entry_type = trial.suggest_categorical("entry_type", ["ft", "dp", "st50"])
            entry = _apply_entry_type(entry, entry_type)

        return {
            "atr_mult": round(atr_mult, 2) if atr_mult is not None else None,
            "sl_points": sl_points,
            "tp_mult": round(tp_mult, 2),
            "entry_bar_offset": offset,
            "trail_activation": round(trail_activation, 1),
            "trail_distance": round(trail_distance, 1),
            "breakeven_rr": round(breakeven_rr, 2),
            "exit_mode": em,
            "session": idea.session,
            "concepts": idea.concepts,
            "entry": entry,
            "zone": zone,
            "context": context,
            "timing": timing,
            "direction": idea.direction if idea.direction != 0 else 1,
        }
