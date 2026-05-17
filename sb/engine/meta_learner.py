"""Meta-Learner: lernt aus Registry welche Konzept-Kombinationen Tier A/B werden.

Workflow:
  1. fit(registry)          – trainiert kalibrierten Random Forest
  2. suggest(n, explore)    – schlaegt n neue Ideen vor (MMR-Diversity + Exploration)
  3. feature_importance()   – zeigt welche Konzepte den groessten Einfluss haben
  4. save(path) / load(path)– Modell-Persistenz via joblib
"""

from __future__ import annotations

import itertools
import logging
import random
import re as _re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KNOWN_CONCEPTS: list[str] = [
    "BOS",
    "CHOCH",
    "MSS",
    "DISPLACEMENT",
    "FAILURE_SWING",
    "FVG",
    "OB",
    "MB",
    "IFVG",
    "BPR",
    "BREAKER",
    "RB",
    "VB",
    "PB",
    "VI",
    "FIRST_FVG",
    "OPENING_RANGE",
    "POST_FVG",
    "SWEEP",
    "ERL",
    "IRL",
    "CE",
    "REH",
    "REL",
    "OTE",
    "JUDAS",
    "SILVER_BULLET",
    "MMXM",
    "AMD",
    "MANIP",
    "CBDR",
    "CISD",
    "NDOG",
    "NWOG",
    "SMC_BOS",
    "HURST",
    "GARCH",
    "KALMAN",
    "REGIME",
    "CHANGEPOINT",
    "ENTROPY",
    "CUSUM",
    "SHANNON_CAPACITY",
    "PRIGOGINE",
    "KILLZONE",
    "MACRO_TIMING",
    "SESSION_TIMING",
    "NARRATIVE",
    "LIQUIDITY_BIAS",
    "PREMIUM",
    "DISCOUNT",
    "DEALING_RANGE",
    "CONSOLIDATION",
    "EXPANSION",
    "DRAWN_LIQUIDITY",
    "MARKTPHASE",
    "SESSION_BIAS",
]
KNOWN_CONCEPTS_UPPER = [c.upper() for c in KNOWN_CONCEPTS]
SESSIONS = ["NY", "LONDON", "ASIA"]


# Token-basiertes Matching verhindert dass BOS in SMC_BOS matcht
def _concept_tokens(text: str) -> set[str]:
    """Splittet einen Ideen-String in Tokens (getrennt durch +, Leerzeichen)."""
    return set(_re.split(r"[\s+]+", text.upper()))


# Feature-Namen in derselben Reihenfolge wie _encode_idea_rich()
_FEATURE_NAMES: list[str] = (
    KNOWN_CONCEPTS_UPPER
    + SESSIONS
    + ["n_concepts", "ctx_avg_pf", "ctx_avg_trades", "ctx_avg_mc"]
)


@dataclass
class SuggestionResult:
    """Ergebnis eines einzelnen Meta-Learner Vorschlags."""

    idea: str
    prob_ab: float  # kalibrierte Wahrscheinlichkeit fuer Tier A oder B
    uncertainty: float  # Std der Einzel-Baum-Predictions (Tree-Dispersion)
    novelty: float  # 1.0 = komplett neu, 0.0 = sehr aehnlich zur Registry
    band: str  # "auto_reject" | "human_review" | "auto_queue"
    top_feature: str  # Wichtigstes Konzept fuer diese Idee laut Feature Importance


class MetaLearner:
    """Kalibrierter Random-Forest-basierter Ideen-Generator mit MMR-Diversity."""

    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state
        self._model = None
        self._base_rf = None
        self._existing_ideas: set[str] = set()
        self._registry_stats: dict[str, dict] = {}
        self._feature_importances: list[tuple[str, float]] = []
        self.is_fitted = False

    def fit(self, registry: list[dict]) -> "MetaLearner":
        """Trainiert kalibrierten Random Forest auf Registry-Eintraegen."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier

        if len(registry) < 20:
            raise ValueError(
                f"MetaLearner benoetigt mindestens 20 Runs, hat aber {len(registry)}. "
                "Mehr Batches laufen lassen."
            )

        self._existing_ideas = {r["idea"].strip().lower() for r in registry}
        self._registry_stats = self._build_registry_stats(registry)

        X = [self._encode_idea_rich(r["idea"]) for r in registry]
        y = [1 if r.get("tier") in ("A", "B") else 0 for r in registry]

        # base_rf auf ALLEN Daten trainieren – fuer Feature Importance + Uncertainty
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=self._random_state,
            class_weight="balanced",
        )
        base_rf.fit(X, y)
        self._base_rf = base_rf

        # Kalibriertes Modell fuer P(Tier A/B) – separater Estimator um Fold-Artefakte zu vermeiden
        n_positive = sum(y)
        if n_positive >= 6:
            cv_folds: int | str = 3
            cal_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=self._random_state,
                class_weight="balanced",
            )
            self._model = CalibratedClassifierCV(cal_rf, method="sigmoid", cv=cv_folds)
        elif n_positive >= 4:
            cv_folds = 2
            cal_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=self._random_state,
                class_weight="balanced",
            )
            self._model = CalibratedClassifierCV(cal_rf, method="sigmoid", cv=cv_folds)
        else:
            # prefit: base_rf bereits auf allen Daten – direkt kalibrieren
            self._model = CalibratedClassifierCV(base_rf, method="sigmoid", cv="prefit")
        self._model.fit(X, y)

        importances = self._base_rf.feature_importances_
        self._feature_importances = sorted(
            zip(_FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
        )
        self.is_fitted = True

        logger.info(
            "MetaLearner trainiert: %d Runs, %d Tier-A/B (%.0f%%)",
            len(registry),
            n_positive,
            100 * n_positive / len(registry),
        )
        return self

    def suggest(
        self, n: int = 10, explore_ratio: float = 0.3
    ) -> list[SuggestionResult]:
        """Schlaegt n neue Ideen vor: (1-explore_ratio) Exploit (MMR) + explore_ratio Explore."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("MetaLearner.fit() muss zuerst aufgerufen werden.")

        n_explore = max(1, round(n * explore_ratio))
        n_exploit = n - n_explore

        candidates = self._generate_candidates_full()
        if not candidates:
            return []

        X_cand = [self._encode_idea_rich(c) for c in candidates]
        probs = self._model.predict_proba(X_cand)[:, 1]
        uncertainties = [self._compute_uncertainty(x) for x in X_cand]

        exploit_indices = [i for i, p in enumerate(probs) if p >= 0.40]
        exploit_cands = [candidates[i] for i in exploit_indices]
        exploit_probs = probs[exploit_indices] if exploit_indices else np.array([])

        exploit_ideas: list[str] = []
        if exploit_cands:
            sel_idx = self._mmr_select(exploit_cands, exploit_probs, n_exploit)
            exploit_ideas = [exploit_cands[i] for i in sel_idx]

        exploit_set = set(exploit_ideas)
        remaining = [c for c in candidates if c not in exploit_set]
        rng = random.Random(self._random_state + 1)
        explore_ideas = rng.sample(remaining, min(n_explore, len(remaining)))

        all_ideas = exploit_ideas + explore_ideas
        results: list[SuggestionResult] = []
        cand_idx = {c: i for i, c in enumerate(candidates)}

        for idea in all_ideas:
            idx = cand_idx.get(idea, -1)
            prob = float(probs[idx]) if idx >= 0 else 0.3
            unc = float(uncertainties[idx]) if idx >= 0 else 0.3
            novelty = self._compute_novelty(idea)
            band = self._classify_band(prob, unc)
            top_feat = self._top_feature_for(idea)
            results.append(
                SuggestionResult(
                    idea=idea,
                    prob_ab=round(prob, 4),
                    uncertainty=round(unc, 4),
                    novelty=round(novelty, 4),
                    band=band,
                    top_feature=top_feat,
                )
            )

        return results

    def feature_importance(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Gibt die top_n wichtigsten Features zurueck."""
        if not self._feature_importances:
            raise RuntimeError("Modell nicht trainiert.")
        return self._feature_importances[:top_n]

    def save(self, path: Path) -> None:
        """Speichert Modell-Zustand nach path (joblib)."""
        import joblib

        state = {
            "model": self._model,
            "base_rf": self._base_rf,
            "existing_ideas": self._existing_ideas,
            "registry_stats": self._registry_stats,
            "feature_importances": self._feature_importances,
            "is_fitted": self.is_fitted,
            "random_state": self._random_state,
        }
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, resolved)
        logger.info("MetaLearner gespeichert: %s", resolved)

    @classmethod
    def load(cls, path: Path) -> "MetaLearner":
        """Laedt Modell-Zustand von path (joblib)."""
        import joblib

        state = joblib.load(path)
        learner = cls(random_state=state["random_state"])
        learner._model = state["model"]
        learner._base_rf = state["base_rf"]
        learner._existing_ideas = state["existing_ideas"]
        learner._registry_stats = state["registry_stats"]
        learner._feature_importances = state["feature_importances"]
        learner.is_fitted = state["is_fitted"]
        return learner

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _build_registry_stats(self, registry: list[dict]) -> dict[str, dict]:
        """Berechnet Konzept-Statistiken aus der Registry."""
        stats: dict[str, dict] = {}
        for concept in KNOWN_CONCEPTS_UPPER:
            matching = [
                r for r in registry if concept in _concept_tokens(r.get("idea", ""))
            ]
            if matching:
                stats[concept] = {
                    "avg_pf": sum(r.get("avg_oos_pf") or 1.0 for r in matching)
                    / len(matching),
                    "avg_trades": sum(r.get("avg_trades") or 0 for r in matching)
                    / len(matching),
                    "avg_mc": sum(r.get("mc_pct_profitable") or 0.5 for r in matching)
                    / len(matching),
                }
            else:
                stats[concept] = {"avg_pf": 1.0, "avg_trades": 0.0, "avg_mc": 0.5}
        return stats

    def _encode_idea_rich(self, idea: str) -> list[float]:
        """Kodiert eine Idee als 64-dimensionalen Feature-Vektor."""
        tokens = _concept_tokens(idea)
        vec: list[float] = [1.0 if c in tokens else 0.0 for c in KNOWN_CONCEPTS_UPPER]
        for session in SESSIONS:
            vec.append(1.0 if session in tokens else 0.0)
        present = [c for c in KNOWN_CONCEPTS_UPPER if c in tokens]
        vec.append(float(min(len(present), 5)))

        if self._registry_stats and present:
            avg_pf = sum(self._registry_stats[c]["avg_pf"] for c in present) / len(
                present
            )
            avg_trades = sum(
                self._registry_stats[c]["avg_trades"] for c in present
            ) / len(present)
            avg_mc = sum(self._registry_stats[c]["avg_mc"] for c in present) / len(
                present
            )
        else:
            avg_pf, avg_trades, avg_mc = 1.0, 0.0, 0.5
        vec.extend([avg_pf, avg_trades / 100.0, avg_mc])
        return vec

    def _encode_idea(self, idea: str) -> list[float]:
        """Rueckwaertskompatibilitaet: delegiert an _encode_idea_rich."""
        return self._encode_idea_rich(idea)

    def _compute_uncertainty(self, x: list[float]) -> float:
        """Tree-Dispersion: Std der Einzel-Baum-Predictions."""
        if self._base_rf is None:
            return 0.3
        tree_probs = np.array(
            [tree.predict_proba([x])[0][1] for tree in self._base_rf.estimators_]
        )
        return float(np.std(tree_probs))

    def _compute_novelty(self, idea: str) -> float:
        """Novelty: 1 - max Jaccard-Aehnlichkeit zur Registry."""
        if not self._existing_ideas:
            return 1.0
        idea_concepts = set(
            c for c in KNOWN_CONCEPTS_UPPER if c in _concept_tokens(idea)
        )
        similarities = []
        for existing in self._existing_ideas:
            existing_concepts = set(
                c for c in KNOWN_CONCEPTS_UPPER if c in _concept_tokens(existing)
            )
            if not idea_concepts and not existing_concepts:
                similarities.append(1.0)
            elif not idea_concepts or not existing_concepts:
                similarities.append(0.0)
            else:
                inter = len(idea_concepts & existing_concepts)
                union = len(idea_concepts | existing_concepts)
                similarities.append(inter / union)
        return round(1.0 - max(similarities), 4) if similarities else 1.0

    def _classify_band(self, prob: float, uncertainty: float) -> str:
        """Ordnet einen Vorschlag einem der drei Baender zu."""
        if prob < 0.40:
            return "auto_reject"
        if prob >= 0.65 and uncertainty <= 0.25:
            return "auto_queue"
        return "human_review"

    def _top_feature_for(self, idea: str) -> str:
        """Gibt das wichtigste Konzept fuer diese Idee laut Feature Importance zurueck."""
        if not self._feature_importances:
            return "-"
        tokens = _concept_tokens(idea)
        for feat, imp in self._feature_importances:
            if feat in tokens and imp > 0:
                return f"{feat} ({imp:.3f})"
        return self._feature_importances[0][0] if self._feature_importances else "-"

    def _mmr_select(
        self, candidates: list[str], probs: np.ndarray, n: int, lambda_mmr: float = 0.7
    ) -> list[int]:
        """MMR: score = lambda * prob - (1-lambda) * max_sim_to_selected."""
        if not candidates:
            return []
        vecs = np.array([self._encode_idea_rich(c) for c in candidates], dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs_norm = vecs / norms

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(n, len(candidates))):
            if not remaining:
                break
            if not selected:
                best_idx = remaining[int(np.argmax(probs[remaining]))]
            else:
                sel_vecs = vecs_norm[selected]
                best_score = -999.0
                best_idx = remaining[0]
                for i in remaining:
                    sim = float(np.max(vecs_norm[i] @ sel_vecs.T))
                    score = lambda_mmr * float(probs[i]) - (1 - lambda_mmr) * sim
                    if score > best_score:
                        best_score = score
                        best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    def _generate_candidates_full(self, max_candidates: int = 3000) -> list[str]:
        """Generiert Kandidaten aus ALLEN bekannten Konzepten."""
        rng = random.Random(self._random_state)
        candidates: list[str] = []

        pairs = list(itertools.combinations(KNOWN_CONCEPTS_UPPER, 2))
        rng.shuffle(pairs)
        for c1, c2 in pairs:
            for session in SESSIONS:
                idea = f"{c1} + {c2} {session}"
                if idea.lower() not in self._existing_ideas:
                    candidates.append(idea)
            if len(candidates) >= max_candidates // 2:
                break

        triples = list(itertools.combinations(KNOWN_CONCEPTS_UPPER, 3))
        rng.shuffle(triples)
        for c1, c2, c3 in triples:
            for session in SESSIONS:
                idea = f"{c1} + {c2} + {c3} {session}"
                if idea.lower() not in self._existing_ideas:
                    candidates.append(idea)
            if len(candidates) >= max_candidates:
                break

        return candidates
