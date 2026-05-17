"""
analyse.py – Strategie-Testmaschine
====================================
Analysiert builder.db: Welcher Baustein bringt Edge?
6 Phasen: Ablation, LASSO, Paar-Synergien, SHAP, MI, Overfitting-Check.

Usage: ./sb.py analyse -o output_v3
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from pathlib import Path
from dataclasses import dataclass
from scipy import stats as scipy_stats

from sb.engine.meta_learner import KNOWN_CONCEPTS, SESSIONS

ALL_FEATURES = KNOWN_CONCEPTS + SESSIONS


@dataclass
class BlockScore:
    """Ergebnis für einen einzelnen Baustein."""

    name: str
    avg_pf_with: float
    avg_pf_without: float
    delta: float
    count_with: int
    count_without: int
    ttest_pvalue: float
    lasso_coef: float = 0.0
    shap_importance: float = 0.0
    mutual_info: float = 0.0


@dataclass
class PairScore:
    """Synergie-Score für ein Baustein-Paar."""

    block_a: str
    block_b: str
    interaction_effect: float
    combined_avg_pf: float
    individual_sum: float
    synergy: float
    count: int


@dataclass
class AnalyseResult:
    """Gesamtergebnis der 6-Phasen-Analyse."""

    n_strategies: int
    tier_dist: dict[str, int]
    block_scores: list[BlockScore]
    pair_scores: list[PairScore]
    noise_blocks: list[str]
    overfitting_risk: float
    report_path: Path | None = None


# ── Feature-Matrix ──────────��───────────────────────────────────────────


def build_feature_matrix(
    registry: list[dict],
) -> tuple[pd.DataFrame, pd.Series]:
    """Baut binäre Feature-Matrix aus Registry.

    Fix #1: Wort-Tokenisierung statt Substring-Match (MB != MMXM).
    Fix #10: None-safe idea parsing.
    Fix #11: NaN/None-safe avg_oos_pf.
    """
    rows: list[dict[str, float]] = []
    targets: list[float] = []

    for run in registry:
        idea = str(run.get("idea") or "").upper()
        # Fix #1: Tokenisierung – "BOS + FVG NY" → {"BOS", "FVG", "NY"}
        tokens = set(re.findall(r"[A-Z][A-Z0-9_]*", idea))
        row = {}
        for feat in ALL_FEATURES:
            row[feat] = 1.0 if feat.upper() in tokens else 0.0
        rows.append(row)

        # Fix #11: NaN/None-safe target
        value = run.get("avg_oos_pf")
        try:
            value = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        targets.append(value)

    X = pd.DataFrame(rows, columns=ALL_FEATURES)
    y = pd.Series(targets, name="avg_oos_pf")

    # Spalten entfernen die nie vorkommen
    used_cols = X.columns[X.sum() > 0]
    X = X[used_cols]

    return X, y


# ── Phase 1: Ablation + t-Test ───────────���─────────────────────────────


def phase1_ablation(X: pd.DataFrame, y: pd.Series) -> list[BlockScore]:
    """Für jeden Baustein: PF MIT vs OHNE + Welch t-Test."""
    scores: list[BlockScore] = []

    for col in X.columns:
        mask = X[col] == 1
        pf_with = y[mask]
        pf_without = y[~mask]

        if len(pf_with) < 3 or len(pf_without) < 3:
            continue

        _, pval = scipy_stats.ttest_ind(pf_with, pf_without, equal_var=False)

        # Fix #12: NaN p-value guard
        if not np.isfinite(pval):
            pval = 1.0

        scores.append(
            BlockScore(
                name=col,
                avg_pf_with=float(pf_with.mean()),
                avg_pf_without=float(pf_without.mean()),
                delta=float(pf_with.mean() - pf_without.mean()),
                count_with=int(mask.sum()),
                count_without=int((~mask).sum()),
                ttest_pvalue=float(pval),
            )
        )

    scores.sort(key=lambda s: s.delta, reverse=True)
    return scores


# ── Phase 2: LASSO Regression ─────���────────────────────────────────────


def phase2_lasso(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], list[str]]:
    """LASSO: Noise-Bausteine werden auf Koeffizient 0 gedrückt.

    Fix #6: cv cap bei kleinen Datensätzen.
    Fix #7: Guard für leere Feature-Matrix.
    """
    if X.shape[1] == 0 or X.shape[0] < 5:
        return {}, list(X.columns)

    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fix #6: cv darf nicht größer als n_samples sein
    n_cv = min(5, X.shape[0])
    lasso = LassoCV(cv=n_cv, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)

    coefficients: dict[str, float] = {}
    noise_blocks: list[str] = []

    for name, coef in zip(X.columns, lasso.coef_):
        coefficients[name] = float(coef)
        if coef == 0.0:
            noise_blocks.append(name)

    return coefficients, noise_blocks


# ── Phase 3: Paar-Synergien ───────────────���───────────────────────────


def phase3_interactions(
    X: pd.DataFrame, y: pd.Series, top_n: int = 15
) -> list[PairScore]:
    """Berechnet Interaktionseffekte zwischen den Top-N Baustein-Paaren.

    Fix #8: Umbenannt von "ANOVA" zu "Paar-Synergien" (ist mean-basierte Heuristik).
    """
    freq_cols = [c for c in X.columns if X[c].sum() >= 10][:top_n]
    pairs: list[PairScore] = []

    for a, b in combinations(freq_cols, 2):
        both = (X[a] == 1) & (X[b] == 1)
        a_only = (X[a] == 1) & (X[b] == 0)
        b_only = (X[a] == 0) & (X[b] == 1)
        neither = (X[a] == 0) & (X[b] == 0)

        n_both = int(both.sum())
        if n_both < 5:
            continue

        pf_both = float(y[both].mean())
        pf_a = float(y[a_only].mean()) if a_only.sum() > 0 else float(y.mean())
        pf_b = float(y[b_only].mean()) if b_only.sum() > 0 else float(y.mean())
        pf_none = float(y[neither].mean()) if neither.sum() > 0 else float(y.mean())

        interaction = (pf_both - pf_a - pf_b + pf_none) / 2
        individual_sum = (pf_a - pf_none) + (pf_b - pf_none) + pf_none
        synergy = pf_both - individual_sum

        pairs.append(
            PairScore(
                block_a=a,
                block_b=b,
                interaction_effect=float(interaction),
                combined_avg_pf=pf_both,
                individual_sum=float(individual_sum),
                synergy=float(synergy),
                count=n_both,
            )
        )

    pairs.sort(key=lambda p: p.synergy, reverse=True)
    return pairs


# ── Phase 4: XGBoost + SHAP ──��──────────────────���─────────────────────


def phase4_shap(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """XGBoost + SHAP: nicht-linearer Beitrag jedes Bausteins.

    Fix #7: Guard für leere Matrix.
    Fix #9: Import-Fallback mit klarer Fehlermeldung.
    """
    if X.shape[1] == 0:
        return {}

    try:
        import shap
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "Phase 4 braucht 'shap' und 'xgboost': pip install shap xgboost"
        ) from exc

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance: dict[str, float] = {}
    for i, col in enumerate(X.columns):
        importance[col] = float(np.abs(shap_values[:, i]).mean())

    return importance


# ── Phase 5: Mutual Information ─────────────────────────────────��──────


def phase5_mutual_info(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Mutual Information: erkennt nicht-lineare Abhängigkeiten.

    Fix #7: Guard für leere Matrix.
    """
    if X.shape[1] == 0:
        return {}

    from sklearn.feature_selection import mutual_info_regression

    mi_scores = mutual_info_regression(
        X,
        y,
        discrete_features=True,
        random_state=42,  # type: ignore[arg-type]
    )
    return {col: float(score) for col, score in zip(X.columns, mi_scores)}


# ── Phase 6: Overfitting-Check (Bootstrap Multiple Testing) ────────────


def phase6_overfitting_check(
    X: pd.DataFrame, y: pd.Series, n_bootstrap: int = 1000
) -> float:
    """Bootstrap-basierter Overfitting-Test für Cross-Section von Strategien.

    Idee (White's Reality Check / Hansen's SPA):
    1. Nimm die beste Strategie (höchster PF).
    2. Bootstrappe die PF-Verteilung 1000× unter der Null-Hypothese
       (alle Bausteine sind gleich gut → PF-Labels werden geshufffled).
    3. Wie oft ist der Bootstrap-Max >= echtem Max?
       → Das ist die Overfitting-Wahrscheinlichkeit.

    Returns:
        overfitting_risk: 0.0 (echter Edge) bis 1.0 (pures Overfitting)
    """
    T = len(y)
    if T < 20:
        return 1.0

    rng = np.random.default_rng(42)
    observed_max = float(y.max())
    observed_best_mean = float(y.nlargest(10).mean())  # Top-10 Durchschnitt

    # Bootstrap: Shuffle PF-Labels, finde Max → Null-Verteilung
    bootstrap_maxes = np.empty(n_bootstrap)
    bootstrap_top10 = np.empty(n_bootstrap)
    y_arr = y.values.copy()

    for i in range(n_bootstrap):
        rng.shuffle(y_arr)
        bootstrap_maxes[i] = y_arr.max()
        bootstrap_top10[i] = np.sort(y_arr)[-10:].mean()

    # P-Wert: Wie oft ist Bootstrap-Max >= beobachteter Max?
    p_max = float(np.mean(bootstrap_maxes >= observed_max))
    p_top10 = float(np.mean(bootstrap_top10 >= observed_best_mean))

    # Kombinierter Risk-Score (Gewichtung: 60% Top-10, 40% Max)
    # Top-10 ist robuster als einzelner Max-Wert
    risk = 0.4 * p_max + 0.6 * p_top10

    return round(max(0.0, min(1.0, risk)), 4)


# ── Hauptfunktion ────────────────────────────────────────────��─────────


def run_analyse(registry: list[dict], output_dir: Path) -> AnalyseResult:
    """Führt alle 6 Phasen aus und gibt AnalyseResult zurück.

    Fix #7: Early exit bei leerer Feature-Matrix.
    """
    from rich.console import Console

    console = Console()

    X, y = build_feature_matrix(registry)

    # Fix #7: Guard für leere Matrix
    if X.shape[1] == 0:
        console.print("[red]Keine Bausteine erkannt – Feature-Matrix leer.[/red]")
        return AnalyseResult(
            n_strategies=len(registry),
            tier_dist={},
            block_scores=[],
            pair_scores=[],
            noise_blocks=[],
            overfitting_risk=1.0,
        )

    console.print(
        f"\n[bold]Strategie-Testmaschine[/bold]"
        f" – {len(registry)} Strategien, {X.shape[1]} Bausteine\n"
    )

    # Tier-Verteilung
    tier_dist: dict[str, int] = {}
    for run in registry:
        t = run.get("tier", "C")
        tier_dist[t] = tier_dist.get(t, 0) + 1

    # Phase 1
    console.print("[bold cyan]Phase 1:[/bold cyan] Ablation + t-Test...")
    block_scores = phase1_ablation(X, y)

    # Phase 2
    console.print("[bold cyan]Phase 2:[/bold cyan] LASSO Regression...")
    lasso_coefs, noise_blocks = phase2_lasso(X, y)
    for bs in block_scores:
        bs.lasso_coef = lasso_coefs.get(bs.name, 0.0)

    # Phase 3
    console.print("[bold cyan]Phase 3:[/bold cyan] Paar-Synergien...")
    pair_scores = phase3_interactions(X, y)

    # Phase 4
    console.print("[bold cyan]Phase 4:[/bold cyan] XGBoost + SHAP...")
    shap_scores = phase4_shap(X, y)
    for bs in block_scores:
        bs.shap_importance = shap_scores.get(bs.name, 0.0)

    # Phase 5
    console.print("[bold cyan]Phase 5:[/bold cyan] Mutual Information...")
    mi_scores = phase5_mutual_info(X, y)
    for bs in block_scores:
        bs.mutual_info = mi_scores.get(bs.name, 0.0)

    # Phase 6
    console.print("[bold cyan]Phase 6:[/bold cyan] Overfitting-Check...")
    overfitting_risk = phase6_overfitting_check(X, y)

    result = AnalyseResult(
        n_strategies=len(registry),
        tier_dist=tier_dist,
        block_scores=block_scores,
        pair_scores=pair_scores,
        noise_blocks=noise_blocks,
        overfitting_risk=overfitting_risk,
    )

    result.report_path = _save_report(result, output_dir)
    _print_results(result, console)

    return result


# ── Terminal-Output ────────────────────���───────────────────────────────


def _print_results(result: AnalyseResult, console) -> None:  # type: ignore[no-untyped-def]
    """Zeigt Ergebnisse als Rich-Tabellen."""
    from rich.table import Table

    console.print(
        f"\n[bold]Tier-Verteilung:[/bold]"
        f" A={result.tier_dist.get('A', 0)}"
        f" B={result.tier_dist.get('B', 0)}"
        f" C={result.tier_dist.get('C', 0)}"
    )
    console.print(f"[bold]Overfitting-Risiko:[/bold] {result.overfitting_risk:.1%}\n")

    # Baustein-Rangliste
    table = Table(title="Baustein-Rangliste (Top 20)")
    table.add_column("Baustein", style="bold")
    table.add_column("Delta PF", justify="right")
    table.add_column("p-Wert", justify="right")
    table.add_column("LASSO", justify="right")
    table.add_column("SHAP", justify="right")
    table.add_column("MI", justify="right")
    table.add_column("n", justify="right")
    table.add_column("Urteil", justify="center")

    for bs in result.block_scores[:20]:
        if bs.ttest_pvalue < 0.05 and bs.delta > 0 and bs.lasso_coef > 0:
            sig = "EDGE"
        elif bs.name in result.noise_blocks:
            sig = "NOISE"
        else:
            sig = "UNKLAR"

        # Fix #12: NaN p-value guard
        if not np.isfinite(bs.ttest_pvalue):
            pval_str = "n/a"
        elif bs.ttest_pvalue >= 0.001:
            pval_str = f"{bs.ttest_pvalue:.4f}"
        else:
            pval_str = "<0.001"

        table.add_row(
            bs.name,
            f"{bs.delta:+.4f}",
            pval_str,
            f"{bs.lasso_coef:+.4f}",
            f"{bs.shap_importance:.4f}",
            f"{bs.mutual_info:.4f}",
            str(bs.count_with),
            sig,
        )

    console.print(table)

    if result.noise_blocks:
        console.print(
            f"\n[bold red]Noise (LASSO=0):[/bold red] {', '.join(result.noise_blocks)}"
        )

    if result.pair_scores:
        pair_table = Table(title="Top 10 Paar-Synergien")
        pair_table.add_column("Paar", style="bold")
        pair_table.add_column("Synergie", justify="right")
        pair_table.add_column("Avg PF", justify="right")
        pair_table.add_column("n", justify="right")

        for ps in result.pair_scores[:10]:
            pair_table.add_row(
                f"{ps.block_a} + {ps.block_b}",
                f"{ps.synergy:+.4f}",
                f"{ps.combined_avg_pf:.3f}",
                str(ps.count),
            )
        console.print(pair_table)


# ── Markdown-Report ────────────��───────────────────────────────────���───


def _save_report(result: AnalyseResult, output_dir: Path) -> Path:
    """Speichert Analyse-Report als Markdown."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = output_dir / f"analyse_{ts}.md"

    lines = [
        f"# Strategie-Testmaschine – Analyse ({ts})",
        "",
        f"**Strategien:** {result.n_strategies}",
        f"**Tier:** A={result.tier_dist.get('A', 0)}"
        f" B={result.tier_dist.get('B', 0)}"
        f" C={result.tier_dist.get('C', 0)}",
        f"**Overfitting-Risiko:** {result.overfitting_risk:.1%}",
        "",
        "## Baustein-Rangliste",
        "",
        "| Baustein | Delta PF | p-Wert | LASSO | SHAP | MI | n | Urteil |",
        "|----------|----------|--------|-------|------|----|---|--------|",
    ]

    for bs in result.block_scores:
        if bs.ttest_pvalue < 0.05 and bs.delta > 0 and bs.lasso_coef > 0:
            sig = "EDGE"
        elif bs.name in result.noise_blocks:
            sig = "NOISE"
        else:
            sig = "UNKLAR"

        pval_str = (
            "n/a" if not np.isfinite(bs.ttest_pvalue) else f"{bs.ttest_pvalue:.4f}"
        )
        lines.append(
            f"| {bs.name} | {bs.delta:+.4f} | {pval_str} | "
            f"{bs.lasso_coef:+.4f} | {bs.shap_importance:.4f} | "
            f"{bs.mutual_info:.4f} | {bs.count_with} | {sig} |"
        )

    lines.extend(
        [
            "",
            "## Noise-Bausteine (LASSO=0)",
            "",
            ", ".join(result.noise_blocks) if result.noise_blocks else "Keine",
            "",
            "## Top 10 Paar-Synergien",
            "",
            "| Paar | Synergie | Avg PF | n |",
            "|------|----------|--------|---|",
        ]
    )

    for ps in result.pair_scores[:10]:
        lines.append(
            f"| {ps.block_a} + {ps.block_b} | {ps.synergy:+.4f}"
            f" | {ps.combined_avg_pf:.3f} | {ps.count} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
