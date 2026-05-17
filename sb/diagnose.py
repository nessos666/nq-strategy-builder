from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Modul 1: Regime-Shift ────────────────────────────────────────────────────


def _compute_atr(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def _compute_adx(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    """Vereinfachter ADX (nur DX, kein echtes +DI/-DI Smoothing)."""
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    dm_plus = (df["high"] - df["high"].shift(1)).clip(lower=0)
    dm_minus = (df["low"].shift(1) - df["low"]).clip(lower=0)
    atr = _compute_atr(df, window)
    di_plus = (
        100 * dm_plus.rolling(window, min_periods=1).mean() / atr.replace(0, np.nan)
    )
    di_minus = (
        100 * dm_minus.rolling(window, min_periods=1).mean() / atr.replace(0, np.nan)
    )
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.rolling(window, min_periods=1).mean().fillna(0)


def analyse_regime_shift(ohlcv: pd.DataFrame) -> dict[str, Any]:
    """Vergleicht Marktregime quartalsweise: ATR, ADX.

    Returns:
        {
          "quarters": [{"period": "2024Q1", "atr_mean": ..., "atr_std": ...,
                        "adx_mean": ..., "n_bars": ...}, ...],
          "shift_detected": bool,
          "shift_details": str,
        }
    """
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]

    atr = _compute_atr(df)
    adx = _compute_adx(df)

    quarters: list[dict[str, Any]] = []
    for period, grp in atr.groupby(atr.index.to_period("Q")):
        adx_grp = adx.loc[grp.index]
        quarters.append(
            {
                "period": str(period),
                "atr_mean": round(float(grp.mean()), 4),
                "atr_std": round(float(grp.std()), 4),
                "adx_mean": round(float(adx_grp.mean()), 4),
                "n_bars": len(grp),
            }
        )

    shift_detected = False
    shift_details = "Nicht genug Daten"
    if len(quarters) >= 3:
        prev_atrs = [q["atr_mean"] for q in quarters[:-1]]
        last_atr = quarters[-1]["atr_mean"]
        median_prev = float(np.median(prev_atrs))
        ratio = last_atr / median_prev if median_prev > 0 else 1.0
        if ratio > 1.4 or ratio < 0.7:
            shift_detected = True
            shift_details = (
                f"ATR-Ratio letztes Quartal vs. Median: {ratio:.2f} "
                f"({'höher' if ratio > 1 else 'niedriger'} als Baseline)"
            )
        else:
            shift_details = f"ATR-Ratio: {ratio:.2f} (kein signifikanter Shift)"

    return {
        "quarters": quarters,
        "shift_detected": shift_detected,
        "shift_details": shift_details,
    }


# ── Modul 2: Overfitting-Diagnose ───────────────────────────────────────────


def _classify_overfitting(
    avg_oos_pf: float, holdout_pf: float | None, holdout_trades: int | None
) -> str:
    if holdout_pf is None or holdout_trades is None or holdout_trades < 10:
        return "Nicht validiert"
    degradation = (avg_oos_pf - holdout_pf) / avg_oos_pf if avg_oos_pf > 0 else 0
    if holdout_pf >= avg_oos_pf * 0.8:
        return "Echte Edge"
    if holdout_pf >= 1.0:
        return "Partial"
    if degradation > 0.5:
        return "Curve-Fitted"
    return "Regime-Shift"


def analyse_overfitting(
    db_path: Path | str,
    tier: str | None = None,
) -> list[dict[str, Any]]:
    """Analysiert Overfitting für alle Tier-A/B Strategien.

    Returns:
        Liste von Dicts mit run_id, idea, avg_oos_pf, holdout_pf,
        holdout_trades, degradation, classification.
    """
    conn = sqlite3.connect(db_path)
    where = ""
    params: tuple = ()
    if tier:
        where = "WHERE tier = ?"
        params = (tier.upper(),)
    else:
        where = "WHERE tier IN ('A','B')"

    rows = conn.execute(
        f"SELECT id, idea, avg_oos_pf, holdout_pf, holdout_trades, pbo_score "
        f"FROM build_runs {where} ORDER BY avg_oos_pf DESC",
        params,
    ).fetchall()
    conn.close()

    result = []
    for run_id, idea, oos_pf, ho_pf, ho_t, pbo in rows:
        deg = (oos_pf - (ho_pf or 0)) / oos_pf if oos_pf > 0 else 0
        result.append(
            {
                "run_id": run_id,
                "idea": idea,
                "avg_oos_pf": round(oos_pf, 3),
                "holdout_pf": round(ho_pf, 3) if ho_pf else None,
                "holdout_trades": ho_t,
                "pbo_score": pbo,
                "degradation": round(deg, 3),
                "classification": _classify_overfitting(oos_pf, ho_pf, ho_t),
            }
        )
    return result


# ── Modul 3: Baustein-Qualität ───────────────────────────────────────────────


def analyse_bausteine(
    cache_path: Path | str,
    db_path: Path | str,
) -> dict[str, Any]:
    """Analysiert welche Bausteine im Cache aktiv sind und Edge bringen.

    Returns:
        {
          "bausteine": [{"name": ..., "status": "aktiv"|"tot",
                         "fire_rate": ..., "avg_oos_pf": ...}, ...],
          "dead_bausteine": [...],
          "active_bausteine": [...],
        }
    """
    cache_path = Path(cache_path)
    # Sharded Cache: signal_cache.parquet ist nur Marker {"sharded": true}
    # Echte Daten liegen in pda_cache/*.parquet
    try:
        cache_df = pd.read_parquet(cache_path)
    except Exception:
        pda_dir = cache_path.parent / "pda_cache"
        if pda_dir.exists():
            parts = [pd.read_parquet(p) for p in sorted(pda_dir.glob("*.parquet"))]
            if parts:
                cache_df = pd.concat(parts, axis=1)
                cache_df = cache_df.loc[:, ~cache_df.columns.duplicated()]
            else:
                cache_df = pd.DataFrame()
        else:
            cache_df = pd.DataFrame()
    n_bars = len(cache_df)

    col_stats: dict[str, dict] = {}
    for col in cache_df.columns:
        non_zero = float((cache_df[col] != 0).sum())
        fire_rate = round(non_zero / n_bars, 4) if n_bars > 0 else 0.0
        col_stats[col] = {
            "fire_rate": fire_rate,
            "is_dead": fire_rate == 0.0,
        }

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT idea, avg_oos_pf FROM build_runs WHERE tier IN ('A','B')"
    ).fetchall()
    conn.close()

    concept_pfs: dict[str, list[float]] = {}
    for idea, oos_pf in rows:
        parts = idea.split(" + ")
        for p in parts:
            concept = p.strip().split(" ")[0].upper()
            concept_pfs.setdefault(concept, []).append(float(oos_pf))

    bausteine = []
    for col, stats in sorted(col_stats.items(), key=lambda x: -x[1]["fire_rate"]):
        concept_key = col.upper().split("_")[0]
        pfs = concept_pfs.get(concept_key, concept_pfs.get(col.upper(), []))
        bausteine.append(
            {
                "name": col,
                "status": "tot" if stats["is_dead"] else "aktiv",
                "fire_rate": stats["fire_rate"],
                "avg_oos_pf": round(float(np.mean(pfs)), 3) if pfs else None,
                "n_strategies": len(pfs),
            }
        )

    dead = [b["name"] for b in bausteine if b["status"] == "tot"]
    active = [b["name"] for b in bausteine if b["status"] == "aktiv"]

    return {
        "bausteine": bausteine,
        "dead_bausteine": dead,
        "active_bausteine": active,
    }


# ── Modul 4: Trade-Verteilung ────────────────────────────────────────────────


def analyse_trade_distribution(output_dir: Path | str) -> dict[str, Any]:
    """Analysiert Trade-Verteilung aus allen Parquets in output_dir/trades/.

    Returns:
        {
          "by_session": {"ny": {"avg_pnl_pts": ..., "win_rate": ..., "n": ...}, ...},
          "by_hour_weekday": dict (Heatmap-Daten),
          "by_regime": {...},
          "total_trades": int,
          "cumulative_pnl": list[float],
        }
    """
    trades_dir = Path(output_dir) / "trades"
    empty: dict[str, Any] = {
        "by_session": {},
        "by_hour_weekday": {},
        "by_regime": {},
        "total_trades": 0,
        "cumulative_pnl": [],
    }
    if not trades_dir.exists():
        return empty

    parquets = list(trades_dir.glob("run_*_trades.parquet"))
    if not parquets:
        return empty

    all_trades = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    all_trades = all_trades[all_trades["pnl_points"].notna()]

    def _group_stats(grp: pd.DataFrame) -> dict:
        return {
            "avg_pnl_pts": round(float(grp["pnl_points"].mean()), 3),
            "avg_pnl_usd": round(float(grp["pnl_usd"].mean()), 2),
            "win_rate": round(float((grp["pnl_points"] > 0).mean()), 3),
            "n": int(len(grp)),
        }

    by_session: dict[str, dict] = {}
    for sess, grp in all_trades.groupby("session"):
        by_session[str(sess)] = _group_stats(grp)

    by_regime: dict[str, dict] = {}
    for reg, grp in all_trades.groupby("regime"):
        by_regime[str(reg)] = _group_stats(grp)

    pivot = all_trades.pivot_table(
        values="pnl_points",
        index="day_of_week",
        columns="hour_of_day",
        aggfunc="mean",
    ).round(2)

    if "entry_time" in all_trades.columns:
        sorted_trades = all_trades.sort_values("entry_time")
    else:
        sorted_trades = all_trades
    cumulative = list(sorted_trades["pnl_usd"].cumsum().round(2))

    return {
        "by_session": by_session,
        "by_hour_weekday": pivot.to_dict(),
        "by_regime": by_regime,
        "total_trades": int(len(all_trades)),
        "cumulative_pnl": cumulative,
    }
