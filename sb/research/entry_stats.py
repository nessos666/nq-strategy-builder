"""
Entry-Stats: MAE/MFE-Analyse für Entry-Algo-Signale.

Für jedes Entry-Signal wird gemessen:
  MAE (Max Adverse Excursion)  – wie weit geht der Preis GEGEN uns
  MFE (Max Favorable Excursion) – wie weit geht der Preis FÜR uns
  Win-Rate bei verschiedenen TP-Levels (1:1 RR, SL = TP)

Aus diesen Daten werden SL- und TP-Regeln abgeleitet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EntryStatsResult:
    col: str  # z.B. "fvg_std_bull_ft"
    direction: str  # "bull" oder "bear"
    n_signals: int  # Anzahl Signale
    signals_per_day: float  # Ø Signale pro Tag
    mae_p50: float  # MAE Median
    mae_p80: float  # MAE 80%-Percentile → SL-Empfehlung
    mae_p90: float  # MAE 90%-Percentile
    mae_p95: float  # MAE 95%-Percentile
    mfe_p50: float  # MFE Median
    mfe_p75: float  # MFE 75%-Percentile
    mfe_p90: float  # MFE 90%-Percentile
    win_rates: dict[float, float]  # TP-Level → Win-Rate (1:1 RR)
    forward_bars: int


def _detect_entry_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Erkennt Entry-Signal-Spalten automatisch.
    Gibt Liste von (spaltenname, direction) zurück.
    direction = "bull" oder "bear".
    """
    result = []
    for col in df.columns:
        col_lower = col.lower()
        if "_bull_" in col_lower or col_lower.endswith("_bull"):
            result.append((col, "bull"))
        elif "_bear_" in col_lower or col_lower.endswith("_bear"):
            result.append((col, "bear"))
    return result


def _compute_mae_mfe(
    signals: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: str,
    forward_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet MAE und MFE für jeden Signal-Bar.

    Returns: (mae_arr, mfe_arr) – Arrays der Länge = Anzahl Signale.
    """
    n = len(signals)
    mae_list: list[float] = []
    mfe_list: list[float] = []

    for i in range(n):
        if not signals[i]:
            continue
        entry = close[i]
        end = min(i + 1 + forward_bars, n)
        fwd_high = high[i + 1 : end]
        fwd_low = low[i + 1 : end]

        if len(fwd_high) == 0:
            continue

        if direction == "bull":
            mfe = float(np.max(fwd_high) - entry)
            mae = float(entry - np.min(fwd_low))
        else:
            mfe = float(entry - np.min(fwd_low))
            mae = float(np.max(fwd_high) - entry)

        mae_list.append(max(0.0, mae))
        mfe_list.append(max(0.0, mfe))

    return np.array(mae_list), np.array(mfe_list)


def _compute_win_rates(
    signals: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    direction: str,
    forward_bars: int,
    tp_levels: list[float],
) -> dict[float, float]:
    """
    Win-Rate bei 1:1 RR: TP und SL = gleicher Abstand vom Entry.
    Win = Preis erreicht TP bevor SL getroffen wird (oder Timeout = Verlust).
    """
    n = len(signals)
    win_counts: dict[float, int] = {tp: 0 for tp in tp_levels}
    total = 0

    for i in range(n):
        if not signals[i]:
            continue
        entry = close[i]
        end = min(i + 1 + forward_bars, n)
        total += 1

        for tp in tp_levels:
            tp_price = entry + tp if direction == "bull" else entry - tp
            sl_price = entry - tp if direction == "bull" else entry + tp
            won = False

            for j in range(i + 1, end):
                if direction == "bull":
                    if high[j] >= tp_price:
                        won = True
                        break
                    if low[j] <= sl_price:
                        break
                else:
                    if low[j] <= tp_price:
                        won = True
                        break
                    if high[j] >= sl_price:
                        break

            if won:
                win_counts[tp] += 1

    if total == 0:
        return {tp: 0.0 for tp in tp_levels}
    return {tp: win_counts[tp] / total for tp in tp_levels}


def analyze_entry_stats(
    df: pd.DataFrame,
    forward_bars: int = 200,
    tp_levels: list[float] | None = None,
) -> list[EntryStatsResult]:
    """
    Analysiert alle Entry-Signal-Spalten im DataFrame.

    Args:
        df:           DataFrame mit OHLC + Entry-Signal-Spalten (bool)
        forward_bars: Wie viele Bars voraus gemessen wird (default 200 = ~3h auf 1min)
        tp_levels:    TP-Abstände in Punkten für Win-Rate-Berechnung

    Returns:
        Liste von EntryStatsResult, eine pro Signal-Spalte.
    """
    if tp_levels is None:
        tp_levels = [5.0, 10.0, 20.0, 50.0]

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    # Anzahl Tage im DataFrame
    try:
        idx_et = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    except Exception:
        idx_et = pd.DatetimeIndex(df.index)
    n_days = max(1, idx_et.normalize().nunique())

    entry_cols = _detect_entry_columns(df)
    results: list[EntryStatsResult] = []

    for col, direction in entry_cols:
        if col not in df.columns:
            continue
        signals = df[col].to_numpy(dtype=bool)
        n_signals = int(signals.sum())
        if n_signals == 0:
            continue

        mae_arr, mfe_arr = _compute_mae_mfe(
            signals, high, low, close, direction, forward_bars
        )

        win_rates = _compute_win_rates(
            signals, high, low, close, direction, forward_bars, tp_levels
        )

        results.append(
            EntryStatsResult(
                col=col,
                direction=direction,
                n_signals=n_signals,
                signals_per_day=round(n_signals / n_days, 2),
                mae_p50=float(np.percentile(mae_arr, 50)),
                mae_p80=float(np.percentile(mae_arr, 80)),
                mae_p90=float(np.percentile(mae_arr, 90)),
                mae_p95=float(np.percentile(mae_arr, 95)),
                mfe_p50=float(np.percentile(mfe_arr, 50)),
                mfe_p75=float(np.percentile(mfe_arr, 75)),
                mfe_p90=float(np.percentile(mfe_arr, 90)),
                win_rates=win_rates,
                forward_bars=forward_bars,
            )
        )

    return results
