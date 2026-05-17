"""Standalone rolling Hurst exponent filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class Config:
    """Configuration for rolling Hurst calculation."""

    window: int = 100
    trend_threshold: float = 0.50
    revert_threshold: float = 0.45


def _hurst(values: np.ndarray) -> float:
    if len(values) < 16:
        return 0.5
    log_prices = np.log(np.maximum(values, 1e-9))
    lags = np.arange(2, min(20, len(values) // 2 + 1))
    tau = []
    valid_lags = []
    for lag in lags:
        # k-Perioden Log-Returns: log(P[t+k]) - log(P[t])
        diffs = log_prices[lag:] - log_prices[:-lag]
        std = np.std(diffs, ddof=1)
        if np.isfinite(std) and std > 0.0:
            tau.append(std)
            valid_lags.append(lag)
    if len(tau) < 2:
        return 0.5
    slope = float(np.polyfit(np.log(valid_lags), np.log(tau), 1)[0])
    return float(np.clip(slope, 0.0, 1.0))


def compute_hurst_exponent(df: pd.DataFrame, config: Config = Config()) -> pd.DataFrame:
    """Compute a causal rolling Hurst exponent and derived regime columns."""

    logger.debug("Computing Hurst exponent for {} rows", len(df))
    close_col = next((c for c in ("Close", "close") if c in df.columns), None)
    if close_col is None:
        logger.warning(
            "Hurst: keine Close-Spalte gefunden – gebe neutrale Werte zurück"
        )
        out = df.copy()
        out["hurst_exp"] = 0.5
        out["hurst_regime"] = 0
        out["hurst_strength"] = 0.0
        out["hurst_trend"] = 0
        out["hurst_passes"] = False
        return out
    window = max(config.window, 1)
    close = (
        pd.to_numeric(df[close_col], errors="coerce")
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )
    hurst = np.full(len(close), 0.5, dtype=float)
    for i in range(window, len(close)):
        hurst[i] = _hurst(close[i - window : i])
    regime = np.zeros(len(close), dtype=np.int8)
    regime[hurst > config.trend_threshold] = 1
    regime[hurst < config.revert_threshold] = -1
    trend = (
        np.sign(pd.Series(hurst).diff().fillna(0.0))
        .replace(0.0, 0)
        .astype("int8")
        .to_numpy()
    )
    strength = np.abs(hurst - 0.5)
    out = df.copy()
    out["hurst_exp"] = hurst
    out["hurst_regime"] = regime
    out["hurst_strength"] = strength
    out["hurst_trend"] = trend
    out["hurst_passes"] = hurst > config.trend_threshold
    return out
