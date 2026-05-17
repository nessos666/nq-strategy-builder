"""
PDA-Algo 31 v2: Premium/Discount (ICT-Definition, eigenständig).

ICT-Definition:
- Obere Hälfte der Range = Premium (Sell-Bias, Verkaufszone)
- Untere Hälfte der Range = Discount (Buy-Bias, Kaufzone)
- Equilibrium = 50 % der Range (neutrale Zone)

FIX gegenüber v1:
v1 verwendete `groupby(date).transform("max")` → Look-Ahead!
(kennt das Tageshoch bereits an Bar 1 eines Tages).
v2 verwendet rolling max/min → kein Look-Ahead.

Eingabe: DataFrame mit OHLC-Spalten.
Ausgabe: DataFrame mit pd_-Spalten (gleicher Index, kein Look-Ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

LOOKBACK: int = 20


@dataclass(frozen=True)
class PremiumDiscountConfig:
    """Konfiguration für den Premium/Discount-Algo v2."""

    lookback: int = LOOKBACK


def compute_premium_discount(
    df: pd.DataFrame,
    config: PremiumDiscountConfig | None = None,
) -> pd.DataFrame:
    """
    Berechnet Premium/Discount-Zonen nach ICT-Definition.

    Parameter
    ----------
    df : DataFrame mit Spalten Open, High, Low, Close.
    config : PremiumDiscountConfig (optional).

    Rückgabe
    --------
    DataFrame mit Spalten:
      - pd_in_premium   : bool  – Close > Equilibrium
      - pd_in_discount  : bool  – Close < Equilibrium
      - pd_equilibrium  : float – 50 % der rolling Range
      - pd_pct_of_range : float – Position in Range (0–100 %)
      - pd_range_high   : float – rolling High
      - pd_range_low    : float – rolling Low
    """
    if config is None:
        config = PremiumDiscountConfig()

    n = len(df)
    out = pd.DataFrame(index=df.index)

    empty_cols = (
        "pd_in_premium",
        "pd_in_discount",
        "pd_equilibrium",
        "pd_pct_of_range",
        "pd_range_high",
        "pd_range_low",
    )

    if n == 0:
        logger.warning("compute_premium_discount: leerer DataFrame – leeres Ergebnis")
        for col in empty_cols:
            out[col] = False if col in ("pd_in_premium", "pd_in_discount") else np.nan
        return out

    h = df["High"]
    lo = df["Low"]
    c = df["Close"]

    lookback = max(1, min(config.lookback, n))

    # Rolling max/min: min_periods=1 damit auch kurze DFs funktionieren.
    # shift(0) = inkludiert aktuellen Bar → kein Look-Ahead, da Bar i bereits geschlossen.
    range_high = h.rolling(window=lookback, min_periods=1).max()
    range_low = lo.rolling(window=lookback, min_periods=1).min()

    equilibrium = (range_high + range_low) / 2.0
    span = range_high - range_low

    # pct_of_range: 0 % = range_low, 100 % = range_high
    # Guard gegen Division durch 0 (flat market)
    pct_of_range = np.where(
        span > 0,
        (c - range_low) / span * 100.0,
        50.0,  # Flat Market → neutral (50 %)
    )

    in_premium = c > equilibrium
    in_discount = c < equilibrium

    out["pd_in_premium"] = in_premium
    out["pd_in_discount"] = in_discount
    out["pd_equilibrium"] = equilibrium
    out["pd_pct_of_range"] = pct_of_range
    out["pd_range_high"] = range_high
    out["pd_range_low"] = range_low

    logger.debug(
        f"compute_premium_discount: {int(in_premium.sum())} premium, "
        f"{int(in_discount.sum())} discount Bars"
    )
    return out
