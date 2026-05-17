"""
PDA-Algo 33 v2: Dealing Range (ICT-Definition, eigenständig).

ICT-Definition:
- Die Range in der IPDA handelt = Rolling High bis Rolling Low.
- Equilibrium = Mitte der Range (50 %).
- In-Range = Preis befindet sich zwischen Range-High und Range-Low.

FIX gegenüber v1:
v1 verwendete `groupby(date).transform("max")` → Look-Ahead!
v2 verwendet rolling max/min → kein Look-Ahead.

Eingabe: DataFrame mit OHLC-Spalten.
Ausgabe: DataFrame mit dr_-Spalten (gleicher Index, kein Look-Ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

LOOKBACK: int = 20


@dataclass(frozen=True)
class DealingRangeConfig:
    """Konfiguration für den Dealing-Range-Algo v2."""

    lookback: int = LOOKBACK


def compute_dealing_range(
    df: pd.DataFrame,
    config: DealingRangeConfig | None = None,
) -> pd.DataFrame:
    """
    Berechnet Dealing-Range-Zonen nach ICT-Definition.

    Parameter
    ----------
    df : DataFrame mit Spalten Open, High, Low, Close.
    config : DealingRangeConfig (optional).

    Rückgabe
    --------
    DataFrame mit Spalten:
      - dr_high     : float – rolling High (obere Range-Grenze)
      - dr_low      : float – rolling Low (untere Range-Grenze)
      - dr_mid      : float – 50 % der Range (Equilibrium)
      - dr_in_range : bool  – Close innerhalb [dr_low, dr_high]
      - dr_span     : float – Breite der Range (dr_high - dr_low)
    """
    if config is None:
        config = DealingRangeConfig()

    n = len(df)
    out = pd.DataFrame(index=df.index)

    empty_cols = ("dr_high", "dr_low", "dr_mid", "dr_in_range", "dr_span")

    if n == 0:
        logger.warning("compute_dealing_range: leerer DataFrame – leeres Ergebnis")
        for col in empty_cols:
            out[col] = False if col == "dr_in_range" else np.nan
        return out

    h = df["High"]
    lo = df["Low"]
    c = df["Close"]

    lookback = max(1, min(config.lookback, n))

    range_high = h.rolling(window=lookback, min_periods=1).max()
    range_low = lo.rolling(window=lookback, min_periods=1).min()
    dr_mid = (range_high + range_low) / 2.0
    dr_span = range_high - range_low
    dr_in_range = (c >= range_low) & (c <= range_high)

    out["dr_high"] = range_high
    out["dr_low"] = range_low
    out["dr_mid"] = dr_mid
    out["dr_in_range"] = dr_in_range
    out["dr_span"] = dr_span

    logger.debug(
        f"compute_dealing_range: {int(dr_in_range.sum())} Bars in Range "
        f"(lookback={lookback})"
    )
    return out
