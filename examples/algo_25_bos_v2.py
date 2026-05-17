"""
PDA-Algo 25 v2: BOS (Break of Structure) – sauber neu.

**ICT-Definition:**
  Bullish BOS : Close bricht über ein vorheriges Swing High → Trend-Fortsetzung
  Bearish BOS : Close bricht unter ein vorheriges Swing Low  → Trend-Fortsetzung

  BOS ≠ CHoCH: BOS ist Trend-Fortsetzung, CHoCH ist Trendwechsel.

**Swing-Erkennung ohne Look-ahead (center=False):**
  Swing High = Bar dessen High das höchste der letzten swing_window Bars ist
  Swing Low  = Bar dessen Low das niedrigste der letzten swing_window Bars ist

Signal: Nur der ERSTE Bar des Bruchs (Übergang False→True).
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

SWING_LOOKBACK: int = 5


def compute_bos(
    df: pd.DataFrame,
    swing_lookback: int = SWING_LOOKBACK,
) -> pd.DataFrame:
    """
    Erkennt Break of Structure (BOS) ohne Look-ahead.

    Parameter
    ----------
    df : DataFrame mit Open, High, Low, Close
    swing_lookback : Bars für Rolling-Swing-Erkennung

    Rückgabe
    ---------
    DataFrame mit Spalten:
        bos_bullish : bool – True auf dem ERSTEN Bar wo Close > letztes Swing High
        bos_bearish : bool – True auf dem ERSTEN Bar wo Close < letztes Swing Low
        bos_level   : float – Swing-Level das gebrochen wurde (NaN wenn kein BOS)
    """
    n = len(df)

    # center=False → kein Look-ahead
    rolling_max = (
        df["High"].rolling(window=swing_lookback, center=False, min_periods=1).max()
    )
    rolling_min = (
        df["Low"].rolling(window=swing_lookback, center=False, min_periods=1).min()
    )

    is_swing_high = df["High"] == rolling_max
    is_swing_low = df["Low"] == rolling_min

    # Letztes Swing-Level vorwärts propagieren (unbegrenzt)
    last_sh = df["High"].where(is_swing_high).ffill()
    last_sl = df["Low"].where(is_swing_low).ffill()

    # shift(1): aktueller Bar soll nicht sein eigenes Level brechen
    prev_sh = last_sh.shift(1)
    prev_sl = last_sl.shift(1)

    above_sh = (df["Close"] > prev_sh) & prev_sh.notna()
    below_sl = (df["Close"] < prev_sl) & prev_sl.notna()

    # Nur der ERSTE Bar des Bruchs
    bos_bull = above_sh & ~above_sh.shift(1, fill_value=False)
    bos_bear = below_sl & ~below_sl.shift(1, fill_value=False)

    # bos_level = das Level das gebrochen wurde
    bos_level = pd.Series(index=df.index, dtype=float)
    bos_level = bos_level.where(~bos_bull, prev_sh)
    bos_level = bos_level.where(~bos_bear, prev_sl)

    logger.debug(
        "BOS: {} bullish, {} bearish aus {} Bars",
        int(bos_bull.sum()),
        int(bos_bear.sum()),
        n,
    )

    out = pd.DataFrame(index=df.index)
    out["bos_bullish"] = bos_bull.fillna(False)
    out["bos_bearish"] = bos_bear.fillna(False)
    out["bos_level"] = bos_level
    return out
