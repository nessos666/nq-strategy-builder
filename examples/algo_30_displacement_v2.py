"""
PDA-Algo 30 v2: Displacement – ATR-basiert, ICT-Definition.

**ICT-Definition (institutionelles Sponsorship):**
  Starke Einwegbewegung die zeigt, dass Smart Money hinter der Bewegung steckt.
  Merkmale:
    1. Body-Range > ATR × DISPLACEMENT_ATR_FACTOR (Volatilitäts-Ausreißer)
    2. Body-Ratio >= BODY_RATIO_MIN (wenig Wick → dominanter Body)
    3. Richtung: Bullish wenn Close > Open, Bearish wenn Close < Open

**Unterschied zu v1:**
  - v1: Schwellenwert über rolling avg von Body UND Range
  - v2: Schwellenwert nur über ATR(14) × Faktor – ICT-standardisiert

Displacement ≠ einfach große Kerze:
  Eine Kerze mit 80% Wick ist KEIN Displacement, auch wenn sie groß ist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

DISPLACEMENT_ATR_FACTOR: float = 1.5
BODY_RATIO_MIN: float = 0.5
ATR_PERIOD: int = 14


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """True Range Average über `period` Bars (Wilder-Methode via rolling mean)."""
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    # shift(1): aktuelle Kerze beeinflusst nicht ihre eigene ATR-Basis
    return tr.shift(1).rolling(period, min_periods=1).mean()


def compute_displacement(
    df: pd.DataFrame,
    atr_factor: float = DISPLACEMENT_ATR_FACTOR,
    body_ratio_min: float = BODY_RATIO_MIN,
    atr_period: int = ATR_PERIOD,
) -> pd.DataFrame:
    """
    Erkennt Displacement-Kerzen nach ICT-Standard (ATR-basiert).

    Parameter
    ----------
    df : DataFrame mit Open, High, Low, Close
    atr_factor : Body-Range muss > ATR × atr_factor
    body_ratio_min : Body muss mind. X% der Kerzen-Range ausmachen
    atr_period : Periode für ATR-Berechnung

    Rückgabe
    ---------
    DataFrame mit Spalten:
        displacement         : bool – True wenn Displacement-Kriterien erfüllt
        displacement_bullish : bool – Bullish Displacement (Close > Open)
        displacement_bearish : bool – Bearish Displacement (Close < Open)
        displacement_body_ratio : float – Body-Anteil (abs(Close-Open) / (High-Low))
        displacement_atr     : float – ATR-Wert für diese Bar
    """
    n = len(df)

    body_range = (df["Close"] - df["Open"]).abs()
    bar_range = df["High"] - df["Low"]
    bar_range_safe = bar_range.replace(0.0, np.nan)

    body_ratio = body_range / bar_range_safe

    atr = _compute_atr(df, atr_period)

    # Bedingung 1: Body größer als ATR × Faktor
    big_body = body_range > (atr * atr_factor)

    # Bedingung 2: Body-Anteil der Kerze >= Minimum
    strong_body = body_ratio >= body_ratio_min

    is_displacement = big_body & strong_body

    bullish = is_displacement & (df["Close"] > df["Open"])
    bearish = is_displacement & (df["Close"] < df["Open"])

    logger.debug(
        "Displacement: {} bullish, {} bearish aus {} Bars",
        int(bullish.sum()),
        int(bearish.sum()),
        n,
    )

    return pd.DataFrame(
        {
            "displacement": is_displacement.fillna(False),
            "displacement_bullish": bullish.fillna(False),
            "displacement_bearish": bearish.fillna(False),
            "displacement_body_ratio": body_ratio,
            "displacement_atr": atr,
        },
        index=df.index,
    )
