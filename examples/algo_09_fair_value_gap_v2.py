"""
PDA-Algo 09 v2: Fair Value Gap (ICT-Definition, eigenständig).

ICT-Definition (3-Kerzen-Muster):
- Kerze 1 = Bar[i-2], Kerze 2 = Bar[i-1], Kerze 3 = Bar[i]
- Bullish FVG : Low[i]  > High[i-2] → Lücke nach oben
  Zone: fvg_bull_low = High[i-2], fvg_bull_high = Low[i]
- Bearish FVG: High[i] < Low[i-2]  → Lücke nach unten
  Zone: fvg_bear_low = High[i], fvg_bear_high = Low[i-2]
- Signal an Bar i (kein Look-Ahead: alle 3 Kerzen bereits geschlossen)
- CE (Consequent Encroachment): 50 % der Zone
- Mitigation Bullisch: Low[j] <= fvg_bull_low → FVG gefüllt
- Mitigation Bearisch: High[j] >= fvg_bear_high → FVG gefüllt

FIX gegenüber v1:
v1 propagiert Zonen per ffill() und stoppt nie.
v2: fvg_bull_active / fvg_bear_active werden nach Mitigation auf False gesetzt.

Eingabe: DataFrame mit OHLC-Spalten.
Ausgabe: DataFrame mit fvg_-Spalten (gleicher Index, kein Look-Ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

MIN_GAP_PTS: float = 0.0
MAX_ACTIVE: int = 5


@dataclass(frozen=True)
class FVGConfig:
    """Konfiguration für den FVG-Algo v2."""

    min_gap_pts: float = MIN_GAP_PTS
    max_active: int = MAX_ACTIVE


def compute_fvg(
    df: pd.DataFrame,
    config: FVGConfig | None = None,
) -> pd.DataFrame:
    """
    Erkennt Fair Value Gaps nach ICT-Definition mit Mitigation-Tracking.

    Parameter
    ----------
    df : DataFrame mit Spalten Open, High, Low, Close.
    config : FVGConfig (optional).

    Rückgabe
    --------
    DataFrame mit Spalten:
      - fvg_bullish    : bool  – neues bullisches FVG an diesem Bar
      - fvg_bearish    : bool  – neues bearisches FVG an diesem Bar
      - fvg_bull_low   : float – Zone-Unterkante (aktives bull. FVG)
      - fvg_bull_high  : float – Zone-Oberkante (aktives bull. FVG)
      - fvg_bear_low   : float – Zone-Unterkante (aktives bear. FVG)
      - fvg_bear_high  : float – Zone-Oberkante (aktives bear. FVG)
      - fvg_bull_ce    : float – CE-Level bullisch (50 % der Zone)
      - fvg_bear_ce    : float – CE-Level bearisch
      - fvg_bull_active: bool  – bull. FVG noch aktiv (nicht gefüllt)
      - fvg_bear_active: bool  – bear. FVG noch aktiv
      - fvg_bull_filled: bool  – bull. FVG in diesem Bar gefüllt
      - fvg_bear_filled: bool  – bear. FVG in diesem Bar gefüllt
    """
    if config is None:
        config = FVGConfig()

    n = len(df)
    out = pd.DataFrame(index=df.index)

    bool_cols = (
        "fvg_bullish",
        "fvg_bearish",
        "fvg_bull_active",
        "fvg_bear_active",
        "fvg_bull_filled",
        "fvg_bear_filled",
    )
    float_cols = (
        "fvg_bull_low",
        "fvg_bull_high",
        "fvg_bear_low",
        "fvg_bear_high",
        "fvg_bull_ce",
        "fvg_bear_ce",
    )

    if n < 3:
        logger.warning("compute_fvg: zu wenige Bars (< 3) – leeres Ergebnis")
        for col in bool_cols:
            out[col] = False
        for col in float_cols:
            out[col] = np.nan
        return out

    h = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)

    fvg_bullish = np.zeros(n, dtype=bool)
    fvg_bearish = np.zeros(n, dtype=bool)
    fvg_bull_low_arr = np.full(n, np.nan)
    fvg_bull_high_arr = np.full(n, np.nan)
    fvg_bear_low_arr = np.full(n, np.nan)
    fvg_bear_high_arr = np.full(n, np.nan)
    fvg_bull_ce_arr = np.full(n, np.nan)
    fvg_bear_ce_arr = np.full(n, np.nan)
    fvg_bull_active = np.zeros(n, dtype=bool)
    fvg_bear_active = np.zeros(n, dtype=bool)
    fvg_bull_filled = np.zeros(n, dtype=bool)
    fvg_bear_filled = np.zeros(n, dtype=bool)

    # Aktives bull FVG (zuletzt gesehen, noch nicht gefüllt)
    act_bull_low = np.nan
    act_bull_high = np.nan
    bull_active = False

    # Aktives bear FVG
    act_bear_low = np.nan
    act_bear_high = np.nan
    bear_active = False

    for i in range(n):
        # --- Mitigation prüfen (vor Erkennung, da Bar i bereits geschlossen) ---
        if bull_active:
            # Bullische FVG gefüllt wenn Low[i] <= fvg_bull_low (Preis dringt in Zone ein)
            if lo[i] <= act_bull_low:
                fvg_bull_filled[i] = True
                bull_active = False

        if bear_active:
            # Bearische FVG gefüllt wenn High[i] >= fvg_bear_high
            if h[i] >= act_bear_high:
                fvg_bear_filled[i] = True
                bear_active = False

        # --- Neue FVG erkennen (i >= 2, kein Look-Ahead: alle 3 Bars geschlossen) ---
        if i >= 2:
            # Bullisch: Low[i] > High[i-2]
            gap_bull = lo[i] - h[i - 2]
            if gap_bull > config.min_gap_pts:
                zone_low = h[i - 2]
                zone_high = lo[i]
                fvg_bullish[i] = True
                fvg_bull_low_arr[i] = zone_low
                fvg_bull_high_arr[i] = zone_high
                fvg_bull_ce_arr[i] = (zone_low + zone_high) / 2.0
                # Neue FVG überschreibt aktive (last active wins)
                act_bull_low = zone_low
                act_bull_high = zone_high
                bull_active = True

            # Bearisch: High[i] < Low[i-2]
            gap_bear = lo[i - 2] - h[i]
            if gap_bear > config.min_gap_pts:
                zone_low = h[i]
                zone_high = lo[i - 2]
                fvg_bearish[i] = True
                fvg_bear_low_arr[i] = zone_low
                fvg_bear_high_arr[i] = zone_high
                fvg_bear_ce_arr[i] = (zone_low + zone_high) / 2.0
                act_bear_low = zone_low
                act_bear_high = zone_high
                bear_active = True

        # --- Aktive FVG-Werte für diesen Bar propagieren ---
        if bull_active:
            fvg_bull_low_arr[i] = act_bull_low
            fvg_bull_high_arr[i] = act_bull_high
            fvg_bull_ce_arr[i] = (act_bull_low + act_bull_high) / 2.0
            fvg_bull_active[i] = True

        if bear_active:
            fvg_bear_low_arr[i] = act_bear_low
            fvg_bear_high_arr[i] = act_bear_high
            fvg_bear_ce_arr[i] = (act_bear_low + act_bear_high) / 2.0
            fvg_bear_active[i] = True

    out["fvg_bullish"] = fvg_bullish
    out["fvg_bearish"] = fvg_bearish
    out["fvg_bull_low"] = fvg_bull_low_arr
    out["fvg_bull_high"] = fvg_bull_high_arr
    out["fvg_bear_low"] = fvg_bear_low_arr
    out["fvg_bear_high"] = fvg_bear_high_arr
    out["fvg_bull_ce"] = fvg_bull_ce_arr
    out["fvg_bear_ce"] = fvg_bear_ce_arr
    out["fvg_bull_active"] = fvg_bull_active
    out["fvg_bear_active"] = fvg_bear_active
    out["fvg_bull_filled"] = fvg_bull_filled
    out["fvg_bear_filled"] = fvg_bear_filled

    logger.debug(
        f"compute_fvg: {int(fvg_bullish.sum())} bull FVGs, "
        f"{int(fvg_bearish.sum())} bear FVGs gefunden"
    )
    return out
