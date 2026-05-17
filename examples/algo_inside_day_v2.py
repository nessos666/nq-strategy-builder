"""
PDA-Algo: Inside Day v2 – per-Bar-DataFrame.

ICT-Definition:
  Inside Day: ein Handelstag liegt komplett innerhalb des vorangegangenen Ranges.
  High(day-1) < High(day-2) UND Low(day-1) > Low(day-2)
  Bedeutung: Kompression/Konsolidierung → Expansion folgt.

Kein Look-Ahead:
  inside_day[bar] = True wenn GESTERN ein Inside Day war (vs. vorgestern).
  Nur abgeschlossene Tages-H/L werden verwendet (shift(1) und shift(2)).
  Heutiges intraday-H/L wird NICHT verwendet → kein Look-Ahead.
  Breakout = Close > äußeres PDH (vorgestern) am aktuellen Bar.

v2-Bug: transform("max/min") gibt intraday jedem Bar das finale Tageshoch/-tief
  → Look-Ahead für Intraday-Bars.
v2-Fix: shift(2) auf Tages-Ebene (gestern inside vs. vorgestern).

v1-Problem: detect_inside_days() gibt dict[date, InsideDayResult] → falsche API.
v2-Änderungen:
- compute_inside_day() gibt per-Bar-DataFrame zurück
- @dataclass(frozen=True)
- tz-naive Index → ValueError
- Look-Ahead Fix: shift(2) statt transform("max/min")
- loguru statt print
- from __future__ import annotations
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class InsideDayConfig:
    """Konfiguration Inside Day v2."""

    timezone: str = "America/New_York"
    strict: bool = True  # True = High < PDH (strikt), False = <=


def compute_inside_day(
    df: pd.DataFrame,
    config: InsideDayConfig | None = None,
) -> pd.DataFrame:
    """
    Inside Day: per-Bar-Flags.

    Input : DataFrame mit tz-aware DatetimeIndex.
    Output: DataFrame mit inside_day-Spalten (gleicher Index wie df).

    inside_day           : True wenn gestrigen Tag ein Inside Day war (vs. vorgestern)
    inside_day_pdh       : Äußeres High (vorgestern) – Breakout-Ziel oben
    inside_day_pdl       : Äußeres Low  (vorgestern) – Breakout-Ziel unten
    inside_day_breakout_bull: Close > äußeres PDH → bullischer Breakout
    inside_day_breakout_bear: Close < äußeres PDL → bärischer Breakout
    """
    if config is None:
        config = InsideDayConfig()

    def _empty_result() -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["inside_day"] = False
        out["inside_day_pdh"] = float("nan")
        out["inside_day_pdl"] = float("nan")
        out["inside_day_breakout_bull"] = False
        out["inside_day_breakout_bear"] = False
        return out

    required = {"High", "Low", "Close"}
    if df.empty or not required.issubset(df.columns):
        logger.debug("Inside Day v2: leerer DataFrame oder fehlende Spalten")
        return _empty_result()

    if df.index.tz is None:
        raise ValueError(
            "Inside Day v2: DatetimeIndex muss tz-aware sein. "
            "Lösung: df.index = df.index.tz_localize('America/New_York')"
        )

    idx_et = df.index.tz_convert(config.timezone)
    dates = pd.Series(idx_et.date, index=df.index)

    # ── Per-Datum: finales High/Low (komplett bekannt nach Tagesende) ─────────
    per_date_high = df.groupby(dates)["High"].max()
    per_date_low = df.groupby(dates)["Low"].min()

    # shift(1) = gestern, shift(2) = vorgestern (beide vollständig bekannt)
    prev1_high = per_date_high.shift(1)
    prev2_high = per_date_high.shift(2)
    prev1_low = per_date_low.shift(1)
    prev2_low = per_date_low.shift(2)

    # ── Inside Day: war GESTERN inside (vs. VORGESTERN)? ──────────────────────
    # Kein Look-Ahead: nur abgeschlossene Tages-H/L werden verglichen.
    if config.strict:
        is_inside_date = (prev1_high < prev2_high) & (prev1_low > prev2_low)
    else:
        is_inside_date = (prev1_high <= prev2_high) & (prev1_low >= prev2_low)

    # Auf alle Bars des heutigen Tages mappen
    is_inside_today = dates.map(is_inside_date).fillna(False)

    # Äußerer Range (vorgestern) = Breakout-Ziele für heute
    pdh_outer = dates.map(prev2_high)
    pdl_outer = dates.map(prev2_low)

    # ── Breakout ──────────────────────────────────────────────────────────────
    breakout_bull = is_inside_today & (df["Close"] > pdh_outer) & pdh_outer.notna()
    breakout_bear = is_inside_today & (df["Close"] < pdl_outer) & pdl_outer.notna()

    out = pd.DataFrame(index=df.index)
    out["inside_day"] = is_inside_today
    out["inside_day_pdh"] = pdh_outer.where(is_inside_today)
    out["inside_day_pdl"] = pdl_outer.where(is_inside_today)
    out["inside_day_breakout_bull"] = breakout_bull.fillna(False)
    out["inside_day_breakout_bear"] = breakout_bear.fillna(False)

    logger.debug(
        "Inside Day v2: {} Inside-Day-Bars",
        int(is_inside_today.sum()),
    )
    return out
