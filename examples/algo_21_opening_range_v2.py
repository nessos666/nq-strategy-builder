"""
PDA-Algo 21: Opening Range v2.

ICT-Definition:
  Die erste N Minuten des NY-Handels (Standard: 9:30–10:00 ET) bilden die
  Opening Range. OR-High und OR-Low gelten als Level für den restlichen Tag.

v1-Bug (Look-Ahead):
  groupby(dates).apply(or_high_per_day) mappt OR-High auf ALLE Bars des Tages,
  auch auf Bars INNERHALB der OR → Bars bei 9:31 kennen schon OR-High von 9:58!

v2-Fix:
  - in_or: Bar-Flag, kein Look-Ahead
  - OR-High/Low aus .where(in_or) → groupby().max/min (nur in_or-Bars)
  - Bars IN der OR: or_high=NaN, or_low=NaN
  - Post-OR-Bars: Level aus .where(post_or) gemappt → kein Look-Ahead
  - or_completed: True ab erster Bar nach OR-Ende

v2 Änderungen gegenüber v1:
- compute_opening_range() statt run()
- @dataclass(frozen=True)
- loguru statt print
- from __future__ import annotations
- Echter tz_convert (kein manueller Offset)
- Look-Ahead-Bug gefixt
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class OpeningRangeConfig:
    """Konfiguration Opening Range v2."""

    timezone: str = "America/New_York"
    or_start_hour: int = 9
    or_start_minute: int = 30
    or_duration_min: int = 30  # 9:30–10:00 ET


def compute_opening_range(
    df: pd.DataFrame,
    config: OpeningRangeConfig | None = None,
) -> pd.DataFrame:
    """
    Opening Range: Tages-Level und Flags ohne Look-Ahead.

    Input : OHLCV DataFrame (beliebiger DatetimeIndex, UTC-aware oder tz-naiv als UTC).
    Output: DataFrame mit or_*-Spalten (gleicher Index wie df).

    in_opening_range : True für Bars innerhalb des OR-Fensters
    or_high          : OR-High des Tages – NaN für Bars IN der OR!
    or_low           : OR-Low  des Tages – NaN für Bars IN der OR!
    or_completed     : True ab erster Bar nach OR-Ende (bar_min >= or_end)
    """
    if config is None:
        config = OpeningRangeConfig()

    def _empty_result() -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["in_opening_range"] = False
        out["or_high"] = float("nan")
        out["or_low"] = float("nan")
        out["or_completed"] = False
        return out

    required = {"High", "Low"}
    if df.empty or not required.issubset(df.columns):
        logger.debug("Opening Range v2: leerer DataFrame oder fehlende Spalten")
        return _empty_result()

    tz = config.timezone

    # Index in Zielzeitzone konvertieren
    try:
        idx_et = df.index.tz_convert(tz)
    except TypeError:
        idx_et = df.index.tz_localize("UTC").tz_convert(tz)

    bar_min = pd.Series(idx_et.hour * 60 + idx_et.minute, index=df.index)
    date_ser = pd.Series(idx_et.date, index=df.index)

    or_start_min = config.or_start_hour * 60 + config.or_start_minute
    or_end_min = or_start_min + config.or_duration_min

    in_or = (bar_min >= or_start_min) & (bar_min < or_end_min)
    post_or = bar_min >= or_end_min

    # OR-Level: groupby pro Tag, nur in_or-Bars berücksichtigen
    # .where(in_or) → NaN für Bars außerhalb der OR → groupby().max/min ignoriert NaN
    or_high_per_day = df["High"].where(in_or).groupby(date_ser).max()
    or_low_per_day = df["Low"].where(in_or).groupby(date_ser).min()

    # Level auf alle Bars mappen, dann per .where(post_or) auf Post-OR beschränken
    or_high_mapped = date_ser.map(or_high_per_day)
    or_low_mapped = date_ser.map(or_low_per_day)

    or_high = or_high_mapped.where(post_or)
    or_low = or_low_mapped.where(post_or)

    # or_completed: True ab erster Bar nach OR-Ende
    or_completed = post_or.copy()

    out = pd.DataFrame(index=df.index)
    out["in_opening_range"] = in_or
    out["or_high"] = or_high
    out["or_low"] = or_low
    out["or_completed"] = or_completed

    logger.debug(
        "Opening Range v2: {} in_or Bars, {} post_or Bars",
        int(in_or.sum()),
        int(post_or.sum()),
    )
    return out
