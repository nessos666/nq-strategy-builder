"""
PDA-Algo 20: Sessions Asia / London / NY + ICT Killzones v2.

ICT-Definition (ET = US/Eastern):
  Asia   : 19:00–04:00 ET  (über Mitternacht)
  London : 02:00–11:00 ET
  NY     : 08:00–17:00 ET

ICT Killzones (Hochwahrscheinlichkeits-Fenster):
  London Open KZ : 02:00–05:00 ET
  NY Open KZ     : 07:00–11:00 ET  (NY AM)
  NY PM KZ       : 13:30–16:00 ET  (Silver Bullet)

v2 Änderungen gegenüber v1:
- compute_sessions() statt run()
- @dataclass(frozen=True)
- loguru statt print
- from __future__ import annotations
- Echter tz_convert statt manuellem -5h-Offset (v1-Bug gefixt!)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class SessionsConfig:
    """Konfiguration Sessions v2."""

    timezone: str = "America/New_York"
    # Breite Sessions (Stunden ET)
    asia_start: int = 19
    asia_end: int = 4
    london_start: int = 2
    london_end: int = 11
    ny_start: int = 8
    ny_end: int = 17
    # Killzones (ICT-Standard)
    kz_london_start: int = 2
    kz_london_end: int = 5
    kz_ny_am_start: int = 7
    kz_ny_am_end: int = 11
    kz_ny_pm_start_h: int = 13
    kz_ny_pm_start_m: int = 30
    kz_ny_pm_end_h: int = 16
    kz_ny_pm_end_m: int = 0


def _in_window(hour: pd.Series, start: int, end: int) -> pd.Series:
    """True wenn hour im Fenster [start, end) liegt – unterstützt Übernacht-Fenster."""
    if start < end:
        return (hour >= start) & (hour < end)
    # Übernacht-Fenster, z.B. Asia 19:00–04:00
    return (hour >= start) | (hour < end)


def compute_sessions(
    df: pd.DataFrame,
    config: SessionsConfig | None = None,
) -> pd.DataFrame:
    """
    Session-Flags und ICT-Killzone-Flags für jeden Bar.

    Input : OHLCV DataFrame (beliebiger DatetimeIndex, UTC-aware oder tz-naiv als UTC).
    Output: DataFrame mit session_*/kz_*-Spalten (gleicher Index wie df).

    session_asia    : Bool – Bar liegt in Asia-Session
    session_london  : Bool – Bar liegt in London-Session
    session_ny      : Bool – Bar liegt in NY-Session
    session_name    : "asia" | "london" | "ny" | "overlap" | ""
    kz_london_open  : Bool – London-Open-Killzone 02:00–05:00 ET
    kz_ny_open      : Bool – NY-AM-Killzone 07:00–11:00 ET
    kz_ny_pm        : Bool – NY-PM-Killzone 13:30–16:00 ET
    """
    if config is None:
        config = SessionsConfig()

    def _empty_result() -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["session_asia"] = False
        out["session_london"] = False
        out["session_ny"] = False
        out["session_name"] = ""
        out["kz_london_open"] = False
        out["kz_ny_open"] = False
        out["kz_ny_pm"] = False
        return out

    if df.empty:
        logger.debug("Sessions v2: leerer DataFrame")
        return _empty_result()

    tz = config.timezone

    # Index in Zielzeitzone konvertieren (kein manueller Offset!)
    try:
        idx_et = df.index.tz_convert(tz)
    except TypeError:
        idx_et = df.index.tz_localize("UTC").tz_convert(tz)

    hour = pd.Series(idx_et.hour, index=df.index)
    minute = pd.Series(idx_et.minute, index=df.index)

    # Breite Sessions
    asia = _in_window(hour, config.asia_start, config.asia_end)
    london = _in_window(hour, config.london_start, config.london_end)
    ny = _in_window(hour, config.ny_start, config.ny_end)

    # session_name: bei mehreren aktiven Sessions → "overlap"
    active_count = asia.astype(int) + london.astype(int) + ny.astype(int)
    session_name = pd.Series("", index=df.index, dtype=str)
    session_name[asia & (active_count == 1)] = "asia"
    session_name[london & (active_count == 1)] = "london"
    session_name[ny & (active_count == 1)] = "ny"
    session_name[active_count >= 2] = "overlap"

    # Killzones (stundengenaue Fenster)
    kz_london_open = _in_window(hour, config.kz_london_start, config.kz_london_end)
    kz_ny_open = _in_window(hour, config.kz_ny_am_start, config.kz_ny_am_end)

    # NY PM Killzone: Minutenpräzision
    t_min = hour * 60 + minute
    kz_ny_pm_start_min = config.kz_ny_pm_start_h * 60 + config.kz_ny_pm_start_m
    kz_ny_pm_end_min = config.kz_ny_pm_end_h * 60 + config.kz_ny_pm_end_m
    kz_ny_pm = (t_min >= kz_ny_pm_start_min) & (t_min < kz_ny_pm_end_min)

    out = pd.DataFrame(index=df.index)
    out["session_asia"] = asia
    out["session_london"] = london
    out["session_ny"] = ny
    out["session_name"] = session_name
    out["kz_london_open"] = kz_london_open
    out["kz_ny_open"] = kz_ny_open
    out["kz_ny_pm"] = kz_ny_pm

    logger.debug(
        "Sessions v2: {} Asia / {} London / {} NY Bars",
        int(asia.sum()),
        int(london.sum()),
        int(ny.sum()),
    )
    return out
