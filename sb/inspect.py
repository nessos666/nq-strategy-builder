"""
Baustein-Inspektor – Analysiert wann ein einzelner Algo Signale erzeugt.

Kein Backtest, kein Strategy-Builder. Nur:
    1. Algo-Datei finden (nach Namens-Teilstring)
    2. Algo auf vollem Datensatz ausführen
    3. Signal-Spalten auto-erkennen
    4. Signale nach NY-Zeitfenster (30 min) gruppieren
    5. Rate normalisieren (Signale pro Tag, nicht absolut)
    6. Optionale Heatmap: Zeit × Wochentag
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

_OHLCV_COLS: frozenset[str] = frozenset(
    {"open", "high", "low", "close", "volume", "Open", "High", "Low", "Close", "Volume"}
)

_DOW_NAMES = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# ── NY-Zeit Hilfsfunktion ───────────────────────────────────────────────────────


def _to_ny_series(index: pd.Index) -> pd.Series:
    """Konvertiert Index zu NY-Zeit und gibt als pd.Series zurück.

    Identisches Muster wie in '2. ATR Session.py'.
    """
    s = pd.Series(pd.DatetimeIndex(index))
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    else:
        s = s.dt.tz_convert("US/Eastern")
    return s


# ── Algo-Suche ─────────────────────────────────────────────────────────────────


def find_algo_file(name: str, dirs: list[Path]) -> Path | None:
    """Sucht eine Algo-Datei anhand eines Namens-Teilstrings.

    Matching: name.lower() muss im Stem der .py-Datei vorkommen.
    Erste Übereinstimmung aus erster passender Directory wird zurückgegeben.
    """
    name_lower = name.lower()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            if name_lower in f.stem.lower():
                return f
    return None


# ── Algo dynamisch laden ────────────────────────────────────────────────────────


def run_algo(algo_file: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Lädt Algo-Modul dynamisch und ruft run(df) auf.

    Erwartet dass df Spalten High/Low/Close/Open hat (Großbuchstaben).
    Wirft AttributeError wenn kein run() im Modul.
    """
    algo_dir = str(algo_file.parent)
    nq_root = str(algo_file.parent.parent.parent)
    added: list[str] = []
    for p in (algo_dir, nq_root):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        spec = importlib.util.spec_from_file_location(algo_file.stem, algo_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Kann Algo nicht laden: {algo_file}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = ""
        sys.modules[algo_file.stem] = mod
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        finally:
            sys.modules.pop(algo_file.stem, None)
        if hasattr(mod, "run"):
            return mod.run(df)  # type: ignore[no-any-return]
        raise AttributeError(f"kein run() in {algo_file.name}")
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


# ── Signal-Spalten erkennen ─────────────────────────────────────────────────────


def detect_signal_columns(
    df: pd.DataFrame,
    original_cols: set[str] | None = None,
) -> list[str]:
    """Erkennt Signal-Spalten im Algo-Output.

    Signal = bool-Spalte ODER numerische Spalte die nur 0, 1 und NaN enthält.
    OHLCV-Spalten und original_cols (Spalten vor Algo-Lauf) werden ausgeschlossen.
    """
    exclude = _OHLCV_COLS | (original_cols or set())
    result: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if s.dtype == bool or str(s.dtype) == "boolean":
            result.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            unique_vals = set(s.dropna().unique())
            if unique_vals <= {0, 1, 0.0, 1.0, True, False}:
                result.append(col)
    return result


# ── Zeitfenster-Gruppierung ─────────────────────────────────────────────────────


def _make_event_masks(df: pd.DataFrame, signal_cols: list[str]) -> dict[str, pd.Series]:
    """Gibt pro Signal-Spalte eine bool-Maske zurück die nur beim 0→1 Übergang True ist.

    Beispiel FVG: fvg_bull bleibt 1 für 200 Bars → event_mask ist nur am ersten Bar True.
    """
    masks: dict[str, pd.Series] = {}
    for col in signal_cols:
        s = df[col].fillna(0)
        masks[col] = (s == 1) & (s.shift(1).fillna(0) != 1)
    return masks


def group_signals_by_window(
    df: pd.DataFrame,
    signal_cols: list[str],
    window_minutes: int = 30,
    events_only: bool = False,
) -> list[dict]:
    """Gruppiert Signal-Bars nach NY-Zeitfenster.

    events_only=True: zählt nur den ersten Bar eines Signals (0→1 Übergang).
    events_only=False: zählt jeden Bar wo das Signal aktiv ist (Standard).

    Gibt nur Fenster zurück die mindestens 1 Signal haben.
    Jeder Eintrag: {
        window: "HH:MM",
        bull: int, bear: int, other: int, total: int,
        rate_per_day: float  – Signale pro Handelstag (normalisiert)
    }
    """
    if df.empty or not signal_cols:
        return []

    ny = _to_ny_series(df.index)

    # Anzahl eindeutiger Handelstage (für Rate-Normalisierung)
    total_days = max(len(set(ny.dt.date)), 1)

    # Minuten-seit-Mitternacht → Fenster-Bucket
    minutes = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(dtype=int)
    bucket = (minutes // window_minutes) * window_minutes

    # Events-Only: nur 0→1 Übergänge zählen
    event_masks = _make_event_masks(df, signal_cols) if events_only else None

    buckets: dict[int, dict[str, int]] = {}
    for i, b in enumerate(bucket):
        row = df.iloc[i]
        for col in signal_cols:
            val = event_masks[col].iloc[i] if event_masks is not None else row[col]
            if not val:
                continue
            if b not in buckets:
                buckets[b] = {"bull": 0, "bear": 0, "other": 0}
            if "_bull" in col:
                buckets[b]["bull"] += 1
            elif "_bear" in col:
                buckets[b]["bear"] += 1
            else:
                buckets[b]["other"] += 1

    result = []
    for b in sorted(buckets):
        h = b // 60
        m = b % 60
        d = buckets[b]
        total = d["bull"] + d["bear"] + d["other"]
        if total > 0:
            result.append(
                {
                    "window": f"{h:02d}:{m:02d}",
                    "bull": d["bull"],
                    "bear": d["bear"],
                    "other": d["other"],
                    "total": total,
                    "rate_per_day": round(total / total_days, 3),
                }
            )
    return result


def build_heatmap(
    df: pd.DataFrame,
    signal_cols: list[str],
    window_minutes: int = 30,
    events_only: bool = False,
) -> dict[str, dict[str, int]]:
    """Erstellt Heatmap-Daten: {window → {wochentag → count}}.

    events_only=True: nur 0→1 Übergänge zählen.
    Wochentage: "Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"
    Gibt nur Fenster zurück mit mindestens 1 Signal.
    """
    if df.empty or not signal_cols:
        return {}

    ny = _to_ny_series(df.index)
    minutes = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(dtype=int)
    bucket = (minutes // window_minutes) * window_minutes
    weekdays = ny.dt.weekday.to_numpy(dtype=int)  # 0=Mo … 6=So

    event_masks = _make_event_masks(df, signal_cols) if events_only else None

    heatmap: dict[str, dict[str, int]] = {}
    for i, b in enumerate(bucket):
        for col in signal_cols:
            val = (
                event_masks[col].iloc[i] if event_masks is not None else df.iloc[i][col]
            )
            if not val:
                continue
            h = b // 60
            m = b % 60
            key = f"{h:02d}:{m:02d}"
            dow = _DOW_NAMES[weekdays[i]]
            if key not in heatmap:
                heatmap[key] = {}
            heatmap[key][dow] = heatmap[key].get(dow, 0) + 1

    return heatmap


# ── Zone Outcome-Analyse (generisch) ────────────────────────────────────────────


def detect_zone_prefixes(df: pd.DataFrame) -> list[str]:
    """Erkennt Zone-Prefixe automatisch in einem Algo-Output DataFrame.

    Unterstützt zwei Signal-Namenskonventionen:
      - {prefix}_bull / {prefix}_bear         (FVG Standard)
      - {prefix}_bullish / {prefix}_bearish   (iFVG – original name, Strategy Builder)

    Benötigt immer: {prefix}_bull_high, {prefix}_bull_low, {prefix}_bull_filled.

    Gibt z.B. ["fvg", "ifvg"] zurück.
    """
    cols = set(df.columns)
    prefixes: list[str] = []
    for col in df.columns:
        # Beide Muster prüfen
        if col.endswith("_bull"):
            prefix = col[:-5]
        elif col.endswith("_bullish"):
            prefix = col[:-8]
        else:
            continue
        needed = {
            f"{prefix}_bull_high",
            f"{prefix}_bull_low",
            f"{prefix}_bull_filled",
        }
        if needed.issubset(cols) and prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


# ── Level-Spalten erkennen ───────────────────────────────────────────────────────

_ZONE_SUFFIXES: frozenset[str] = frozenset(
    {
        "_bull_high",
        "_bull_low",
        "_bear_high",
        "_bear_low",
        "_bull_filled",
        "_bear_filled",
    }
)


def detect_level_columns(df: pd.DataFrame) -> list[str]:
    """Erkenne alle Preis-Level-Spalten im DataFrame.

    Eingeschlossen: float/int-Spalten die auf _high oder _low enden
                    und KEINE OHLCV- oder Zonen-Spalten sind.
    Ausgeschlossen: Open/High/Low/Close/Volume, fvg_bull_high etc.
    """
    result = []
    for col in df.columns:
        if col in _OHLCV_COLS:
            continue
        if any(col.endswith(s) for s in _ZONE_SUFFIXES):
            continue
        if not (col.endswith("_high") or col.endswith("_low")):
            continue
        if df[col].dtype.kind not in ("f", "i"):
            continue
        result.append(col)
    return result


def compute_level_touch_heatmap(
    df: pd.DataFrame,
    level_col: str,
    threshold: float = 3.0,
    window_minutes: int = 30,
) -> dict[str, dict[str, int]]:
    """Touch-Heatmap für einen Preis-Level (PDH, PDL, PWH, PWL etc.).

    Touch-Definition: High >= level - threshold OR Low <= level + threshold.
    Rising-Edge: nur der erste Bar einer zusammenhängenden Touch-Episode zählt.

    Gibt {fenster → {wochentag → count}} zurück (identisches Format wie build_heatmap).
    """
    import numpy as np

    lv = df[level_col].to_numpy(dtype=float)
    hi = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)

    valid_level = ~np.isnan(lv)
    touch_raw = (hi >= lv - threshold) & (lo <= lv + threshold) & valid_level

    touch_series = pd.Series(touch_raw, index=df.index)
    rising = touch_series & ~touch_series.shift(1).fillna(False)

    ny = _to_ny_series(df.index)
    minutes = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(dtype=int)
    bucket = (minutes // window_minutes) * window_minutes
    weekdays = ny.dt.weekday.to_numpy(dtype=int)

    heatmap: dict[str, dict[str, int]] = {}
    rising_arr = rising.to_numpy()
    for i in range(len(rising_arr)):
        if not rising_arr[i]:
            continue
        h = bucket[i] // 60
        m = bucket[i] % 60
        key = f"{h:02d}:{m:02d}"
        dow = _DOW_NAMES[weekdays[i]]
        if key not in heatmap:
            heatmap[key] = {}
        heatmap[key][dow] = heatmap[key].get(dow, 0) + 1

    return heatmap


def analyze_zone_outcomes(df: pd.DataFrame, prefix: str) -> dict:
    """Generische Zone-Outcome-Analyse für jeden Zonen-Baustein.

    Funktioniert für jeden Algo der folgende Spalten ausgibt:
      {prefix}_bull / {prefix}_bear          – Signal (neue Zone)
      {prefix}_bull_high / {prefix}_bull_low – Zonen-Grenzen (ffill)
      {prefix}_bull_filled                   – Zone gefüllt (bool)
      {prefix}_bear_low  / {prefix}_bear_high
      {prefix}_bear_filled

    Für jede Zone-Episode:
      - Touch (Bull, von oben):  Low <= {prefix}_bull_high
      - Touch (Bear, von unten): High >= {prefix}_bear_low
      - Durch:  filled = True
      - Bounce: Touch aber kein Durch
      - Keine Berührung: Preis kommt nie rein

    Penetrations-Tiefe (nur bei Berührungen):
      depth_pts: Punkte die der Preis IN die Zone eindringt
      depth_pct: Tiefe als % der Zonengröße (0–100%+)
      Kategorien: <25% | 25–50% | 50–75% (CE-Bereich) | 75–100% | ≥100% (Durch)

    Wirft ValueError wenn Spalten fehlen.
    """
    # Signal-Spalte: _bull/_bear (FVG) oder _bullish/_bearish (iFVG)
    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    def _side_stats(
        signal_col: str,
        zone_entry_col: str,  # Kante die berührt wird (bull_high oder bear_low)
        zone_far_col: str,  # gegenüberliegende Kante (bull_low oder bear_high)
        ohlc_col: str,  # "Low" (von oben) oder "High" (von unten)
        touch_op: str,  # "le" oder "ge"
        filled_col: str,
        depth_ohlc: str,  # "Low" für bull (tiefster Punkt), "High" für bear
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        subsequent = ~df[signal_col].astype(bool) & (zone_id > 0)

        entry_price = pd.to_numeric(df[zone_entry_col], errors="coerce")
        far_price = pd.to_numeric(df[zone_far_col], errors="coerce")
        has_zone = entry_price.notna()  # type: ignore[union-attr]

        if touch_op == "le":
            touched_per_bar = subsequent & has_zone & (df[ohlc_col] <= entry_price)
        else:
            touched_per_bar = subsequent & has_zone & (df[ohlc_col] >= entry_price)

        filled_per_bar = subsequent & df[filled_col].fillna(False).astype(bool)

        # Aggregation nach Zone-Episode
        touched_by_zone = touched_per_bar.groupby(zone_id).any()
        filled_by_zone = filled_per_bar.groupby(zone_id).any()

        valid = touched_by_zone.index[touched_by_zone.index > 0]
        touched_by_zone = touched_by_zone.loc[valid]
        filled_by_zone = filled_by_zone.reindex(valid, fill_value=False)

        total_zones = len(valid)
        no_touch = int((~touched_by_zone).sum())
        through = int((touched_by_zone & filled_by_zone).sum())
        bounce = int((touched_by_zone & ~filled_by_zone).sum())
        touches = bounce + through

        # ── Penetrations-Tiefe ─────────────────────────────────────────────────
        # Tiefster Punkt bis zum ersten Fill-Bar (danach ignorieren).
        # Bounce-Zonen (nie gefüllt): gesamte Episode wird gewertet.
        # cumsum <= 1: Bars vor + am ersten Fill-Bar; >1 = Post-Fill → ausblenden.
        filled_cumsum = filled_per_bar.groupby(zone_id).cumsum()  # type: ignore[union-attr]
        depth_mask = subsequent & (filled_cumsum <= 1)
        depth_series = pd.to_numeric(df[depth_ohlc], errors="coerce")
        if touch_op == "le":
            extreme_by_zone = depth_series.where(depth_mask).groupby(zone_id).min()  # type: ignore[union-attr]
        else:
            extreme_by_zone = depth_series.where(depth_mask).groupby(zone_id).max()  # type: ignore[union-attr]

        # Zonen-Grenzen (konstant innerhalb Episode da ffill)
        entry_by_zone = entry_price.where(subsequent).groupby(zone_id).first()  # type: ignore[union-attr]
        far_by_zone = far_price.where(subsequent).groupby(zone_id).first()  # type: ignore[union-attr]
        zone_size_by_zone = (entry_by_zone - far_by_zone).abs()

        # Eindringstiefe in Punkten (von der Eintritts-Kante)
        if touch_op == "le":
            depth_pts_by_zone = (entry_by_zone - extreme_by_zone).clip(lower=0)
        else:
            depth_pts_by_zone = (extreme_by_zone - entry_by_zone).clip(lower=0)

        depth_pct_by_zone = (
            depth_pts_by_zone / zone_size_by_zone.replace(0, float("nan")) * 100
        )

        # Nur berührte Zonen für Tiefenstatistik
        touched_idx = valid[touched_by_zone.values]
        dp = depth_pct_by_zone.reindex(touched_idx).dropna()
        dpts = depth_pts_by_zone.reindex(touched_idx).dropna()

        depth_stats: dict = {}
        if len(dp) > 0:
            depth_stats = {
                "depth_pts_mean": round(float(dpts.mean()), 2),
                "depth_pts_median": round(float(dpts.median()), 2),
                "depth_pct_mean": round(float(dp.mean()), 1),
                "depth_pct_median": round(float(dp.median()), 1),
                "pct_lt25": round(float((dp < 25).mean() * 100), 1),
                "pct_25_50": round(float(((dp >= 25) & (dp < 50)).mean() * 100), 1),
                "pct_50_75": round(float(((dp >= 50) & (dp < 75)).mean() * 100), 1),
                "pct_75_100": round(float(((dp >= 75) & (dp < 100)).mean() * 100), 1),
                "pct_gte100": round(float((dp >= 100).mean() * 100), 1),
            }

        return {
            "total_zones": total_zones,
            "no_touch": no_touch,
            "touches": touches,
            "bounce": bounce,
            "through": through,
            "bounce_pct": round(bounce / touches * 100, 1) if touches > 0 else 0.0,
            "through_pct": round(through / touches * 100, 1) if touches > 0 else 0.0,
            "depth": depth_stats,
        }

    bull = _side_stats(
        bull_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        "Low",
        "le",
        f"{prefix}_bull_filled",
        "Low",
    )
    bear = _side_stats(
        bear_signal,
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        "High",
        "ge",
        f"{prefix}_bear_filled",
        "High",
    )
    return {"bull": bull, "bear": bear}


def analyze_fvg_outcomes(result_df: pd.DataFrame) -> dict:
    """Rückwärtskompatible Wrapper – ruft analyze_zone_outcomes mit prefix='fvg' auf."""
    return analyze_zone_outcomes(result_df, "fvg")


# ── Zone-Return-After-Through Analyse ────────────────────────────────────────


def analyze_zone_return_after_through(
    df: pd.DataFrame,
    prefix: str,
    return_windows: list[int] | None = None,
) -> dict:
    """Return-Rate nach Durch: Wie oft kommt Preis nach einem 'Durch' zurück?

    Für jede Zone die durchbrochen wurde (filled=True):
      - Bull-Zone (von oben): Preis durchsticht bull_low nach unten.
        Return = High kommt innerhalb N Bars wieder über bull_low
      - Bear-Zone (von unten): Preis durchsticht bear_high nach oben.
        Return = Low kommt innerhalb N Bars wieder unter bear_high

    return_windows: Liste von Bar-Anzahlen (auf 1min-Daten = Minuten).
                    Default: [15, 30, 60, 120]

    Gibt für jede Seite + jedes Fenster zurück:
      - n_through: Anzahl durchbrochener Zonen
      - return_N: Anzahl Returns innerhalb N Bars
      - return_pct_N: Return-Rate in %
    """
    if return_windows is None:
        return_windows = [15, 30, 60, 120]

    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    def _return_stats(
        signal_col: str,
        far_col: str,  # Grenze die durchbrochen wird (bull_low oder bear_high)
        filled_col: str,
        price_col: str,  # "High" (bull-Return) oder "Low" (bear-Return)
        return_op: str,  # "ge" (High >= far = return in bull) oder "le"
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        subsequent = ~df[signal_col].astype(bool) & (zone_id > 0)

        filled_per_bar = subsequent & df[filled_col].fillna(False).astype(bool)
        filled_by_zone = filled_per_bar.groupby(zone_id).any()
        valid = filled_by_zone.index[filled_by_zone.index > 0]
        through_zones = valid[filled_by_zone.loc[valid]]

        n_through = len(through_zones)
        if n_through == 0:
            result: dict = {"n_through": 0}
            for w in return_windows:
                result[f"return_{w}"] = 0
                result[f"return_pct_{w}"] = 0.0
            return result

        # Für jede durchbrochene Zone: Fill-Bar-Index finden
        # Dann in den nächsten N Bars prüfen ob Price zurückkommt
        far_prices = pd.to_numeric(df[far_col], errors="coerce")
        price_series = pd.to_numeric(df[price_col], errors="coerce")

        # Positionen im df (integer index)
        df.reset_index(drop=True)
        zone_id_arr = zone_id.values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        far_arr = far_prices.values
        price_arr = price_series.values

        returns: dict[int, int] = {w: 0 for w in return_windows}

        for zid in through_zones:
            # Fill-Bar: erster Bar wo filled=True für diese Zone
            mask = (zone_id_arr == zid) & filled_arr
            fill_positions = np.where(mask)[0]
            if len(fill_positions) == 0:
                continue
            fill_pos = fill_positions[0]

            # far_price dieser Zone (konstant wegen ffill)
            zone_mask = zone_id_arr == zid
            zone_far_vals = far_arr[zone_mask & ~df[signal_col].astype(bool).values]
            if len(zone_far_vals) == 0:
                continue
            far_val = zone_far_vals[0]
            if np.isnan(far_val):
                continue

            # Für jedes Return-Fenster: Bars NACH dem Fill-Bar prüfen
            for w in return_windows:
                end_pos = min(fill_pos + w + 1, len(price_arr))
                window_prices = price_arr[fill_pos + 1 : end_pos]
                if len(window_prices) == 0:
                    continue
                if return_op == "ge":
                    returned = np.any(window_prices >= far_val)
                else:
                    returned = np.any(window_prices <= far_val)
                if returned:
                    returns[w] += 1

        result = {"n_through": n_through}
        for w in return_windows:
            result[f"return_{w}"] = returns[w]
            result[f"return_pct_{w}"] = round(returns[w] / n_through * 100, 1)
        return result

    bull = _return_stats(
        bull_signal,
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        "High",  # Return: High kommt wieder über bull_low
        "ge",
    )
    bear = _return_stats(
        bear_signal,
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
        "Low",  # Return: Low kommt wieder unter bear_high
        "le",
    )
    return {"bull": bull, "bear": bear, "windows": return_windows}


# ── Zone-Break-Profile Analyse ───────────────────────────────────────────────


def analyze_zone_break_profile(
    df: pd.DataFrame,
    prefix: str,
    no_return_window: int = 60,
) -> dict:
    """Kategorisiert jede Zone in 3 Typen + Heatmap nach NY-Stunde.

    Kategorien:
      Hold         – Zone nie durchbrochen (filled immer False)
      Fake-Out     – durchbrochen + Preis kehrt innerhalb no_return_window Bars zurück
      Echter Break – durchbrochen + kein Return in no_return_window Bars

    Zeitdimension:
      - Hold-Zonen: Stunde des Signal-Bars (Zone-Entstehung)
      - Fake-Out + Break: Stunde des Fill-Bars (Durchbruch-Zeitpunkt)
      Alle Zeiten in NY (America/New_York).

    Gibt zurück:
      {
        "bull": {"hold": N, "fake_out": N, "real_break": N,
                 "hold_pct": %, "fake_out_pct": %, "real_break_pct": %,
                 "hold_by_hour": {0..23: count}, "fake_out_by_hour": {...}, "break_by_hour": {...}},
        "bear": {...},
        "no_return_window": int,
      }
    """
    import zoneinfo

    NY = zoneinfo.ZoneInfo("America/New_York")
    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    # NY-Stunden-Array vorbereiten
    if hasattr(df.index, "tz") and df.index.tz is not None:
        ny_hours = df.index.tz_convert(NY).hour
    else:
        ny_hours = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(NY).hour
    ny_hours_arr = np.array(ny_hours)

    def _profile(
        signal_col: str,
        filled_col: str,
        far_col: str,
        price_col: str,  # "High" für Bull, "Low" für Bear
        return_op: str,  # "ge" oder "le"
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        far_arr = pd.to_numeric(df[far_col], errors="coerce").values
        price_arr = pd.to_numeric(df[price_col], errors="coerce").values

        hold = 0
        fake_out = 0
        real_break = 0
        hold_by_hour: dict[int, int] = {}
        fake_out_by_hour: dict[int, int] = {}
        break_by_hour: dict[int, int] = {}

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid

            # Signal-Bar-Position (Entstehung der Zone)
            sig_positions = np.where(zone_mask & signal_arr)[0]
            if len(sig_positions) == 0:
                continue
            sig_pos = sig_positions[0]

            # Wurde die Zone je durchbrochen?
            sub_filled = filled_arr[zone_mask & ~signal_arr]
            ever_filled = np.any(sub_filled)

            if not ever_filled:
                # Hold
                hold += 1
                h = int(ny_hours_arr[sig_pos])
                hold_by_hour[h] = hold_by_hour.get(h, 0) + 1
                continue

            # Fill-Bar: erster Bar wo filled=True
            fill_positions = np.where(zone_mask & filled_arr)[0]
            if len(fill_positions) == 0:
                continue
            fill_pos = fill_positions[0]
            fill_hour = int(ny_hours_arr[fill_pos])

            # far_val dieser Zone
            zone_sub = zone_mask & ~signal_arr
            zone_far_vals = far_arr[zone_sub]
            if len(zone_far_vals) == 0:
                continue
            far_val = zone_far_vals[0]
            if np.isnan(far_val):
                continue

            # Return-Check in no_return_window Bars nach Fill
            end_pos = min(fill_pos + no_return_window + 1, len(price_arr))
            window_prices = price_arr[fill_pos + 1 : end_pos]
            if return_op == "ge":
                returned = len(window_prices) > 0 and np.any(window_prices >= far_val)
            else:
                returned = len(window_prices) > 0 and np.any(window_prices <= far_val)

            if returned:
                fake_out += 1
                fake_out_by_hour[fill_hour] = fake_out_by_hour.get(fill_hour, 0) + 1
            else:
                real_break += 1
                break_by_hour[fill_hour] = break_by_hour.get(fill_hour, 0) + 1

        total = hold + fake_out + real_break
        return {
            "hold": hold,
            "fake_out": fake_out,
            "real_break": real_break,
            "hold_pct": round(hold / total * 100, 1) if total else 0.0,
            "fake_out_pct": round(fake_out / total * 100, 1) if total else 0.0,
            "real_break_pct": round(real_break / total * 100, 1) if total else 0.0,
            "total": total,
            "hold_by_hour": hold_by_hour,
            "fake_out_by_hour": fake_out_by_hour,
            "break_by_hour": break_by_hour,
        }

    bull = _profile(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_low",
        "High",
        "ge",
    )
    bear = _profile(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        "Low",
        "le",
    )
    return {"bull": bull, "bear": bear, "no_return_window": no_return_window}


# ── Zone-Break-Duration Analyse ──────────────────────────────────────────────


def analyze_zone_break_duration(
    df: pd.DataFrame,
    prefix: str,
    max_window: int = 120,
) -> dict:
    """Wie lange ist der Preis AUSSERHALB einer Zone nach einem Ausbruch?

    Für jeden Ausbruch (filled):
      - Bull OB: Preis bricht unter bull_low → zähle Bars bis High >= bull_low
      - Bear OB: Preis bricht über bear_high → zähle Bars bis Low <= bear_high

    Gibt zurück:
      - Verteilung der Draußen-Dauer in Buckets [1-5, 6-10, 11-15, 16-20, 21-30, 31-60, 61-120, 120+]
      - Median + Mean Draußen-Dauer (in Bars / Minuten auf 1min-Daten)
      - % die nach 5 / 10 / 15 / 20 / 30 / 60 / 120 Bars noch draußen sind
      - Separiert nach erstem und zweitem+ Ausbruch (via Zone-Reihenfolge)

    max_window: Bars nach fill_pos die maximal beobachtet werden (= "permanent break" wenn überschritten)
    """
    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    BUCKETS = [5, 10, 15, 20, 30, 60, 120]  # Schwellenwerte in Bars

    def _duration_stats(
        signal_col: str,
        filled_col: str,
        boundary_col: str,  # bull_low (Bull) oder bear_high (Bear)
        price_col: str,  # "High" (bull-Return) oder "Low" (bear-Return)
        return_op: str,  # "ge" oder "le"
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        boundary_arr = pd.to_numeric(df[boundary_col], errors="coerce").values
        price_arr = pd.to_numeric(df[price_col], errors="coerce").values

        durations: list[int] = []  # Bars draußen bis Return (oder max_window)
        permanent: int = 0  # kein Return im Fenster

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid
            sub_filled = filled_arr[zone_mask & ~signal_arr]
            if not np.any(sub_filled):
                continue  # Hold – kein Ausbruch

            # Fill-Bar: erster Bar wo filled=True
            fill_positions = np.where(zone_mask & filled_arr)[0]
            if len(fill_positions) == 0:
                continue
            fill_pos = fill_positions[0]

            # Boundary-Wert dieser Zone (ffill → konstant)
            zone_sub = zone_mask & ~signal_arr
            bvals = boundary_arr[zone_sub]
            if len(bvals) == 0 or np.isnan(bvals[0]):
                continue
            boundary_val = bvals[0]

            # Beobachtungsfenster: max_window Bars nach fill_pos
            end_pos = min(fill_pos + max_window + 1, len(price_arr))
            window_prices = price_arr[fill_pos + 1 : end_pos]

            # Finde ersten Bar wo Preis zurückkommt
            if return_op == "ge":
                returned_mask = window_prices >= boundary_val
            else:
                returned_mask = window_prices <= boundary_val

            if not np.any(returned_mask):
                permanent += 1
                durations.append(max_window + 1)  # als "zu lange" markiert
            else:
                first_return = (
                    int(np.argmax(returned_mask)) + 1
                )  # +1 weil ab fill_pos+1
                durations.append(first_return)

        n = len(durations)
        if n == 0:
            return {"n": 0, "durations": [], "permanent": 0, "buckets": {}}

        dur_arr = np.array(durations)
        valid = dur_arr[dur_arr <= max_window]

        # Verteilung: Anteil der Ausbrüche die nach N Bars NOCH draußen sind
        still_out: dict[int, float] = {}
        for t in BUCKETS:
            still_out[t] = round(float(np.mean(dur_arr > t)) * 100, 1)

        # Bucket-Zähler: wie viele Return in diesem Fenster
        bucket_counts: dict[str, int] = {}
        prev = 0
        for b in BUCKETS:
            count = int(np.sum((dur_arr > prev) & (dur_arr <= b)))
            bucket_counts[f"{prev + 1}-{b}"] = count
            prev = b
        bucket_counts[f"{max_window + 1}+"] = permanent

        return {
            "n": n,
            "n_permanent": permanent,
            "permanent_pct": round(permanent / n * 100, 1) if n else 0.0,
            "median_bars": int(np.median(valid)) if len(valid) else max_window,
            "mean_bars": round(float(np.mean(valid)), 1)
            if len(valid)
            else float(max_window),
            "still_out_pct": still_out,  # % die nach N Bars noch draußen
            "bucket_counts": bucket_counts,  # Absolute Zahlen pro Bucket
        }

    bull = _duration_stats(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_low",
        "High",
        "ge",
    )
    bear = _duration_stats(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        "Low",
        "le",
    )
    return {"bull": bull, "bear": bear, "max_window": max_window}


# ── Zone-Bounce-Depth Analyse ─────────────────────────────────────────────────


def analyze_zone_bounce_depth(
    df: pd.DataFrame,
    prefix: str,
) -> dict:
    """Penetrations-Tiefe bei Bounces: wie tief geht Preis in die Zone?

    Nur für Zonen die NIE gefüllt wurden (Hold-Zonen).
    Tiefe = (max_penetration / zone_height) * 100 in %.

    Bull OB [low, high]: Preis kommt von oben rein.
      max_penetration = max(zone_high - bar_low) für alle Bars in Zone
      zone_height = zone_high - zone_low
    Bear OB [low, high]: Preis kommt von unten rein.
      max_penetration = max(bar_high - zone_low) für alle Bars in Zone
      zone_height = zone_high - zone_low

    Gibt zurück:
      Buckets: 0-10%, 10-25%, 25-50%, 50-75%, 75-100%
      Median + Mean Tiefe in %
      Tiefe in Punkten (Median + Mean)
    """
    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    DEPTH_BUCKETS = [10, 25, 50, 75, 100]

    def _depth_stats(
        signal_col: str,
        filled_col: str,
        zone_high_col: str,
        zone_low_col: str,
        bar_extreme_col: str,  # "Low" für Bull, "High" für Bear
        penetration_op: str,  # "from_top" (bull) oder "from_bottom" (bear)
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        zone_high_arr = pd.to_numeric(df[zone_high_col], errors="coerce").values
        zone_low_arr = pd.to_numeric(df[zone_low_col], errors="coerce").values
        extreme_arr = pd.to_numeric(df[bar_extreme_col], errors="coerce").values

        depths_pct: list[float] = []
        depths_pts: list[float] = []

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid
            sub_filled = filled_arr[zone_mask & ~signal_arr]
            if np.any(sub_filled):
                continue  # Nur Hold-Zonen

            # Zone-Grenzen (aus Signal-Bar)
            sig_positions = np.where(zone_mask & signal_arr)[0]
            if len(sig_positions) == 0:
                continue
            sig_pos = sig_positions[0]
            z_high = zone_high_arr[sig_pos]
            z_low = zone_low_arr[sig_pos]
            zone_height = z_high - z_low
            if zone_height <= 0 or np.isnan(zone_height):
                continue

            # Alle Bars innerhalb dieser Zone-Periode
            sub_mask = zone_mask & ~signal_arr
            extremes = extreme_arr[sub_mask]
            if len(extremes) == 0:
                continue

            if penetration_op == "from_top":
                # Bull OB: Preis kommt von oben. Tiefe = wie weit unter zone_high
                max_pen_pts = float(np.nanmax(z_high - extremes))
            else:
                # Bear OB: Preis kommt von unten. Tiefe = wie weit über zone_low
                max_pen_pts = float(np.nanmax(extremes - z_low))

            max_pen_pts = max(0.0, max_pen_pts)
            depth_pct = min(100.0, (max_pen_pts / zone_height) * 100)
            depths_pct.append(depth_pct)
            depths_pts.append(max_pen_pts)

        n = len(depths_pct)
        if n == 0:
            return {"n": 0}

        arr_pct = np.array(depths_pct)
        arr_pts = np.array(depths_pts)

        bucket_counts: dict[str, int] = {}
        prev = 0
        for b in DEPTH_BUCKETS:
            bucket_counts[f"{prev}-{b}%"] = int(
                np.sum((arr_pct > prev) & (arr_pct <= b))
            )
            prev = b

        return {
            "n": n,
            "median_pct": round(float(np.median(arr_pct)), 1),
            "mean_pct": round(float(np.mean(arr_pct)), 1),
            "median_pts": round(float(np.median(arr_pts)), 1),
            "mean_pts": round(float(np.mean(arr_pts)), 1),
            "bucket_counts": bucket_counts,
        }

    bull = _depth_stats(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        "Low",
        "from_top",
    )
    bear = _depth_stats(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        f"{prefix}_bear_low",
        "High",
        "from_bottom",
    )
    return {"bull": bull, "bear": bear}


# ── Zone-By-Session Analyse ───────────────────────────────────────────────────


def analyze_zone_by_session(
    df: pd.DataFrame,
    prefix: str,
    no_return_window: int = 60,
) -> dict:
    """Hold/Break-Rate nach Session in der der OB entstanden ist.

    Sessions (NY-Zeit):
      Asia:    18:00–00:00 (Vorabend)
      London:  02:00–07:59
      PreM:    08:00–09:29
      AM:      09:30–11:59
      Lunch:   12:00–13:29
      PM:      13:30–15:59
      AH:      16:00–17:59

    Gibt pro Session + Seite zurück: n, hold, fake_out, real_break (%)
    """
    import zoneinfo

    NY = zoneinfo.ZoneInfo("America/New_York")

    SESSION_MAP = [
        ("Asia", 18, 24),
        ("Asia", 0, 2),
        ("London", 2, 8),
        ("PreM", 8, 9.5),
        ("AM", 9.5, 12),
        ("Lunch", 12, 13.5),
        ("PM", 13.5, 16),
        ("AH", 16, 18),
    ]

    def _hour_to_session(h: float) -> str:
        for name, start, end in SESSION_MAP:
            if start <= h < end:
                return name
        return "Other"

    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    if hasattr(df.index, "tz") and df.index.tz is not None:
        ny_hours = df.index.tz_convert(NY).hour + df.index.tz_convert(NY).minute / 60
    else:
        idx_ny = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(NY)
        ny_hours = idx_ny.hour + idx_ny.minute / 60
    ny_hours_arr = np.array(ny_hours)

    def _session_stats(
        signal_col: str,
        filled_col: str,
        far_col: str,
        price_col: str,
        return_op: str,
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        far_arr = pd.to_numeric(df[far_col], errors="coerce").values
        price_arr = pd.to_numeric(df[price_col], errors="coerce").values

        sessions = ["Asia", "London", "PreM", "AM", "Lunch", "PM", "AH"]
        result: dict = {
            s: {"hold": 0, "fake_out": 0, "real_break": 0} for s in sessions
        }
        result["Other"] = {"hold": 0, "fake_out": 0, "real_break": 0}

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid
            sig_positions = np.where(zone_mask & signal_arr)[0]
            if len(sig_positions) == 0:
                continue
            sig_pos = sig_positions[0]
            session = _hour_to_session(float(ny_hours_arr[sig_pos]))

            sub_filled = filled_arr[zone_mask & ~signal_arr]
            ever_filled = np.any(sub_filled)

            if not ever_filled:
                result[session]["hold"] += 1
                continue

            fill_positions = np.where(zone_mask & filled_arr)[0]
            if len(fill_positions) == 0:
                continue
            fill_pos = fill_positions[0]

            zone_sub = zone_mask & ~signal_arr
            far_vals = far_arr[zone_sub]
            if len(far_vals) == 0 or np.isnan(far_vals[0]):
                continue
            far_val = far_vals[0]

            end_pos = min(fill_pos + no_return_window + 1, len(price_arr))
            window = price_arr[fill_pos + 1 : end_pos]
            if return_op == "ge":
                returned = len(window) > 0 and np.any(window >= far_val)
            else:
                returned = len(window) > 0 and np.any(window <= far_val)

            if returned:
                result[session]["fake_out"] += 1
            else:
                result[session]["real_break"] += 1

        # Prozente berechnen
        for s in result:
            d = result[s]
            total = d["hold"] + d["fake_out"] + d["real_break"]
            d["total"] = total
            d["hold_pct"] = round(d["hold"] / total * 100, 1) if total else 0.0
            d["fake_out_pct"] = round(d["fake_out"] / total * 100, 1) if total else 0.0
            d["real_break_pct"] = (
                round(d["real_break"] / total * 100, 1) if total else 0.0
            )
        return result

    bull = _session_stats(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_low",
        "High",
        "ge",
    )
    bear = _session_stats(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        "Low",
        "le",
    )
    return {"bull": bull, "bear": bear, "no_return_window": no_return_window}


# ── Zone-Context-Filter Analyse ───────────────────────────────────────────────


def analyze_zone_with_context(
    df: pd.DataFrame,
    prefix: str,
    context_col: str,
    no_return_window: int = 60,
) -> dict:
    """Vergleicht Hold/Break-Rate einer Zone mit vs. ohne Context-Signal.

    context_col: Boolean-Spalte im df (z.B. 'manip_bear') die auf True steht
                 wenn der Kontext aktiv ist.

    Gibt für beide Gruppen (context=True, context=False) zurück:
      n, hold_pct, fake_out_pct, real_break_pct
    """
    cols = set(df.columns)
    if context_col not in cols:
        raise ValueError(f"Context-Spalte '{context_col}' nicht gefunden")

    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    context_arr = df[context_col].fillna(False).astype(bool).values

    def _context_stats(
        signal_col: str,
        filled_col: str,
        far_col: str,
        price_col: str,
        return_op: str,
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        far_arr = pd.to_numeric(df[far_col], errors="coerce").values
        price_arr = pd.to_numeric(df[price_col], errors="coerce").values

        groups: dict = {
            True: {"hold": 0, "fake_out": 0, "real_break": 0},
            False: {"hold": 0, "fake_out": 0, "real_break": 0},
        }

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid
            sig_positions = np.where(zone_mask & signal_arr)[0]
            if len(sig_positions) == 0:
                continue
            # Context = aktiv wenn am Signal-Bar oder innerhalb der Zone aktiv
            ctx_in_zone = context_arr[zone_mask]
            ctx_active = bool(np.any(ctx_in_zone))

            sub_filled = filled_arr[zone_mask & ~signal_arr]
            ever_filled = np.any(sub_filled)

            if not ever_filled:
                groups[ctx_active]["hold"] += 1
                continue

            fill_positions = np.where(zone_mask & filled_arr)[0]
            if len(fill_positions) == 0:
                continue
            fill_pos = fill_positions[0]

            zone_sub = zone_mask & ~signal_arr
            far_vals = far_arr[zone_sub]
            if len(far_vals) == 0 or np.isnan(far_vals[0]):
                continue
            far_val = far_vals[0]

            end_pos = min(fill_pos + no_return_window + 1, len(price_arr))
            window = price_arr[fill_pos + 1 : end_pos]
            if return_op == "ge":
                returned = len(window) > 0 and np.any(window >= far_val)
            else:
                returned = len(window) > 0 and np.any(window <= far_val)

            if returned:
                groups[ctx_active]["fake_out"] += 1
            else:
                groups[ctx_active]["real_break"] += 1

        for key in groups:
            d = groups[key]
            total = d["hold"] + d["fake_out"] + d["real_break"]
            d["total"] = total
            d["hold_pct"] = round(d["hold"] / total * 100, 1) if total else 0.0
            d["fake_out_pct"] = round(d["fake_out"] / total * 100, 1) if total else 0.0
            d["real_break_pct"] = (
                round(d["real_break"] / total * 100, 1) if total else 0.0
            )
        return groups

    bull = _context_stats(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_low",
        "High",
        "ge",
    )
    bear = _context_stats(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        "Low",
        "le",
    )
    return {"bull": bull, "bear": bear, "context_col": context_col}


# ── Zone-Visit-Sequence Analyse ───────────────────────────────────────────────


def analyze_zone_visit_sequence(
    df: pd.DataFrame,
    prefix: str,
    max_visits: int = 3,
    max_window: int = 120,
) -> dict:
    """Analysiert Besuche in einer Zone: wie verhält sich Preis beim 1., 2., 3. Besuch?

    Ein "Besuch" = eine zusammenhängende Sequenz von Bars wo Preis innerhalb der Zone ist.
    Für Bull OB [low, high]: drin = Low <= zone_high AND High >= zone_low
    Ausgang: "bounce" = nächste Bar nach Besuch hat Low > zone_high (Preis über Zone raus)
             "break"  = nächste Bar nach Besuch hat High < zone_low (Preis unter Zone raus)
             "unclear" = unentschlossen

    Gibt pro Visit-Nummer (1, 2, 3) zurück:
      n, avg_duration_bars, bounce_pct, break_pct, unclear_pct
    """
    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    def _visit_stats(
        signal_col: str,
        filled_col: str,
        zone_high_col: str,
        zone_low_col: str,
        bar_high_col: str,
        bar_low_col: str,
        direction: str,  # "bull" oder "bear"
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        zone_high_arr = pd.to_numeric(df[zone_high_col], errors="coerce").values
        zone_low_arr = pd.to_numeric(df[zone_low_col], errors="coerce").values
        bar_high_arr = pd.to_numeric(df[bar_high_col], errors="coerce").values
        bar_low_arr = pd.to_numeric(df[bar_low_col], errors="coerce").values

        visit_data: dict = {
            v: {"durations": [], "bounce": 0, "break_": 0, "unclear": 0}
            for v in range(1, max_visits + 1)
        }

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid
            sig_positions = np.where(zone_mask & signal_arr)[0]
            if len(sig_positions) == 0:
                continue
            sig_pos = sig_positions[0]
            z_high = zone_high_arr[sig_pos]
            z_low = zone_low_arr[sig_pos]
            if np.isnan(z_high) or np.isnan(z_low):
                continue

            # Alle Sub-Bars dieser Zone (nach Signal-Bar)
            sub_positions = np.where(zone_mask & ~signal_arr)[0]
            if len(sub_positions) == 0:
                continue

            # "In Zone" = Low <= z_high AND High >= z_low
            in_zone = (bar_low_arr[sub_positions] <= z_high) & (
                bar_high_arr[sub_positions] >= z_low
            )

            # Besuche finden: zusammenhängende True-Sequenzen
            visit_num = 0
            i = 0
            while i < len(in_zone) and visit_num < max_visits:
                if in_zone[i]:
                    # Beginn eines Besuchs
                    visit_num += 1
                    visit_start = i
                    while i < len(in_zone) and in_zone[i]:
                        i += 1
                    visit_end = i  # erster Bar nach Besuch

                    duration = visit_end - visit_start
                    visit_data[visit_num]["durations"].append(duration)

                    # Ausgang bestimmen
                    if visit_end < len(sub_positions):
                        exit_pos = sub_positions[visit_end]
                        if direction == "bull":
                            # Bounce: High > z_high (Preis über Zone raus)
                            # Break: Low < z_low (Preis unter Zone raus)
                            if bar_high_arr[exit_pos] > z_high:
                                visit_data[visit_num]["bounce"] += 1
                            elif bar_low_arr[exit_pos] < z_low:
                                visit_data[visit_num]["break_"] += 1
                            else:
                                visit_data[visit_num]["unclear"] += 1
                        else:  # bear
                            if bar_low_arr[exit_pos] < z_low:
                                visit_data[visit_num]["bounce"] += 1
                            elif bar_high_arr[exit_pos] > z_high:
                                visit_data[visit_num]["break_"] += 1
                            else:
                                visit_data[visit_num]["unclear"] += 1
                    else:
                        visit_data[visit_num]["unclear"] += 1
                else:
                    i += 1

        result: dict = {}
        for v in range(1, max_visits + 1):
            d = visit_data[v]
            n = d["bounce"] + d["break_"] + d["unclear"]
            durs = d["durations"]
            result[f"visit_{v}"] = {
                "n": n,
                "avg_duration": round(float(np.mean(durs)), 1) if durs else 0.0,
                "median_duration": int(np.median(durs)) if durs else 0,
                "bounce_pct": round(d["bounce"] / n * 100, 1) if n else 0.0,
                "break_pct": round(d["break_"] / n * 100, 1) if n else 0.0,
                "unclear_pct": round(d["unclear"] / n * 100, 1) if n else 0.0,
            }
        return result

    bull = _visit_stats(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        "High",
        "Low",
        "bull",
    )
    bear = _visit_stats(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        f"{prefix}_bear_low",
        "High",
        "Low",
        "bear",
    )
    return {"bull": bull, "bear": bear}


# ── Zone-Visit-Dauer Analyse ──────────────────────────────────────────────────


def analyze_zone_visit_duration(
    df: pd.DataFrame,
    prefix: str,
    max_visits: int = 4,
    post_death_window: int = 120,
) -> dict:
    """Wie lange bleibt Preis in der Zone? Wie lange draußen zwischen Besuchen?

    Pro Visit-Nummer (1..max_visits):
      n                     – Anzahl dieser Besuche über alle Zonen
      median_duration_inside – Median Bars drin
      avg_duration_inside    – Ø Bars drin
      median_outside_before  – Median Bars draußen vor diesem Besuch (0 für Visit 1)
      exit_return_pct        – % Exit durch Return (Preis über Zone hinaus)
      exit_break_pct         – % Exit durch Break (Preis unter Zone durch)
      exit_unclear_pct       – % unklar

    post_death (nach finalen Break-Exit):
      n                     – Anzahl finaler Break-Exits
      permanent             – kein Re-Touch in post_death_window Bars
      permanent_pct         – in %
      median_bars_to_retouch – Median Bars bis nächste Berührung (nur Non-Permanent)
    """
    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    def _dur_stats(
        signal_col: str,
        zone_high_col: str,
        zone_low_col: str,
        bar_high_col: str,
        bar_low_col: str,
        direction: str,  # "bull" oder "bear"
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        zone_high_arr = pd.to_numeric(df[zone_high_col], errors="coerce").values
        zone_low_arr = pd.to_numeric(df[zone_low_col], errors="coerce").values
        bar_high_arr = pd.to_numeric(df[bar_high_col], errors="coerce").values
        bar_low_arr = pd.to_numeric(df[bar_low_col], errors="coerce").values

        visit_data: dict = {
            v: {
                "durations": [],
                "outside_before": [],
                "exit_return": 0,
                "exit_break": 0,
                "exit_unclear": 0,
            }
            for v in range(1, max_visits + 1)
        }
        post_death: dict = {"n": 0, "permanent": 0, "bars_to_retouch": []}

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid
            sig_positions = np.where(zone_mask & signal_arr)[0]
            if len(sig_positions) == 0:
                continue
            sig_pos = sig_positions[0]
            z_high = zone_high_arr[sig_pos]
            z_low = zone_low_arr[sig_pos]
            if np.isnan(z_high) or np.isnan(z_low):
                continue

            sub_positions = np.where(zone_mask & ~signal_arr)[0]
            if len(sub_positions) == 0:
                continue

            # Bull zone: Low must be within [z_low, z_high] (price resting in support zone)
            # Bear zone: High must be within [z_low, z_high] (price resting in resistance zone)
            if direction == "bull":
                in_zone = (bar_low_arr[sub_positions] >= z_low) & (
                    bar_low_arr[sub_positions] <= z_high
                )
            else:
                in_zone = (bar_high_arr[sub_positions] >= z_low) & (
                    bar_high_arr[sub_positions] <= z_high
                )

            visit_num = 0
            i = 0
            last_visit_end_i = 0

            while i < len(in_zone) and visit_num < max_visits:
                if in_zone[i]:
                    visit_num += 1
                    visit_start_i = i
                    while i < len(in_zone) and in_zone[i]:
                        i += 1
                    visit_end_i = i

                    visit_data[visit_num]["durations"].append(
                        visit_end_i - visit_start_i
                    )
                    if visit_num > 1:
                        visit_data[visit_num]["outside_before"].append(
                            visit_start_i - last_visit_end_i
                        )

                    # Exit-Richtung bestimmen
                    if visit_end_i < len(sub_positions):
                        exit_pos = sub_positions[visit_end_i]
                        is_break = False
                        if direction == "bull":
                            if bar_high_arr[exit_pos] > z_high:
                                visit_data[visit_num]["exit_return"] += 1
                            elif bar_low_arr[exit_pos] < z_low:
                                visit_data[visit_num]["exit_break"] += 1
                                is_break = True
                            else:
                                visit_data[visit_num]["exit_unclear"] += 1
                        else:  # bear
                            if bar_low_arr[exit_pos] < z_low:
                                visit_data[visit_num]["exit_return"] += 1
                            elif bar_high_arr[exit_pos] > z_high:
                                visit_data[visit_num]["exit_break"] += 1
                                is_break = True
                            else:
                                visit_data[visit_num]["exit_unclear"] += 1
                        # Track post-death immediately on each break exit
                        if is_break:
                            post_death["n"] += 1
                            remaining = in_zone[visit_end_i:]
                            window = remaining[:post_death_window]
                            if np.any(window):
                                post_death["bars_to_retouch"].append(
                                    int(np.argmax(window))
                                )
                            else:
                                post_death["permanent"] += 1
                    else:
                        visit_data[visit_num]["exit_unclear"] += 1

                    last_visit_end_i = visit_end_i
                else:
                    i += 1

        result: dict = {}
        for v in range(1, max_visits + 1):
            d = visit_data[v]
            n = d["exit_return"] + d["exit_break"] + d["exit_unclear"]
            durs = d["durations"]
            out = d["outside_before"]
            result[f"visit_{v}"] = {
                "n": n,
                "median_duration_inside": int(np.median(durs)) if durs else 0,
                "avg_duration_inside": round(float(np.mean(durs)), 1) if durs else 0.0,
                "median_outside_before": int(np.median(out)) if out else 0,
                "exit_return_pct": round(d["exit_return"] / n * 100, 1) if n else 0.0,
                "exit_break_pct": round(d["exit_break"] / n * 100, 1) if n else 0.0,
                "exit_unclear_pct": round(d["exit_unclear"] / n * 100, 1) if n else 0.0,
            }

        n_pd = post_death["n"]
        btr = post_death["bars_to_retouch"]
        result["post_death"] = {
            "n": n_pd,
            "permanent": post_death["permanent"],
            "permanent_pct": round(post_death["permanent"] / n_pd * 100, 1)
            if n_pd
            else 0.0,
            "median_bars_to_retouch": int(np.median(btr)) if btr else post_death_window,
        }
        return result

    bull = _dur_stats(
        bull_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        "High",
        "Low",
        "bull",
    )
    bear = _dur_stats(
        bear_signal,
        f"{prefix}_bear_high",
        f"{prefix}_bear_low",
        "High",
        "Low",
        "bear",
    )
    return {"bull": bull, "bear": bear}


# ── Zone-Return-Excursion Analyse ─────────────────────────────────────────────


def analyze_zone_return_excursion(
    df: pd.DataFrame,
    prefix: str,
    return_window: int = 120,
) -> dict:
    """Maximale Gegenbewegung bei Fake-Outs: SL-Bereich Analyse.

    Nur für Fake-Outs (Zone gefüllt UND Preis kehrt innerhalb return_window zurück).

    Bull: Preis bricht unter bull_low → Return = High >= bull_low.
          Excursion = bull_low - min(Low) zwischen Fill und Return.
    Bear: Preis bricht über bear_high → Return = Low <= bear_high.
          Excursion = max(High) - bear_high zwischen Fill und Return.

    Output pro Seite:
      n_fake_outs, p50, p75, p80, p90, mean (alle in Punkten)
    """
    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    def _excursion_stats(
        signal_col: str,
        filled_col: str,
        boundary_col: str,
        return_price_col: str,
        extreme_col: str,
        return_op: str,
        excursion_op: str,
    ) -> dict:
        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        boundary_arr = pd.to_numeric(df[boundary_col], errors="coerce").values
        return_price_arr = pd.to_numeric(df[return_price_col], errors="coerce").values
        extreme_arr = pd.to_numeric(df[extreme_col], errors="coerce").values

        filled_by_zone = pd.Series(filled_arr).groupby(zone_id_arr).any()
        valid = filled_by_zone.index[filled_by_zone.index > 0]
        through_zones = valid[filled_by_zone.loc[valid]]

        excursions: list = []

        for zid in through_zones:
            zone_mask = zone_id_arr == zid
            fill_positions = np.where(zone_mask & filled_arr)[0]
            if len(fill_positions) == 0:
                continue
            fill_pos = fill_positions[0]

            zone_sub = zone_mask & ~signal_arr
            bvals = boundary_arr[zone_sub]
            if len(bvals) == 0 or np.isnan(bvals[0]):
                continue
            boundary_val = bvals[0]

            end_pos = min(fill_pos + return_window + 1, len(return_price_arr))
            # Include the fill bar itself so max excursion includes the break bar
            window_return = return_price_arr[fill_pos:end_pos]
            window_extreme = extreme_arr[fill_pos:end_pos]

            if len(window_return) == 0:
                continue

            if return_op == "ge":
                returned_mask = window_return >= boundary_val
            else:
                returned_mask = window_return <= boundary_val

            if not np.any(returned_mask):
                continue

            return_idx = int(np.argmax(returned_mask))

            extreme_window = window_extreme[: return_idx + 1]
            if len(extreme_window) == 0:
                continue

            if excursion_op == "sub":
                extreme_val = float(np.nanmin(extreme_window))
                excursion = boundary_val - extreme_val
            else:
                extreme_val = float(np.nanmax(extreme_window))
                excursion = extreme_val - boundary_val

            excursions.append(max(0.0, excursion))

        n = len(excursions)
        if n == 0:
            return {"n_fake_outs": 0}

        arr = np.array(excursions)
        return {
            "n_fake_outs": n,
            "p50": round(float(np.percentile(arr, 50)), 1),
            "p75": round(float(np.percentile(arr, 75)), 1),
            "p80": round(float(np.percentile(arr, 80)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "mean": round(float(np.mean(arr)), 1),
            "median": round(float(np.median(arr)), 1),
        }

    bull = _excursion_stats(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_low",
        "High",
        "Low",
        "ge",
        "sub",
    )
    bear = _excursion_stats(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        "Low",
        "High",
        "le",
        "add",
    )
    return {"bull": bull, "bear": bear, "return_window": return_window}


# ── MANIP-Tages-Bias Analyse ──────────────────────────────────────────────────


def analyze_manip_day_bias(
    df: pd.DataFrame,
    signal_col: str = "manip_bear",
) -> dict:
    """Tages-Richtung an MANIP-Signal-Tagen vs. normalen Tagen.

    Beantwortet: Warum verbessert MANIP Bear die Bear-Zonen?
    Hypothese: MANIP-Tage sind systematisch bearische Tage.

    Pro Tag (NY):
      - RTH Open = Close des ersten Bars ab 09:30
      - RTH Close = Close des letzten Bars bis 16:00
      - day_delta = RTH_Close - RTH_Open

    signal_col: Boolean-Spalte (manip_bear, manip_bull etc.)

    Output:
      manip_active: {n, avg_delta, median_delta, pct_bearish, pct_bullish}
      no_manip:     {n, avg_delta, median_delta, pct_bearish, pct_bullish}
    """
    import zoneinfo

    NY = zoneinfo.ZoneInfo("America/New_York")

    if signal_col not in df.columns:
        raise ValueError(f"Spalte '{signal_col}' nicht gefunden")

    signal_arr = df[signal_col].fillna(False).astype(bool)
    close_arr = pd.to_numeric(df["Close"], errors="coerce")

    if hasattr(df.index, "tz") and df.index.tz is not None:
        ny_idx = df.index.tz_convert(NY)
    else:
        ny_idx = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(NY)

    dates = pd.Series([t.date() for t in ny_idx], index=df.index)
    hours = pd.Series([t.hour + t.minute / 60 for t in ny_idx], index=df.index)

    manip_days: list = []
    no_manip_days: list = []

    for d in sorted(dates.unique()):
        day_mask = dates == d
        day_hours = hours[day_mask]
        day_close = close_arr[day_mask]
        day_signal = signal_arr[day_mask]

        rth_mask = (day_hours >= 9.5) & (day_hours < 16.0)
        rth_close = day_close[rth_mask]
        if len(rth_close) < 2:
            continue

        delta = float(rth_close.iloc[-1] - rth_close.iloc[0])
        manip_active = bool(day_signal.any())

        if manip_active:
            manip_days.append(delta)
        else:
            no_manip_days.append(delta)

    def _stats(deltas: list) -> dict:
        if not deltas:
            return {
                "n": 0,
                "avg_delta": 0.0,
                "median_delta": 0.0,
                "pct_bearish": 0.0,
                "pct_bullish": 0.0,
            }
        arr = np.array(deltas)
        return {
            "n": len(deltas),
            "avg_delta": round(float(np.mean(arr)), 1),
            "median_delta": round(float(np.median(arr)), 1),
            "pct_bearish": round(float(np.mean(arr < 0)) * 100, 1),
            "pct_bullish": round(float(np.mean(arr > 0)) * 100, 1),
            "std_delta": round(float(np.std(arr)), 1),
        }

    return {
        "signal_col": signal_col,
        "manip_active": _stats(manip_days),
        "no_manip": _stats(no_manip_days),
    }


# ── Zone-Session-Break-Return Analyse ─────────────────────────────────────────


def analyze_zone_session_break_return(
    df: pd.DataFrame,
    prefix: str,
    return_windows: "list[int] | None" = None,
) -> dict:
    """Per Session: Wie viele echte Breaks kehren innerhalb N Bars zurück?

    Beantwortet: London/AM Breaks – permanent oder kehren sie zurück?

    Für jede Zone die durchbrochen wurde (filled=True):
      1. Session der Zone-Entstehung bestimmen
      2. Return-Check in mehreren Fenstern (15/30/60/120 Bars)

    Gibt pro Session zurück:
      n_through, return_N, return_pct_N (für jedes Fenster N)

    Sessions (NY-Zeit): Asia, London, PreM, AM, Lunch, PM, AH
    """
    import zoneinfo

    if return_windows is None:
        return_windows = [15, 30, 60, 120]

    NY = zoneinfo.ZoneInfo("America/New_York")

    SESSION_MAP = [
        ("Asia", 18, 24),
        ("Asia", 0, 2),
        ("London", 2, 8),
        ("PreM", 8, 9.5),
        ("AM", 9.5, 12),
        ("Lunch", 12, 13.5),
        ("PM", 13.5, 16),
        ("AH", 16, 18),
    ]

    def _hour_to_session(h: float) -> str:
        for name, start, end in SESSION_MAP:
            if start <= h < end:
                return name
        return "Other"

    cols = set(df.columns)
    bull_signal = f"{prefix}_bull" if f"{prefix}_bull" in cols else f"{prefix}_bullish"
    bear_signal = f"{prefix}_bear" if f"{prefix}_bear" in cols else f"{prefix}_bearish"

    required = {
        bull_signal,
        bear_signal,
        f"{prefix}_bull_high",
        f"{prefix}_bull_low",
        f"{prefix}_bull_filled",
        f"{prefix}_bear_low",
        f"{prefix}_bear_high",
        f"{prefix}_bear_filled",
    }
    missing = required - cols
    if missing:
        raise ValueError(f"Spalten fehlen für Prefix '{prefix}': {sorted(missing)}")

    if hasattr(df.index, "tz") and df.index.tz is not None:
        ny_idx = df.index.tz_convert(NY)
    else:
        ny_idx = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(NY)
    ny_hours_arr = np.array([t.hour + t.minute / 60 for t in ny_idx])

    def _session_return(
        signal_col: str,
        filled_col: str,
        far_col: str,
        price_col: str,
        return_op: str,
    ) -> dict:
        sessions = ["Asia", "London", "PreM", "AM", "Lunch", "PM", "AH", "Other"]
        empty = {"n_through": 0, **{f"return_{w}": 0 for w in return_windows}}
        result: dict = {s: dict(empty) for s in sessions}

        zone_id = df[signal_col].cumsum()
        zone_id_arr = zone_id.values
        signal_arr = df[signal_col].astype(bool).values
        filled_arr = df[filled_col].fillna(False).astype(bool).values
        far_arr = pd.to_numeric(df[far_col], errors="coerce").values
        price_arr = pd.to_numeric(df[price_col], errors="coerce").values

        unique_zones = np.unique(zone_id_arr[zone_id_arr > 0])
        for zid in unique_zones:
            zone_mask = zone_id_arr == zid
            sig_positions = np.where(zone_mask & signal_arr)[0]
            if len(sig_positions) == 0:
                continue
            sig_pos = sig_positions[0]
            session = _hour_to_session(float(ny_hours_arr[sig_pos]))

            sub_filled = filled_arr[zone_mask & ~signal_arr]
            if not np.any(sub_filled):
                continue

            fill_positions = np.where(zone_mask & filled_arr)[0]
            if len(fill_positions) == 0:
                continue
            fill_pos = fill_positions[0]

            zone_sub = zone_mask & ~signal_arr
            far_vals = far_arr[zone_sub]
            if len(far_vals) == 0 or np.isnan(far_vals[0]):
                continue
            far_val = far_vals[0]

            result[session]["n_through"] += 1

            for w in return_windows:
                end_pos = min(fill_pos + w + 1, len(price_arr))
                window = price_arr[fill_pos + 1 : end_pos]
                if return_op == "ge":
                    ret = len(window) > 0 and np.any(window >= far_val)
                else:
                    ret = len(window) > 0 and np.any(window <= far_val)
                if ret:
                    result[session][f"return_{w}"] += 1

        for s in result:
            n = result[s]["n_through"]
            for w in return_windows:
                result[s][f"return_pct_{w}"] = (
                    round(result[s][f"return_{w}"] / n * 100, 1) if n else 0.0
                )

        return result

    bull = _session_return(
        bull_signal,
        f"{prefix}_bull_filled",
        f"{prefix}_bull_low",
        "High",
        "ge",
    )
    bear = _session_return(
        bear_signal,
        f"{prefix}_bear_filled",
        f"{prefix}_bear_high",
        "Low",
        "le",
    )
    return {"bull": bull, "bear": bear, "return_windows": return_windows}


# ── Zone-Near-Level Analyse ──────────────────────────────────────────────────


def analyze_zone_near_level(
    df: pd.DataFrame,
    zone_prefix: str,
    level_col: str,
    proximity_pts: float = 20.0,
) -> dict:
    """Analysiere Bounce/Durch-Rate für Zonen in der Nähe eines Preis-Levels.

    Splittet Zone-Touches in zwei Gruppen:
    - 'near': zone_mid innerhalb von proximity_pts vom Level
    - 'far': zone_mid weiter als proximity_pts vom Level (oder Level NaN)

    Args:
        df: DataFrame mit Zone-Spalten ({prefix}_bull_high etc.) und Level-Spalte
        zone_prefix: z.B. "fvg" oder "ifvg"
        level_col: z.B. "prev_week_high"
        proximity_pts: Maximale Distanz zone_mid ↔ level in Punkten (default 20)
    """
    import numpy as np

    bull_high_col = f"{zone_prefix}_bull_high"
    bull_low_col = f"{zone_prefix}_bull_low"
    bull_filled_col = f"{zone_prefix}_bull_filled"

    for col in (bull_high_col, bull_low_col, bull_filled_col, level_col):
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' nicht im DataFrame")

    h = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    zh = df[bull_high_col].to_numpy(dtype=float)
    zl = df[bull_low_col].to_numpy(dtype=float)
    zf = df[bull_filled_col].to_numpy(dtype=float)
    lv = df[level_col].to_numpy(dtype=float)

    n = len(df)

    near: dict = {"touches": 0, "bounce": 0, "through": 0}
    far: dict = {"touches": 0, "bounce": 0, "through": 0}

    i = 0
    while i < n:
        # Episode-Start: Zone aktiv (nicht NaN, nicht filled)
        if np.isnan(zh[i]) or zf[i] >= 1.0:
            i += 1
            continue

        zone_h_val = zh[i]
        zone_l_val = zl[i]

        # Episode-Ende finden
        j = i
        while j < n and not np.isnan(zh[j]) and zh[j] == zone_h_val and zf[j] < 1.0:
            j += 1

        zone_mid = (zone_h_val + zone_l_val) / 2.0

        # Ersten Touch suchen
        for k in range(i, j):
            if lo[k] <= zone_mid <= h[k]:
                # Proximity-Check
                level_val = lv[k]
                if (
                    not np.isnan(level_val)
                    and abs(level_val - zone_mid) <= proximity_pts
                ):
                    group = near
                else:
                    group = far

                group["touches"] += 1
                if c[k] < zone_mid:
                    group["bounce"] += 1
                else:
                    group["through"] += 1
                break

        i = j

    def _pct(d: dict) -> dict:
        t = d["touches"]
        return {
            **d,
            "bounce_pct": round(d["bounce"] / t * 100, 1) if t > 0 else 0.0,
            "through_pct": round(d["through"] / t * 100, 1) if t > 0 else 0.0,
        }

    return {
        "zone_prefix": zone_prefix,
        "level_col": level_col,
        "proximity_pts": proximity_pts,
        "near": _pct(near),
        "far": _pct(far),
    }


# ── Zone-Overlap-Analyse ─────────────────────────────────────────────────────────


def analyze_zone_overlap_outcomes(df: pd.DataFrame, prefix: str) -> dict:
    """Analysiere Bounce/Durch-Rate für Zonen mit/ohne Überlappung anderer aktiver Zonen.

    Für jeden FVG-Touch: prüfe ob eine andere aktive FVG-Zone (gleiche oder
    entgegengesetzte Richtung) mit der berührten Zone überlappt.

    Overlap-Definition: NOT (high_A < low_B OR high_B < low_A)
    """
    import numpy as np

    bull_h_col = f"{prefix}_bull_high"
    bull_l_col = f"{prefix}_bull_low"
    bull_f_col = f"{prefix}_bull_filled"
    bear_h_col = f"{prefix}_bear_high"
    bear_l_col = f"{prefix}_bear_low"
    bear_f_col = f"{prefix}_bear_filled"

    for col in (bull_h_col, bull_l_col, bull_f_col, bear_h_col, bear_l_col, bear_f_col):
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' nicht im DataFrame")

    H = df["High"].to_numpy(dtype=float)
    Lo = df["Low"].to_numpy(dtype=float)
    C = df["Close"].to_numpy(dtype=float)
    bh = df[bull_h_col].to_numpy(dtype=float)
    bl = df[bull_l_col].to_numpy(dtype=float)
    bf = df[bull_f_col].to_numpy(dtype=float)
    rh = df[bear_h_col].to_numpy(dtype=float)
    rl = df[bear_l_col].to_numpy(dtype=float)
    rf = df[bear_f_col].to_numpy(dtype=float)
    n = len(df)

    def _empty() -> dict:
        return {"touches": 0, "bounce": 0, "through": 0}

    def _pct(d: dict) -> dict:
        t = d["touches"]
        return {
            **d,
            "bounce_pct": round(d["bounce"] / t * 100, 1) if t else 0.0,
            "through_pct": round(d["through"] / t * 100, 1) if t else 0.0,
        }

    bull_single: dict = _empty()
    bull_double: dict = _empty()
    bull_same: dict = _empty()
    bull_opp: dict = _empty()
    bear_single: dict = _empty()
    bear_double: dict = _empty()
    bear_same: dict = _empty()
    bear_opp: dict = _empty()

    # ── Episoden-Erkennung (Episode = konstanter high-Wert + not filled) ───────
    def _episode_end(
        arr_h: "np.ndarray", arr_f: "np.ndarray", start: int, h_val: float
    ) -> int:
        j = start
        while j < n and not np.isnan(arr_h[j]) and arr_h[j] == h_val and arr_f[j] < 1.0:
            j += 1
        return j

    def _collect_episodes(
        arr_h: "np.ndarray", arr_l: "np.ndarray", arr_f: "np.ndarray"
    ) -> list:
        """Gibt Liste von (start, end, high, low) Tupeln zurück."""
        eps = []
        i = 0
        while i < n:
            if np.isnan(arr_h[i]) or arr_f[i] >= 1.0:
                i += 1
                continue
            h_val = arr_h[i]
            l_val = arr_l[i]
            j = _episode_end(arr_h, arr_f, i, h_val)
            eps.append((i, j, h_val, l_val))
            i = j
        return eps

    bull_eps = _collect_episodes(bh, bl, bf)
    bear_eps = _collect_episodes(rh, rl, rf)

    # ── Bar-Level-Überlappungs-Check ─────────────────────────────────────────
    # Für einen Bull-Touch bei Bar k mit Zone (zh, zl):
    # - Bull×Bull auf gleichem TF: unmöglich, da bh[k] nur einen Wert hat.
    #   has_same ist immer False.
    # - Bull×Bear: prüfe ob Bear bei Bar k aktiv ist (rh[k] nicht NaN, rf[k]<1)
    #   und sich mit der Bull-Zone überlappt.
    #
    # "Aktiv bei Bar k" bedeutet: Wert genau an diesem Bar nicht NaN und nicht filled.
    # Das vermeidet das historische Artefakt des alten [:k+1]-Slicings das auch
    # längst beendete Episoden als "aktiv" erkannte.

    def _has_bear_overlap_at_k(k: int, zh: float, zl: float) -> bool:
        """Gibt True zurück wenn bei Bar k eine Bear-Zone aktiv ist die mit Bull-Zone überlappt."""
        bear_h = rh[k]
        bear_l = rl[k]
        bear_fi = rf[k]
        # Bear aktiv: nicht NaN, nicht filled
        if np.isnan(bear_h) or bear_fi >= 1.0:
            return False
        # Überlappung: NOT (zh < bear_l OR bear_h < zl)
        return not (zh < bear_l or bear_h < zl)

    def _has_bull_overlap_at_k(k: int, zh: float, zl: float) -> bool:
        """Gibt True zurück wenn bei Bar k eine Bull-Zone aktiv ist die mit Bear-Zone überlappt."""
        bull_h = bh[k]
        bull_l = bl[k]
        bull_fi = bf[k]
        # Bull aktiv: nicht NaN, nicht filled
        if np.isnan(bull_h) or bull_fi >= 1.0:
            return False
        # Überlappung: NOT (zh < bull_l OR bull_h < zl)
        return not (zh < bull_l or bull_h < zl)

    # ── Bull-Episoden analysieren ─────────────────────────────────────────────
    for ep_start, ep_end, zh, zl in bull_eps:
        zone_mid = (zh + zl) / 2.0
        # Ersten Touch suchen
        touch_k = -1
        for k in range(ep_start, ep_end):
            if Lo[k] <= zone_mid <= H[k]:
                touch_k = k
                break
        if touch_k < 0:
            continue

        has_same = (
            False  # Bull×Bull auf gleichem TF nicht möglich (bh[k] hat nur 1 Wert)
        )
        has_opp = _has_bear_overlap_at_k(touch_k, zh, zl)

        is_double = has_same or has_opp
        outcome = bull_double if is_double else bull_single
        outcome["touches"] += 1
        if C[touch_k] < zone_mid:
            outcome["bounce"] += 1
        else:
            outcome["through"] += 1
        if has_same:
            bull_same["touches"] += 1
            if C[touch_k] < zone_mid:
                bull_same["bounce"] += 1
            else:
                bull_same["through"] += 1
        if has_opp:
            bull_opp["touches"] += 1
            if C[touch_k] < zone_mid:
                bull_opp["bounce"] += 1
            else:
                bull_opp["through"] += 1

    # ── Bear-Episoden analysieren ─────────────────────────────────────────────
    for ep_start, ep_end, zh, zl in bear_eps:
        zone_mid = (zh + zl) / 2.0
        touch_k = -1
        for k in range(ep_start, ep_end):
            if Lo[k] <= zone_mid <= H[k]:
                touch_k = k
                break
        if touch_k < 0:
            continue

        has_same = (
            False  # Bear×Bear auf gleichem TF nicht möglich (rh[k] hat nur 1 Wert)
        )
        has_opp = _has_bull_overlap_at_k(touch_k, zh, zl)

        is_double = has_same or has_opp
        outcome = bear_double if is_double else bear_single
        outcome["touches"] += 1
        if C[touch_k] > zone_mid:
            outcome["bounce"] += 1
        else:
            outcome["through"] += 1
        if has_same:
            bear_same["touches"] += 1
            if C[touch_k] > zone_mid:
                bear_same["bounce"] += 1
            else:
                bear_same["through"] += 1
        if has_opp:
            bear_opp["touches"] += 1
            if C[touch_k] > zone_mid:
                bear_opp["bounce"] += 1
            else:
                bear_opp["through"] += 1

    return {
        "prefix": prefix,
        "bull": {
            "single": _pct(bull_single),
            "double": _pct(bull_double),
            "double_same": _pct(bull_same),
            "double_opposite": _pct(bull_opp),
        },
        "bear": {
            "single": _pct(bear_single),
            "double": _pct(bear_double),
            "double_same": _pct(bear_same),
            "double_opposite": _pct(bear_opp),
        },
    }


# ── Level-Outcome-Analyse ────────────────────────────────────────────────────────


def analyze_level_outcomes(df: pd.DataFrame, level_col: str) -> dict:
    """Analysiere Bounce/Durch-Rate für ein einzelnes Preis-Level.

    Pro Level-Episode (solange Level-Wert gleich bleibt):
    - Erkennt die Approach-Seite (von oben oder unten) anhand des letzten
      Closes VOR dem ersten Touch. Bei Touch am ersten Episoden-Bar wird
      lo[k] < level als Indikator für "von unten" verwendet.
    - Erster Touch der Episode zählt (Low <= Level <= High)
    - Bounce = Close zurück auf Approach-Seite
    - Durch = Close auf anderer Seite
    - Eindringstiefe in Punkten (wie weit Wick ins Level eindringt)

    Wirft ValueError wenn level_col nicht im DataFrame.
    """
    import numpy as np

    if level_col not in df.columns:
        raise ValueError(f"Spalte '{level_col}' nicht im DataFrame")

    for col in ("High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame fehlt Spalte '{col}'")

    h = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    level = df[level_col].to_numpy(dtype=float)

    n = len(df)
    touches = 0
    bounce = 0
    through = 0
    depth_pts: list[float] = []

    i = 0
    while i < n:
        lv = level[i]
        if np.isnan(lv):
            i += 1
            continue

        # Episode-Ende finden (solange gleicher Level-Wert)
        j = i
        while j < n and level[j] == lv:
            j += 1

        # Ersten Touch in Episode suchen
        for k in range(i, j):
            if lo[k] <= lv <= h[k]:
                touches += 1
                # Approach-Seite bestimmen:
                # Wenn k > i → letzter Close vor Touch = c[k-1]
                # Wenn k == i → keine Vorperiode im Episode; nutze Low[k] < lv
                #   (Wick unterhalb Level = Price kam von unten)
                if k > i:
                    start_above = c[k - 1] > lv
                else:
                    # k == i: erste Bar der Episode ist bereits eine Touch-Bar → kein Prior-Close verfügbar
                    # Fallback: lo[k] >= lv → Wick bleibt über Level → Ankunft von oben
                    start_above = lo[k] >= lv

                if start_above:
                    depth_pts.append(max(0.0, lv - lo[k]))
                    if c[k] > lv:
                        bounce += 1
                    else:
                        through += 1
                else:
                    depth_pts.append(max(0.0, h[k] - lv))
                    if c[k] < lv:
                        bounce += 1
                    else:
                        through += 1
                break

        i = j

    depth_arr = np.array(depth_pts) if depth_pts else np.array([])

    return {
        "level_col": level_col,
        "touches": touches,
        "bounce": bounce,
        "through": through,
        "bounce_pct": round(bounce / touches * 100, 1) if touches > 0 else 0.0,
        "through_pct": round(through / touches * 100, 1) if touches > 0 else 0.0,
        "depth_pts_median": round(float(np.median(depth_arr)), 2)
        if len(depth_arr) > 0
        else 0.0,
        "depth_pts_mean": round(float(np.mean(depth_arr)), 2)
        if len(depth_arr) > 0
        else 0.0,
    }


# ── Research-Pipeline ────────────────────────────────────────────────────────────


def save_zone_research(
    stats: dict,
    algo_file: Path,
    data_info: dict,
) -> Path:
    """Speichert Zone-Forschungsdaten neben der Algo-Datei in _research/.

    Erstellt:
      {algo_dir}/_research/{algo_stem}_zone_stats_{datum}.json
      {algo_dir}/_research/{algo_stem}_zone_stats_{datum}.md

    Gibt den Pfad zum _research/-Ordner zurück.
    """
    research_dir = algo_file.parent / "_research"
    research_dir.mkdir(exist_ok=True)

    stem = algo_file.stem
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_name = f"{stem}_zone_stats_{date_str}"

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {
        "algo": algo_file.name,
        "generated": date_str,
        "data": data_info,
        "zones": stats,
    }
    json_path = research_dir / f"{base_name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_lines = [
        f"# Zone-Stats: {algo_file.name}",
        "",
        f"**Generiert:** {date_str}  ",
        f"**Daten:** {data_info.get('from', '?')} → {data_info.get('to', '?')}  ",
        f"**Bars:** {data_info.get('bars', '?'):,}  ",
        f"**Handelstage:** {data_info.get('days', '?')}  ",
        "",
    ]
    for prefix, pdata in stats.items():
        md_lines.append(f"## Prefix: `{prefix}`")
        for side, s in pdata.items():
            label = "Bull (von oben)" if side == "bull" else "Bear (von unten)"
            md_lines += [
                "",
                f"### {label}",
                "| Kennzahl | Wert |",
                "|----------|------|",
                f"| Zonen gesamt | {s['total_zones']:,} |",
                f"| Keine Berührung | {s['no_touch']:,} |",
                f"| Berührungen | {s['touches']:,} |",
                f"| Bounce ↩ | {s['bounce']:,} ({s['bounce_pct']}%) |",
                f"| Durch → | {s['through']:,} ({s['through_pct']}%) |",
            ]
            d = s.get("depth", {})
            if d:
                md_lines += [
                    "",
                    "**Penetrations-Tiefe** (nur Berührungen):",
                    "| Tiefe | Wert |",
                    "|-------|------|",
                    f"| Ø Punkte | {d['depth_pts_mean']} |",
                    f"| Median Punkte | {d['depth_pts_median']} |",
                    f"| Ø % der Zone | {d['depth_pct_mean']}% |",
                    f"| < 25% (kaum berührt) | {d['pct_lt25']}% |",
                    f"| 25–50% (vor CE) | {d['pct_25_50']}% |",
                    f"| 50–75% (CE-Bereich) | {d['pct_50_75']}% |",
                    f"| 75–100% (tief) | {d['pct_75_100']}% |",
                    f"| ≥100% (Durch) | {d['pct_gte100']}% |",
                ]
        md_lines.append("")

    md_path = research_dir / f"{base_name}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return research_dir


# ── Haupt-Funktion ──────────────────────────────────────────────────────────────


@dataclass
class InspectResult:
    algo_file: Path
    total_bars: int
    total_days: int
    signal_cols: list[str]
    windows: list[dict]
    events_only: bool = False
    heatmap: dict[str, dict[str, int]] = field(default_factory=dict)


def inspect_algo(
    name: str,
    algo_dirs: list[Path],
    data_path: Path,
    window_minutes: int = 30,
    with_heatmap: bool = False,
    events_only: bool = False,
) -> InspectResult:
    """Orchestriert Baustein-Inspektion: Suche → Laden → Analyse → Gruppierung.

    Wirft FileNotFoundError wenn kein Algo-Match oder data_path fehlt.
    """
    algo_file = find_algo_file(name, [Path(d) for d in algo_dirs])
    if algo_file is None:
        raise FileNotFoundError(
            f"kein Algo gefunden für '{name}' in {[str(d) for d in algo_dirs]}"
        )

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Daten nicht gefunden: {data_path}")

    # NQ-Daten laden; Spalten auf Title-Case umbenennen (Algo erwartet High/Low/Close/Open)
    df = pd.read_parquet(data_path)
    rename_map = {
        c: c.title()
        for c in df.columns
        if c.lower() in {"open", "high", "low", "close", "volume"}
    }
    df = df.rename(columns=rename_map)

    # Anzahl Handelstage
    ny = _to_ny_series(df.index)
    total_days = max(len(set(ny.dt.date)), 1)

    # Spalten vor Algo-Lauf merken → nur neue Spalten als Signale werten
    original_cols = set(df.columns)

    # Algo ausführen
    result_df = run_algo(algo_file, df)

    # Signal-Spalten finden (nur neu hinzugefügte Spalten)
    signal_cols = detect_signal_columns(result_df, original_cols=original_cols)

    # Zeitfenster-Gruppierung
    windows = group_signals_by_window(
        result_df, signal_cols, window_minutes, events_only=events_only
    )

    # Optionale Heatmap
    hm = (
        build_heatmap(result_df, signal_cols, window_minutes, events_only=events_only)
        if with_heatmap
        else {}
    )

    return InspectResult(
        algo_file=algo_file,
        total_bars=len(df),
        total_days=total_days,
        signal_cols=signal_cols,
        windows=windows,
        events_only=events_only,
        heatmap=hm,
    )


# ── Multi-TF Nesting Analyse ──────────────────────────────────────────────────────


def analyze_zone_mtf_nesting(
    df_htf: pd.DataFrame,
    df_ltf: pd.DataFrame,
    prefix: str,
    nesting: str = "contained",
) -> dict:
    """Multi-TF Nesting: Bounce-Rate wenn HTF-Zone eine aktive LTF-Zone enthält.

    Für jeden HTF-Touch: prüfe ob eine LTF-Zone (gleiche Richtung) aktiv ist
    die innerhalb der HTF-Zone liegt.

    nesting='contained': LTF vollständig in HTF (htf_low <= ltf_low AND ltf_high <= htf_high)
    nesting='overlap': beliebige Überschneidung (NOT (zh < ltf_l OR ltf_h < zl))

    Alignment: HTF-Touch-Timestamp wird in LTF-Index gesucht (searchsorted).
    Beide DataFrames müssen gleiche Timezone haben.

    Wirft ValueError wenn Pflicht-Spalten fehlen.
    """
    import numpy as np

    bull_h_col = f"{prefix}_bull_high"
    bull_l_col = f"{prefix}_bull_low"
    bull_f_col = f"{prefix}_bull_filled"
    bear_h_col = f"{prefix}_bear_high"
    bear_l_col = f"{prefix}_bear_low"
    bear_f_col = f"{prefix}_bear_filled"

    needed = (bull_h_col, bull_l_col, bull_f_col, bear_h_col, bear_l_col, bear_f_col)
    for df_name, df in (("df_htf", df_htf), ("df_ltf", df_ltf)):
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"Spalte '{col}' fehlt in {df_name}")

    # ── HTF Arrays ──────────────────────────────────────────────────────────────
    H_htf = df_htf["High"].to_numpy(dtype=float)
    Lo_htf = df_htf["Low"].to_numpy(dtype=float)
    C_htf = df_htf["Close"].to_numpy(dtype=float)
    bh_htf = df_htf[bull_h_col].to_numpy(dtype=float)
    df_htf[bull_l_col].to_numpy(dtype=float)
    bf_htf = df_htf[bull_f_col].to_numpy(dtype=float)
    rh_htf = df_htf[bear_h_col].to_numpy(dtype=float)
    df_htf[bear_l_col].to_numpy(dtype=float)
    rf_htf = df_htf[bear_f_col].to_numpy(dtype=float)
    n_htf = len(df_htf)

    # ── LTF Arrays ──────────────────────────────────────────────────────────────
    bh_ltf = df_ltf[bull_h_col].to_numpy(dtype=float)
    bl_ltf = df_ltf[bull_l_col].to_numpy(dtype=float)
    bf_ltf = df_ltf[bull_f_col].to_numpy(dtype=float)
    rh_ltf = df_ltf[bear_h_col].to_numpy(dtype=float)
    rl_ltf = df_ltf[bear_l_col].to_numpy(dtype=float)
    rf_ltf = df_ltf[bear_f_col].to_numpy(dtype=float)
    ltf_idx = df_ltf.index
    n_ltf = len(df_ltf)

    def _is_nested(ltf_h: float, ltf_l: float, zh: float, zl: float) -> bool:
        if nesting == "contained":
            return zl <= ltf_l and ltf_h <= zh
        # overlap
        return not (zh < ltf_l or ltf_h < zl)

    def _has_bull_ltf_at(ts: "pd.Timestamp", zh: float, zl: float) -> bool:
        pos = ltf_idx.searchsorted(ts, side="left")
        if pos >= n_ltf:
            return False
        if np.isnan(bh_ltf[pos]) or bf_ltf[pos] >= 1.0:
            return False
        return _is_nested(bh_ltf[pos], bl_ltf[pos], zh, zl)

    def _has_bear_ltf_at(ts: "pd.Timestamp", zh: float, zl: float) -> bool:
        pos = ltf_idx.searchsorted(ts, side="left")
        if pos >= n_ltf:
            return False
        if np.isnan(rh_ltf[pos]) or rf_ltf[pos] >= 1.0:
            return False
        return _is_nested(rh_ltf[pos], rl_ltf[pos], zh, zl)

    def _empty() -> dict:
        return {"touches": 0, "bounce": 0, "through": 0}

    def _pct(d: dict) -> dict:
        t = d["touches"]
        return {
            **d,
            "bounce_pct": round(d["bounce"] / t * 100, 1) if t else 0.0,
            "through_pct": round(d["through"] / t * 100, 1) if t else 0.0,
        }

    bull_nested: dict = _empty()
    bull_single: dict = _empty()
    bear_nested: dict = _empty()
    bear_single: dict = _empty()

    htf_timestamps = df_htf.index

    # ── Episoden-Erkennung HTF ───────────────────────────────────────────────────
    def _collect_episodes(arr_h: "np.ndarray", arr_f: "np.ndarray") -> list:
        eps = []
        i = 0
        while i < n_htf:
            if np.isnan(arr_h[i]) or arr_f[i] >= 1.0:
                i += 1
                continue
            h_val = arr_h[i]
            j = i
            while (
                j < n_htf
                and not np.isnan(arr_h[j])
                and arr_h[j] == h_val
                and arr_f[j] < 1.0
            ):
                j += 1
            eps.append((i, j, h_val))
            i = j
        return eps

    bull_l_arr = df_htf[bull_l_col].to_numpy(dtype=float)
    bear_l_arr = df_htf[bear_l_col].to_numpy(dtype=float)

    # ── Bull-Episoden ────────────────────────────────────────────────────────────
    for ep_start, ep_end, zh in _collect_episodes(bh_htf, bf_htf):
        zl = bull_l_arr[ep_start]
        zone_mid = (zh + zl) / 2.0
        touch_k = -1
        for k in range(ep_start, ep_end):
            if Lo_htf[k] <= zone_mid <= H_htf[k]:
                touch_k = k
                break
        if touch_k < 0:
            continue
        ts = htf_timestamps[touch_k]
        has_nested = _has_bull_ltf_at(ts, zh, zl)
        bucket = bull_nested if has_nested else bull_single
        bucket["touches"] += 1
        if C_htf[touch_k] < zone_mid:
            bucket["bounce"] += 1
        else:
            bucket["through"] += 1

    # ── Bear-Episoden ────────────────────────────────────────────────────────────
    for ep_start, ep_end, zh in _collect_episodes(rh_htf, rf_htf):
        zl = bear_l_arr[ep_start]
        zone_mid = (zh + zl) / 2.0
        touch_k = -1
        for k in range(ep_start, ep_end):
            if Lo_htf[k] <= zone_mid <= H_htf[k]:
                touch_k = k
                break
        if touch_k < 0:
            continue
        ts = htf_timestamps[touch_k]
        has_nested = _has_bear_ltf_at(ts, zh, zl)
        bucket = bear_nested if has_nested else bear_single
        bucket["touches"] += 1
        if C_htf[touch_k] > zone_mid:
            bucket["bounce"] += 1
        else:
            bucket["through"] += 1

    return {
        "prefix": prefix,
        "nesting": nesting,
        "bull": {
            "nested": _pct(bull_nested),
            "single": _pct(bull_single),
        },
        "bear": {
            "nested": _pct(bear_nested),
            "single": _pct(bear_single),
        },
    }
