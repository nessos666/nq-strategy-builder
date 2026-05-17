"""
Settlement Spezialtests – 6 einzigartige Tests die Settlements spezifische Eigenschaften nutzen.

1. Magnet/Attraktor: Zieht Settlement den Preis an?
2. Gap-Fill: Schließt sich die Lücke Globex-Open → Settlement?
3. Tagesrichtung: Settlement-Position als Bias-Signal
4. Cluster: Mehrere nahe Settlements = stärkeres Level?
5. Tagestyp: Settlement × KER/ATR Regime
6. Breakout-Momentum: MFE nach Settlement-Break

Ausführung:
    .venv/bin/python3 scripts/settlement_spezialtests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Projekt-Root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = Path(
    Path(__file__).resolve().parent.parent
    "nq_backtest/data/nq_1m_databento_2024_2026.parquet"
)
ALGO_PATH = ROOT / "david_bibliothek/01_Hoch_Tief/8. Settlement Levels.py"
RESEARCH_DIR = ROOT / "david_bibliothek/01_Hoch_Tief/_research"

# ---------- Helpers ----------


def load_bars() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # Spaltennamen normalisieren
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    return df


def run_settlement_algo(df: pd.DataFrame) -> pd.DataFrame:
    """Settlement-Algo ausführen."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("stl", str(ALGO_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run(df)


def to_ny(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Index nach NY-Zeit konvertieren."""
    try:
        return idx.tz_convert("America/New_York")
    except TypeError:
        return idx.tz_localize("UTC").tz_convert("America/New_York")


def to_ct(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Index nach CT konvertieren."""
    try:
        return idx.tz_convert("America/Chicago")
    except TypeError:
        return idx.tz_localize("UTC").tz_convert("America/Chicago")


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Tägliche Kennzahlen: Open/High/Low/Close/Settlement."""
    ny = to_ny(pd.DatetimeIndex(df.index))
    dates = ny.date
    df_tmp = df.copy()
    df_tmp["_date"] = dates

    daily = df_tmp.groupby("_date").agg(
        day_open=("close", "first"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
        day_range=("high", lambda x: x.max() - df_tmp.loc[x.index, "low"].min()),
    )

    # Settlement-Preis pro Tag
    stl_bars = df_tmp[df_tmp["is_settlement_bar"]]
    stl_daily = stl_bars.groupby("_date")["stl_price"].first()
    daily["settlement"] = stl_daily
    daily = daily.dropna(subset=["settlement"])

    return daily


# ---------- Test 1: Magnet/Attraktor ----------


def test_magnet(df: pd.DataFrame) -> dict:
    """Wie nah kommt der Preis zum nächsten Settlement im Tagesverlauf?"""
    print("  Test 1: Magnet/Attraktor...")

    ny = to_ny(pd.DatetimeIndex(df.index))
    hours = pd.Series(ny).dt.hour.values

    # Nur Bars wo ein Settlement in der Nähe ist
    has_above = ~np.isnan(df["stl_nearest_above"].values)
    has_below = ~np.isnan(df["stl_nearest_below"].values)
    has_either = has_above | has_below

    close = df["close"].values
    above = df["stl_nearest_above"].values
    below = df["stl_nearest_below"].values

    # Distanz zum nächsten Settlement
    dist_above = np.where(has_above, above - close, np.inf)
    dist_below = np.where(has_below, close - below, np.inf)
    dist_nearest = np.minimum(dist_above, dist_below)
    dist_nearest = np.where(has_either, dist_nearest, np.nan)

    # Tagesrange für Normalisierung
    ny_dates = pd.Series(ny).dt.date.values
    df_tmp = pd.DataFrame(
        {"date": ny_dates, "high": df["high"].values, "low": df["low"].values}
    )
    day_range = df_tmp.groupby("date").apply(lambda g: g["high"].max() - g["low"].min())

    # Distanz nach Stunde (NY-Zeit)
    dist_by_hour = {}
    for h in range(24):
        mask = (hours == h) & has_either
        if mask.sum() > 100:
            d = dist_nearest[mask]
            dist_by_hour[str(h)] = {
                "mean": round(float(np.nanmean(d)), 1),
                "median": round(float(np.nanmedian(d)), 1),
                "p25": round(float(np.nanpercentile(d, 25)), 1),
                "n": int(mask.sum()),
            }

    # Wie oft innerhalb X Punkte?
    valid = ~np.isnan(dist_nearest)
    total_valid = int(valid.sum())
    within_5 = int((dist_nearest[valid] <= 5).sum())
    within_10 = int((dist_nearest[valid] <= 10).sum())
    within_20 = int((dist_nearest[valid] <= 20).sum())
    within_50 = int((dist_nearest[valid] <= 50).sum())

    # Random-Vergleich: Ø Distanz zu zufälligem Punkt im Tagesrange
    # Bei gleichverteiltem Preis in Range R: E[dist] = R/4
    avg_range = float(day_range.mean())
    random_expected_dist = avg_range / 4

    return {
        "test": "Magnet/Attraktor",
        "frage": "Zieht Settlement den Preis an?",
        "gesamt_mean_dist": round(float(np.nanmean(dist_nearest[valid])), 1),
        "gesamt_median_dist": round(float(np.nanmedian(dist_nearest[valid])), 1),
        "random_expected_dist": round(random_expected_dist, 1),
        "within_5pt_pct": round(100 * within_5 / total_valid, 1),
        "within_10pt_pct": round(100 * within_10 / total_valid, 1),
        "within_20pt_pct": round(100 * within_20 / total_valid, 1),
        "within_50pt_pct": round(100 * within_50 / total_valid, 1),
        "dist_by_hour_ny": dist_by_hour,
        "n_bars": total_valid,
    }


# ---------- Test 2: Gap-Fill ----------


def test_gap_fill(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Schließt sich die Lücke zwischen Globex-Open und Settlement?"""
    print("  Test 2: Gap-Fill...")

    ny = to_ny(pd.DatetimeIndex(df.index))
    hours = pd.Series(ny).dt.hour.values
    dates = pd.Series(ny).dt.date.values

    results = []
    sorted_dates = sorted(daily.index)

    for i in range(1, len(sorted_dates)):
        prev_date = sorted_dates[i - 1]
        curr_date = sorted_dates[i]

        stl_price = daily.loc[prev_date, "settlement"]
        if np.isnan(stl_price):
            continue

        # Globex-Open des nächsten Tages (18:00 ET)
        mask_day = dates == curr_date
        mask_18 = mask_day & (hours == 18)
        if not mask_18.any():
            continue

        globex_idx = np.where(mask_18)[0][0]
        globex_open = df["close"].values[globex_idx]

        gap = globex_open - stl_price
        gap_abs = abs(gap)
        if gap_abs < 1:
            continue  # Kein Gap

        gap_dir = "above" if gap > 0 else "below"

        # Prüfe ob Gap gefüllt wird (Preis erreicht Settlement vor EOD)
        day_bars = np.where(mask_day)[0]
        after_open = day_bars[day_bars >= globex_idx]

        filled = False
        fill_bars = 0
        for j in after_open:
            h = df["high"].values[j]
            lo = df["low"].values[j]
            if gap_dir == "above" and lo <= stl_price:
                filled = True
                fill_bars = j - globex_idx
                break
            elif gap_dir == "below" and h >= stl_price:
                filled = True
                fill_bars = j - globex_idx
                break

        results.append(
            {
                "gap": round(gap, 2),
                "gap_abs": round(gap_abs, 2),
                "gap_dir": gap_dir,
                "filled": filled,
                "fill_bars": fill_bars if filled else None,
            }
        )

    if not results:
        return {"test": "Gap-Fill", "error": "keine Gaps gefunden"}

    rdf = pd.DataFrame(results)
    total = len(rdf)
    filled_pct = round(100 * rdf["filled"].sum() / total, 1)

    # Nach Gap-Größe
    by_size = {}
    for label, lo, hi in [
        ("klein_0-20", 0, 20),
        ("mittel_20-50", 20, 50),
        ("gross_50-100", 50, 100),
        ("extrem_100+", 100, 9999),
    ]:
        mask = (rdf["gap_abs"] >= lo) & (rdf["gap_abs"] < hi)
        n = int(mask.sum())
        if n >= 10:
            by_size[label] = {
                "n": n,
                "fill_pct": round(100 * rdf.loc[mask, "filled"].sum() / n, 1),
                "mean_gap": round(float(rdf.loc[mask, "gap_abs"].mean()), 1),
                "median_fill_bars": int(
                    rdf.loc[mask & rdf["filled"], "fill_bars"].median()
                )
                if rdf.loc[mask & rdf["filled"], "fill_bars"].any()
                else None,
            }

    # Fill-Zeit
    filled_bars = rdf.loc[rdf["filled"], "fill_bars"]

    return {
        "test": "Gap-Fill (Globex Open → Settlement)",
        "frage": "Schließt sich die Lücke zum Vortags-Settlement?",
        "total_gaps": total,
        "fill_rate_pct": filled_pct,
        "mean_gap_pts": round(float(rdf["gap_abs"].mean()), 1),
        "median_gap_pts": round(float(rdf["gap_abs"].median()), 1),
        "fill_time_median_bars": int(filled_bars.median())
        if len(filled_bars) > 0
        else None,
        "fill_time_mean_bars": round(float(filled_bars.mean()), 1)
        if len(filled_bars) > 0
        else None,
        "by_gap_size": by_size,
        "above_fill_pct": round(
            100 * rdf.loc[rdf["gap_dir"] == "above", "filled"].mean(), 1
        )
        if (rdf["gap_dir"] == "above").sum() > 10
        else None,
        "below_fill_pct": round(
            100 * rdf.loc[rdf["gap_dir"] == "below", "filled"].mean(), 1
        )
        if (rdf["gap_dir"] == "below").sum() > 10
        else None,
    }


# ---------- Test 3: Tagesrichtung ----------


def test_daily_bias(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Settlement-Position als Tagesrichtungs-Signal."""
    print("  Test 3: Tagesrichtung...")

    sorted_dates = sorted(daily.index)
    results = []

    ny = to_ny(pd.DatetimeIndex(df.index))
    hours = pd.Series(ny).dt.hour.values
    dates = pd.Series(ny).dt.date.values

    for i in range(1, len(sorted_dates)):
        prev_date = sorted_dates[i - 1]
        curr_date = sorted_dates[i]

        stl = daily.loc[prev_date, "settlement"]
        if np.isnan(stl):
            continue

        # RTH Open (9:30) und Close (16:00) des aktuellen Tages
        mask_day = dates == curr_date
        mask_930 = mask_day & (hours == 9)
        mask_1530 = mask_day & (hours == 15)

        if not mask_930.any() or not mask_1530.any():
            continue

        # Erster Bar um 9:30
        np.where(mask_930)[0]
        # Finde 9:30 genauer
        minutes = pd.Series(ny).dt.minute.values
        mask_exact = mask_day & (hours == 9) & (minutes >= 30)
        if not mask_exact.any():
            continue

        rth_open_idx = np.where(mask_exact)[0][0]
        rth_open = df["close"].values[rth_open_idx]

        # Letzter Bar um 15:59 CT = 16:59 ET? Nein, 15:00 CT = RTH Close
        mask_close = mask_day & (hours == 15) & (minutes >= 55)
        if not mask_close.any():
            mask_close = mask_day & (hours == 16) & (minutes == 0)
        if not mask_close.any():
            continue

        rth_close_idx = np.where(mask_close)[0][-1]
        rth_close = df["close"].values[rth_close_idx]

        # Globex Open (18:00 ET Vortag)
        mask_globex = mask_day & (hours == 18)
        if mask_globex.any():
            df["close"].values[np.where(mask_globex)[0][0]]
        else:
            pass

        # Position relativ zu Settlement
        above_stl = rth_open > stl
        day_bullish = rth_close > rth_open
        day_return = rth_close - rth_open

        results.append(
            {
                "above_stl": above_stl,
                "day_bullish": day_bullish,
                "day_return": round(day_return, 2),
                "dist_to_stl": round(rth_open - stl, 2),
            }
        )

    rdf = pd.DataFrame(results)
    above = rdf[rdf["above_stl"]]
    below = rdf[~rdf["above_stl"]]

    return {
        "test": "Tagesrichtung (Settlement als Bias)",
        "frage": "Open > Settlement → bullisch? Open < Settlement → bärisch?",
        "total_days": len(rdf),
        "open_above_stl": {
            "n": len(above),
            "bullish_pct": round(100 * above["day_bullish"].mean(), 1)
            if len(above) > 0
            else None,
            "mean_return_pts": round(float(above["day_return"].mean()), 1)
            if len(above) > 0
            else None,
            "median_return_pts": round(float(above["day_return"].median()), 1)
            if len(above) > 0
            else None,
        },
        "open_below_stl": {
            "n": len(below),
            "bearish_pct": round(100 * (1 - below["day_bullish"].mean()), 1)
            if len(below) > 0
            else None,
            "mean_return_pts": round(float(below["day_return"].mean()), 1)
            if len(below) > 0
            else None,
            "median_return_pts": round(float(below["day_return"].median()), 1)
            if len(below) > 0
            else None,
        },
        "overall_bullish_pct": round(100 * rdf["day_bullish"].mean(), 1),
        "random_baseline": 50.0,
    }


# ---------- Test 4: Cluster ----------


def test_cluster(df: pd.DataFrame) -> dict:
    """Mehrere nahe Settlements = stärkeres Level?"""
    print("  Test 4: Cluster...")

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    to_ny(pd.DatetimeIndex(df.index))

    # Aktive Settlements pro Bar (aus stl_nearest_above/below nicht ausreichend)
    # Wir laufen den Algo nochmal durch und tracken alle aktiven Levels
    ct = to_ct(pd.DatetimeIndex(df.index))
    ct_hours = pd.Series(ct).dt.hour.values
    prev_hours = np.roll(ct_hours, 1)
    prev_hours[0] = ct_hours[0]
    is_stl = (prev_hours < 15) & (ct_hours >= 15)

    active: dict[float, str | None] = {}
    cluster_touches = []  # (bar_idx, n_nearby, bounced_15)

    for i in range(len(close)):
        c = close[i]
        h = high[i]
        lo_val = low[i]

        if is_stl[i]:
            active[c] = None

        # Remove filled
        to_remove = []
        for level, side in active.items():
            if side is None:
                if c > level:
                    active[level] = "above"
                elif c < level:
                    active[level] = "below"
                continue
            if side == "above" and c < level:
                to_remove.append(level)
            elif side == "below" and c > level:
                to_remove.append(level)
            else:
                if c > level:
                    active[level] = "above"
                elif c < level:
                    active[level] = "below"
        for lv in to_remove:
            del active[lv]

        # Touch-Check: berührt ein Level?
        touched_levels = [
            lv for lv in active if lo_val <= lv <= h and active[lv] is not None
        ]
        if not touched_levels:
            continue

        for lv in touched_levels:
            # Wie viele andere Settlements innerhalb 20 Punkte?
            nearby = sum(1 for other in active if abs(other - lv) <= 20 and other != lv)

            # Bounce: 15 Bars nach Touch
            if i + 15 < len(close):
                side = active.get(lv)
                if side == "above":
                    bounced = close[i + 15] > lv
                elif side == "below":
                    bounced = close[i + 15] < lv
                else:
                    bounced = None

                if bounced is not None:
                    cluster_touches.append(
                        {
                            "nearby": nearby,
                            "bounced": bounced,
                        }
                    )

    if not cluster_touches:
        return {"test": "Cluster", "error": "keine Daten"}

    cdf = pd.DataFrame(cluster_touches)

    by_cluster = {}
    for label, lo_n, hi_n in [
        ("0_allein", 0, 0),
        ("1_nearby", 1, 1),
        ("2_nearby", 2, 2),
        ("3plus", 3, 99),
    ]:
        mask = (cdf["nearby"] >= lo_n) & (cdf["nearby"] <= hi_n)
        n = int(mask.sum())
        if n >= 20:
            by_cluster[label] = {
                "n": n,
                "bounce_pct": round(100 * cdf.loc[mask, "bounced"].mean(), 1),
            }

    return {
        "test": "Settlement-Cluster (nahe Settlements = stärker?)",
        "frage": "Mehrere Settlements in 20pt-Band → höhere Bounce-Rate?",
        "total_touches": len(cdf),
        "overall_bounce_pct": round(100 * cdf["bounced"].mean(), 1),
        "by_cluster_size": by_cluster,
    }


# ---------- Test 5: Tagestyp (KER × Settlement) ----------


def test_regime(df: pd.DataFrame) -> dict:
    """Settlement-Bounce nach Markt-Regime (KER + ATR)."""
    print("  Test 5: Tagestyp (KER × ATR)...")

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    # KER (Kaufman Efficiency Ratio) auf 20-Bar-Fenster
    ker_period = 20
    ker = np.full(n, np.nan)
    for i in range(ker_period, n):
        direction = abs(close[i] - close[i - ker_period])
        volatility = sum(
            abs(close[j] - close[j - 1]) for j in range(i - ker_period + 1, i + 1)
        )
        if volatility > 0:
            ker[i] = direction / volatility

    # ATR auf 20-Bar-Fenster
    atr = np.full(n, np.nan)
    tr = np.maximum(
        high - low,
        np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    for i in range(ker_period, n):
        atr[i] = np.mean(tr[i - ker_period + 1 : i + 1])

    # Regime-Schwellwerte (Median-Split)
    valid_ker = ker[~np.isnan(ker)]
    valid_atr = atr[~np.isnan(atr)]
    ker_median = np.median(valid_ker)
    atr_median = np.median(valid_atr)

    # Touch-Events mit Regime-Info
    touch_mask = df["stl_touch"].values.astype(bool)
    results = []

    for i in range(n):
        if not touch_mask[i] or np.isnan(ker[i]) or np.isnan(atr[i]):
            continue
        if i + 15 >= n:
            continue

        c = close[i]
        above_val = df["stl_nearest_above"].values[i]
        below_val = df["stl_nearest_below"].values[i]

        # Welches Level wird berührt?
        dist_above = abs(c - above_val) if not np.isnan(above_val) else np.inf
        dist_below = abs(c - below_val) if not np.isnan(below_val) else np.inf
        if dist_above <= dist_below and not np.isnan(above_val):
            level = above_val
            side = "below"  # Preis ist unter dem Level
        elif not np.isnan(below_val):
            level = below_val
            side = "above"  # Preis ist über dem Level
        else:
            continue

        # Bounce
        if side == "above":
            bounced = close[i + 15] > level
        else:
            bounced = close[i + 15] < level

        trending = ker[i] > ker_median
        volatile = atr[i] > atr_median

        if trending and volatile:
            regime = "trending_volatil"
        elif trending and not volatile:
            regime = "trending_ruhig"
        elif not trending and volatile:
            regime = "choppy_volatil"
        else:
            regime = "choppy_ruhig"

        results.append(
            {"regime": regime, "bounced": bounced, "ker": ker[i], "atr": atr[i]}
        )

    rdf = pd.DataFrame(results)
    by_regime = {}
    for regime in [
        "trending_volatil",
        "trending_ruhig",
        "choppy_volatil",
        "choppy_ruhig",
    ]:
        mask = rdf["regime"] == regime
        n_r = int(mask.sum())
        if n_r >= 20:
            by_regime[regime] = {
                "n": n_r,
                "bounce_pct": round(100 * rdf.loc[mask, "bounced"].mean(), 1),
            }

    return {
        "test": "Settlement × Tagestyp (KER/ATR Regime)",
        "frage": "Bounce-Rate nach Marktphase?",
        "total_touches": len(rdf),
        "overall_bounce_pct": round(100 * rdf["bounced"].mean(), 1),
        "ker_median": round(float(ker_median), 4),
        "atr_median": round(float(atr_median), 2),
        "by_regime": by_regime,
    }


# ---------- Test 6: Breakout-Momentum ----------


def test_breakout_momentum(df: pd.DataFrame) -> dict:
    """MFE nach Settlement-Break."""
    print("  Test 6: Breakout-Momentum...")

    df["stl_filled"].values.astype(bool)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    ny = to_ny(pd.DatetimeIndex(df.index))
    hours = pd.Series(ny).dt.hour.values

    # Bei Fill wissen wir die Richtung: Preis hat Settlement durchbrochen
    # Wir brauchen den Settlement-Preis und die Richtung
    ct = to_ct(pd.DatetimeIndex(df.index))
    ct_hours = pd.Series(ct).dt.hour.values
    prev_ct = np.roll(ct_hours, 1)
    prev_ct[0] = ct_hours[0]
    is_stl = (prev_ct < 15) & (ct_hours >= 15)

    active: dict[float, str | None] = {}
    breakouts = []

    for i in range(n):
        c = close[i]

        if is_stl[i]:
            active[c] = None

        to_remove = []
        for level, side in active.items():
            if side is None:
                if c > level:
                    active[level] = "above"
                elif c < level:
                    active[level] = "below"
                continue

            if side == "above" and c < level:
                # Breakout nach unten
                to_remove.append(level)
                if i + 60 < n:
                    mfe_5 = level - min(low[i + 1 : i + 6])
                    mfe_15 = level - min(low[i + 1 : i + 16])
                    mfe_60 = level - min(low[i + 1 : i + 61])
                    mae_15 = max(high[i + 1 : i + 16]) - level

                    breakouts.append(
                        {
                            "direction": "bearish",
                            "level": level,
                            "mfe_5": round(mfe_5, 2),
                            "mfe_15": round(mfe_15, 2),
                            "mfe_60": round(mfe_60, 2),
                            "mae_15": round(mae_15, 2),
                            "hour_ny": int(hours[i]),
                        }
                    )

            elif side == "below" and c > level:
                # Breakout nach oben
                to_remove.append(level)
                if i + 60 < n:
                    mfe_5 = max(high[i + 1 : i + 6]) - level
                    mfe_15 = max(high[i + 1 : i + 16]) - level
                    mfe_60 = max(high[i + 1 : i + 61]) - level
                    mae_15 = level - min(low[i + 1 : i + 16])

                    breakouts.append(
                        {
                            "direction": "bullish",
                            "level": level,
                            "mfe_5": round(mfe_5, 2),
                            "mfe_15": round(mfe_15, 2),
                            "mfe_60": round(mfe_60, 2),
                            "mae_15": round(mae_15, 2),
                            "hour_ny": int(hours[i]),
                        }
                    )
            else:
                if c > level:
                    active[level] = "above"
                elif c < level:
                    active[level] = "below"

        for lv in to_remove:
            del active[lv]

    if not breakouts:
        return {"test": "Breakout-Momentum", "error": "keine Breaks"}

    bdf = pd.DataFrame(breakouts)

    by_dir = {}
    for d in ["bullish", "bearish"]:
        mask = bdf["direction"] == d
        sub = bdf[mask]
        if len(sub) >= 20:
            by_dir[d] = {
                "n": len(sub),
                "mfe_5_mean": round(float(sub["mfe_5"].mean()), 1),
                "mfe_5_median": round(float(sub["mfe_5"].median()), 1),
                "mfe_15_mean": round(float(sub["mfe_15"].mean()), 1),
                "mfe_15_median": round(float(sub["mfe_15"].median()), 1),
                "mfe_60_mean": round(float(sub["mfe_60"].mean()), 1),
                "mfe_60_median": round(float(sub["mfe_60"].median()), 1),
                "mae_15_mean": round(float(sub["mae_15"].mean()), 1),
                "rr_15": round(float(sub["mfe_15"].mean() / sub["mae_15"].mean()), 2)
                if sub["mae_15"].mean() > 0
                else None,
            }

    # Breakout-Profit (MFE > MAE?)
    overall_rr = (
        round(float(bdf["mfe_15"].mean() / bdf["mae_15"].mean()), 2)
        if bdf["mae_15"].mean() > 0
        else None
    )

    # Profitable Breakouts (MFE_15 > MAE_15)
    profitable = (bdf["mfe_15"] > bdf["mae_15"]).mean()

    return {
        "test": "Breakout-Momentum (MFE nach Settlement-Break)",
        "frage": "Gibt es Momentum in Breakout-Richtung?",
        "total_breaks": len(bdf),
        "mfe_15_mean": round(float(bdf["mfe_15"].mean()), 1),
        "mfe_15_median": round(float(bdf["mfe_15"].median()), 1),
        "mae_15_mean": round(float(bdf["mae_15"].mean()), 1),
        "rr_15_overall": overall_rr,
        "profitable_pct": round(100 * profitable, 1),
        "by_direction": by_dir,
    }


# ---------- Main ----------


def main():
    print("Settlement Spezialtests – 6 Tests")
    print("=" * 50)

    print("\nLade Daten...")
    df = load_bars()
    print(f"  {len(df):,} Bars geladen")

    print("Laufe Settlement-Algo...")
    df = run_settlement_algo(df)
    print(f"  {int(df['is_settlement_bar'].sum())} Settlements erkannt")

    print("\nBaue Daily-Tabelle...")
    daily = build_daily(df)
    print(f"  {len(daily)} Handelstage mit Settlement")

    print("\n--- Starte Tests ---\n")

    results = {}

    # Test 1
    results["magnet"] = test_magnet(df)

    # Test 2
    results["gap_fill"] = test_gap_fill(df, daily)

    # Test 3
    results["daily_bias"] = test_daily_bias(df, daily)

    # Test 4
    results["cluster"] = test_cluster(df)

    # Test 5
    results["regime"] = test_regime(df)

    # Test 6
    results["breakout_momentum"] = test_breakout_momentum(df)

    # Speichern
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESEARCH_DIR / "settlement_spezialtests_2026-05-09.json"
    md_path = RESEARCH_DIR / "settlement_spezialtests_2026-05-09.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ JSON: {json_path}")

    # MD-Report
    lines = ["# Settlement Spezialtests (2026-05-09, Chat 406)\n"]
    lines.append("717.665 1min-Bars, 642 Handelstage, 529 Settlements\n")
    lines.append("---\n")

    for key, data in results.items():
        lines.append(f"## {data.get('test', key)}\n")
        lines.append(f"**Frage**: {data.get('frage', '?')}\n")
        for k, v in data.items():
            if k in ("test", "frage"):
                continue
            if isinstance(v, dict):
                lines.append(f"\n**{k}**:\n")
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        items = ", ".join(f"{k3}={v3}" for k3, v3 in v2.items())
                        lines.append(f"- {k2}: {items}\n")
                    else:
                        lines.append(f"- {k2}: {v2}\n")
            else:
                lines.append(f"- **{k}**: {v}\n")
        lines.append("\n---\n")

    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"✓ MD:   {md_path}")

    # Zusammenfassung
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)

    for key, data in results.items():
        print(f"\n  {data.get('test', key)}:")
        # Kernzahl ausgeben
        if "fill_rate_pct" in data:
            print(f"    Gap-Fill-Rate: {data['fill_rate_pct']}%")
        if "overall_bounce_pct" in data:
            print(f"    Bounce: {data['overall_bounce_pct']}%")
        if "gesamt_mean_dist" in data:
            print(
                f"    Ø Distanz: {data['gesamt_mean_dist']}pt (Random: {data['random_expected_dist']}pt)"
            )
        if "rr_15_overall" in data:
            print(f"    RR @15min: {data['rr_15_overall']}")
        if "open_above_stl" in data:
            above = data["open_above_stl"]
            below = data["open_below_stl"]
            print(
                f"    Über STL → bullisch: {above.get('bullish_pct')}% (n={above.get('n')})"
            )
            print(
                f"    Unter STL → bärisch: {below.get('bearish_pct')}% (n={below.get('n')})"
            )
        if "by_regime" in data:
            for regime, vals in data["by_regime"].items():
                print(f"    {regime}: {vals['bounce_pct']}% (n={vals['n']})")


if __name__ == "__main__":
    main()
