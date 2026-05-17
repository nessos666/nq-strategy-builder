"""
Settlement Breakout-Momentum – Tiefenforschung.

Nur Fills die NACH dem Settlement-Tag passieren (echte Breakouts).
16h-Fills sind Artefakte (Settlement erzeugt + sofort durchquert).

Aufschlüsselung:
- Alter des Settlements bei Fill (Stunden/Tage)
- Uhrzeit des Fills (NY)
- Richtung (Bull/Bear)
- Session (Globex/Pre-Market/RTH)
- MFE/MAE auf 5/15/60 Bars

Ausführung:
    .venv/bin/python3 scripts/settlement_breakout_deep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = Path(
    Path(__file__).resolve().parent.parent
    "nq_backtest/data/nq_1m_databento_2024_2026.parquet"
)
ALGO_PATH = ROOT / "david_bibliothek/01_Hoch_Tief/8. Settlement Levels.py"
RESEARCH_DIR = ROOT / "david_bibliothek/01_Hoch_Tief/_research"


def load_and_run():
    import importlib.util

    df = pd.read_parquet(DATA_PATH)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    spec = importlib.util.spec_from_file_location("stl", str(ALGO_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run(df)


def to_ny(idx):
    try:
        return idx.tz_convert("America/New_York")
    except TypeError:
        return idx.tz_localize("UTC").tz_convert("America/New_York")


def to_ct(idx):
    try:
        return idx.tz_convert("America/Chicago")
    except TypeError:
        return idx.tz_localize("UTC").tz_convert("America/Chicago")


def classify_session(hour_ny: int) -> str:
    if 18 <= hour_ny <= 23:
        return "globex_abend"
    elif 0 <= hour_ny < 5:
        return "asia_london"
    elif 5 <= hour_ny < 9:
        return "pre_market"
    elif 9 <= hour_ny < 12:
        return "rth_am"
    elif 12 <= hour_ny < 14:
        return "rth_lunch"
    elif 14 <= hour_ny < 16:
        return "rth_pm"
    elif 16 <= hour_ny < 18:
        return "settlement_hour"
    return "unknown"


def main():
    print("Settlement Breakout-Momentum – Tiefenforschung")
    print("=" * 55)

    print("\nLade Daten + Algo...")
    df = load_and_run()
    print(f"  {len(df):,} Bars")

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    ny = to_ny(pd.DatetimeIndex(df.index))
    ct = to_ct(pd.DatetimeIndex(df.index))
    ny_hours = pd.Series(ny).dt.hour.values
    ct_hours = pd.Series(ct).dt.hour.values
    ny_dates = pd.Series(ny).dt.date.values

    # Settlement-Bar erkennen
    prev_ct = np.roll(ct_hours, 1)
    prev_ct[0] = ct_hours[0]
    is_stl = (prev_ct < 15) & (ct_hours >= 15)

    # Aktive Settlements tracken: level → (side, creation_bar_idx, creation_date)
    active: dict[float, tuple[str | None, int, object]] = {}
    breakouts = []

    print("Scanne Breakouts...")
    for i in range(n):
        c = close[i]
        h = high[i]
        low[i]

        if is_stl[i]:
            active[c] = (None, i, ny_dates[i])

        to_remove = []
        for level, (side, created_idx, created_date) in active.items():
            if side is None:
                if c > level:
                    active[level] = ("above", created_idx, created_date)
                elif c < level:
                    active[level] = ("below", created_idx, created_date)
                continue

            filled = False
            direction = None
            if side == "above" and c < level:
                filled = True
                direction = "bearish"
            elif side == "below" and c > level:
                filled = True
                direction = "bullish"

            if filled:
                to_remove.append(level)
                if i + 60 >= n:
                    continue

                age_bars = i - created_idx
                age_hours = round(age_bars / 60, 1)
                same_day = ny_dates[i] == created_date

                if direction == "bearish":
                    mfe_5 = level - min(low[i + 1 : i + 6])
                    mfe_15 = level - min(low[i + 1 : i + 16])
                    mfe_60 = level - min(low[i + 1 : i + 61])
                    mae_5 = max(high[i + 1 : i + 6]) - level
                    mae_15 = max(high[i + 1 : i + 16]) - level
                    mae_60 = max(high[i + 1 : i + 61]) - level
                else:
                    mfe_5 = max(high[i + 1 : i + 6]) - level
                    mfe_15 = max(high[i + 1 : i + 16]) - level
                    mfe_60 = max(high[i + 1 : i + 61]) - level
                    mae_5 = level - min(low[i + 1 : i + 6])
                    mae_15 = level - min(low[i + 1 : i + 16])
                    mae_60 = level - min(low[i + 1 : i + 61])

                breakouts.append(
                    {
                        "direction": direction,
                        "level": round(level, 2),
                        "hour_ny": int(ny_hours[i]),
                        "session": classify_session(int(ny_hours[i])),
                        "same_day": same_day,
                        "age_bars": age_bars,
                        "age_hours": age_hours,
                        "mfe_5": round(mfe_5, 2),
                        "mfe_15": round(mfe_15, 2),
                        "mfe_60": round(mfe_60, 2),
                        "mae_5": round(mae_5, 2),
                        "mae_15": round(mae_15, 2),
                        "mae_60": round(mae_60, 2),
                    }
                )
            else:
                if c > level:
                    active[level] = ("above", created_idx, created_date)
                elif c < level:
                    active[level] = ("below", created_idx, created_date)

        for lv in to_remove:
            del active[lv]

    bdf = pd.DataFrame(breakouts)
    print(f"  {len(bdf)} Breakouts total")

    # --- Aufspaltung: Same-Day vs. Delayed ---
    same = bdf[bdf["same_day"]]
    delayed = bdf[~bdf["same_day"]]

    print(f"  Same-Day: {len(same)}  |  Delayed (echter Break): {len(delayed)}")

    results = {
        "test": "Settlement Breakout-Momentum – Tiefenforschung",
        "total_breaks": len(bdf),
        "same_day_breaks": len(same),
        "delayed_breaks": len(delayed),
    }

    def stats(sub, label):
        if len(sub) < 5:
            return {label: {"n": len(sub), "zu_wenig_daten": True}}
        return {
            "n": len(sub),
            "mfe_5_mean": round(float(sub["mfe_5"].mean()), 1),
            "mfe_5_median": round(float(sub["mfe_5"].median()), 1),
            "mfe_15_mean": round(float(sub["mfe_15"].mean()), 1),
            "mfe_15_median": round(float(sub["mfe_15"].median()), 1),
            "mfe_60_mean": round(float(sub["mfe_60"].mean()), 1),
            "mfe_60_median": round(float(sub["mfe_60"].median()), 1),
            "mae_15_mean": round(float(sub["mae_15"].mean()), 1),
            "mae_15_median": round(float(sub["mae_15"].median()), 1),
            "mae_60_mean": round(float(sub["mae_60"].mean()), 1),
            "rr_5": round(float(sub["mfe_5"].mean() / sub["mae_5"].mean()), 2)
            if sub["mae_5"].mean() > 0
            else None,
            "rr_15": round(float(sub["mfe_15"].mean() / sub["mae_15"].mean()), 2)
            if sub["mae_15"].mean() > 0
            else None,
            "rr_60": round(float(sub["mfe_60"].mean() / sub["mae_60"].mean()), 2)
            if sub["mae_60"].mean() > 0
            else None,
            "profitable_15_pct": round(100 * (sub["mfe_15"] > sub["mae_15"]).mean(), 1),
            "profitable_60_pct": round(100 * (sub["mfe_60"] > sub["mae_60"]).mean(), 1),
        }

    # Same-Day gesamt
    results["same_day"] = stats(same, "same_day")

    # Delayed gesamt
    results["delayed"] = stats(delayed, "delayed")

    # Delayed nach Richtung
    results["delayed_bullish"] = stats(
        delayed[delayed["direction"] == "bullish"], "delayed_bullish"
    )
    results["delayed_bearish"] = stats(
        delayed[delayed["direction"] == "bearish"], "delayed_bearish"
    )

    # Delayed nach Session
    results["delayed_by_session"] = {}
    for sess in [
        "globex_abend",
        "asia_london",
        "pre_market",
        "rth_am",
        "rth_lunch",
        "rth_pm",
    ]:
        sub = delayed[delayed["session"] == sess]
        if len(sub) >= 10:
            results["delayed_by_session"][sess] = stats(sub, sess)

    # Delayed nach Alter
    results["delayed_by_age"] = {}
    for label, lo_h, hi_h in [
        ("1-6h", 1, 6),
        ("6-24h", 6, 24),
        ("24-72h", 24, 72),
        ("72h+", 72, 99999),
    ]:
        sub = delayed[(delayed["age_hours"] >= lo_h) & (delayed["age_hours"] < hi_h)]
        if len(sub) >= 10:
            results["delayed_by_age"][label] = stats(sub, label)

    # Delayed nach Stunde (NY)
    results["delayed_by_hour"] = {}
    for h in range(24):
        sub = delayed[delayed["hour_ny"] == h]
        if len(sub) >= 10:
            results["delayed_by_hour"][str(h)] = {
                "n": len(sub),
                "mfe_15_mean": round(float(sub["mfe_15"].mean()), 1),
                "mae_15_mean": round(float(sub["mae_15"].mean()), 1),
                "rr_15": round(float(sub["mfe_15"].mean() / sub["mae_15"].mean()), 2)
                if sub["mae_15"].mean() > 0
                else None,
                "profitable_15_pct": round(
                    100 * (sub["mfe_15"] > sub["mae_15"]).mean(), 1
                ),
            }

    # Speichern
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESEARCH_DIR / "settlement_breakout_deep_2026-05-09.json"
    md_path = RESEARCH_DIR / "settlement_breakout_deep_2026-05-09.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # MD Report
    lines = [
        "# Settlement Breakout-Momentum – Tiefenforschung (2026-05-09)\n\n",
        f"Total Breaks: {len(bdf)} | Same-Day: {len(same)} | Delayed: {len(delayed)}\n\n",
        "---\n\n",
    ]

    def write_stats(d, title):
        lines.append(f"### {title}\n\n")
        if isinstance(d, dict) and "zu_wenig_daten" in d:
            lines.append(f"n={d['n']} – zu wenig Daten\n\n")
            return
        for k, v in d.items():
            lines.append(f"- **{k}**: {v}\n")
        lines.append("\n")

    write_stats(results["same_day"], "Same-Day Breaks (Artefakt)")
    write_stats(results["delayed"], "Delayed Breaks (ECHTE Breakouts)")
    write_stats(results["delayed_bullish"], "Delayed Bullish")
    write_stats(results["delayed_bearish"], "Delayed Bearish")

    lines.append("## Delayed nach Session\n\n")
    for sess, d in results["delayed_by_session"].items():
        write_stats(d, sess)

    lines.append("## Delayed nach Alter\n\n")
    for age, d in results["delayed_by_age"].items():
        write_stats(d, age)

    lines.append("## Delayed nach Stunde (NY)\n\n")
    lines.append("| Stunde | n | MFE 15 | MAE 15 | RR | Profitable |\n")
    lines.append("|--------|---|--------|--------|-----|------------|\n")
    for h in sorted(results["delayed_by_hour"].keys(), key=int):
        d = results["delayed_by_hour"][h]
        lines.append(
            f"| {h}h | {d['n']} | {d['mfe_15_mean']} | {d['mae_15_mean']} | {d['rr_15']} | {d['profitable_15_pct']}% |\n"
        )

    with open(md_path, "w") as f:
        f.writelines(lines)

    print(f"\n✓ JSON: {json_path}")
    print(f"✓ MD:   {md_path}")

    # Zusammenfassung
    print("\n" + "=" * 55)
    print("ZUSAMMENFASSUNG")
    print("=" * 55)

    print(f"\n  Same-Day ({len(same)} Breaks):")
    if len(same) >= 5:
        sd = results["same_day"]
        print(
            f"    MFE 15min: Ø {sd['mfe_15_mean']}pt | MAE: Ø {sd['mae_15_mean']}pt | RR: {sd['rr_15']}"
        )

    print(f"\n  DELAYED ({len(delayed)} echte Breaks):")
    if len(delayed) >= 5:
        dd = results["delayed"]
        print(
            f"    MFE 15min: Ø {dd['mfe_15_mean']}pt | MAE: Ø {dd['mae_15_mean']}pt | RR: {dd['rr_15']}"
        )
        print(
            f"    MFE 60min: Ø {dd['mfe_60_mean']}pt | MAE: Ø {dd['mae_60_mean']}pt | RR: {dd['rr_60']}"
        )
        print(f"    Profitable @15min: {dd['profitable_15_pct']}%")
        print(f"    Profitable @60min: {dd['profitable_60_pct']}%")

    if len(delayed[delayed["direction"] == "bullish"]) >= 5:
        db = results["delayed_bullish"]
        print(
            f"\n    Bullish:  RR15={db['rr_15']} | MFE={db['mfe_15_mean']}pt | n={db['n']}"
        )
    if len(delayed[delayed["direction"] == "bearish"]) >= 5:
        db = results["delayed_bearish"]
        print(
            f"    Bearish:  RR15={db['rr_15']} | MFE={db['mfe_15_mean']}pt | n={db['n']}"
        )

    print("\n  Nach Session:")
    for sess, d in results["delayed_by_session"].items():
        print(
            f"    {sess:20s}: RR={d['rr_15']} | MFE={d['mfe_15_mean']}pt | n={d['n']}"
        )

    print("\n  Nach Alter:")
    for age, d in results["delayed_by_age"].items():
        print(f"    {age:10s}: RR={d['rr_15']} | MFE={d['mfe_15_mean']}pt | n={d['n']}")


if __name__ == "__main__":
    main()
