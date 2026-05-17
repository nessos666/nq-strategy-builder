"""
Premium/Discount Tiefentests – 6 Tests außerhalb der Baumaschine.

1. Inspect: Grundstatistik
2. EQ-Linie als S/R (Bounce-Rate)
3. Premium=Short, Discount=Long (Richtungs-Edge)
4. Session-Vergleich (welche Zone ist am stärksten)
5. Confluence (alle Sessions gleichzeitig)
6. Filter für OBs/FVGs (Verbesserung?)

Ausführung:
    .venv/bin/python3 scripts/premium_discount_tests.py
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
PD_ALGO = (
    ROOT / "david_bibliothek/12_Alte_Algos_Noch_Nicht_Getestet/premium_discount.py"
)
RESEARCH_DIR = ROOT / "david_bibliothek/12_Alte_Algos_Noch_Nicht_Getestet/_research"

SESSIONS = ["asia", "ldn", "am", "pm"]


def load_and_run():
    import importlib.util

    df_raw = pd.read_parquet(DATA_PATH)
    # P/D Algo braucht lowercase
    df = df_raw.rename(columns={c: c.lower() for c in df_raw.columns})
    spec = importlib.util.spec_from_file_location("pd_algo", str(PD_ALGO))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # OB/FVG Algos brauchen Großbuchstaben
    cap_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df_cap = df_raw.rename(
        columns={k: v for k, v in cap_map.items() if k in df_raw.columns}
    )
    return mod.run(df), df_cap


def to_ny(idx):
    try:
        return idx.tz_convert("America/New_York")
    except TypeError:
        return idx.tz_localize("UTC").tz_convert("America/New_York")


# ─── Test 1: Inspect ───
def test_inspect(df):
    print("  Test 1: Inspect...")
    ny = to_ny(pd.DatetimeIndex(df.index))
    hours = pd.Series(ny).dt.hour.values
    n = len(df)

    result = {"test": "Inspect (Grundstatistik)", "bars": n}

    for s in SESSIONS:
        frozen = df[f"pd_{s}_frozen"].sum()
        prem = df[f"pd_{s}_premium"].sum()
        disc = df[f"pd_{s}_discount"].sum()
        hi = df[f"pd_{s}_high"].dropna()
        lo = df[f"pd_{s}_low"].dropna()
        spans = hi.values - lo.values if len(hi) > 0 else np.array([])
        valid_spans = spans[spans > 0] if len(spans) > 0 else np.array([])

        result[s] = {
            "frozen_pct": round(100 * frozen / n, 1),
            "premium_pct": round(100 * prem / n, 1),
            "discount_pct": round(100 * disc / n, 1),
            "zone_span_mean": round(float(valid_spans.mean()), 1)
            if len(valid_spans) > 0
            else None,
            "zone_span_median": round(float(np.median(valid_spans)), 1)
            if len(valid_spans) > 0
            else None,
        }

    # Stunden-Verteilung der aggregierten Premium/Discount
    by_hour = {}
    for h in range(24):
        mask = hours == h
        if mask.sum() > 500:
            prem_h = df.loc[mask, "pd_in_premium"].mean()
            disc_h = df.loc[mask, "pd_in_discount"].mean()
            by_hour[str(h)] = {
                "premium_pct": round(100 * prem_h, 1),
                "discount_pct": round(100 * disc_h, 1),
                "n": int(mask.sum()),
            }
    result["by_hour"] = by_hour
    return result


# ─── Test 2: EQ-Bounce ───
def test_eq_bounce(df):
    print("  Test 2: EQ-Linie als S/R...")
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    result = {
        "test": "EQ-Linie Bounce-Rate",
        "frage": "Reagiert der Preis auf die 50%-Linie?",
    }

    for s in SESSIONS:
        eq = df[f"pd_{s}_eq"].values
        frozen = df[f"pd_{s}_frozen"].values

        touches = []
        for i in range(n - 15):
            if not frozen[i] or np.isnan(eq[i]):
                continue
            e = eq[i]
            # Touch: Low <= EQ <= High
            if low[i] <= e <= high[i]:
                # Bounce: Close 15 Bars später auf gleicher Seite wie vorher?
                if close[i] > e:
                    side = "above"
                    bounced = close[i + 15] > e
                else:
                    side = "below"
                    bounced = close[i + 15] < e
                touches.append({"bounced": bounced, "side": side})

        if len(touches) > 50:
            tdf = pd.DataFrame(touches)
            result[s] = {
                "total_touches": len(tdf),
                "bounce_pct": round(100 * tdf["bounced"].mean(), 1),
                "touches_per_day": round(len(tdf) / 642, 1),
            }
    return result


# ─── Test 3: Premium=Short, Discount=Long ───
def test_direction(df):
    print("  Test 3: Premium=Short, Discount=Long...")
    close = df["close"].values
    n = len(close)

    result = {
        "test": "Richtungs-Edge",
        "frage": "Kaufe in Discount, verkaufe in Premium → Edge?",
    }

    for s in SESSIONS:
        prem = df[f"pd_{s}_premium"].values
        disc = df[f"pd_{s}_discount"].values
        frozen = df[f"pd_{s}_frozen"].values

        prem_returns = []
        disc_returns = []

        for horizon in [5, 15, 60]:
            # Premium → erwarten Short-Profit (Preis fällt)
            p_mask = prem & frozen
            p_idx = np.where(p_mask)[0]
            p_idx = p_idx[p_idx + horizon < n]
            if len(p_idx) > 100:
                p_ret = close[p_idx] - close[p_idx + horizon]  # Short-Return
                prem_returns.append(
                    {
                        "horizon": horizon,
                        "mean_return": round(float(p_ret.mean()), 2),
                        "positive_pct": round(100 * (p_ret > 0).mean(), 1),
                        "n": len(p_idx),
                    }
                )

            # Discount → erwarten Long-Profit (Preis steigt)
            d_mask = disc & frozen
            d_idx = np.where(d_mask)[0]
            d_idx = d_idx[d_idx + horizon < n]
            if len(d_idx) > 100:
                d_ret = close[d_idx + horizon] - close[d_idx]  # Long-Return
                disc_returns.append(
                    {
                        "horizon": horizon,
                        "mean_return": round(float(d_ret.mean()), 2),
                        "positive_pct": round(100 * (d_ret > 0).mean(), 1),
                        "n": len(d_idx),
                    }
                )

        if prem_returns or disc_returns:
            result[s] = {
                "premium_short": prem_returns,
                "discount_long": disc_returns,
            }
    return result


# ─── Test 4: Session-Vergleich ───
def test_session_compare(df):
    print("  Test 4: Session-Vergleich...")
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    result = {
        "test": "Session-Vergleich",
        "frage": "Welche Session-Zone wird am meisten respektiert?",
    }

    for s in SESSIONS:
        sh = df[f"pd_{s}_high"].values
        sl = df[f"pd_{s}_low"].values
        frozen = df[f"pd_{s}_frozen"].values

        # Wie oft berührt der Preis High/Low und prallt ab?
        hi_touches = 0
        hi_bounces = 0
        lo_touches = 0
        lo_bounces = 0

        for i in range(n - 15):
            if not frozen[i] or np.isnan(sh[i]):
                continue
            # High-Touch
            if high[i] >= sh[i] and low[i] < sh[i]:
                hi_touches += 1
                if close[i + 15] < sh[i]:
                    hi_bounces += 1
            # Low-Touch
            if low[i] <= sl[i] and high[i] > sl[i]:
                lo_touches += 1
                if close[i + 15] > sl[i]:
                    lo_bounces += 1

        result[s] = {
            "high_touches": hi_touches,
            "high_bounce_pct": round(100 * hi_bounces / hi_touches, 1)
            if hi_touches > 20
            else None,
            "low_touches": lo_touches,
            "low_bounce_pct": round(100 * lo_bounces / lo_touches, 1)
            if lo_touches > 20
            else None,
        }
    return result


# ─── Test 5: Confluence ───
def test_confluence(df):
    print("  Test 5: Confluence (alle Sessions gleichzeitig)...")
    close = df["close"].values
    n = len(close)

    result = {
        "test": "Confluence",
        "frage": "Preis in Discount ALLER Sessions → stärker?",
    }

    # Zähle wie viele Sessions gleichzeitig Discount/Premium zeigen
    disc_count = np.zeros(n, dtype=int)
    prem_count = np.zeros(n, dtype=int)

    for s in SESSIONS:
        disc_count += df[f"pd_{s}_discount"].values.astype(int)
        prem_count += df[f"pd_{s}_premium"].values.astype(int)

    # Return nach Confluence-Level
    for label, counts, direction in [
        ("discount_long", disc_count, 1),
        ("premium_short", prem_count, -1),
    ]:
        by_level = {}
        for level in range(1, 5):
            mask = counts >= level
            idx = np.where(mask)[0]
            idx = idx[idx + 15 < n]
            if len(idx) > 200:
                ret = direction * (close[idx + 15] - close[idx])
                by_level[f"{level}_sessions"] = {
                    "n": len(idx),
                    "mean_return": round(float(ret.mean()), 2),
                    "positive_pct": round(100 * (ret > 0).mean(), 1),
                }
        result[label] = by_level
    return result


# ─── Test 6: Filter für OBs/FVGs ───
def test_zone_filter(df, df_raw):
    print("  Test 6: Premium/Discount als Filter für OBs/FVGs...")

    import importlib.util

    ob_path = (
        ROOT / "david_bibliothek/03_Order_Blocks/7d. Session Hoch-Tief Orderblock.py"
    )
    fvg_path = ROOT / "david_bibliothek/02_FVG_Zonen/3. FVG Standard.py"

    result = {
        "test": "P/D als Filter für OBs/FVGs",
        "frage": "OB/FVG in Discount → bessere Bounce-Rate als in Premium?",
    }

    close = df["close"].values
    n = len(close)

    for algo_name, algo_path, touch_col_pattern in [
        ("s_ob", ob_path, "s_ob"),
        ("fvg", fvg_path, "fvg"),
    ]:
        if not algo_path.exists():
            result[algo_name] = {"error": f"{algo_path} nicht gefunden"}
            continue

        try:
            spec = importlib.util.spec_from_file_location(algo_name, str(algo_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # OB/FVG Algos brauchen Großbuchstaben-Spalten
            zone_df = mod.run(df_raw)
        except Exception as e:
            result[algo_name] = {"error": str(e)}
            continue

        # Finde Touch-Spalten
        touch_cols = [
            c for c in zone_df.columns if "_touch" in c and touch_col_pattern in c
        ]
        if not touch_cols:
            touch_cols = [
                c
                for c in zone_df.columns
                if touch_col_pattern in c and zone_df[c].dtype == bool
            ]

        if not touch_cols:
            result[algo_name] = {
                "error": "keine Touch-Spalten",
                "cols": list(zone_df.columns)[:20],
            }
            continue

        # Alle Touch-Spalten OR-verknüpfen für maximale Abdeckung
        touch = np.zeros(n, dtype=bool)
        for tc in touch_cols:
            touch |= zone_df[tc].values.astype(bool)

        algo_result = {
            "touch_cols_used": len(touch_cols),
            "total_touches": int(touch.sum()),
        }
        for s in SESSIONS:
            prem = df[f"pd_{s}_premium"].values
            disc = df[f"pd_{s}_discount"].values
            frozen = df[f"pd_{s}_frozen"].values

            p_mask = touch & prem & frozen
            p_idx = np.where(p_mask)[0]
            p_idx = p_idx[p_idx + 15 < n]

            d_mask = touch & disc & frozen
            d_idx = np.where(d_mask)[0]
            d_idx = d_idx[d_idx + 15 < n]

            if len(p_idx) > 30 and len(d_idx) > 30:
                p_bounce = (close[p_idx + 15] < close[p_idx]).mean()
                d_bounce = (close[d_idx + 15] > close[d_idx]).mean()

                algo_result[s] = {
                    "premium_n": len(p_idx),
                    "premium_short_pct": round(100 * p_bounce, 1),
                    "discount_n": len(d_idx),
                    "discount_long_pct": round(100 * d_bounce, 1),
                    "delta": round(100 * (d_bounce - p_bounce), 1),
                }

        result[algo_name] = algo_result

    return result


# ─── Main ───
def main():
    print("Premium/Discount Tiefentests – 6 Tests")
    print("=" * 50)

    print("\nLade Daten + Algo...")
    df, df_raw = load_and_run()
    print(f"  {len(df):,} Bars, 31 P/D Spalten")

    results = {}
    results["inspect"] = test_inspect(df)
    results["eq_bounce"] = test_eq_bounce(df)
    results["direction"] = test_direction(df)
    results["session_compare"] = test_session_compare(df)
    results["confluence"] = test_confluence(df)
    results["zone_filter"] = test_zone_filter(df, df_raw)

    # Speichern
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESEARCH_DIR / "premium_discount_tiefentests_2026-05-10.json"
    md_path = RESEARCH_DIR / "premium_discount_tiefentests_2026-05-10.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ JSON: {json_path}")

    # MD Report
    lines = ["# Premium/Discount Tiefentests (2026-05-10)\n\n"]
    lines.append("717.665 1min-Bars, 642 Handelstage, 4 Sessions\n\n---\n\n")

    for key, data in results.items():
        lines.append(f"## {data.get('test', key)}\n\n")
        if "frage" in data:
            lines.append(f"**Frage**: {data['frage']}\n\n")
        for k, v in data.items():
            if k in ("test", "frage"):
                continue
            if isinstance(v, dict):
                lines.append(f"### {k}\n\n")
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        items = ", ".join(f"{k3}={v3}" for k3, v3 in v2.items())
                        lines.append(f"- {k2}: {items}\n")
                    elif isinstance(v2, list):
                        for item in v2:
                            if isinstance(item, dict):
                                items = ", ".join(
                                    f"{k3}={v3}" for k3, v3 in item.items()
                                )
                                lines.append(f"  - {items}\n")
                    else:
                        lines.append(f"- {k2}: {v2}\n")
                lines.append("\n")
            else:
                lines.append(f"- **{k}**: {v}\n")
        lines.append("\n---\n\n")

    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"✓ MD:   {md_path}")

    # Zusammenfassung
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)

    # Test 2: EQ Bounce
    eq = results.get("eq_bounce", {})
    for s in SESSIONS:
        if s in eq:
            d = eq[s]
            print(
                f"\n  EQ-Bounce {s}: {d['bounce_pct']}% ({d['total_touches']} touches)"
            )

    # Test 3: Direction
    dr = results.get("direction", {})
    for s in SESSIONS:
        if s in dr:
            d = dr[s]
            if d.get("discount_long"):
                dl = d["discount_long"]
                for h in dl:
                    if h["horizon"] == 15:
                        print(
                            f"  {s} Discount Long @15: {h['positive_pct']}% positiv, Ø {h['mean_return']}pt"
                        )
            if d.get("premium_short"):
                ps = d["premium_short"]
                for h in ps:
                    if h["horizon"] == 15:
                        print(
                            f"  {s} Premium Short @15: {h['positive_pct']}% positiv, Ø {h['mean_return']}pt"
                        )

    # Test 5: Confluence
    cf = results.get("confluence", {})
    if "discount_long" in cf:
        print("\n  Confluence Discount Long:")
        for k, v in cf["discount_long"].items():
            print(
                f"    {k}: {v['positive_pct']}% positiv, Ø {v['mean_return']}pt (n={v['n']})"
            )

    # Test 6: Zone Filter
    zf = results.get("zone_filter", {})
    for algo in ["s_ob", "fvg"]:
        if algo in zf and isinstance(zf[algo], dict) and "error" not in zf[algo]:
            print(f"\n  {algo} × P/D Filter:")
            for s, d in zf[algo].items():
                if isinstance(d, dict) and "delta" in d:
                    print(
                        f"    {s}: Premium Short {d['premium_short_pct']}% | Discount Long {d['discount_long_pct']}% | Delta {d['delta']}%"
                    )


if __name__ == "__main__":
    main()
