#!/usr/bin/env python3
"""
Cross-Algo Research: 7 Tests auf verifizierte Bibliothek
NQ 1min | 717k Bars | 2024–2026

Tests:
  1. Entry-Typ-Vergleich (FT vs DP vs ST50 pro Zone)
  2. OB-Typ-Ranking (Chaos vs Tageshoch vs Session vs Session HTief)
  3. ATR-Multiplier-Sweep (SL-Optimierung)
  4. Regime × Exit (Entries in LRL vs HRL)
  5. Regime × Macro (Macro-Timing + Regime Kombination)
  6. Zone-Alter-Decay (FVG + OB Alterung)
  7. P/D × Regime (Premium/Discount + HRL/LRL)
"""

from __future__ import annotations

import sys as _sys

# sb/inspect.py schattet stdlib inspect — Script-Dir aus sys.path entfernen
if _sys.path and _sys.path[0].endswith("/sb"):
    _sys.path.pop(0)

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=130)

# ─── Pfade ───────────────────────────────────────────────
BIB = Path(__file__).resolve().parent.parent / "david_bibliothek"
DATA_1M = Path(__file__).resolve().parent.parent / "data" / "nq_1m_databento_2024_2026.parquet"

ALGO_PATHS = {
    "fvg_std": BIB / "02_FVG_Zonen/3. FVG Standard.py",
    "ifvg_1w": BIB / "02_FVG_Zonen/4a. iFVG 1Woche.py",
    "ifvg_sd": BIB / "02_FVG_Zonen/4b. iFVG SameDay.py",
    "fvg_2t": BIB / "02_FVG_Zonen/5a. FVG 2Tage.py",
    "fvg_12w": BIB / "02_FVG_Zonen/5b. FVG 1-2Wochen.py",
    "ob_chaos": BIB / "03_Order_Blocks/7a. Chaos Order Block.py",
    "ob_tagh": BIB / "03_Order_Blocks/7b. Tageshoch-Tief Orderblock.py",
    "ob_sess": BIB / "03_Order_Blocks/7c. Session Orderblock.py",
    "ob_sess_ht": BIB / "03_Order_Blocks/7d. Session Hoch-Tief Orderblock.py",
    "entry_ft": BIB / "09_Entry_Logik/entry_first_touch.py",
    "entry_dp": BIB / "09_Entry_Logik/entry_displacement.py",
    "entry_st50": BIB / "09_Entry_Logik/entry_second_touch_50.py",
    "atr": BIB / "05_Stoploss_TakeProfit/1. ATR Standard.py",
    "macro_short": BIB / "06_Time_Zeit/6a. Macro Time Short.py",
    "hrl_lrl": BIB / "03_Context/HRL LRL Internet 15min Trend Filter.py",
    "pd": BIB / "03_Context/premium_discount.py",
}

ZONE_TYPES = [
    "fvg_std",
    "ifvg_1w",
    "ifvg_sd",
    "fvg_2t",
    "fvg_12w",
    "ob_chaos",
    "ob_tagh",
    "ob_sess",
    "s_ob_sess",
]
ENTRY_SUFFIXES = {"ft": "entry_ft", "dp": "entry_dp", "st50": "entry_st50"}


# ─── Hilfsfunktionen ────────────────────────────────────


def _load_mod(path: Path):
    mod_name = f"_algo_{path.stem.replace(' ', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Kann {path} nicht laden")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        _sys.modules.pop(mod_name, None)
    return mod


def run_algo(name: str, df: pd.DataFrame, cache: dict) -> pd.DataFrame | None:
    if name in cache:
        return cache[name]
    path = ALGO_PATHS[name]
    t0 = time.time()
    try:
        result = _load_mod(path).run(df)
        console.print(f"  [dim]✓ {name} ({time.time() - t0:.1f}s)[/dim]")
        cache[name] = result
        return result
    except Exception as e:
        console.print(f"  [red]✗ {name}: {e}[/red]")
        cache[name] = None
        return None


def run_hrl_on_15m(df_1m: pd.DataFrame, cache: dict) -> np.ndarray:
    """HRL/LRL auf 15min berechnen, auf 1min forward-fillen."""
    key = "_hrl_15m_regime"
    if key in cache:
        return cache[key]
    console.print("  [dim]Resampling 1m→15m für HRL/LRL...[/dim]")
    df_15m = (
        df_1m.resample("15min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    df_15m = df_15m.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    mod = _load_mod(ALGO_PATHS["hrl_lrl"])
    t0 = time.time()
    r15 = mod.run(df_15m)
    console.print(f"  [dim]✓ HRL/LRL 15min ({time.time() - t0:.1f}s)[/dim]")
    regime_15m = r15["stat_regime"]
    regime_1m = regime_15m.reindex(df_1m.index, method="ffill").fillna("NEUTRAL").values
    cache[key] = regime_1m
    return regime_1m


def fwd_walk(c, h, l, idx, direction, n=60):
    """Forward walk: return (mfe, mae, net) oder None."""
    if idx + n >= len(c):
        return None
    entry = c[idx]
    fh = h[idx + 1 : idx + n + 1]
    fl = l[idx + 1 : idx + n + 1]
    c_end = c[idx + n]
    if direction == "long":
        return (np.max(fh) - entry, entry - np.min(fl), c_end - entry)
    return (entry - np.min(fl), np.max(fh) - entry, entry - c_end)


def analyze_col(df, col, direction, c, h, l, sample=0.15, n_fwd=60):
    """Forward-Analyse für eine Boolean-Spalte."""
    if col not in df.columns:
        return None
    mask = df[col].fillna(False).astype(bool)
    total = int(mask.sum())
    if total < 5:
        return None
    np.random.seed(42)
    idxs = np.where(mask.values)[0]
    n_keep = max(20, int(len(idxs) * sample))
    if n_keep < len(idxs):
        idxs = np.random.choice(idxs, n_keep, replace=False)
    results = [
        r for i in idxs if (r := fwd_walk(c, h, l, i, direction, n_fwd)) is not None
    ]
    if len(results) < 5:
        return None
    mfe = np.array([r[0] for r in results])
    mae = np.array([r[1] for r in results])
    net = np.array([r[2] for r in results])
    med_mae = np.median(mae)
    return {
        "total": total,
        "sampled": len(results),
        "mfe": float(np.median(mfe)),
        "mae": float(med_mae),
        "net": float(np.median(net)),
        "wr_2to1": float(np.mean(mfe >= med_mae * 2)) if med_mae > 0 else 0.0,
        "ratio": float(np.median(mfe) / med_mae) if med_mae > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════
# TEST 1: Entry-Typ-Vergleich
# ═══════════════════════════════════════════════════════════


def test_1(df, cache, c, h, l):
    console.print(
        Panel(
            "[bold]Test 1: Entry-Typ-Vergleich (FT vs DP vs ST50)[/bold]",
            subtitle="Welcher Entry-Typ funktioniert am besten pro Zone?",
        )
    )
    console.print("[dim]Lade Entry-Algos (jeder lädt intern alle Zone-Algos)...[/dim]")
    entry_data = {}
    for suffix, algo_key in ENTRY_SUFFIXES.items():
        entry_data[suffix] = run_algo(algo_key, df, cache)

    table = Table(title="Entry-Typ × Zone → MFE/MAE Ratio + WR 2:1")
    table.add_column("Zone", style="white", min_width=12)
    for et in ["ft", "dp", "st50"]:
        table.add_column(f"{et.upper()} n", style="dim", justify="right")
        table.add_column(f"{et.upper()} Ratio", justify="right")
        table.add_column(f"{et.upper()} WR", justify="right")

    best = {}
    for zone in ZONE_TYPES:
        row = [zone]
        best_ratio = -1
        best_et = ""
        for et in ["ft", "dp", "st50"]:
            data = entry_data.get(et)
            if data is None:
                row += ["—", "—", "—"]
                continue
            bull = analyze_col(data, f"{zone}_bull_{et}", "long", c, h, l)
            bear = analyze_col(data, f"{zone}_bear_{et}", "short", c, h, l)
            if bull and bear:
                n = bull["total"] + bear["total"]
                ratio = (bull["ratio"] + bear["ratio"]) / 2
                wr = (bull["wr_2to1"] + bear["wr_2to1"]) / 2
            elif bull:
                n, ratio, wr = bull["total"], bull["ratio"], bull["wr_2to1"]
            elif bear:
                n, ratio, wr = bear["total"], bear["ratio"], bear["wr_2to1"]
            else:
                row += ["—", "—", "—"]
                continue
            style = "[green]" if ratio > 1.2 else "[red]" if ratio < 0.8 else ""
            end = "[/green]" if ratio > 1.2 else "[/red]" if ratio < 0.8 else ""
            row += [str(n), f"{style}{ratio:.2f}{end}", f"{wr:.0%}"]
            if ratio > best_ratio:
                best_ratio, best_et = ratio, et
        if best_et:
            best[zone] = (best_et, best_ratio)
        table.add_row(*row)

    console.print(table)
    console.print("\n[bold]Bester Entry pro Zone:[/bold]")
    for zone, (et, ratio) in sorted(best.items(), key=lambda x: -x[1][1]):
        console.print(f"  {zone}: [bold]{et.upper()}[/bold] (MFE/MAE {ratio:.2f})")
    return best


# ═══════════════════════════════════════════════════════════
# TEST 2: OB-Typ-Ranking
# ═══════════════════════════════════════════════════════════


def test_2(df, cache, c, h, l):
    console.print(
        Panel(
            "[bold]Test 2: OB-Typ-Ranking[/bold]",
            subtitle="Welcher Order Block hat den höchsten Bounce?",
        )
    )
    ob_map = {
        "ob_chaos": "7a Chaos",
        "ob_tagh": "7b Tageshoch",
        "ob_sess": "7c Session (Daily)",
        "ob_sess_ht": "7d Session HTief (Daily)",
    }
    table = Table(title="Order Block Typ-Vergleich")
    table.add_column("OB Typ", style="white", min_width=22)
    table.add_column("Bull", justify="right")
    table.add_column("Bear", justify="right")
    table.add_column("Mitigation%", justify="right")
    table.add_column("Ø Größe", justify="right")
    table.add_column("MFE/MAE", justify="right")
    table.add_column("WR 2:1", justify="right")

    results = {}
    for key, label in ob_map.items():
        data = run_algo(key, df, cache)
        if data is None:
            table.add_row(label, *["—"] * 6)
            continue
        bc = int(data["ob_bull"].sum()) if "ob_bull" in data.columns else 0
        brc = int(data["ob_bear"].sum()) if "ob_bear" in data.columns else 0
        bm = int(data.get("ob_bull_mitigated", pd.Series(dtype=float)).sum())
        brm = int(data.get("ob_bear_mitigated", pd.Series(dtype=float)).sum())
        total = bc + brc
        mit = (bm + brm) / total if total else 0
        sizes = []
        if "ob_bull_high" in data.columns:
            s = (
                data.loc[data["ob_bull"], "ob_bull_high"]
                - data.loc[data["ob_bull"], "ob_bull_low"]
            )
            sizes.extend(s.dropna().tolist())
        if "ob_bear_high" in data.columns:
            s = (
                data.loc[data["ob_bear"], "ob_bear_high"]
                - data.loc[data["ob_bear"], "ob_bear_low"]
            )
            sizes.extend(s.dropna().tolist())
        avg_size = np.median(sizes) if sizes else np.nan

        b_r = analyze_col(data, "ob_bull", "long", c, h, l, sample=0.15)
        br_r = analyze_col(data, "ob_bear", "short", c, h, l, sample=0.15)
        if b_r and br_r:
            ratio = (b_r["ratio"] + br_r["ratio"]) / 2
            wr = (b_r["wr_2to1"] + br_r["wr_2to1"]) / 2
        elif b_r:
            ratio, wr = b_r["ratio"], b_r["wr_2to1"]
        elif br_r:
            ratio, wr = br_r["ratio"], br_r["wr_2to1"]
        else:
            ratio, wr = np.nan, np.nan

        results[key] = {
            "total": total,
            "mit": mit,
            "size": avg_size,
            "ratio": ratio,
            "wr": wr,
        }
        table.add_row(
            label,
            str(bc),
            str(brc),
            f"{mit:.0%}",
            f"{avg_size:.1f}" if not np.isnan(avg_size) else "—",
            f"{ratio:.2f}" if not np.isnan(ratio) else "—",
            f"{wr:.0%}" if not np.isnan(wr) else "—",
        )
    console.print(table)
    ranked = sorted(
        results.items(), key=lambda x: x[1].get("ratio", 0) or 0, reverse=True
    )
    console.print("\n[bold]Ranking:[/bold]")
    for i, (k, r) in enumerate(ranked, 1):
        console.print(
            f"  {i}. {ob_map[k]}: MFE/MAE {r['ratio']:.2f}, {r['total']} Zonen, {r['mit']:.0%} mitigated"
        )
    return results


# ═══════════════════════════════════════════════════════════
# TEST 3: ATR-Multiplier-Sweep
# ═══════════════════════════════════════════════════════════


def test_3(df, cache, c, h, l):
    console.print(
        Panel(
            "[bold]Test 3: ATR-Multiplier-Sweep (SL-Optimierung)[/bold]",
            subtitle="Welcher ATR-Multiplikator gibt bestes RR?",
        )
    )
    atr_data = run_algo("atr", df, cache)
    ft_data = run_algo("entry_ft", df, cache)
    if atr_data is None or ft_data is None:
        console.print("[red]Daten fehlen![/red]")
        return {}

    atr_arr = atr_data["atr"].values
    bull_m = (
        ft_data.get("fvg_std_bull_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )
    bear_m = (
        ft_data.get("fvg_std_bear_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )

    np.random.seed(42)
    bull_i = np.where(bull_m)[0]
    bear_i = np.where(bear_m)[0]
    sr = 0.2
    if len(bull_i) > 80:
        bull_i = np.random.choice(bull_i, max(80, int(len(bull_i) * sr)), replace=False)
    if len(bear_i) > 80:
        bear_i = np.random.choice(bear_i, max(80, int(len(bear_i) * sr)), replace=False)

    mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    n = len(c)
    max_bars = 120

    table = Table(title="ATR-Multiplier × Ergebnis (FVG_STD_FT, RR 2:1)")
    table.add_column("Mult", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Ø Win", justify="right")
    table.add_column("Ø Loss", justify="right")
    table.add_column("PF", justify="right")

    results = {}
    for mult in mults:
        w_pts, l_pts = [], []
        for idx in np.concatenate([bull_i, bear_i]):
            if idx + max_bars >= n or np.isnan(atr_arr[idx]) or atr_arr[idx] <= 0:
                continue
            entry = c[idx]
            sl = atr_arr[idx] * mult
            tp = sl * 2
            d = "long" if bull_m[idx] else "short"
            for j in range(1, max_bars + 1):
                if idx + j >= n:
                    break
                hj, lj = h[idx + j], l[idx + j]
                if d == "long":
                    if lj <= entry - sl:
                        l_pts.append(sl)
                        break
                    if hj >= entry + tp:
                        w_pts.append(tp)
                        break
                else:
                    if hj >= entry + sl:
                        l_pts.append(sl)
                        break
                    if lj <= entry - tp:
                        w_pts.append(tp)
                        break
        total = len(w_pts) + len(l_pts)
        wr = len(w_pts) / total if total else 0
        pf = sum(w_pts) / sum(l_pts) if l_pts else 0
        aw = np.mean(w_pts) if w_pts else 0
        al = np.mean(l_pts) if l_pts else 0
        results[mult] = {"trades": total, "wr": wr, "pf": pf}
        style_pf = "[green]" if pf > 1.5 else "[red]" if pf < 1.0 else ""
        end_pf = "[/green]" if pf > 1.5 else "[/red]" if pf < 1.0 else ""
        table.add_row(
            f"{mult:.2f}x",
            str(total),
            f"{wr:.1%}",
            f"{aw:.1f}",
            f"{al:.1f}",
            f"{style_pf}{pf:.2f}{end_pf}",
        )
    console.print(table)
    best = max(results.items(), key=lambda x: x[1]["pf"])
    console.print(
        f"\n[bold]Bester Multiplier: {best[0]:.2f}x (PF {best[1]['pf']:.2f}, WR {best[1]['wr']:.0%})[/bold]"
    )
    return results


# ═══════════════════════════════════════════════════════════
# TEST 4: Regime × Exit
# ═══════════════════════════════════════════════════════════


def test_4(df, cache, c, h, l):
    console.print(
        Panel(
            "[bold]Test 4: Regime × Exit (LRL vs HRL vs NEUTRAL)[/bold]",
            subtitle="Haben Trades in LRL bessere Forward-Qualität?",
        )
    )
    regime = run_hrl_on_15m(df, cache)
    ft_data = run_algo("entry_ft", df, cache)
    if ft_data is None:
        console.print("[red]Entry-Daten fehlen![/red]")
        return {}

    bull_m = (
        ft_data.get("fvg_std_bull_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )
    bear_m = (
        ft_data.get("fvg_std_bear_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )

    table = Table(title="Regime → Forward 60 Bars nach FVG_STD_FT Entry")
    for col in ["Regime", "Entries", "Ø MFE", "Ø MAE", "MFE/MAE", "WR 2:1", "Ø Net"]:
        table.add_column(col, justify="right")

    results = {}
    for rn in ["LRL", "HRL", "NEUTRAL"]:
        rm = regime == rn
        bi = np.where(bull_m & rm)[0]
        bri = np.where(bear_m & rm)[0]
        np.random.seed(42)
        s = 0.2
        if len(bi) > 40:
            bi = np.random.choice(bi, max(40, int(len(bi) * s)), replace=False)
        if len(bri) > 40:
            bri = np.random.choice(bri, max(40, int(len(bri) * s)), replace=False)
        mfes, maes, nets = [], [], []
        for i in bi:
            r = fwd_walk(c, h, l, i, "long")
            if r:
                mfes.append(r[0])
                maes.append(r[1])
                nets.append(r[2])
        for i in bri:
            r = fwd_walk(c, h, l, i, "short")
            if r:
                mfes.append(r[0])
                maes.append(r[1])
                nets.append(r[2])
        nn = len(mfes)
        if nn < 5:
            table.add_row(rn, str(nn), *["—"] * 5)
            continue
        mfe = np.median(mfes)
        mae = np.median(maes)
        ratio = mfe / mae if mae > 0 else 0
        wr = np.mean(np.array(mfes) >= np.median(maes) * 2) if mae > 0 else 0
        net = np.median(nets)
        results[rn] = {
            "n": nn,
            "mfe": mfe,
            "mae": mae,
            "ratio": ratio,
            "wr": wr,
            "net": net,
        }
        table.add_row(
            rn,
            str(nn),
            f"{mfe:.1f}",
            f"{mae:.1f}",
            f"{ratio:.2f}",
            f"{wr:.0%}",
            f"{net:+.1f}",
        )
    console.print(table)

    if "LRL" in results and "HRL" in results:
        lr, hr = results["LRL"], results["HRL"]
        console.print("\n[bold]Delta LRL vs HRL:[/bold]")
        console.print(
            f"  MFE/MAE: {lr['ratio']:.2f} vs {hr['ratio']:.2f} (Δ {lr['ratio'] - hr['ratio']:+.2f})"
        )
        console.print(
            f"  MAE: {lr['mae']:.1f} vs {hr['mae']:.1f} (weniger MAE = weniger DD)"
        )
        if lr["ratio"] > hr["ratio"]:
            console.print(
                "  [green]→ LRL-Entries haben bessere Forward-Qualität[/green]"
            )
        else:
            console.print("  [red]→ HRL-Entries gleich oder besser (unerwartet!)[/red]")
    return results


# ═══════════════════════════════════════════════════════════
# TEST 5: Regime × Macro
# ═══════════════════════════════════════════════════════════


def test_5(df, cache, c, h, l):
    console.print(
        Panel(
            "[bold]Test 5: Regime × Macro (Verstärkung?)[/bold]",
            subtitle="Verstärkt Macro-Timing den Regime-Filter?",
        )
    )
    regime = run_hrl_on_15m(df, cache)
    macro_data = run_algo("macro_short", df, cache)
    ft_data = run_algo("entry_ft", df, cache)
    if macro_data is None or ft_data is None:
        console.print("[red]Daten fehlen![/red]")
        return {}

    in_macro = macro_data["in_macro_short"].values
    bull_m = (
        ft_data.get("fvg_std_bull_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )
    bear_m = (
        ft_data.get("fvg_std_bear_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )
    entry = bull_m | bear_m

    table = Table(title="Regime × Macro → Forward MFE/MAE")
    for col in ["Kombination", "Entries", "Ø MFE", "Ø MAE", "MFE/MAE", "WR 2:1"]:
        table.add_column(col, justify="right")

    combos = [
        ("LRL + Macro", (regime == "LRL") & in_macro & entry),
        ("LRL + NoMacro", (regime == "LRL") & ~in_macro & entry),
        ("HRL + Macro", (regime == "HRL") & in_macro & entry),
        ("HRL + NoMacro", (regime == "HRL") & ~in_macro & entry),
        ("NEUTRAL + Macro", (regime == "NEUTRAL") & in_macro & entry),
        ("NEUTRAL + NoMacro", (regime == "NEUTRAL") & ~in_macro & entry),
    ]
    results = {}
    for label, mask in combos:
        idxs = np.where(mask)[0]
        np.random.seed(42)
        if len(idxs) > 60:
            idxs = np.random.choice(idxs, max(60, int(len(idxs) * 0.15)), replace=False)
        mfes, maes = [], []
        for i in idxs:
            d = "long" if bull_m[i] else "short"
            r = fwd_walk(c, h, l, i, d)
            if r:
                mfes.append(r[0])
                maes.append(r[1])
        nn = len(mfes)
        if nn < 5:
            table.add_row(label, str(nn), *["—"] * 4)
            continue
        mfe = np.median(mfes)
        mae = np.median(maes)
        ratio = mfe / mae if mae > 0 else 0
        wr = np.mean(np.array(mfes) >= np.median(maes) * 2) if mae > 0 else 0
        results[label] = {"n": nn, "mfe": mfe, "mae": mae, "ratio": ratio, "wr": wr}
        table.add_row(
            label, str(nn), f"{mfe:.1f}", f"{mae:.1f}", f"{ratio:.2f}", f"{wr:.0%}"
        )
    console.print(table)

    lm = results.get("LRL + Macro", {})
    ln = results.get("LRL + NoMacro", {})
    if lm.get("ratio") and ln.get("ratio"):
        d = lm["ratio"] - ln["ratio"]
        console.print(
            f"\n[bold]Macro-Effekt in LRL: {lm['ratio']:.2f} vs {ln['ratio']:.2f} (Δ {d:+.2f})[/bold]"
        )
        if d > 0.1:
            console.print("  [green]→ Macro verstärkt LRL![/green]")
        elif d < -0.1:
            console.print("  [yellow]→ Macro schadet in LRL — besser ohne![/yellow]")
        else:
            console.print("  [dim]→ Kein signifikanter Effekt[/dim]")
    return results


# ═══════════════════════════════════════════════════════════
# TEST 6: Zone-Alter-Decay
# ═══════════════════════════════════════════════════════════


def test_6(df, cache, c, h, l):
    console.print(
        Panel(
            "[bold]Test 6: Zone-Alter-Decay (FVG + OB)[/bold]",
            subtitle="Wann verlieren Zonen ihren Edge?",
        )
    )
    fvg_data = run_algo("fvg_std", df, cache)
    ob_data = run_algo("ob_chaos", df, cache)

    buckets = [
        ("0-60 (≤1h)", 0, 60),
        ("60-240 (1-4h)", 60, 240),
        ("240-780 (4h-1d)", 240, 780),
        ("780-1560 (1-2d)", 780, 1560),
        ("1560-3900 (2-5d)", 1560, 3900),
        ("3900+ (5d+)", 3900, 999999),
    ]

    table = Table(title="Zone-Alter → Bounce-Qualität (MFE/MAE nach Bull-Touch)")
    table.add_column("Alter", style="white", min_width=18)
    table.add_column("FVG Touches", justify="right")
    table.add_column("FVG Ratio", justify="right")
    table.add_column("OB Touches", justify="right")
    table.add_column("OB Ratio", justify="right")

    results = {}

    for zone_label, data, creation_col, zhigh_col in [
        ("FVG", fvg_data, "fvg_bull", "fvg_bull_high"),
        ("OB", ob_data, "ob_bull", "ob_bull_high"),
    ]:
        if data is None:
            continue
        creation = (
            data[creation_col].values
            if creation_col in data.columns
            else np.zeros(len(data), bool)
        )
        zh = (
            data[zhigh_col].values
            if zhigh_col in data.columns
            else np.full(len(data), np.nan)
        )

        # Berechne Alter (Bars seit letzter Creation)
        age = np.zeros(len(data), dtype=int)
        last = -1
        for i in range(len(data)):
            if creation[i]:
                last = i
            if last >= 0:
                age[i] = i - last

        for blabel, amin, amax in buckets:
            touch_mask = (age >= amin) & (age < amax) & ~np.isnan(zh) & (l <= zh)
            idxs = np.where(touch_mask)[0]
            np.random.seed(42)
            if len(idxs) > 100:
                idxs = np.random.choice(idxs, 100, replace=False)
            mfes, maes = [], []
            for i in idxs:
                r = fwd_walk(c, h, l, i, "long")
                if r:
                    mfes.append(r[0])
                    maes.append(r[1])
            nn = len(mfes)
            ratio = (
                np.median(mfes) / np.median(maes)
                if nn >= 5 and np.median(maes) > 0
                else np.nan
            )
            results.setdefault(blabel, {})[zone_label] = {"n": nn, "ratio": ratio}

    for blabel, _, _ in buckets:
        r = results.get(blabel, {})
        fvg = r.get("FVG", {"n": 0, "ratio": np.nan})
        ob = r.get("OB", {"n": 0, "ratio": np.nan})
        table.add_row(
            blabel,
            str(fvg["n"]),
            f"{fvg['ratio']:.2f}" if not np.isnan(fvg["ratio"]) else "—",
            str(ob["n"]),
            f"{ob['ratio']:.2f}" if not np.isnan(ob["ratio"]) else "—",
        )
    console.print(table)

    console.print("\n[bold]Erkenntnisse:[/bold]")
    for zt in ["FVG", "OB"]:
        young = results.get("0-60 (≤1h)", {}).get(zt, {}).get("ratio", np.nan)
        old = results.get("1560-3900 (2-5d)", {}).get(zt, {}).get("ratio", np.nan)
        if not np.isnan(young) and not np.isnan(old):
            if old > young * 1.1:
                console.print(
                    f"  [green]{zt}: Alte Zonen STÄRKER ({old:.2f} vs {young:.2f})[/green]"
                )
            elif young > old * 1.1:
                console.print(
                    f"  [yellow]{zt}: Junge Zonen besser ({young:.2f} vs {old:.2f}) — schnell handeln![/yellow]"
                )
            else:
                console.print(
                    f"  [dim]{zt}: Kein klarer Decay ({young:.2f} → {old:.2f})[/dim]"
                )
    return results


# ═══════════════════════════════════════════════════════════
# TEST 7: P/D × Regime
# ═══════════════════════════════════════════════════════════


def test_7(df, cache, c, h, l):
    console.print(
        Panel(
            "[bold]Test 7: Premium/Discount × Regime[/bold]",
            subtitle="Verstärkt Regime den P/D-Edge?",
        )
    )
    pd_data = run_algo("pd", df, cache)
    regime = run_hrl_on_15m(df, cache)
    ft_data = run_algo("entry_ft", df, cache)
    if pd_data is None or ft_data is None:
        console.print("[red]Daten fehlen![/red]")
        return {}

    in_prem = pd_data["pd_in_premium"].values
    in_disc = pd_data["pd_in_discount"].values
    bull_m = (
        ft_data.get("fvg_std_bull_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )
    bear_m = (
        ft_data.get("fvg_std_bear_ft", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
        .values
    )

    table = Table(title="P/D × Regime → Entry-Qualität (Long MFE/MAE)")
    for col in ["Kombination", "Long", "Short", "L Ratio", "S Ratio", "WR 2:1"]:
        table.add_column(col, justify="right")

    combos = [
        ("Discount + LRL", in_disc & (regime == "LRL")),
        ("Discount + HRL", in_disc & (regime == "HRL")),
        ("Discount + NEUTRAL", in_disc & (regime == "NEUTRAL")),
        ("Premium + LRL", in_prem & (regime == "LRL")),
        ("Premium + HRL", in_prem & (regime == "HRL")),
        ("Premium + NEUTRAL", in_prem & (regime == "NEUTRAL")),
        ("Discount (gesamt)", in_disc),
        ("Premium (gesamt)", in_prem),
    ]
    results = {}
    for label, zmask in combos:
        li = np.where(bull_m & zmask)[0]
        si = np.where(bear_m & zmask)[0]
        np.random.seed(42)
        if len(li) > 60:
            li = np.random.choice(li, max(60, int(len(li) * 0.2)), replace=False)
        if len(si) > 60:
            si = np.random.choice(si, max(60, int(len(si) * 0.2)), replace=False)
        lm, la, sm, sa = [], [], [], []
        for i in li:
            r = fwd_walk(c, h, l, i, "long")
            if r:
                lm.append(r[0])
                la.append(r[1])
        for i in si:
            r = fwd_walk(c, h, l, i, "short")
            if r:
                sm.append(r[0])
                sa.append(r[1])
        lr = (
            np.median(lm) / np.median(la)
            if len(la) >= 5 and np.median(la) > 0
            else np.nan
        )
        sr = (
            np.median(sm) / np.median(sa)
            if len(sa) >= 5 and np.median(sa) > 0
            else np.nan
        )
        all_mfe = lm + sm
        all_mae = la + sa
        wr = (
            np.mean(np.array(all_mfe) >= np.median(all_mae) * 2)
            if all_mae and np.median(all_mae) > 0
            else np.nan
        )
        results[label] = {"ln": len(lm), "sn": len(sm), "lr": lr, "sr": sr, "wr": wr}
        table.add_row(
            label,
            str(len(lm)),
            str(len(sm)),
            f"{lr:.2f}" if not np.isnan(lr) else "—",
            f"{sr:.2f}" if not np.isnan(sr) else "—",
            f"{wr:.0%}" if not np.isnan(wr) else "—",
        )
    console.print(table)

    dl = results.get("Discount + LRL", {})
    dh = results.get("Discount + HRL", {})
    if (
        dl.get("lr")
        and dh.get("lr")
        and not np.isnan(dl["lr"])
        and not np.isnan(dh["lr"])
    ):
        delta = dl["lr"] - dh["lr"]
        console.print("\n[bold]Discount Long — LRL vs HRL:[/bold]")
        console.print(f"  LRL: {dl['lr']:.2f} vs HRL: {dh['lr']:.2f} (Δ {delta:+.2f})")
        if delta > 0.15:
            console.print(
                "  [green]→ Regime VERSTÄRKT P/D-Edge! Discount+LRL = stärkste Kombi[/green]"
            )
        else:
            console.print("  [dim]→ Kein starker Verstärkungs-Effekt[/dim]")
    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════


def main():
    console.print(
        Panel(
            "[bold]Cross-Algo Research: 7 Tests auf verifizierte Bibliothek[/bold]\n"
            "NQ 1min | 717k Bars | 2024–2026\n"
            "Sampling 15-20% für Forward-Walk | 60 Bars Forward",
            style="blue",
        )
    )
    t0 = time.time()

    df = pd.read_parquet(DATA_1M)
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    console.print(f"[dim]{len(df):,} Bars geladen[/dim]")

    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    cache: dict = {}

    sep = "\n" + "═" * 70

    console.print(sep)
    test_1(df, cache, c, h, l)

    console.print(sep)
    test_2(df, cache, c, h, l)

    console.print(sep)
    test_3(df, cache, c, h, l)

    console.print(sep)
    test_4(df, cache, c, h, l)

    console.print(sep)
    test_5(df, cache, c, h, l)

    console.print(sep)
    test_6(df, cache, c, h, l)

    console.print(sep)
    test_7(df, cache, c, h, l)

    dt = time.time() - t0
    console.print(
        Panel(
            f"[bold]Alle 7 Tests abgeschlossen in {dt / 60:.1f} Minuten[/bold]",
            style="green",
        )
    )


if __name__ == "__main__":
    main()
