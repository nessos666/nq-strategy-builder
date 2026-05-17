#!/usr/bin/env python3
"""
Test 7 Vertiefung: P/D × Regime — Tiefenanalyse

Fragen:
  1. Gilt P/D+HRL-Vorteil auch für andere Entry-Typen (DP, ST50)?
  2. Gilt es für alle Zone-Typen (FVG + OB)?
  3. Session-Effekt: Welche Session profitiert am meisten?
  4. Richtung: Discount Long vs Short, Premium Long vs Short
  5. Forward-Perioden: 30/60/120 Bars — ist der Effekt kurzfristig oder anhaltend?
  6. Zone-Position: OB am Zone-Rand vs Mitte (P/D Chat 406 Fund)
"""

from __future__ import annotations

import sys as _sys

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

BIB = Path(__file__).resolve().parent.parent / "david_bibliothek"
DATA_1M = Path(
    "data/nq_1m_databento_2024_2026.parquet"
)

ALGO_PATHS = {
    "entry_ft": BIB / "09_Entry_Logik/entry_first_touch.py",
    "entry_dp": BIB / "09_Entry_Logik/entry_displacement.py",
    "entry_st50": BIB / "09_Entry_Logik/entry_second_touch_50.py",
    "hrl_lrl": BIB / "03_Context/HRL LRL Internet 15min Trend Filter.py",
    "pd": BIB / "03_Context/premium_discount.py",
    "macro_short": BIB / "06_Time_Zeit/6a. Macro Time Short.py",
}


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


def run_algo(name, df, cache):
    if name in cache:
        return cache[name]
    t0 = time.time()
    try:
        result = _load_mod(ALGO_PATHS[name]).run(df)
        console.print(f"  [dim]✓ {name} ({time.time() - t0:.1f}s)[/dim]")
        cache[name] = result
        return result
    except Exception as e:
        console.print(f"  [red]✗ {name}: {e}[/red]")
        cache[name] = None
        return None


def run_hrl_15m(df_1m, cache):
    key = "_hrl_15m"
    if key in cache:
        return cache[key]
    df_15m = (
        df_1m.resample("15min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    mod = _load_mod(ALGO_PATHS["hrl_lrl"])
    t0 = time.time()
    r15 = mod.run(df_15m)
    console.print(f"  [dim]✓ HRL/LRL 15min ({time.time() - t0:.1f}s)[/dim]")
    regime = (
        r15["stat_regime"].reindex(df_1m.index, method="ffill").fillna("NEUTRAL").values
    )
    cache[key] = regime
    return regime


def fwd_walk(c, h, l, idx, direction, n=60):
    if idx + n >= len(c):
        return None
    entry = c[idx]
    fh = h[idx + 1 : idx + n + 1]
    fl = l[idx + 1 : idx + n + 1]
    c_end = c[idx + n]
    if direction == "long":
        return (np.max(fh) - entry, entry - np.min(fl), c_end - entry)
    return (entry - np.min(fl), np.max(fh) - entry, entry - c_end)


def compute_stats(mfes, maes, nets):
    """Compute stats from lists of MFE/MAE/net."""
    if len(mfes) < 5:
        return {
            "n": len(mfes),
            "mfe": np.nan,
            "mae": np.nan,
            "ratio": np.nan,
            "wr": np.nan,
            "net": np.nan,
        }
    mfe = np.median(mfes)
    mae = np.median(maes)
    ratio = mfe / mae if mae > 0 else 0
    wr = np.mean(np.array(mfes) >= np.median(maes) * 2) if mae > 0 else 0
    net = np.median(nets)
    return {
        "n": len(mfes),
        "mfe": mfe,
        "mae": mae,
        "ratio": ratio,
        "wr": wr,
        "net": net,
    }


def analyze_group(idxs, directions, c, h, l, n_fwd=60, max_sample=200):
    """Forward-walk für eine Gruppe von Entry-Indizes."""
    np.random.seed(42)
    if len(idxs) > max_sample:
        choice = np.random.choice(len(idxs), max_sample, replace=False)
        idxs = [idxs[i] for i in choice]
        directions = [directions[i] for i in choice]
    mfes, maes, nets = [], [], []
    for idx, d in zip(idxs, directions):
        r = fwd_walk(c, h, l, idx, d, n_fwd)
        if r:
            mfes.append(r[0])
            maes.append(r[1])
            nets.append(r[2])
    return compute_stats(mfes, maes, nets)


def get_ny_session(idx_ny):
    """Bestimme NY-Session aus Minutenzahl."""
    mins = idx_ny.hour * 60 + idx_ny.minute
    # Asia 18:00-02:00 (1080-1440 + 0-120)
    if mins >= 1080 or mins < 120:
        return "Asia"
    # London 02:00-08:00 (120-480)
    elif mins < 480:
        return "London"
    # NY AM 08:00-12:00 (480-720)
    elif mins < 720:
        return "NY_AM"
    # NY PM 12:00-16:00 (720-960)
    elif mins < 960:
        return "NY_PM"
    else:
        return "Post"


def main():
    console.print(
        Panel(
            "[bold]Test 7 Vertiefung: P/D × Regime — Tiefenanalyse[/bold]\n"
            "NQ 1min | 717k Bars | Sampling 200 pro Gruppe",
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
    cache = {}

    # Algos laden
    console.print("\n[bold]Algos laden...[/bold]")
    ft_data = run_algo("entry_ft", df, cache)
    dp_data = run_algo("entry_dp", df, cache)
    st50_data = run_algo("entry_st50", df, cache)
    pd_data = run_algo("pd", df, cache)
    regime = run_hrl_15m(df, cache)

    if pd_data is None or ft_data is None:
        console.print("[red]Daten fehlen![/red]")
        return

    in_prem = pd_data["pd_in_premium"].values
    in_disc = pd_data["pd_in_discount"].values

    # NY-Session bestimmen
    try:
        idx_ny = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    except Exception:
        idx_ny = pd.DatetimeIndex(df.index)
    sessions = np.array([get_ny_session(t) for t in idx_ny])

    # ═══════════════════════════════════════════════════════════
    # TEIL 1: P/D × Regime für ALLE Entry-Typen
    # ═══════════════════════════════════════════════════════════
    console.print(Panel("[bold]Teil 1: P/D × Regime für alle Entry-Typen[/bold]"))

    entry_configs = {
        "FVG_STD_FT": (ft_data, "fvg_std_bull_ft", "fvg_std_bear_ft"),
        "FVG_STD_DP": (dp_data, "fvg_std_bull_dp", "fvg_std_bear_dp"),
        "FVG_STD_ST50": (st50_data, "fvg_std_bull_st50", "fvg_std_bear_st50"),
        "OB_CHAOS_FT": (ft_data, "ob_chaos_bull_ft", "ob_chaos_bear_ft"),
        "OB_SESS_ST50": (st50_data, "ob_sess_bull_st50", "ob_sess_bear_st50"),
        "S_OB_SESS_FT": (ft_data, "s_ob_sess_bull_ft", "s_ob_sess_bear_ft"),
    }

    table = Table(title="P/D × Regime für verschiedene Entry-Typen (MFE/MAE)")
    table.add_column("Entry-Typ", style="white", min_width=16)
    table.add_column("Disc+LRL", justify="right")
    table.add_column("Disc+HRL", justify="right")
    table.add_column("Prem+LRL", justify="right")
    table.add_column("Prem+HRL", justify="right")
    table.add_column("Δ Disc(HRL-LRL)", justify="right")

    for entry_name, (edata, bull_col, bear_col) in entry_configs.items():
        if edata is None or bull_col not in edata.columns:
            table.add_row(entry_name, *["—"] * 5)
            continue

        bull_m = edata[bull_col].fillna(False).astype(bool).values
        bear_m = (
            edata[bear_col].fillna(False).astype(bool).values
            if bear_col in edata.columns
            else np.zeros(len(df), bool)
        )

        combos = {}
        for label, zmask in [
            ("Disc+LRL", in_disc & (regime == "LRL")),
            ("Disc+HRL", in_disc & (regime == "HRL")),
            ("Prem+LRL", in_prem & (regime == "LRL")),
            ("Prem+HRL", in_prem & (regime == "HRL")),
        ]:
            idxs = []
            dirs = []
            for idx in np.where(bull_m & zmask)[0]:
                idxs.append(idx)
                dirs.append("long")
            for idx in np.where(bear_m & zmask)[0]:
                idxs.append(idx)
                dirs.append("short")
            combos[label] = analyze_group(idxs, dirs, c, h, l)

        dl = combos.get("Disc+LRL", {}).get("ratio", np.nan)
        dh = combos.get("Disc+HRL", {}).get("ratio", np.nan)
        delta = dh - dl if not (np.isnan(dl) or np.isnan(dh)) else np.nan

        row = [entry_name]
        for k in ["Disc+LRL", "Disc+HRL", "Prem+LRL", "Prem+HRL"]:
            r = combos.get(k, {}).get("ratio", np.nan)
            row.append(f"{r:.2f}" if not np.isnan(r) else "—")

        style = "[green]" if not np.isnan(delta) and delta > 0.1 else ""
        end = "[/green]" if style else ""
        row.append(f"{style}{delta:+.2f}{end}" if not np.isnan(delta) else "—")
        table.add_row(*row)

    console.print(table)

    # ═══════════════════════════════════════════════════════════
    # TEIL 2: Richtungsanalyse (Long vs Short separat)
    # ═══════════════════════════════════════════════════════════
    console.print(
        Panel("[bold]Teil 2: Richtungsanalyse (Long vs Short separat)[/bold]")
    )

    bull_m = ft_data["fvg_std_bull_ft"].fillna(False).astype(bool).values
    bear_m = ft_data["fvg_std_bear_ft"].fillna(False).astype(bool).values

    table = Table(title="P/D × Regime × Richtung → MFE/MAE")
    for col in ["Kombination", "n", "MFE", "MAE", "MFE/MAE", "WR 2:1", "Net"]:
        table.add_column(col, justify="right")

    for label, entry_mask, direction, zmask in [
        ("Disc+LRL Long", bull_m, "long", in_disc & (regime == "LRL")),
        ("Disc+LRL Short", bear_m, "short", in_disc & (regime == "LRL")),
        ("Disc+HRL Long", bull_m, "long", in_disc & (regime == "HRL")),
        ("Disc+HRL Short", bear_m, "short", in_disc & (regime == "HRL")),
        ("Prem+LRL Long", bull_m, "long", in_prem & (regime == "LRL")),
        ("Prem+LRL Short", bear_m, "short", in_prem & (regime == "LRL")),
        ("Prem+HRL Long", bull_m, "long", in_prem & (regime == "HRL")),
        ("Prem+HRL Short", bear_m, "short", in_prem & (regime == "HRL")),
    ]:
        idxs = np.where(entry_mask & zmask)[0].tolist()
        dirs = [direction] * len(idxs)
        s = analyze_group(idxs, dirs, c, h, l)
        table.add_row(
            label,
            str(s["n"]),
            f"{s['mfe']:.1f}" if not np.isnan(s["mfe"]) else "—",
            f"{s['mae']:.1f}" if not np.isnan(s["mae"]) else "—",
            f"{s['ratio']:.2f}" if not np.isnan(s["ratio"]) else "—",
            f"{s['wr']:.0%}" if not np.isnan(s["wr"]) else "—",
            f"{s['net']:+.1f}" if not np.isnan(s["net"]) else "—",
        )
    console.print(table)

    # ═══════════════════════════════════════════════════════════
    # TEIL 3: Session-Effekt (P/D × Regime × Session)
    # ═══════════════════════════════════════════════════════════
    console.print(Panel("[bold]Teil 3: Session-Effekt[/bold]"))

    entry_all = bull_m | bear_m
    table = Table(title="P/D × Regime × Session → MFE/MAE")
    table.add_column("Session", style="white")
    table.add_column("Disc+LRL", justify="right")
    table.add_column("Disc+HRL", justify="right")
    table.add_column("Prem+LRL", justify="right")
    table.add_column("Prem+HRL", justify="right")

    for sess in ["Asia", "London", "NY_AM", "NY_PM"]:
        sess_mask = sessions == sess
        row = [sess]
        for label, zmask in [
            ("dl", in_disc & (regime == "LRL") & sess_mask & entry_all),
            ("dh", in_disc & (regime == "HRL") & sess_mask & entry_all),
            ("pl", in_prem & (regime == "LRL") & sess_mask & entry_all),
            ("ph", in_prem & (regime == "HRL") & sess_mask & entry_all),
        ]:
            idxs = np.where(zmask)[0].tolist()
            dirs = ["long" if bull_m[i] else "short" for i in idxs]
            s = analyze_group(idxs, dirs, c, h, l)
            r = s["ratio"]
            style = (
                "[green]"
                if not np.isnan(r) and r > 1.1
                else "[red]"
                if not np.isnan(r) and r < 0.85
                else ""
            )
            end = (
                "[/green]"
                if "[green]" in style
                else "[/red]"
                if "[red]" in style
                else ""
            )
            row.append(f"{style}{r:.2f}{end} (n={s['n']})" if not np.isnan(r) else "—")
        table.add_row(*row)
    console.print(table)

    # ═══════════════════════════════════════════════════════════
    # TEIL 4: Forward-Perioden (30/60/120 Bars)
    # ═══════════════════════════════════════════════════════════
    console.print(Panel("[bold]Teil 4: Forward-Perioden (30/60/120 Bars)[/bold]"))

    table = Table(title="P/D × Regime × Forward-Periode → MFE/MAE")
    table.add_column("Kombination", style="white", min_width=16)
    table.add_column("30 Bars", justify="right")
    table.add_column("60 Bars", justify="right")
    table.add_column("120 Bars", justify="right")

    for label, zmask in [
        ("Disc+LRL", in_disc & (regime == "LRL") & entry_all),
        ("Disc+HRL", in_disc & (regime == "HRL") & entry_all),
        ("Prem+LRL", in_prem & (regime == "LRL") & entry_all),
        ("Prem+HRL", in_prem & (regime == "HRL") & entry_all),
    ]:
        idxs = np.where(zmask)[0].tolist()
        dirs = ["long" if bull_m[i] else "short" for i in idxs]
        row = [label]
        for n_fwd in [30, 60, 120]:
            s = analyze_group(idxs, dirs, c, h, l, n_fwd=n_fwd)
            r = s["ratio"]
            row.append(f"{r:.2f}" if not np.isnan(r) else "—")
        table.add_row(*row)
    console.print(table)

    # ═══════════════════════════════════════════════════════════
    # TEIL 5: Zone-Position (nahe Rand vs nahe EQ)
    # ═══════════════════════════════════════════════════════════
    console.print(Panel("[bold]Teil 5: P/D Zone-Position (Rand vs Mitte)[/bold]"))

    pd_nearest_high = pd_data["pd_nearest_high"].values
    pd_nearest_low = pd_data["pd_nearest_low"].values
    pd_nearest_eq = pd_data["pd_nearest_eq"].values

    # Berechne Position: Abstand zu EQ vs Abstand zu Rand
    zone_span = pd_nearest_high - pd_nearest_low
    dist_to_eq = np.abs(c - pd_nearest_eq)
    # Position 0 = am Rand, 1 = an EQ
    with np.errstate(divide="ignore", invalid="ignore"):
        zone_pos = np.where(zone_span > 0, dist_to_eq / (zone_span / 2), np.nan)
    # Rand = zone_pos < 0.3, Mitte = zone_pos > 0.7
    near_rand = zone_pos < 0.3
    near_eq = zone_pos > 0.7

    table = Table(title="P/D × Regime × Zone-Position → MFE/MAE")
    table.add_column("Kombination", style="white", min_width=18)
    table.add_column("Rand (<30%)", justify="right")
    table.add_column("Mitte (>70%)", justify="right")
    table.add_column("Δ (Rand-Mitte)", justify="right")

    for label, zmask in [
        ("Disc+LRL", in_disc & (regime == "LRL") & entry_all),
        ("Disc+HRL", in_disc & (regime == "HRL") & entry_all),
        ("Prem+LRL", in_prem & (regime == "LRL") & entry_all),
        ("Prem+HRL", in_prem & (regime == "HRL") & entry_all),
    ]:
        rand_idxs = np.where(zmask & near_rand)[0].tolist()
        eq_idxs = np.where(zmask & near_eq)[0].tolist()
        rand_dirs = ["long" if bull_m[i] else "short" for i in rand_idxs]
        eq_dirs = ["long" if bull_m[i] else "short" for i in eq_idxs]

        rand_s = analyze_group(rand_idxs, rand_dirs, c, h, l)
        eq_s = analyze_group(eq_idxs, eq_dirs, c, h, l)

        rr = rand_s["ratio"]
        er = eq_s["ratio"]
        delta = rr - er if not (np.isnan(rr) or np.isnan(er)) else np.nan

        table.add_row(
            label,
            f"{rr:.2f} (n={rand_s['n']})" if not np.isnan(rr) else "—",
            f"{er:.2f} (n={eq_s['n']})" if not np.isnan(er) else "—",
            f"{delta:+.2f}" if not np.isnan(delta) else "—",
        )
    console.print(table)

    # ═══════════════════════════════════════════════════════════
    # ZUSAMMENFASSUNG
    # ═══════════════════════════════════════════════════════════
    dt = time.time() - t0
    console.print(
        Panel(
            f"[bold]Test 7 Vertiefung abgeschlossen in {dt / 60:.1f} Minuten[/bold]",
            style="green",
        )
    )


if __name__ == "__main__":
    main()
