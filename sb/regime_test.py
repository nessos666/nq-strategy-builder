"""
Regime-Filter Validierungs-Test

Misst ob der HRL/LRL Indikator tatsaechlich vorhersagt was DANACH passiert.
Forward-Looking: Fuer jedes Regime-Signal (LRL/HRL/NEUTRAL) messen wir
was in den naechsten N Bars passiert.

Wenn LRL korrekt ist → Preis laeuft danach effizient (hoher KER, grosse Moves)
Wenn HRL korrekt ist → Preis zickzackt danach (niedriger KER, kleine Netto-Moves)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

FORWARD_PERIODS = [15, 30, 60, 120]

DATA_PATHS = {
    "1m": Path("data/nq_1m_databento_2024_2026.parquet"),
    "15m": Path("data/nq_15m_2024_2026.parquet"),
}

ALGO_MAP = {
    "petar": "HRL LRL Petar Drawdown Filter",
    "internet": "HRL LRL Internet 15min Trend Filter",
}

REGIME_COL_MAP = {
    "petar": ("pda_regime", "is_pda_lrl", "is_pda_hrl"),
    "internet": ("stat_regime", "is_stat_lrl", "is_stat_hrl"),
}


def _load_algo(algo_name: str) -> object:
    """Lade Algo-Modul dynamisch."""
    import importlib
    import sys

    algo_dir = Path(__file__).parent.parent / "david_bibliothek" / "03_Context"
    sys.path.insert(0, str(algo_dir))
    mod = importlib.import_module(ALGO_MAP[algo_name])
    sys.path.pop(0)
    return mod


def _compute_forward_metrics(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    opn: pd.Series,
    mask: pd.Series,
    n: int,
) -> dict:
    """Berechne Forward-Looking Metriken fuer Bars wo mask=True."""
    if mask.sum() == 0:
        return {
            "count": 0,
            "fwd_net_move": np.nan,
            "fwd_ker": np.nan,
            "fwd_max_dd": np.nan,
            "fwd_consistency": np.nan,
            "fwd_body_ratio": np.nan,
        }

    indices = mask[mask].index
    results = []

    for idx in indices:
        pos = close.index.get_loc(idx)
        if pos + n >= len(close):
            continue

        fwd_close = close.iloc[pos : pos + n + 1]
        fwd_high = high.iloc[pos : pos + n + 1]
        fwd_low = low.iloc[pos : pos + n + 1]
        fwd_open = opn.iloc[pos : pos + n + 1]

        c0 = fwd_close.iloc[0]
        cn = fwd_close.iloc[-1]

        # Netto-Bewegung
        net_move = abs(cn - c0)

        # KER der Forward-Bars
        direction = abs(cn - c0)
        volatility = fwd_close.diff().abs().sum()
        ker = direction / volatility if volatility > 0 else 0.0

        # Max Drawdown (vom Einstieg)
        if cn > c0:  # bullisch
            dd = c0 - fwd_low.min()
        else:  # baerisch
            dd = fwd_high.max() - c0

        # Konsistenz (% Bars in Hauptrichtung)
        diffs = fwd_close.diff().dropna()
        if len(diffs) > 0:
            if cn > c0:
                consistency = (diffs > 0).mean()
            else:
                consistency = (diffs < 0).mean()
        else:
            consistency = 0.5

        # Body Ratio
        candle_range = fwd_high - fwd_low
        body = (fwd_close - fwd_open).abs()
        body_pct = body / candle_range.replace(0, np.nan)
        body_ratio = body_pct.mean()

        results.append(
            {
                "fwd_net_move": net_move,
                "fwd_ker": ker,
                "fwd_max_dd": dd,
                "fwd_consistency": consistency,
                "fwd_body_ratio": body_ratio,
            }
        )

    if not results:
        return {
            "count": 0,
            "fwd_net_move": np.nan,
            "fwd_ker": np.nan,
            "fwd_max_dd": np.nan,
            "fwd_consistency": np.nan,
            "fwd_body_ratio": np.nan,
        }

    rdf = pd.DataFrame(results)
    return {
        "count": len(rdf),
        "fwd_net_move": rdf["fwd_net_move"].median(),
        "fwd_ker": rdf["fwd_ker"].median(),
        "fwd_max_dd": rdf["fwd_max_dd"].median(),
        "fwd_consistency": rdf["fwd_consistency"].median(),
        "fwd_body_ratio": rdf["fwd_body_ratio"].median(),
    }


def validate_regime(
    algo_name: str = "petar",
    forward_periods: list[int] | None = None,
    sample_rate: float = 0.1,
    timeframe: str = "15m",
) -> dict:
    """Validiere Regime-Filter mit Forward-Looking Metriken.

    sample_rate: Anteil der Events die getestet werden (0.1 = 10%, schneller).
    timeframe: "1m" oder "15m"
    """
    if forward_periods is None:
        forward_periods = FORWARD_PERIODS

    console.print(f"\n[bold]Regime-Test: {ALGO_MAP[algo_name]} ({timeframe})[/bold]")
    console.print(f"Forward-Perioden: {forward_periods}")

    # Daten laden
    data_path = DATA_PATHS.get(timeframe, DATA_PATHS["15m"])
    console.print(f"Lade Daten: {data_path.name}")
    df = pd.read_parquet(data_path)
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    console.print(f"Bars: {len(df):,}")

    # Algo ausfuehren
    console.print("Algo ausfuehren...")
    mod = _load_algo(algo_name)
    result = mod.run(df)

    regime_col, lrl_col, hrl_col = REGIME_COL_MAP[algo_name]

    # Regime-Wechsel erkennen (Rising Edge)
    regime = result[regime_col]
    regime_changed = regime != regime.shift(1)

    lrl_events = regime_changed & (regime == "LRL")
    hrl_events = regime_changed & (regime == "HRL")
    neutral_events = regime_changed & (regime == "NEUTRAL")

    console.print(
        f"Events: LRL={lrl_events.sum()}, HRL={hrl_events.sum()}, NEUTRAL={neutral_events.sum()}"
    )

    # Sampling (10% default, sonst zu langsam auf 717k Bars)
    if sample_rate < 1.0:
        np.random.seed(42)
        for events in [lrl_events, hrl_events, neutral_events]:
            true_idx = events[events].index
            n_keep = max(1, int(len(true_idx) * sample_rate))
            keep = np.random.choice(true_idx, n_keep, replace=False)
            drop = set(true_idx) - set(keep)
            events.loc[list(drop)] = False
        console.print(
            f"Sampling {sample_rate * 100:.0f}%: LRL={lrl_events.sum()}, HRL={hrl_events.sum()}, NEUTRAL={neutral_events.sum()}"
        )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]

    all_results = {}

    for n in forward_periods:
        console.print(f"  Forward {n} Bars...")
        lrl_m = _compute_forward_metrics(close, high, low, opn, lrl_events, n)
        hrl_m = _compute_forward_metrics(close, high, low, opn, hrl_events, n)
        neu_m = _compute_forward_metrics(close, high, low, opn, neutral_events, n)
        all_results[n] = {"LRL": lrl_m, "HRL": hrl_m, "NEUTRAL": neu_m}

    # Tabelle anzeigen
    for n in forward_periods:
        table = Table(title=f"Forward {n} Bars", show_header=True)
        table.add_column("Metrik", style="white")
        table.add_column("LRL", style="green")
        table.add_column("HRL", style="red")
        table.add_column("NEUTRAL", style="dim")
        table.add_column("Delta (LRL-HRL)", style="yellow")

        r = all_results[n]
        for metric in [
            "count",
            "fwd_net_move",
            "fwd_ker",
            "fwd_max_dd",
            "fwd_consistency",
            "fwd_body_ratio",
        ]:
            lv = r["LRL"].get(metric, np.nan)
            hv = r["HRL"].get(metric, np.nan)
            nv = r["NEUTRAL"].get(metric, np.nan)

            if metric == "count":
                table.add_row(metric, str(lv), str(hv), str(nv), "")
            else:
                delta = lv - hv if not (np.isnan(lv) or np.isnan(hv)) else np.nan
                fmt = ".2f" if metric != "fwd_net_move" else ".1f"
                table.add_row(
                    metric,
                    f"{lv:{fmt}}" if not np.isnan(lv) else "—",
                    f"{hv:{fmt}}" if not np.isnan(hv) else "—",
                    f"{nv:{fmt}}" if not np.isnan(nv) else "—",
                    f"{delta:+{fmt}}" if not np.isnan(delta) else "—",
                )
        console.print(table)

    # Bewertung
    console.print("\n[bold]Bewertung:[/bold]")
    passes = 0
    fails = 0
    for n in forward_periods:
        r = all_results[n]
        lrl_ker = r["LRL"].get("fwd_ker", 0)
        hrl_ker = r["HRL"].get("fwd_ker", 0)
        if not np.isnan(lrl_ker) and not np.isnan(hrl_ker) and lrl_ker > hrl_ker:
            console.print(
                f"  [green]PASS[/green] Forward {n}: LRL KER {lrl_ker:.3f} > HRL KER {hrl_ker:.3f}"
            )
            passes += 1
        else:
            console.print(
                f"  [red]FAIL[/red] Forward {n}: LRL KER {lrl_ker:.3f} <= HRL KER {hrl_ker:.3f}"
            )
            fails += 1

    verdict = "PASS" if passes > fails else "FAIL"
    console.print(
        f"\n[bold]{'[green]' if verdict == 'PASS' else '[red]'}{verdict}[/bold] — {passes}/{passes + fails} Forward-Perioden bestanden"
    )

    return all_results
