from __future__ import annotations

import argparse
import sqlite3
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

SESSION_TOKENS = {"NY", "LONDON", "ASIA", "RTH", "ETH", "PREMARKET"}

BAUSTEIN_GRUPPEN: dict[str, str] = {
    # Struktur – Preis-Struktur Signale
    "BOS": "Struktur",
    "OB": "Struktur",
    "FVG": "Struktur",
    "SWEEP": "Struktur",
    "BPR": "Struktur",
    "AMD": "Struktur",
    "JUDAS": "Struktur",
    "DISPLACEMENT": "Struktur",
    "EQH": "Struktur",
    "EQL": "Struktur",
    "MANIP": "Struktur",
    "IMBALANCE": "Struktur",
    "BB": "Struktur",
    "MMXM": "Struktur",
    "LRS": "Struktur",
    "LRB": "Struktur",
    # Zeit – Zeit/Session-basierte Kontext-Signale
    "ASIA_SWEEP": "Zeit",
    "LONDON_SWEEP": "Zeit",
    "PREMARKT_SWEEP": "Zeit",
    "MIDNIGHT_SWEEP": "Zeit",
    "NDOG": "Zeit",
    "NWOG": "Zeit",
    "WEEK_LEVELS": "Zeit",
    "DWM_LEVELS": "Zeit",
    "KILLZONE": "Zeit",
    # Momentum – Markt-Zustand / Filter-Signale
    "HURST": "Momentum",
    "CBDR": "Momentum",
    "DEALING_RANGE": "Momentum",
    "PREMIUM": "Momentum",
    "DISCOUNT": "Momentum",
    "DRAWN_LIQUIDITY": "Momentum",
    "RANGE": "Momentum",
}


def classify_gruppe(baustein: str) -> str:
    """Gibt die Gruppe eines Bausteins zurück: Struktur, Zeit, Momentum oder Unbekannt."""
    return BAUSTEIN_GRUPPEN.get(baustein, "Unbekannt")


def get_gruppen_profil(bausteine: set[str]) -> frozenset[str]:
    """Welche Gruppen sind in einem Set von Bausteinen vertreten?"""
    return frozenset(classify_gruppe(b) for b in bausteine)


def extract_bausteine(idea: str) -> set[str]:
    """
    'SWEEP + OB NY + HURST' → {'SWEEP', 'OB', 'HURST'}
    Entfernt Session-Token am Ende jedes Teils.
    """
    parts = [p.strip() for p in idea.split("+")]
    result = set()
    for part in parts:
        tokens = part.split()
        clean_tokens = [t for t in tokens if t not in SESSION_TOKENS]
        cleaned = " ".join(clean_tokens).strip()
        if cleaned:
            result.add(cleaned)
    return result


def load_build_runs(db_path: str | Path) -> pd.DataFrame:
    """
    Lädt build_runs + bestes Ergebnis (rank=1) aus results.
    Gibt DataFrame zurück mit: id, idea, avg_oos_pf, tier, session, winrate, num_trades
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            """
            SELECT
                b.id,
                b.idea,
                b.avg_oos_pf,
                b.tier,
                b.session,
                r.winrate,
                r.num_trades
            FROM build_runs b
            LEFT JOIN results r
                ON r.run_id = b.id AND r.rank = 1
        """,
            conn,
        )
    finally:
        conn.close()
    return df


def compute_matrix(df: pd.DataFrame) -> dict:
    """
    Für jeden Baustein: Ø PF und WR MIT vs. OHNE diesen Baustein.
    Gibt dict zurück: {baustein: {pf_mit, pf_ohne, delta, wr_mit, wr_ohne, wr_delta, count_mit, count_ohne}}
    """
    all_bausteine: set[str] = set()
    for idea in df["idea"]:
        all_bausteine.update(extract_bausteine(idea))

    matrix = {}
    for b in sorted(all_bausteine):
        mask = df["idea"].apply(lambda x: b in extract_bausteine(x))
        mit = df[mask]
        ohne = df[~mask]

        pf_mit = round(float(mit["avg_oos_pf"].mean()), 4) if len(mit) > 0 else 0.0
        pf_ohne = round(float(ohne["avg_oos_pf"].mean()), 4) if len(ohne) > 0 else 0.0
        wr_mit = 0.0
        if len(mit) > 0:
            _s = pd.to_numeric(mit["winrate"], errors="coerce").fillna(0.0)  # type: ignore[union-attr]
            wr_mit = round(float(_s.mean()), 4)  # type: ignore[arg-type]
        wr_ohne = 0.0
        if len(ohne) > 0:
            _s = pd.to_numeric(ohne["winrate"], errors="coerce").fillna(0.0)  # type: ignore[union-attr]
            wr_ohne = round(float(_s.mean()), 4)  # type: ignore[arg-type]

        matrix[b] = {
            "count_mit": len(mit),
            "count_ohne": len(ohne),
            "pf_mit": pf_mit,
            "pf_ohne": pf_ohne,
            "delta": round(pf_mit - pf_ohne, 4),
            "wr_mit": wr_mit,
            "wr_ohne": wr_ohne,
            "wr_delta": round(wr_mit - wr_ohne, 4),
        }
    return matrix


def compute_cross_group_matrix(df: pd.DataFrame) -> dict:
    """
    Gruppiert Strategien nach ihrem Gruppen-Profil und berechnet Ø PF/WR pro Muster.
    Gibt dict zurück: {frozenset(gruppen): {count, pf_mean, wr_mean, label}}
    """
    result: dict = {}
    for _, row in df.iterrows():
        bausteine = extract_bausteine(row["idea"])
        profil = get_gruppen_profil(bausteine)
        if profil not in result:
            result[profil] = {"count": 0, "pf_sum": 0.0, "wr_sum": 0.0}
        result[profil]["count"] += 1
        result[profil]["pf_sum"] += (
            float(row["avg_oos_pf"]) if pd.notna(row["avg_oos_pf"]) else 0.0
        )
        wr = pd.to_numeric(row.get("winrate"), errors="coerce")
        result[profil]["wr_sum"] += float(wr) if pd.notna(wr) else 0.0

    for profil, vals in result.items():
        n = vals["count"]
        vals["pf_mean"] = round(vals["pf_sum"] / n, 4) if n > 0 else 0.0
        vals["wr_mean"] = round(vals["wr_sum"] / n, 4) if n > 0 else 0.0
        vals["label"] = " + ".join(sorted(profil))
        vals.pop("pf_sum")
        vals.pop("wr_sum")

    return result


DEFAULT_DB = Path(__file__).parent.parent / "output_v3" / "builder.db"


def print_matrix(
    matrix: dict, min_count: int = 3, console: Console | None = None
) -> None:
    """Gibt Baustein-Wirkungsmatrix als Rich-Tabelle aus, sortiert nach Delta."""
    if console is None:
        console = Console()
    table = Table(title="Baustein-Wirkungsmatrix", show_lines=True)
    table.add_column("Baustein", style="cyan", no_wrap=True)
    table.add_column("n MIT", justify="right")
    table.add_column("Ø PF MIT", justify="right", style="green")
    table.add_column("Ø PF OHNE", justify="right", style="red")
    table.add_column("Delta PF", justify="right")
    table.add_column("Ø WR MIT", justify="right")
    table.add_column("Bewertung", justify="center")

    sorted_items = sorted(matrix.items(), key=lambda x: x[1]["delta"], reverse=True)

    for baustein, m in sorted_items:
        if m["count_mit"] < min_count:
            continue
        delta = m["delta"]
        bewertung = (
            "⬆ besser"
            if delta > 0.05
            else ("⬇ schlechter" if delta < -0.05 else "≈ neutral")
        )
        delta_str = (
            f"[green]+{delta:.4f}[/green]" if delta >= 0 else f"[red]{delta:.4f}[/red]"
        )
        table.add_row(
            baustein,
            str(m["count_mit"]),
            f"{m['pf_mit']:.4f}",
            f"{m['pf_ohne']:.4f}",
            delta_str,
            f"{m['wr_mit']:.1%}",
            bewertung,
        )

    console.print(table)
    shown = sum(1 for _, m in matrix.items() if m["count_mit"] >= min_count)
    console.print(
        f"\n[dim]Min. {min_count} Strategien MIT Baustein. Angezeigt: {shown} Bausteine.[/dim]"
    )


def print_cross_group_matrix(
    cross: dict, min_count: int = 3, console: Console | None = None
) -> None:
    """Gibt Cross-Group-Analyse als Rich-Tabelle aus, sortiert nach Ø PF absteigend."""
    if console is None:
        console = Console()
    table = Table(
        title="Information Asymmetry – Gruppen-Profil Analyse", show_lines=True
    )
    table.add_column("Gruppen-Kombination", style="cyan", no_wrap=True)
    table.add_column("Gruppen", justify="center")
    table.add_column("n", justify="right")
    table.add_column("Ø PF", justify="right", style="green")
    table.add_column("Ø WR", justify="right")
    table.add_column("Signal", justify="center")

    sorted_items = sorted(cross.items(), key=lambda x: x[1]["pf_mean"], reverse=True)

    for profil, vals in sorted_items:
        if vals["count"] < min_count:
            continue
        n_gruppen = len([g for g in profil if g != "Unbekannt"])
        signal = (
            "⬆⬆ max" if n_gruppen >= 3 else ("⬆ cross" if n_gruppen > 1 else "— single")
        )
        table.add_row(
            vals["label"],
            str(n_gruppen),
            str(vals["count"]),
            f"{vals['pf_mean']:.4f}",
            f"{vals['wr_mean']:.1%}",
            signal,
        )

    console.print(table)
    shown = sum(1 for vals in cross.values() if vals["count"] >= min_count)
    console.print(
        f"\n[dim]Min. {min_count} Strategien pro Profil. Angezeigt: {shown} Profile.[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baustein-Wirkungsanalyse")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Pfad zur builder.db")
    parser.add_argument(
        "--min-count", type=int, default=3, help="Min. Strategien pro Baustein/Profil"
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"FEHLER: DB nicht gefunden: {db_path}")
        raise SystemExit(1)

    print(f"Lade Daten aus: {db_path}")
    df = load_build_runs(db_path)
    print(f"Strategien: {len(df)} | Tiers: {df['tier'].value_counts().to_dict()}")

    matrix = compute_matrix(df)
    print_matrix(matrix, min_count=args.min_count)

    cross = compute_cross_group_matrix(df)
    print_cross_group_matrix(cross, min_count=args.min_count)


if __name__ == "__main__":
    main()
