#!/usr/bin/env python3
"""
sb/export_registry.py – Exportiert Registry-Daten (HO) in studies.db.

Erzeugt/ergaenzt die Tabelle 'build_runs' in output_david_1/studies.db
mit allen Daten aus builder.db's build_runs-Tabelle.

Mapping-Strategie:
  1. Berechne idea_hash via Parser fuer jeden build_run
  2. Suche matching study in studies.db (study_name ~ '{hash}_w*')
  3. Bei Treffer: study_id wird gesetzt
  4. Bei keinem Treffer: idea, tier, ho etc. werden trotzdem exportiert

Verwendung:
    python -m sb.export_registry
    python -m sb.export_registry --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Projekt-Root ermitteln (sb/ liegt im Root)
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

# Parser importieren
sys.path.insert(0, str(_PROJECT_ROOT))
from sb.engine.parser import parse_idea
from sb.engine.walk_forward import _idea_hash

OUTPUT_DIR = _PROJECT_ROOT / "output_david_1"
BUILDER_DB = OUTPUT_DIR / "builder.db"
STUDIES_DB = OUTPUT_DIR / "studies.db"

BUILD_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS build_runs (
    study_id        INTEGER PRIMARY KEY,
    study_name      TEXT,
    idea            TEXT,
    tier            TEXT,
    holdout_pf      REAL,
    oos_pf          REAL,
    is_robust       INTEGER,
    trades          INTEGER,
    created_at      TEXT
);
"""


def get_studies(conn: sqlite3.Connection) -> dict[str, int]:
    """Liest alle studies -> {study_name: study_id}."""
    rows = conn.execute(
        "SELECT study_id, study_name FROM studies"
    ).fetchall()
    result: dict[str, int] = {}
    for sid, sname in rows:
        result[sname] = sid
    return result


def get_study_by_hash(studies: dict[str, int], idea_hash: str) -> int | None:
    """Findet eine study_id fuer einen idea_hash.
    Walk-Forward erzeugt Studie: {idea_hash}_w0, _w1, _w2
    Wir nehmen w0 als primaere study_id.
    """
    for w in range(3):
        sname = f"{idea_hash}_w{w}"
        if sname in studies:
            return studies[sname]
    return None


def compute_mapping(
    builder_path: Path, studies_path: Path
) -> list[dict]:
    """Berechnet vollstaendiges Mapping von build_runs auf studies."""
    b_conn = sqlite3.connect(str(builder_path))
    b_conn.row_factory = sqlite3.Row
    s_conn = sqlite3.connect(str(studies_path))
    s_conn.row_factory = sqlite3.Row

    try:
        studies = get_studies(s_conn)
        
        # Alle build_runs lesen
        b_rows = b_conn.execute(
            "SELECT id, idea, tier, holdout_pf, avg_oos_pf, is_robust, "
            "holdout_trades, holdout_validated FROM build_runs ORDER BY id"
        ).fetchall()
        
        results: list[dict] = []
        used_studies: set[int] = set()
        
        for row in b_rows:
            idea = row["idea"]
            tier = row["tier"]
            ho_pf = row["holdout_pf"]
            oos_pf = row["avg_oos_pf"]
            is_robust = bool(row["is_robust"]) if row["is_robust"] is not None else None
            trades = int(row["holdout_trades"]) if row["holdout_trades"] is not None else None
            validated = bool(row["holdout_validated"]) if row["holdout_validated"] is not None else False
            
            # Nur gueltige HO-Daten exportieren
            if validated and ho_pf is not None and ho_pf > 0:
                eff_ho_pf = ho_pf
            else:
                eff_ho_pf = None
            
            # Hash berechnen
            try:
                parsed = parse_idea(idea)
                h = _idea_hash(parsed)
            except Exception:
                h = None
            
            # Study finden
            study_id = None
            study_name = None
            if h:
                # Alle matching studies finden (w0, w1, w2)
                matched = []
                for w in range(3):
                    sname = f"{h}_w{w}"
                    if sname in studies:
                        matched.append((studies[sname], sname))
                
                if matched:
                    # Nimm w0 (oder ersten verfuegbaren)
                    study_id, study_name = matched[0]
                    used_studies.add(study_id)
            
            results.append({
                "study_id": study_id,
                "study_name": study_name,
                "idea": idea,
                "tier": tier,
                "holdout_pf": eff_ho_pf,
                "oos_pf": oos_pf,
                "is_robust": is_robust,
                "trades": trades,
            })
        
        # Nicht-gemappte Studies als "no data" eintragen
        all_study_ids = set(studies.values())
        unmapped = all_study_ids - used_studies
        sid_to_name = {v: k for k, v in studies.items()}
        for sid in sorted(unmapped):
            results.append({
                "study_id": sid,
                "study_name": sid_to_name.get(sid),
                "idea": None,
                "tier": None,
                "holdout_pf": None,
                "oos_pf": None,
                "is_robust": None,
                "trades": None,
            })
        
        return results
    
    finally:
        b_conn.close()
        s_conn.close()


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(BUILD_RUNS_SCHEMA)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_br_study_name 
        ON build_runs(study_name)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_br_idea 
        ON build_runs(idea)
    """)
    conn.commit()


def upsert(conn: sqlite3.Connection, row: dict) -> None:
    """UPSERT: fuege ein oder aktualisiere, ueberschreibe nie mit NULL."""
    conn.execute(
        """
        INSERT INTO build_runs (study_id, study_name, idea, tier, holdout_pf, oos_pf, is_robust, trades, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(study_id) DO UPDATE SET
            study_name   = COALESCE(EXCLUDED.study_name, build_runs.study_name),
            idea         = COALESCE(EXCLUDED.idea, build_runs.idea),
            tier         = COALESCE(EXCLUDED.tier, build_runs.tier),
            holdout_pf   = COALESCE(EXCLUDED.holdout_pf, build_runs.holdout_pf),
            oos_pf       = COALESCE(EXCLUDED.oos_pf, build_runs.oos_pf),
            is_robust    = COALESCE(EXCLUDED.is_robust, build_runs.is_robust),
            trades       = COALESCE(EXCLUDED.trades, build_runs.trades)
        """,
        (
            row["study_id"],
            row["study_name"],
            row["idea"],
            row["tier"],
            row["holdout_pf"],
            row["oos_pf"],
            row["is_robust"],
            row["trades"],
        ),
    )


def print_report(rows: list[dict]) -> None:
    """Gibt eine Uebersicht der Export-Daten aus."""
    n_total = len(rows)
    n_with_study = sum(1 for r in rows if r["study_id"] is not None)
    n_with_idea = sum(1 for r in rows if r["idea"] is not None)
    n_with_tier = sum(1 for r in rows if r["tier"] is not None)
    n_with_ho = sum(1 for r in rows if r["holdout_pf"] is not None)
    n_with_oos = sum(1 for r in rows if r["oos_pf"] is not None)
    n_with_robust = sum(1 for r in rows if r["is_robust"] is not None)
    
    tiers = {}
    for r in rows:
        t = r["tier"]
        tiers[t] = tiers.get(t, 0) + 1
    
    print(f"Export-Uebersicht:")
    print(f"  Eintraege gesamt:     {n_total}")
    print(f"  Mit study_id:         {n_with_study}")
    print(f"  Mit Idea:             {n_with_idea}")
    print(f"  Mit Tier:             {n_with_tier}  (A={tiers.get('A',0)}, B={tiers.get('B',0)}, C={tiers.get('C',0)})")
    print(f"  Mit OOS PF:           {n_with_oos}")
    print(f"  Mit Holdout PF:       {n_with_ho}")
    print(f"  Mit Robust-Status:    {n_with_robust}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exportiere Registry-Daten (HO) in studies.db"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur anzeigen, nichts schreiben")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR),
                        help=f"Output-Verzeichnis (default: {OUTPUT_DIR})")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    builder_path = output_dir / "builder.db"
    studies_path = output_dir / "studies.db"
    
    if not builder_path.exists():
        print(f"FEHLER: builder.db nicht gefunden: {builder_path}")
        sys.exit(1)
    if not studies_path.exists():
        print(f"FEHLER: studies.db nicht gefunden: {studies_path}")
        sys.exit(1)
    
    rows = compute_mapping(builder_path, studies_path)
    if not rows:
        print("FEHLER: Keine Daten.")
        sys.exit(1)
    
    print(f"Quelle: {builder_path}")
    print(f"Ziel:   {studies_path}")
    print()
    print_report(rows)
    print()
    
    if args.dry_run:
        print("=== DRY-RUN – keine Aenderungen ===")
        if rows:
            print(f"\nBeispiel-Eintraege:")
            with_study = [r for r in rows if r["study_id"] is not None]
            for r in with_study[:5]:
                ho_str = f"ho={r['holdout_pf']:.4f}" if r['holdout_pf'] else "ho=—"
                print(f"  study_id={r['study_id']:>3} | tier={r['tier'] or '?'} | "
                      f"{ho_str} | {r['idea'][:50] if r['idea'] else '?'}")
            if len(with_study) > 5:
                print(f"  ... und {len(with_study)-5} weitere")
            no_study = [r for r in rows if r["idea"] is not None and r["study_id"] is None]
            if no_study:
                print(f"\nUnmatched build_runs (nur builder.db, kein study-Match):")
                for r in no_study[:5]:
                    ho_str = f"ho={r['holdout_pf']}" if r['holdout_pf'] else "ho=—"
                    print(f"  {r['idea'][:50]:50s} | tier={r['tier'] or '?'} | {ho_str}")
                if len(no_study) > 5:
                    print(f"  ... und {len(no_study)-5} weitere")
    else:
        s_conn = sqlite3.connect(str(studies_path))
        try:
            ensure_table(s_conn)
            inserted = 0
            for r in rows:
                if r["study_id"] is not None:
                    upsert(s_conn, r)
                    inserted += 1
            s_conn.commit()
            
            cur = s_conn.execute("SELECT COUNT(*) FROM build_runs")
            total = cur.fetchone()[0]
            print(f"Geschrieben: {inserted} Eintraege (UPSERT)")
            print(f"Gesamt in build_runs-Tabelle: {total}")
            print(f"Fertig: {studies_path}")
        finally:
            s_conn.close()


if __name__ == "__main__":
    main()
