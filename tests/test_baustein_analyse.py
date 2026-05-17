import io
import sys
import sqlite3
import tempfile
import os

import pandas as pd
from rich.console import Console

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from scripts.baustein_analyse import (
    classify_gruppe,
    compute_cross_group_matrix,
    compute_matrix,
    extract_bausteine,
    get_gruppen_profil,
    load_build_runs,
    print_cross_group_matrix,
    print_matrix,
)


def test_einfache_idee():
    assert extract_bausteine("SWEEP + OB NY") == {"SWEEP", "OB"}


def test_mit_hurst():
    assert extract_bausteine("BOS + OB NY + HURST") == {"BOS", "OB", "HURST"}


def test_session_wird_entfernt():
    assert extract_bausteine("SWEEP + FVG NY") == {"SWEEP", "FVG"}
    assert extract_bausteine("BOS + BPR LONDON") == {"BOS", "BPR"}
    assert extract_bausteine("AMD + BB ASIA") == {"AMD", "BB"}


def test_mehrere_sessions():
    assert extract_bausteine("JUDAS + ASIA_SWEEP + MIDNIGHT_SWEEP NY") == {
        "JUDAS",
        "ASIA_SWEEP",
        "MIDNIGHT_SWEEP",
    }


def test_load_build_runs():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE build_runs (
                id INTEGER PRIMARY KEY,
                idea TEXT NOT NULL,
                avg_oos_pf REAL,
                tier TEXT,
                session TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE results (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                pf REAL,
                winrate REAL,
                num_trades INTEGER,
                rank INTEGER
            )
        """)
        conn.executemany(
            "INSERT INTO build_runs VALUES (?,?,?,?,?)",
            [
                (1, "SWEEP + OB NY", 1.3, "C", "ny"),
                (2, "BOS + FVG NY", 0.9, "C", "ny"),
                (3, "SWEEP + FVG NY", 1.1, "C", "ny"),
            ],
        )
        conn.executemany(
            "INSERT INTO results VALUES (?,?,?,?,?,?)",
            [
                (1, 1, 1.3, 0.55, 80, 1),
                (2, 2, 0.9, 0.40, 60, 1),
                (3, 3, 1.1, 0.48, 70, 1),
            ],
        )
        conn.commit()
        conn.close()

        df = load_build_runs(db_path)
        assert len(df) == 3
        assert "idea" in df.columns
        assert "avg_oos_pf" in df.columns
        assert "winrate" in df.columns
        assert "num_trades" in df.columns
    finally:
        os.unlink(db_path)


def test_compute_matrix_delta():
    data = pd.DataFrame(
        [
            {
                "idea": "SWEEP + OB NY",
                "avg_oos_pf": 1.3,
                "winrate": 0.55,
                "num_trades": 80,
            },
            {
                "idea": "SWEEP + FVG NY",
                "avg_oos_pf": 1.1,
                "winrate": 0.48,
                "num_trades": 70,
            },
            {
                "idea": "BOS + OB NY",
                "avg_oos_pf": 0.9,
                "winrate": 0.40,
                "num_trades": 60,
            },
            {
                "idea": "BOS + FVG NY",
                "avg_oos_pf": 0.8,
                "winrate": 0.38,
                "num_trades": 55,
            },
        ]
    )
    matrix = compute_matrix(data)

    assert "SWEEP" in matrix
    assert matrix["SWEEP"]["delta"] > 0
    assert matrix["SWEEP"]["pf_mit"] > matrix["SWEEP"]["pf_ohne"]
    assert "OB" in matrix
    assert matrix["OB"]["delta"] > 0


def test_compute_matrix_count():
    data = pd.DataFrame(
        [
            {
                "idea": "SWEEP + OB NY",
                "avg_oos_pf": 1.3,
                "winrate": 0.55,
                "num_trades": 80,
            },
            {
                "idea": "SWEEP + FVG NY",
                "avg_oos_pf": 1.1,
                "winrate": 0.48,
                "num_trades": 70,
            },
            {
                "idea": "BOS + FVG NY",
                "avg_oos_pf": 0.8,
                "winrate": 0.38,
                "num_trades": 55,
            },
        ]
    )
    matrix = compute_matrix(data)
    assert matrix["SWEEP"]["count_mit"] == 2
    assert matrix["SWEEP"]["count_ohne"] == 1
    assert matrix["FVG"]["count_mit"] == 2
    assert matrix["FVG"]["count_ohne"] == 1


def test_print_matrix_min_count_filter():
    """Bausteine mit count_mit < min_count sollen nicht angezeigt werden."""
    matrix = {
        "SWEEP": {
            "count_mit": 5,
            "count_ohne": 2,
            "pf_mit": 1.3,
            "pf_ohne": 1.1,
            "delta": 0.2,
            "wr_mit": 0.55,
            "wr_ohne": 0.45,
            "wr_delta": 0.1,
        },
        "RARE": {
            "count_mit": 1,
            "count_ohne": 6,
            "pf_mit": 1.5,
            "pf_ohne": 1.0,
            "delta": 0.5,
            "wr_mit": 0.60,
            "wr_ohne": 0.45,
            "wr_delta": 0.15,
        },
    }
    buf = io.StringIO()
    console = Console(file=buf, no_color=True, highlight=False)
    print_matrix(matrix, min_count=3, console=console)
    output = buf.getvalue()

    assert "SWEEP" in output
    assert "RARE" not in output


def test_classify_gruppe_struktur():
    assert classify_gruppe("BOS") == "Struktur"
    assert classify_gruppe("OB") == "Struktur"
    assert classify_gruppe("FVG") == "Struktur"
    assert classify_gruppe("SWEEP") == "Struktur"


def test_classify_gruppe_zeit():
    assert classify_gruppe("ASIA_SWEEP") == "Zeit"
    assert classify_gruppe("MIDNIGHT_SWEEP") == "Zeit"
    assert classify_gruppe("NDOG") == "Zeit"


def test_classify_gruppe_momentum():
    assert classify_gruppe("HURST") == "Momentum"
    assert classify_gruppe("CBDR") == "Momentum"


def test_classify_gruppe_unbekannt():
    assert classify_gruppe("UNBEKANNT_XYZ") == "Unbekannt"


def test_get_gruppen_profil_mixed():
    bausteine = {"SWEEP", "ASIA_SWEEP", "HURST"}
    profil = get_gruppen_profil(bausteine)
    assert profil == frozenset({"Struktur", "Zeit", "Momentum"})


def test_get_gruppen_profil_single():
    bausteine = {"BOS", "OB"}
    profil = get_gruppen_profil(bausteine)
    assert profil == frozenset({"Struktur"})


def test_get_gruppen_profil_cross():
    bausteine = {"SWEEP", "MIDNIGHT_SWEEP"}
    profil = get_gruppen_profil(bausteine)
    assert profil == frozenset({"Struktur", "Zeit"})


def test_compute_cross_group_matrix_keys():
    data = pd.DataFrame(
        [
            {
                "idea": "SWEEP + OB NY",
                "avg_oos_pf": 1.3,
                "winrate": 0.55,
                "num_trades": 80,
            },
            {
                "idea": "SWEEP + ASIA_SWEEP NY",
                "avg_oos_pf": 1.5,
                "winrate": 0.58,
                "num_trades": 75,
            },
            {
                "idea": "BOS + HURST NY",
                "avg_oos_pf": 1.4,
                "winrate": 0.56,
                "num_trades": 70,
            },
            {
                "idea": "OB + FVG NY",
                "avg_oos_pf": 0.9,
                "winrate": 0.42,
                "num_trades": 60,
            },
        ]
    )
    result = compute_cross_group_matrix(data)
    assert frozenset({"Struktur"}) in result
    assert frozenset({"Struktur", "Zeit"}) in result


def test_compute_cross_group_matrix_cross_better():
    """Strategien mit Bausteinen aus verschiedenen Gruppen sollen höheres Ø PF haben."""
    data = pd.DataFrame(
        [
            {
                "idea": "SWEEP + ASIA_SWEEP NY",
                "avg_oos_pf": 1.5,
                "winrate": 0.58,
                "num_trades": 75,
            },
            {
                "idea": "BOS + MIDNIGHT_SWEEP NY",
                "avg_oos_pf": 1.6,
                "winrate": 0.60,
                "num_trades": 80,
            },
            {
                "idea": "OB + FVG NY",
                "avg_oos_pf": 0.9,
                "winrate": 0.42,
                "num_trades": 60,
            },
            {
                "idea": "BOS + EQH NY",
                "avg_oos_pf": 0.8,
                "winrate": 0.40,
                "num_trades": 55,
            },
        ]
    )
    result = compute_cross_group_matrix(data)
    pf_cross = result[frozenset({"Struktur", "Zeit"})]["pf_mean"]
    pf_single = result[frozenset({"Struktur"})]["pf_mean"]
    assert pf_cross > pf_single


def test_compute_cross_group_matrix_count():
    data = pd.DataFrame(
        [
            {
                "idea": "SWEEP + ASIA_SWEEP NY",
                "avg_oos_pf": 1.5,
                "winrate": 0.58,
                "num_trades": 75,
            },
            {
                "idea": "BOS + MIDNIGHT_SWEEP NY",
                "avg_oos_pf": 1.6,
                "winrate": 0.60,
                "num_trades": 80,
            },
            {
                "idea": "OB + FVG NY",
                "avg_oos_pf": 0.9,
                "winrate": 0.42,
                "num_trades": 60,
            },
        ]
    )
    result = compute_cross_group_matrix(data)
    assert result[frozenset({"Struktur", "Zeit"})]["count"] == 2
    assert result[frozenset({"Struktur"})]["count"] == 1


def test_compute_cross_group_matrix_no_internal_keys():
    """pf_sum/wr_sum sind interne Akkumulatoren und dürfen nicht im Output erscheinen."""
    data = pd.DataFrame(
        [
            {
                "idea": "SWEEP + ASIA_SWEEP NY",
                "avg_oos_pf": 1.5,
                "winrate": 0.58,
                "num_trades": 75,
            },
        ]
    )
    result = compute_cross_group_matrix(data)
    for vals in result.values():
        assert "pf_sum" not in vals
        assert "wr_sum" not in vals


def test_print_cross_group_matrix_output():
    cross = {
        frozenset({"Struktur"}): {
            "count": 2,
            "pf_mean": 0.95,
            "wr_mean": 0.42,
            "label": "Struktur",
        },
        frozenset({"Struktur", "Zeit"}): {
            "count": 3,
            "pf_mean": 1.45,
            "wr_mean": 0.57,
            "label": "Struktur + Zeit",
        },
    }
    buf = io.StringIO()
    console = Console(file=buf, no_color=True, highlight=False)
    print_cross_group_matrix(cross, console=console)
    output = buf.getvalue()
    assert "Struktur + Zeit" in output
    assert (
        "0.9500" not in output
    )  # Struktur-only (count=2 < min_count=3) darf nicht erscheinen
    assert "1.4500" in output


def test_print_cross_group_matrix_min_count():
    """Einträge mit count < min_count sollen nicht angezeigt werden."""
    cross = {
        frozenset({"Struktur"}): {
            "count": 1,
            "pf_mean": 2.0,
            "wr_mean": 0.70,
            "label": "Struktur",
        },
        frozenset({"Struktur", "Zeit"}): {
            "count": 5,
            "pf_mean": 1.45,
            "wr_mean": 0.57,
            "label": "Struktur + Zeit",
        },
    }
    buf = io.StringIO()
    console = Console(file=buf, no_color=True, highlight=False)
    print_cross_group_matrix(cross, min_count=3, console=console)
    output = buf.getvalue()
    assert "Struktur + Zeit" in output
    assert "2.0000" not in output
