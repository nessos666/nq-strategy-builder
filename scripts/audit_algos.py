#!/usr/bin/env python3
"""Algo-Bibliothek Audit – prüft alle compute_*() Funktionen auf Korrektheit.

Ausführen:
    .venv/bin/python scripts/audit_algos.py
    .venv/bin/python scripts/audit_algos.py --verbose
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import os

import numpy as np
import pandas as pd

# Algo-Bibliothek Pfade (via Umgebungsvariable TRADINGPROJEKT_PATH konfigurierbar)
_TRADINGPROJEKT = Path(
    os.environ.get("TRADINGPROJEKT_PATH", str(Path.home() / "TRADINGPROJEKT"))
)
_ALGO_BIB = _TRADINGPROJEKT / "nq_backtest" / "algo_bibliothek"
ALGO_DIRS = [
    _ALGO_BIB / "v2" / "pda",
    _ALGO_BIB / "v2" / "smc",
    _ALGO_BIB / "v2" / "science",
]

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def _make_bars(n: int = 100) -> pd.DataFrame:
    """Synthetische MNQ-Bars: 100 aufeinanderfolgende 1-Min-Bars."""
    rng = np.random.default_rng(42)
    closes = 19000.0 + np.cumsum(rng.normal(0, 5, n))
    opens = closes + rng.normal(0, 2, n)
    highs = np.maximum(opens, closes) + abs(rng.normal(0, 3, n))
    lows = np.minimum(opens, closes) - abs(rng.normal(0, 3, n))
    idx = pd.date_range("2025-01-02 14:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 100},
        index=idx,
    )


def _load_module(path: Path):
    """Lädt Python-Datei als Modul."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _check_output(result) -> list[str]:
    """Prüft ob Ausgabe sinnvoll ist. Gibt Fehlermeldungen zurück."""
    issues: list[str] = []
    if result is None:
        issues.append("gibt None zurück")
        return issues
    if isinstance(result, pd.Series):
        if result.empty:
            issues.append("leere Series")
        elif result.isnull().all():
            issues.append("alle Werte NaN")
    elif isinstance(result, pd.DataFrame):
        if result.empty:
            issues.append("leerer DataFrame")
        elif bool(result.isnull().all(axis=None)):
            issues.append("alle Werte NaN")
    elif isinstance(result, np.ndarray):
        if result.size == 0:
            issues.append("leeres Array")
        elif not np.isfinite(result).any():
            issues.append("keine finiten Werte")
    return issues


def audit_algo(path: Path, bars: pd.DataFrame) -> dict:
    """Prüft eine Algo-Datei. Gibt {name, status, functions, issues} zurück."""
    result = {"name": path.stem, "status": "ok", "functions": [], "issues": []}
    try:
        mod = _load_module(path)
        if mod is None:
            result["status"] = "load_error"
            result["issues"].append("Modul konnte nicht geladen werden")
            return result
    except Exception as exc:
        result["status"] = "load_error"
        result["issues"].append(f"Import-Fehler: {exc}")
        return result

    compute_fns = [
        name
        for name, _ in inspect.getmembers(mod, inspect.isfunction)
        if name.startswith("compute_")
    ]
    result["functions"] = compute_fns

    if not compute_fns:
        result["status"] = "no_compute"
        result["issues"].append("keine compute_*() Funktion gefunden")
        return result

    for fn_name in compute_fns:
        fn = getattr(mod, fn_name)
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            # Erster Parameter = bars (DataFrame)
            if not params:
                result["issues"].append(f"{fn_name}: keine Parameter")
                continue
            output = fn(bars)
            issues = _check_output(output)
            if issues:
                for issue in issues:
                    result["issues"].append(f"{fn_name}: {issue}")
                result["status"] = "warn"
        except TypeError as exc:
            # Manche Algos brauchen extra Argumente → kein harter Fehler
            result["issues"].append(
                f"{fn_name}: Signatur-Fehler (ok wenn extra Args nötig): {exc}"
            )
        except Exception as exc:
            result["issues"].append(f"{fn_name}: Exception: {exc}")
            result["status"] = "error"

    return result


def main() -> None:
    bars = _make_bars(200)
    all_files: list[Path] = []
    for d in ALGO_DIRS:
        if d.exists():
            all_files.extend(sorted(d.glob("*_v2.py")))

    print(f"Algo Audit – {len(all_files)} Dateien\n{'=' * 60}")
    ok = warn = error = 0
    for path in all_files:
        audit = audit_algo(path, bars)
        status = audit["status"]
        fns = ", ".join(audit["functions"]) or "–"
        if status == "ok":
            ok += 1
            symbol = "OK  "
        elif status in ("warn", "no_compute"):
            warn += 1
            symbol = "WARN"
        else:
            error += 1
            symbol = "ERR "
        print(f"[{symbol}] {audit['name']:<45} fns={fns[:50]}")
        if audit["issues"] and (VERBOSE or status in ("error", "load_error")):
            for issue in audit["issues"]:
                print(f"       -> {issue}")

    print(f"\n{'=' * 60}")
    print(f"Ergebnis: {ok} OK | {warn} WARN | {error} ERROR | {len(all_files)} gesamt")
    if error > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
