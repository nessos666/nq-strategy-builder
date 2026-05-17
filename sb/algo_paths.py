"""Zentrale Pfade zur Algo-Bibliothek (Signal-Cache, Knowledge-Anzeige).

Alle Signal-Generatoren liegen in algo_bibliothek/v2/{pda,smc,science}/.
Jede Datei dort implementiert compute_*() statt run().
LDP (Lopez de Prado) = ML-Utilities, keine Signal-Generatoren → nicht hier.

Konfiguration über Umgebungsvariable:
    TRADINGPROJEKT_PATH=/path/to/nq_backtest_project
"""

from __future__ import annotations

import os
from pathlib import Path

_TRADINGPROJEKT = Path(
    os.environ.get("TRADINGPROJEKT_PATH", str(Path.home() / "TRADINGPROJEKT"))
)

_ALGO_BIB = _TRADINGPROJEKT / "nq_backtest" / "algo_bibliothek"

# Öffentliche Konstante: Standard-Suchpfade für Optuna/Signal-Cache (v2)
DEFAULT_SIGNAL_ALGO_DIRS: tuple[Path, ...] = (
    _ALGO_BIB / "v2" / "pda",
    _ALGO_BIB / "v2" / "smc",
    _ALGO_BIB / "v2" / "science",
    _ALGO_BIB / "v2" / "ldp",
    _ALGO_BIB / "v2" / "pyind",
)
