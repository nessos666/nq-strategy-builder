"""Quality Gate – Prüft ob ROBUST-Ergebnisse wirklich robust sind."""

from dataclasses import dataclass, field

import typer
from loguru import logger

from helfer.common import write_status

app = typer.Typer(help="Quality Gate: Strategie-Integritäts-Wächter")

MIN_OOS_PF = 1.0
MIN_HO_PF = 1.0
MIN_OOS_TRADES = 20
MIN_HO_TRADES = 15
MAX_OOS_DEGRADATION = 0.30
MAX_HO_DEGRADATION = 0.35


@dataclass
class QualityResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)


def check_robust(
    oos_pf: float,
    ho_pf: float,
    oos_trades: int,
    ho_trades: int,
    oos_degradation: float,
    ho_degradation: float,
) -> QualityResult:
    failures = []
    checks = []

    if oos_pf < MIN_OOS_PF:
        failures.append(f"OOS PF {oos_pf:.2f} < {MIN_OOS_PF}")
    else:
        checks.append(f"OOS PF {oos_pf:.2f} ✓")

    if ho_pf < MIN_HO_PF:
        failures.append(f"HO PF {ho_pf:.2f} < {MIN_HO_PF}")
    else:
        checks.append(f"HO PF {ho_pf:.2f} ✓")

    if oos_trades < MIN_OOS_TRADES:
        failures.append(f"OOS Trades {oos_trades} < {MIN_OOS_TRADES}")
    else:
        checks.append(f"OOS Trades {oos_trades} ✓")

    if ho_trades < MIN_HO_TRADES:
        failures.append(f"HO Trades {ho_trades} < {MIN_HO_TRADES}")
    else:
        checks.append(f"HO Trades {ho_trades} ✓")

    if oos_degradation > MAX_OOS_DEGRADATION:
        failures.append(
            f"OOS Degradation {oos_degradation:.0%} > {MAX_OOS_DEGRADATION:.0%}"
        )
    else:
        checks.append(f"OOS Degradation {oos_degradation:.0%} ✓")

    if ho_degradation > MAX_HO_DEGRADATION:
        failures.append(
            f"HO Degradation {ho_degradation:.0%} > {MAX_HO_DEGRADATION:.0%}"
        )
    else:
        checks.append(f"HO Degradation {ho_degradation:.0%} ✓")

    return QualityResult(passed=len(failures) == 0, failures=failures, checks=checks)


@app.command()
def run():
    logger.add("/tmp/helfer_quality_gate.log", rotation="10 MB", retention="7 days")
    logger.info("Quality Gate Check (DB-Integration folgt)")
    write_status("quality_gate", {"state": "idle"})


if __name__ == "__main__":
    app()
