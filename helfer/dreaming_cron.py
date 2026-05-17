"""Dreaming Cron – Automatischer Nacht-Report."""

import subprocess
from datetime import date
from pathlib import Path

import typer
from loguru import logger

from helfer.common import HelferConfig, write_alert, write_status

app = typer.Typer(help="Dreaming Cron: Automatischer Nacht-Report")


@app.command()
def run(
    datum: str = typer.Argument(default="", help="Datum (YYYY-MM-DD), leer = heute"),
):
    logger.add("/tmp/helfer_dreaming_cron.log", rotation="10 MB", retention="7 days")
    config = HelferConfig.from_yaml()
    target = datum or date.today().isoformat()
    dreaming_script = config.tradingprojekt / "scripts" / "dreaming" / "dreaming.py"
    report_path = (
        Path.home() / ".claude/projects/-home-boobi/memory/dreaming" / f"{target}.md"
    )

    if report_path.exists():
        logger.info(f"Report für {target} existiert bereits – überspringe")
        write_status("dreaming_cron", {"state": "skipped", "date": target})
        return

    logger.info(f"Starte Dreaming Bot für {target}")
    try:
        subprocess.run(
            [
                str(config.tradingprojekt / ".venv/bin/python"),
                str(dreaming_script),
                target,
            ],
            capture_output=True,
            text=True,
            timeout=360,
            cwd=str(config.tradingprojekt),
        )
    except subprocess.TimeoutExpired:
        logger.error("Dreaming Bot Timeout (>6min)")
        write_status("dreaming_cron", {"state": "timeout", "date": target})
        return

    if report_path.exists():
        content = report_path.read_text()
        vorschlaege = content.count("Vorschlag") + content.count("vorgeschlagen")
        if vorschlaege >= 3:
            write_alert(
                f"Dreaming Report {target}: {vorschlaege} Vorschläge",
                f"Report: {report_path}\n\nBitte prüfen mit `/dreaming {target}`",
            )
        write_status(
            "dreaming_cron",
            {"state": "done", "date": target, "vorschlaege": vorschlaege},
        )
        logger.info(f"Report generiert: {report_path} ({vorschlaege} Vorschläge)")
    else:
        logger.warning(f"Kein Report generiert für {target}")
        write_status("dreaming_cron", {"state": "no_sessions", "date": target})


if __name__ == "__main__":
    app()
