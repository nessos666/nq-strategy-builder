"""
Einheitliches Logging-Setup für alle Research-Skripte in scripts/.
Import statt print():

    from scripts._logging import logger
    logger.info("Analyse gestartet")
    logger.debug("Werte: %s", values)
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger("research")
