"""
Zentrales Logging-Modul für nq-strategy-builder.
Ersetzt alle print()-Aufrufe durch strukturiertes Logging.

Verwendung:
    from sb.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Backtest gestartet")
    logger.debug("Signal: %s, Bars: %d", signal_name, len(bars))
"""

import logging
import sys

_LOGGER_CACHE: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Logger mit einheitlichem Format holen (gecached)."""
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False

    _LOGGER_CACHE[name] = logger
    return logger


def configure_root(level: int = logging.WARNING) -> None:
    """Root-Logger konfigurieren (für externe Libraries)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
