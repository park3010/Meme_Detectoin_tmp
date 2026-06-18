"""Logging helpers used across scripts and modules."""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    """Return a stream logger with a compact research-friendly format."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO))
    return logger
