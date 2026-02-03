from __future__ import annotations
import logging
import os
from pathlib import Path


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Minimal, production-style logger.

    - Console logging always enabled
    - Optional file logging if log_file is provided
    - LOG_LEVEL env var supported
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def mask_vin(vin: str) -> str:
    vin = (vin or "").strip().upper()
    if len(vin) < 6:
        return "***"
    return vin[:3] + "********" + vin[-3:]
