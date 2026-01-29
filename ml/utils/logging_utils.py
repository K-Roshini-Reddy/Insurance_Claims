from __future__ import annotations
import logging
import os

def setup_logger(name: str) -> logging.Logger:
    """
    Minimal, production-style logger. Uses env LOG_LEVEL if set.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def mask_vin(vin: str) -> str:
    vin = (vin or "").strip().upper()
    if len(vin) < 6:
        return "***"
    return vin[:3] + "********" + vin[-3:]
