"""Logging configuration shared by all package modules."""
from __future__ import annotations

import logging
import os
import sys

_INITIALIZED = False


def _ensure_root_configured() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    level_name = os.environ.get("TMAFC_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger("tmafc")
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.propagate = False
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    _ensure_root_configured()
    if not name.startswith("tmafc"):
        name = f"tmafc.{name}"
    return logging.getLogger(name)
