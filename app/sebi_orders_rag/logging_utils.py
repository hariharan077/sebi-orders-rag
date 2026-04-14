"""Logging helpers for the SEBI Orders RAG package."""

from __future__ import annotations

import logging

from .exceptions import ConfigurationError

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level_name: str) -> None:
    """Configure the root logger for CLI and service execution."""

    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ConfigurationError(f"Unsupported log level: {level_name}")

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
        return

    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
