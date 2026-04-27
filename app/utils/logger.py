"""
Structured JSON logger using Python's standard logging library.
Outputs machine-readable logs in production, colourised in development.
"""

import logging
import sys
from typing import Any

from app.config import get_settings

_LOG_FORMAT_JSON = (
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"module":"%(name)s","message":"%(message)s"}'
)
_LOG_FORMAT_PRETTY = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured for the application.

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"key": "value"})
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Pretty format for development, JSON for production
    fmt = _LOG_FORMAT_PRETTY if settings.log_level == "DEBUG" else _LOG_FORMAT_JSON
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


class LoggerMixin:
    """Mixin that provides a `self.logger` attribute to any class."""

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)


def log_event(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    """Log a structured event with arbitrary key-value context."""
    ctx = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
    logger.info(f"{event} {ctx}".strip())
