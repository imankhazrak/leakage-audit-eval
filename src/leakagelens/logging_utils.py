"""Minimal logging setup for research modules and experiment runs."""

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"
LOGGER_NAME = "leakagelens"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger."""

    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
    overwrite_file: bool = False,
) -> logging.Logger:
    """Configure console logging and, optionally, a result-directory log file.

    Safe to call more than once: existing LeakageLens handlers are replaced.
    Research modules should use this logger instead of ``print()``.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            path,
            mode="w" if overwrite_file else "a",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
