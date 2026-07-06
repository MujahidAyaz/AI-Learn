"""
logger.py
=========

Centralized logging configuration for the project.

Every module should obtain a logger using:

    from src.logger import get_logger

    logger = get_logger(__name__)

Call configure_logger() exactly once at application startup.
"""

from __future__ import annotations

import logging

from .config import (
    LOG_FILE_NAME,
    LOG_LEVEL,
    LOGS_DIR,
)


def configure_logger() -> None:
    """
    Configure the root logger.

    This function should be called once at the start of the
    application before any logging occurs.
    """

    log_file = LOGS_DIR / LOG_FILE_NAME

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                log_file,
                encoding="utf-8",
                delay=True,
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str | None, default=None
        Logger name. Usually __name__.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    return logging.getLogger(name)