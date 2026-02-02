"""Logging configuration for nanoRecSys."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "nanoRecSys", log_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name: Logger name (default: "nanoRecSys")
        log_level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent propagation to root logger to avoid duplicate logs in Jupyter
    logger.propagate = False

    # Remove existing handlers to avoid duplicates (important for Jupyter notebook re-runs)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(logs_dir / "nanoRecSys.log")
    file_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "nanoRecSys") -> logging.Logger:
    """Get or create a logger instance."""
    return logging.getLogger(name)
