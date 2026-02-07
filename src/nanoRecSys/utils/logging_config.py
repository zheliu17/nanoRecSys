# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging configuration for nanoRecSys."""

import logging
import os
import sys
from pathlib import Path


def setup_logger(
    name: str = "nanoRecSys",
    log_level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name (default: "nanoRecSys")
        log_level: Logging level (default: logging.INFO)
        log_to_file: Whether to save logs to file (default: False)
                     Can also be controlled via NANORECYS_LOG_TO_FILE env var
        log_dir: Directory to save logs. Default: ~/.nanoRecSys/logs
                 Can also be controlled via NANORECYS_LOG_DIR env var

    Returns:
        Configured logger instance
    """
    # Check environment variables
    log_to_file = os.getenv("NANORECYS_LOG_TO_FILE", str(log_to_file)).lower() == "true"
    env_log_dir = os.getenv("NANORECYS_LOG_DIR")
    if env_log_dir:
        log_dir = env_log_dir

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent propagation to root logger to avoid duplicate logs in Jupyter
    logger.propagate = False

    # Remove existing handlers to avoid duplicates (important for Jupyter notebook re-runs)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add console handler
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        # Use provided log_dir or default to ~/.nanoRecSys/logs
        if log_dir is None:
            log_path = Path.home() / ".nanoRecSys" / "logs"
        else:
            log_path = Path(log_dir)

        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path / "nanoRecSys.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "nanoRecSys") -> logging.Logger:
    """Get or create a logger instance. Auto-initializes on first call."""
    logger = logging.getLogger(name)

    # Auto-initialize if not already configured (no handlers)
    if not logger.handlers:
        setup_logger(name)
        logger = logging.getLogger(name)

    return logger
