#!/usr/bin/env python3
"""
Logging utilities for WatsonX Tool Tester.

This module provides functions for setting up and managing logging.
"""

import datetime
import logging
import os
import sys
from typing import Dict, Optional, Union


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Optional name for the logger. If None, returns the root logger.

    Returns:
        logging.Logger: The logger instance
    """
    if name:
        logger_name = f"watsonx_tool_tester.{name}"
    else:
        logger_name = "watsonx_tool_tester"

    return logging.getLogger(logger_name)


def get_log_level(level: Union[str, int]) -> int:
    """Convert a log level string to the corresponding logging level.

    Args:
        level: Log level as string (e.g., 'INFO') or int

    Returns:
        int: The logging level constant
    """
    if isinstance(level, int):
        return level

    level_map: Dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_map.get(level.upper(), logging.INFO)


def setup_logger(
    debug: bool = False,
    log_dir: str = "tool_test_logs",
    log_level: str = "INFO",
    file_logging: bool = False,
) -> str:
    """Set up logging with file and console handlers.

    Args:
        debug: Whether to enable debug logging (overrides log_level)
        log_dir: Directory to store log files
        log_level: Logging level as string (e.g., "INFO")
        file_logging: Whether to enable logging to a file

    Returns:
        str: Path to the log file or empty string if file_logging is False
    """
    # Determine log level
    if debug:
        level = logging.DEBUG
    else:
        level = get_log_level(log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_file = ""
    if file_logging:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create a timestamp for the log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"watsonx_tool_test_{timestamp}.log")

        # Create file handler for all logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Log starting message about file
        root_logger.info(f"Logging to {log_file}")

    # Create console handler with appropriate log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(message)s"
    )  # Simpler format for console
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if level == logging.DEBUG:
        root_logger.debug("Debug logging enabled")

    return log_file


# Alias for backward compatibility
setup_logging = setup_logger


def log_with_level(logger: logging.Logger, level: int, message: str) -> None:
    """Log a message with the specified level.

    Args:
        logger: The logger to use
        level: The logging level (e.g., logging.INFO)
        message: The message to log
    """
    logger.log(level, message)


def log_dict(
    logger: logging.Logger,
    level: int,
    message: str,
    data: dict,
    indent: int = 2,
) -> None:
    """Log a dictionary with the specified level.

    Args:
        logger: The logger to use
        level: The logging level (e.g., logging.INFO)
        message: The message to log before the dictionary
        data: The dictionary to log
        indent: Number of spaces for indentation
    """
    import json

    formatted_data = json.dumps(data, indent=indent)
    logger.log(level, f"{message}\n{formatted_data}")
