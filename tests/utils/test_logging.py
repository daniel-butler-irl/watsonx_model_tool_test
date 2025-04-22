#!/usr/bin/env python3
"""
Tests for logging utilities.

This module contains tests for the logging utility functions.
"""

import logging
from unittest import mock

import pytest

from watsonx_tool_tester.utils.logging import (
    get_log_level,
    get_logger,
    log_dict,
    log_with_level,
    setup_logger,
)


def test_get_logger():
    """Test getting a logger with the specified name."""
    # Test getting a namespaced logger
    logger = get_logger("test")
    assert logger.name == "watsonx_tool_tester.test"

    # Test getting the root logger
    logger = get_logger()
    assert logger.name == "watsonx_tool_tester"


def test_get_log_level():
    """Test converting log level strings to logging levels."""
    # Test string levels
    assert get_log_level("DEBUG") == logging.DEBUG
    assert get_log_level("INFO") == logging.INFO
    assert get_log_level("WARNING") == logging.WARNING
    assert get_log_level("WARN") == logging.WARNING
    assert get_log_level("ERROR") == logging.ERROR
    assert get_log_level("CRITICAL") == logging.CRITICAL
    assert get_log_level("UNKNOWN") == logging.INFO  # Default

    # Test numeric levels
    assert get_log_level(10) == 10
    assert get_log_level(20) == 20


@pytest.fixture
def cleanup_logger():
    """Remove all handlers from the root logger after tests."""
    yield
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def test_setup_logger(tmp_path, cleanup_logger):
    """Test setting up a logger with file and console handlers."""
    log_dir = str(tmp_path / "logs")

    # Mock file operations completely
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.makedirs"
    ), mock.patch("logging.FileHandler") as mock_file_handler, mock.patch(
        "logging.StreamHandler"
    ) as mock_stream_handler:

        # Set up mock handlers
        mock_file = mock.MagicMock()
        mock_file.level = logging.DEBUG
        mock_file_handler.return_value = mock_file

        mock_console = mock.MagicMock()
        mock_console.level = logging.INFO
        mock_stream_handler.return_value = mock_console

        # Test with file logging enabled
        log_file = setup_logger(
            debug=True, log_dir=log_dir, log_level="INFO", file_logging=True
        )

        assert log_file != ""
        mock_file_handler.assert_called_once()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG  # Debug overrides log_level

        # Reset for next test
        root_logger.handlers = []
        mock_file_handler.reset_mock()

        # Test with file logging disabled
        log_file = setup_logger(
            debug=False, log_dir=log_dir, log_level="INFO", file_logging=False
        )

        assert (
            log_file == ""
        )  # Should be empty string when file_logging is False
        mock_file_handler.assert_not_called()  # No file handler should be created

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO


def test_log_with_level():
    """Test logging a message with a specific level."""
    logger = mock.MagicMock()
    log_with_level(logger, logging.INFO, "Test message")
    logger.log.assert_called_once_with(logging.INFO, "Test message")


def test_log_dict():
    """Test logging a dictionary."""
    logger = mock.MagicMock()
    data = {"key": "value"}
    log_dict(logger, logging.INFO, "Test message", data)

    # Check that logger.log was called with the right level
    assert logger.log.call_args[0][0] == logging.INFO

    # Check that the message and JSON-formatted data were included
    log_message = logger.log.call_args[0][1]
    assert "Test message" in log_message
    assert '"key": "value"' in log_message
