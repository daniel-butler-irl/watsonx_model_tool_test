#!/usr/bin/env python3
"""
Pytest configuration for watsonx_tool_tester tests.

This module sets up global fixtures for testing.
"""

from unittest import mock

import pytest
import requests  # noqa: F401


@pytest.fixture(autouse=True)
def mock_requests_globally():
    """
    Mock all HTTP requests to prevent actual API calls during tests.

    This fixture is applied automatically to all tests.
    """
    with mock.patch("requests.get"), mock.patch("requests.post"), mock.patch(
        "requests.put"
    ), mock.patch("requests.delete"), mock.patch("requests.patch"):
        yield


@pytest.fixture(autouse=True)
def prevent_actual_file_operations():
    """
    Prevent actual file operations during tests unless explicitly allowed.

    This fixture doesn't mock os.path.exists and other file check operations,
    but does mock file writing operations.
    """
    real_open = open

    # Allow reading actual files but mock writing
    def mock_open(*args, **kwargs):
        if len(args) > 1 and "w" in args[1]:
            mock_file = mock.MagicMock()
            mock_file.__enter__ = mock.MagicMock(return_value=mock_file)
            mock_file.__exit__ = mock.MagicMock()
            mock_file.write = mock.MagicMock()
            return mock_file
        return real_open(*args, **kwargs)

    with mock.patch("builtins.open", mock_open):
        yield
