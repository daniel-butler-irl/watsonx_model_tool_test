#!/usr/bin/env python3
"""
Tests for the error handling utility module.

This module contains tests for the custom exception classes used
in the WatsonX Tool Tester package.
"""

import json

from watsonx_tool_tester.utils.errors import (
    ClientError,
    ConfigurationError,
    CredentialError,
    ToolExecutionError,
    ToolTestError,
    extract_error_details,
    format_error,
)


def test_extract_error_details_watson_format():
    """Test extracting error details from Watson.ai API error format."""
    error_response = json.dumps(
        {
            "errors": [
                {
                    "code": "invalid_api_key",
                    "message": "Invalid API key provided",
                }
            ]
        }
    )

    result = extract_error_details(error_response)

    assert result == "invalid_api_key: Invalid API key provided"


def test_extract_error_details_trace():
    """Test extracting error details with trace ID."""
    error_response = json.dumps(
        {"trace": "abc123", "message": "An error occurred"}
    )

    result = extract_error_details(error_response)

    assert result == "API Error (trace: abc123)"


def test_extract_error_details_nested_error():
    """Test extracting error details from nested error object."""
    error_response = json.dumps({"error": {"message": "Model not found"}})

    result = extract_error_details(error_response)

    assert result == "Model not found"


def test_extract_error_details_string_error():
    """Test extracting error details from string error object."""
    error_response = json.dumps({"error": "Rate limit exceeded"})

    result = extract_error_details(error_response)

    assert result == "Rate limit exceeded"


def test_extract_error_details_html():
    """Test extracting error details from HTML response."""
    html_response = (
        "<html><body><h1>404 Not Found</h1>"
        "<p>The requested URL was not found on this server.</p></body></html>"
    )

    result = extract_error_details(html_response)

    assert "404 Not Found" in result


def test_extract_error_details_plain_text():
    """Test extracting error details from plain text."""
    plain_text = "Unknown server error occurred"

    result = extract_error_details(plain_text)

    assert result == "Unknown server error occurred"


def test_extract_error_details_long_text():
    """Test extracting error details from long text (should truncate)."""
    long_text = "A" * 200

    result = extract_error_details(long_text)

    assert len(result) <= 103  # 100 chars + "..."
    assert result.endswith("...")


def test_tool_test_error_basic():
    """Test basic ToolTestError functionality."""
    error = ToolTestError("Test error")

    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.details == {}


def test_tool_test_error_with_details():
    """Test ToolTestError with details."""
    details = {"code": 404, "reason": "Not found"}
    error = ToolTestError("Resource not found", details)

    assert str(error) == "Resource not found"
    assert error.message == "Resource not found"
    assert error.details == details


def test_error_subclasses():
    """Test that error subclasses are correctly defined."""
    assert issubclass(ClientError, ToolTestError)
    assert issubclass(CredentialError, ToolTestError)
    assert issubclass(ConfigurationError, ToolTestError)
    assert issubclass(ToolExecutionError, ToolTestError)


def test_format_error_tool_test_error():
    """Test formatting a ToolTestError."""
    error = ToolTestError("Test error", {"code": 500, "info": "Server error"})
    formatted = format_error(error)

    assert "Test error" in formatted
    assert "code: 500" in formatted
    assert "info: Server error" in formatted


def test_format_error_standard_exception():
    """Test formatting a standard exception."""
    error = ValueError("Invalid value")
    formatted = format_error(error)

    assert formatted == "Invalid value"


def test_credential_error():
    """Test CredentialError class."""
    error = CredentialError("Invalid API key")
    assert str(error) == "Invalid API key"
    assert isinstance(error, ToolTestError)

    error_with_details = CredentialError(
        "Invalid API key", {"source": "config"}
    )
    assert error_with_details.details["source"] == "config"


def test_configuration_error():
    """Test ConfigurationError class."""
    error = ConfigurationError("Missing config file")
    assert str(error) == "Missing config file"
    assert isinstance(error, ToolTestError)

    error_with_details = ConfigurationError(
        "Invalid config", {"path": "/etc/config.yaml"}
    )
    assert error_with_details.details["path"] == "/etc/config.yaml"


def test_client_error():
    """Test ClientError class."""
    error = ClientError("Connection failed")
    assert str(error) == "Connection failed"
    assert isinstance(error, ToolTestError)

    error_with_details = ClientError("API error", {"status_code": 500})
    assert error_with_details.details["status_code"] == 500


def test_tool_execution_error():
    """Test ToolExecutionError class."""
    error = ToolExecutionError("Tool failed")
    assert str(error) == "Tool failed"
    assert isinstance(error, ToolTestError)

    error_with_details = ToolExecutionError(
        "Invalid parameters", {"tool": "hello_world"}
    )
    assert error_with_details.details["tool"] == "hello_world"
