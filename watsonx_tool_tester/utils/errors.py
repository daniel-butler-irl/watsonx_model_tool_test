#!/usr/bin/env python3
"""
Error handling utilities for WatsonX Tool Tester.

This module provides functions for error extraction, formatting, and handling.
"""

import json
import re
from typing import Any, Dict, Optional


class ToolTestError(Exception):
    """Base exception class for WatsonX Tool Tester errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize with message and optional details.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ClientError(ToolTestError):
    """Exception raised for errors in the API clients."""

    pass


class CredentialError(ToolTestError):
    """Exception raised for authentication and credential issues."""

    pass


class ConfigurationError(ToolTestError):
    """Exception raised for configuration issues."""

    pass


class ToolExecutionError(ToolTestError):
    """Exception raised for errors during tool execution."""

    pass


def extract_error_details(response_text: str) -> str:
    """Extract detailed error information from API error responses.

    Args:
        response_text: The error response text from the API

    Returns:
        str: A more specific error message if available, otherwise the original text
    """
    try:
        # Check if it's HTML content
        if isinstance(response_text, str) and (
            response_text.startswith(("<html", "<!DOCTYPE", "<HTML"))
            or "<html>" in response_text
        ):
            # Return a simplified version for HTML responses
            if "404 Not Found" in response_text:
                return "404 Not Found: Endpoint not available"
            elif "403 Forbidden" in response_text:
                return "403 Forbidden: Access denied"
            else:
                # Extract just the status code or a generic message
                for status_code in ["404", "403", "401", "500"]:
                    if status_code in response_text:
                        return f"{status_code} Error: Server returned an HTML error page"
                return "HTML Error: Server returned an HTML response instead of JSON"

        # Try to parse as JSON
        error_json = json.loads(response_text)

        # Handle Watson.ai API error format
        if (
            "errors" in error_json
            and isinstance(error_json["errors"], list)
            and len(error_json["errors"]) > 0
        ):
            first_error = error_json["errors"][0]
            if "code" in first_error and "message" in first_error:
                return f"{first_error['code']}: {first_error['message']}"

        # Handle trace ID if present
        if "trace" in error_json:
            trace_id = error_json["trace"]
            return f"API Error (trace: {trace_id})"

        # Look for nested error objects
        if "error" in error_json:
            error_obj = error_json["error"]
            if isinstance(error_obj, dict) and "message" in error_obj:
                return error_obj["message"]
            elif isinstance(error_obj, str):
                return error_obj

        # If we got here, just return the entire error JSON
        return json.dumps(error_json, indent=2)

    except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
        # If parsing fails, return a portion of the original text, but clean up HTML
        if isinstance(response_text, str):
            # Check if it contains HTML
            if "<" in response_text and ">" in response_text:
                # Try to remove HTML tags
                cleaned_text = re.sub(r"<[^>]+>", " ", response_text)
                cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
                if len(cleaned_text) > 100:
                    return f"{cleaned_text[:100]}..."
                return cleaned_text
            elif len(response_text) > 100:
                return f"{response_text[:100]}..."
            else:
                return response_text

    # If all else fails, return a generic message
    return "Unknown API error (response could not be parsed)"


def format_error(error: Exception) -> str:
    """Format an exception for display.

    Args:
        error: The exception to format

    Returns:
        str: Formatted error message
    """
    if isinstance(error, ToolTestError):
        if error.details:
            details_str = "\n".join(
                f"  {k}: {v}" for k, v in error.details.items()
            )
            return f"{error.message}\nDetails:\n{details_str}"
        return error.message
    else:
        return str(error)
