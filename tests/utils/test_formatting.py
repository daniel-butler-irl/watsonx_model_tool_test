#!/usr/bin/env python3
"""
Tests for the formatting utilities.

This module contains tests for the formatting functions used for output display.
"""

import json
from unittest.mock import patch

import click

from watsonx_tool_tester.utils.formatting import (
    format_credentials_status,
    format_error_message,
    format_json_output,
    format_model_count,
    format_percentage,
    format_response_success,
    format_response_time,
    format_summary,
    format_test_result,
    format_tool_call_success,
    pluralize,
    print_result_with_color,
    truncate_text,
)


def test_format_json_output():
    """Test formatting results as a JSON string."""
    results = [
        {
            "model": "model1",
            "tool_call_support": True,
            "handles_response": True,
            "details": "Worked well",
            "response_times": {
                "tool_call_time": 1.2,
                "response_processing_time": 0.5,
                "total_time": 1.7,
            },
            "timestamp": "2023-01-01T12:00:00Z",
        }
    ]

    result = format_json_output(results)

    # Check if the result is valid JSON and has expected structure
    parsed = json.loads(result)
    assert "results" in parsed
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["model_id"] == "model1"
    assert parsed["results"][0]["supports_tool_calls"] is True


def test_format_tool_call_success():
    """Test formatting tool call success status."""
    with patch.object(click, "style", return_value="styled_text"):
        success = format_tool_call_success(True)
        failure = format_tool_call_success(False)

    assert success == "styled_text"
    assert failure == "styled_text"


def test_format_response_success():
    """Test formatting response handling success status."""
    with patch.object(click, "style", return_value="styled_text"):
        correct = format_response_success(True)
        incorrect = format_response_success(False)

    assert correct == "styled_text"
    assert incorrect == "styled_text"


def test_format_response_time():
    """Test formatting response time in seconds."""
    with patch.object(click, "style", return_value="styled_text"):
        result = format_response_time(1.2345)

    assert result == "styled_text"


def test_truncate_text():
    """Test truncating text to a maximum length."""
    short_text = "Short text"
    long_text = "This is a very long text that should be truncated"
    empty_text = ""

    # Short text should remain unchanged
    assert truncate_text(short_text, 20) == short_text

    # Long text should be truncated with ellipsis
    truncated = truncate_text(long_text, 20)
    assert len(truncated) == 20
    assert truncated.endswith("...")
    assert truncated.startswith("This is a very")

    # Empty text should remain empty
    assert truncate_text(empty_text) == ""

    # Default max_length should be 100
    very_long = "x" * 200
    default_truncated = truncate_text(very_long)
    assert len(default_truncated) == 100
    assert default_truncated.endswith("...")


def test_pluralize():
    """Test pluralizing words based on count."""
    assert pluralize(0, "item", "items") == "items"
    assert pluralize(1, "item", "items") == "item"
    assert pluralize(2, "item", "items") == "items"


def test_format_test_result():
    """Test formatting a single test result."""
    result = format_test_result(
        model_id="model1",
        supports_tool_calls=True,
        handles_response=True,
        details="Test details",
        response_times={
            "tool_call_time": 1.2,
            "response_processing_time": 0.5,
            "total_time": 1.7,
        },
    )

    assert "model1" in result
    assert "Tool Calls: ✓" in result
    assert "Response Handling: ✓" in result
    assert "Test details" in result


def test_format_summary():
    """Test formatting a summary of all test results."""
    results = [
        {
            "model": "model1",
            "tool_call_support": True,
            "handles_response": True,
        },
        {
            "model": "model2",
            "tool_call_support": True,
            "handles_response": False,
        },
        {
            "model": "model3",
            "tool_call_support": False,
            "handles_response": False,
        },
    ]

    summary = format_summary(results)

    assert "SUMMARY" in summary
    assert "Total models tested: 3" in summary
    assert "Full success" in summary
    assert "Partial success" in summary
    assert "Failure" in summary


def test_print_result_with_color():
    """Test printing result with color."""
    with patch.object(click, "secho") as mock_secho, patch.object(
        click, "echo"
    ) as mock_echo:
        result = "✅ Model: test\n   Tool Calls: ✓\n   Other line"
        print_result_with_color(result)

        # Should call secho for colored lines and echo for normal lines
        assert mock_secho.call_count >= 2
        assert mock_echo.call_count >= 1


def test_format_credentials_status():
    """Test formatting credentials status."""
    with patch.object(click, "style", return_value="styled_text"):
        valid = format_credentials_status(True)
        invalid = format_credentials_status(False)

    assert valid == "styled_text"
    assert invalid == "styled_text"


def test_format_model_count():
    """Test formatting model count with color."""
    with patch.object(click, "style", return_value="styled_text"):
        zero_count = format_model_count(0)
        low_count = format_model_count(3)
        high_count = format_model_count(10)

    assert zero_count == "styled_text"
    assert low_count == "styled_text"
    assert high_count == "styled_text"


def test_format_percentage():
    """Test formatting percentage with color."""
    with patch.object(click, "style", return_value="styled_text"):
        low_percent = format_percentage(10.5)
        mid_percent = format_percentage(45.0)
        high_percent = format_percentage(80.0)

    assert low_percent == "styled_text"
    assert mid_percent == "styled_text"
    assert high_percent == "styled_text"


def test_format_error_message():
    """Test formatting error message with color."""
    with patch.object(click, "style", return_value="styled_text"):
        message = format_error_message("Something went wrong")

    assert message == "styled_text"
