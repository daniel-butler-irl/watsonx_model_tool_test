#!/usr/bin/env python3
"""
Tests for the ResultHandler class.

This module contains tests for the ResultHandler functionality.
"""

import io
import json
import sys

import pytest

from watsonx_tool_tester.testers.result_handler import ResultHandler


@pytest.fixture
def test_results():
    """Create test results fixture."""
    return [
        {
            "model": "model-1",
            "tool_call_support": True,
            "handles_response": True,
            "details": "Success",
            "response_times": {
                "tool_call_time": 1.0,
                "response_processing_time": 0.5,
                "total_time": 1.5,
            },
        },
        {
            "model": "model-2",
            "tool_call_support": True,
            "handles_response": False,
            "details": "Called tool but didn't use result",
            "response_times": {
                "tool_call_time": 2.0,
                "response_processing_time": 1.0,
                "total_time": 3.0,
            },
        },
        {
            "model": "model-3",
            "tool_call_support": False,
            "handles_response": False,
            "details": "Did not call tool",
            "response_times": {
                "tool_call_time": None,
                "response_processing_time": None,
                "total_time": 2.0,
            },
        },
    ]


@pytest.fixture
def result_handler():
    """Create a ResultHandler instance."""
    return ResultHandler()


def test_sort_results(result_handler, test_results):
    """Test sorting results by different keys."""
    # Test sorting by name
    sorted_by_name = result_handler.sort_results(test_results, "name")
    assert sorted_by_name[0]["model"] == "model-1"
    assert sorted_by_name[1]["model"] == "model-2"
    assert sorted_by_name[2]["model"] == "model-3"

    # Test sorting by tool_call_time
    sorted_by_tool_time = result_handler.sort_results(
        test_results, "tool_call_time"
    )
    assert sorted_by_tool_time[0]["model"] == "model-1"
    assert sorted_by_tool_time[1]["model"] == "model-2"
    assert (
        sorted_by_tool_time[2]["model"] == "model-3"
    )  # None sorts to the end

    # Test sorting by response_time
    sorted_by_response_time = result_handler.sort_results(
        test_results, "response_time"
    )
    assert sorted_by_response_time[0]["model"] == "model-1"
    assert sorted_by_response_time[1]["model"] == "model-2"
    assert (
        sorted_by_response_time[2]["model"] == "model-3"
    )  # None sorts to the end

    # Test sorting by total_time
    sorted_by_total_time = result_handler.sort_results(
        test_results, "total_time"
    )
    assert sorted_by_total_time[0]["model"] == "model-1"
    assert sorted_by_total_time[1]["model"] == "model-3"
    assert sorted_by_total_time[2]["model"] == "model-2"

    # Test invalid sort key (falls back to name)
    sorted_by_invalid = result_handler.sort_results(
        test_results, "invalid_key"
    )
    assert sorted_by_invalid[0]["model"] == "model-1"
    assert sorted_by_invalid[1]["model"] == "model-2"
    assert sorted_by_invalid[2]["model"] == "model-3"


def test_format_table(result_handler, test_results):
    """Test formatting results as a table."""
    table = result_handler.format_table(test_results)

    # The table should be a string
    assert isinstance(table, str)

    # The table should include all model names
    assert "model-1" in table
    assert "model-2" in table
    assert "model-3" in table

    # The table should include headers
    assert "MODEL" in table
    assert "TOOL SUPPORT" in table
    assert "HANDLED" in table
    assert "CALL TIME" in table
    assert "RESP TIME" in table
    assert "TOTAL TIME" in table
    assert "DETAILS" in table

    assert "+" not in table
    assert "|" not in table

    # The table should include timing information
    assert "1.00s" in table  # tool_call_time for model-1
    assert "0.50s" in table  # response_time for model-1
    assert "1.50s" in table  # total_time for model-1
    assert "N/A" in table  # null timing values for model-3

    # Check for long details truncation
    # Create a result with very long details
    result_with_long_details = {
        "model": "model-long",
        "tool_call_support": True,
        "handles_response": True,
        "details": "This is a very long detail message that should be truncated because it exceeds the maximum display width for details",
        "response_times": {
            "tool_call_time": 1.0,
            "response_processing_time": 0.5,
            "total_time": 1.5,
        },
    }

    # Add the result with long details to the test results
    test_results_with_long = test_results + [result_with_long_details]

    # Get the table with the long details result
    table_with_long = result_handler.format_table(test_results_with_long)

    # Check that the long details are truncated with ellipsis
    assert "..." in table_with_long


def test_summarize_results(result_handler, test_results):
    """Test calculating summary statistics."""
    summary = result_handler.summarize_results(test_results)

    # Check counts
    assert summary["total_count"] == 3
    assert summary["supported_count"] == 2
    assert summary["handles_response_count"] == 1

    # Check averages
    assert summary["avg_tool_time"] == 1.5  # (1.0 + 2.0) / 2
    assert summary["avg_response_time"] == 0.75  # (0.5 + 1.0) / 2
    assert summary["avg_total_time"] == pytest.approx(
        2.17, 0.01
    )  # (1.5 + 3.0 + 2.0) / 3

    # Check fastest model
    assert summary["fastest_model"]["model"] == "model-1"
    assert summary["fastest_model"]["time"] == 1.5


def test_print_summary(result_handler, test_results):
    """Test printing a summary of results."""
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Print summary
    result_handler.print_summary(test_results)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Check output
    output = captured_output.getvalue()

    # The output should include all model names
    assert "model-1" in output
    assert "model-2" in output
    assert "model-3" in output

    # The output should include summary statistics
    assert "Total models tested: 3" in output
    assert "2 out of 3 models" in output
    assert "1 out of 2 models" in output

    # The output should include performance information
    assert "Average tool call time" in output
    assert "Average response processing time" in output
    assert "Average total time" in output
    assert "Fastest model" in output


def test_format_json_output(result_handler, test_results):
    """Test formatting results as JSON."""
    # Test without raw responses
    json_output = result_handler.format_json_output(
        test_results, include_raw=False
    )

    # The output should be a valid JSON string
    data = json.loads(json_output)

    # The JSON should have results and summary sections
    assert "results" in data
    assert "summary" in data

    # The results should include all models
    assert len(data["results"]) == 3
    assert data["results"][0]["model"] == "model-1"
    assert data["results"][1]["model"] == "model-2"
    assert data["results"][2]["model"] == "model-3"

    # The results should include timing information
    assert data["results"][0]["timings"]["tool_call_time_seconds"] == 1.0
    assert (
        data["results"][0]["timings"]["response_processing_time_seconds"]
        == 0.5
    )
    assert data["results"][0]["timings"]["total_time_seconds"] == 1.5

    # Test with raw responses
    test_results_with_raw = test_results.copy()
    for i, result in enumerate(test_results_with_raw):
        result["raw_response"] = {"id": f"response-{i+1}"}

    json_output = result_handler.format_json_output(
        test_results_with_raw, include_raw=True
    )
    data = json.loads(json_output)

    # The results should include raw responses
    assert "raw_response" in data["results"][0]
    assert data["results"][0]["raw_response"]["id"] == "response-1"
