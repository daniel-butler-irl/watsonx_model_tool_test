#!/usr/bin/env python3
"""
Formatting utilities for WatsonX Tool Tester.

This module provides functions for formatting test results and other
output in a user-friendly way.
"""

import json
from typing import Any, Dict, List

import click


def format_test_result(
    model_id: str,
    supports_tool_calls: bool,
    handles_response: bool,
    details: str,
    response_times: Dict[str, float],
) -> str:
    """Format a test result for display.

    Args:
        model_id: The ID of the model that was tested
        supports_tool_calls: Whether the model supports tool calls
        handles_response: Whether the model properly handles tool responses
        details: Additional details about the test
        response_times: Timing information for the test

    Returns:
        str: A formatted string describing the test result
    """
    status = _get_status_symbol(supports_tool_calls, handles_response)

    # Format response times
    timing_str = _format_timing(response_times)

    # Format full report
    result = (
        f"{status} Model: {model_id}\n"
        f"   Tool Calls: {'✓' if supports_tool_calls else '✗'}\n"
        f"   Response Handling: {'✓' if handles_response else '✗'}\n"
        f"   Details: {details}\n"
        f"   {timing_str}\n"
    )

    return result


def format_summary(
    results: List[Dict[str, Any]], sort_key: str = "name"
) -> str:
    """Format a summary of all test results.

    Args:
        results: List of test result dictionaries
        sort_key: The key to sort results by

    Returns:
        str: A formatted summary of test results
    """
    # Count results by status
    total = len(results)
    success_count = sum(
        1
        for r in results
        if r.get("tool_call_support", False)
        and r.get("handles_response", False)
    )
    partial_count = sum(
        1
        for r in results
        if r.get("tool_call_support", False)
        and not r.get("handles_response", False)
    )
    failure_count = sum(
        1 for r in results if not r.get("tool_call_support", False)
    )

    # Format summary
    summary = (
        f"\n{'=' * 50}\n"
        f"SUMMARY\n"
        f"{'=' * 50}\n"
        f"Total models tested: {total}\n"
        f"✅ Full success (calls tool and uses result): {success_count}\n"
        f"⚠️ Partial success (calls tool but ignores result): {partial_count}\n"
        f"❌ Failure (does not call tool): {failure_count}\n"
        f"{'=' * 50}\n"
    )

    return summary


def format_json_output(results: List[Dict[str, Any]]) -> str:
    """Format test results as JSON.

    Args:
        results: List of test result dictionaries

    Returns:
        str: JSON string of test results
    """
    # Convert to a format suitable for JSON output
    output_results = []

    for result in results:
        output_result = {
            "model_id": result.get("model"),
            "supports_tool_calls": result.get("tool_call_support", False),
            "handles_response": result.get("handles_response", False),
            "details": result.get("details", ""),
            "timings": {
                "tool_call_time_seconds": result.get("response_times", {}).get(
                    "tool_call_time"
                ),
                "response_processing_time_seconds": result.get(
                    "response_times", {}
                ).get("response_processing_time"),
                "total_time_seconds": result.get("response_times", {}).get(
                    "total_time"
                ),
            },
            "timestamp": result.get("timestamp"),
        }
        output_results.append(output_result)

    # Format as pretty JSON
    return json.dumps({"results": output_results}, indent=2)


def print_result_with_color(result: str):
    """Print a test result with appropriate coloring.

    Args:
        result: The formatted test result string
    """
    # Colorize the result based on success/failure indicators
    lines = result.split("\n")

    for line in lines:
        if line.startswith("✅"):
            click.secho(line, fg="green")
        elif line.startswith("⚠️"):
            click.secho(line, fg="yellow")
        elif line.startswith("❌"):
            click.secho(line, fg="red")
        elif line.strip().startswith("Tool Calls: ✓"):
            click.secho(line, fg="green")
        elif line.strip().startswith("Tool Calls: ✗"):
            click.secho(line, fg="red")
        elif line.strip().startswith("Response Handling: ✓"):
            click.secho(line, fg="green")
        elif line.strip().startswith("Response Handling: ✗"):
            click.secho(line, fg="yellow")
        else:
            click.echo(line)


def format_tool_call_success(success: bool) -> str:
    """Format a tool call success indicator.

    Args:
        success: Whether the tool call was successful

    Returns:
        str: Formatted success indicator
    """
    if success:
        return click.style("✅ SUPPORTED", fg="green")
    else:
        return click.style("❌ NOT SUPPORTED", fg="red")


def format_response_success(success: bool) -> str:
    """Format a response handling success indicator.

    Args:
        success: Whether the response was handled correctly

    Returns:
        str: Formatted success indicator
    """
    if success:
        return click.style("✅ CORRECT", fg="green")
    else:
        return click.style("❌ INCORRECT", fg="red")


def format_response_time(seconds: float) -> str:
    """Format a response time.

    Args:
        seconds: Response time in seconds

    Returns:
        str: Formatted response time
    """
    color = "green"
    if seconds > 5.0:
        color = "red"
    elif seconds > 2.0:
        color = "yellow"

    return click.style(f"{seconds:.2f}s", fg=color)


def format_credentials_status(valid: bool) -> str:
    """Format an API credentials status indicator.

    Args:
        valid: Whether the credentials are valid

    Returns:
        str: Formatted status indicator
    """
    if valid:
        return click.style("✓ Valid", fg="green")
    else:
        return click.style("✗ Invalid", fg="red")


def format_model_count(count: int) -> str:
    """Format a model count.

    Args:
        count: Number of models

    Returns:
        str: Formatted model count with color
    """
    if count == 0:
        return click.style(str(count), fg="red")
    elif count < 5:
        return click.style(str(count), fg="yellow")
    else:
        return click.style(str(count), fg="green")


def format_percentage(value: float) -> str:
    """Format a percentage.

    Args:
        value: Percentage value (0-100)

    Returns:
        str: Formatted percentage with color
    """
    formatted = f"{value:.1f}%"

    if value < 25:
        return click.style(formatted, fg="red")
    elif value < 50:
        return click.style(formatted, fg="yellow")
    elif value < 75:
        return click.style(formatted, fg="green")
    else:
        return click.style(formatted, fg="bright_green")


def format_error_message(message: str) -> str:
    """Format an error message.

    Args:
        message: The error message

    Returns:
        str: Formatted error message with color
    """
    return click.style(f"Error: {message}", fg="red", bold=True)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length.

    Args:
        text: The text to truncate
        max_length: Maximum length before truncation

    Returns:
        str: Truncated text
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return f"{text[:max_length-3]}..."


def pluralize(count: int, singular: str, plural: str) -> str:
    """Return singular or plural form based on count.

    Args:
        count: The count to check
        singular: Singular form of the word
        plural: Plural form of the word

    Returns:
        str: Appropriate form of the word
    """
    return singular if count == 1 else plural


def _get_status_symbol(
    supports_tool_calls: bool, handles_response: bool
) -> str:
    """Get a status symbol based on test results.

    Args:
        supports_tool_calls: Whether the model supports tool calls
        handles_response: Whether the model properly handles tool responses

    Returns:
        str: A status symbol (emoji)
    """
    if supports_tool_calls and handles_response:
        return "✅"
    elif supports_tool_calls:
        return "⚠️"
    else:
        return "❌"


def _format_timing(response_times: Dict[str, float]) -> str:
    """Format timing information for display.

    Args:
        response_times: Timing information for the test

    Returns:
        str: A formatted string describing timing information
    """
    # Extract timing values
    tool_call_time = response_times.get("tool_call_time")
    response_processing_time = response_times.get("response_processing_time")
    total_time = response_times.get("total_time")

    # Format timing string
    timing_parts = []

    if tool_call_time is not None:
        timing_parts.append(f"Tool call: {tool_call_time:.2f}s")

    if response_processing_time is not None:
        timing_parts.append(
            f"Response processing: {response_processing_time:.2f}s"
        )

    if total_time is not None:
        timing_parts.append(f"Total: {total_time:.2f}s")

    if timing_parts:
        return "Timing: " + ", ".join(timing_parts)
    else:
        return "Timing: Not available"


def format_dict(data: Dict[str, Any], indent: int = 2) -> str:
    """Format a dictionary as a JSON string.

    Args:
        data: The dictionary to format
        indent: Number of spaces for indentation

    Returns:
        str: Formatted JSON string
    """
    return json.dumps(data, indent=indent)
