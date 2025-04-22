#!/usr/bin/env python3
"""
Result handler for WatsonX Tool Tester.

This module provides the ResultHandler class for processing, sorting,
and formatting tool test results.
"""

import datetime
import json
from typing import Any, Dict, List

import click
from tabulate import tabulate

from watsonx_tool_tester.utils.formatting import (
    format_response_success,
    format_response_time,
    format_tool_call_success,
)


class ResultHandler:
    """Handler for processing and formatting tool test results.

    This class provides methods for sorting, formatting, and summarizing
    the results of tool tests against various models.
    """

    def sort_results(
        self, results: List[Dict[str, Any]], sort_key: str
    ) -> List[Dict[str, Any]]:
        """Sort test results by the specified key.

        Args:
            results: List of test result dictionaries
            sort_key: Key to sort by ('name', 'tool_call_time', 'response_time', 'total_time')

        Returns:
            List[Dict[str, Any]]: Sorted results
        """

        # Define a function to get the sort value based on the sort key
        def get_sort_value(result: Dict[str, Any]) -> Any:
            if sort_key == "name":
                return result["model"]
            elif sort_key == "tool_call_time" and "response_times" in result:
                # Use a large number for None to sort to the end
                return result["response_times"].get("tool_call_time") or float(
                    "inf"
                )
            elif sort_key == "response_time" and "response_times" in result:
                return result["response_times"].get(
                    "response_processing_time"
                ) or float("inf")
            elif sort_key == "total_time" and "response_times" in result:
                return result["response_times"].get("total_time") or float(
                    "inf"
                )
            else:
                # Default to sorting by name
                return result["model"]

        # Sort the results
        return sorted(results, key=get_sort_value)

    def format_table(self, results: List[Dict[str, Any]]) -> str:
        """Format test results as a table.

        Args:
            results: List of test result dictionaries

        Returns:
            str: Formatted table
        """
        table_data = []

        for result in results:
            model_id = result["model"]
            supports_tool_call = result["tool_call_support"]
            handles_response = result["handles_response"]

            # Format tool call support and response handling status
            tool_call_status = format_tool_call_success(supports_tool_call)
            response_status = format_response_success(handles_response)

            # Get timing information
            times = result.get("response_times", {})
            tool_call_time = times.get("tool_call_time")
            response_time = times.get("response_processing_time")
            total_time = times.get("total_time")

            # Format timing information
            tool_call_time_str = (
                format_response_time(tool_call_time)
                if tool_call_time
                else "N/A"
            )
            response_time_str = (
                format_response_time(response_time) if response_time else "N/A"
            )
            total_time_str = (
                format_response_time(total_time) if total_time else "N/A"
            )

            # Truncate details to keep table width reasonable
            details = result.get("details", "")
            if len(details) > 80:  # Truncate long details
                details = details[:77] + "..."

            # Add row to table
            table_data.append(
                [
                    model_id,
                    tool_call_status,
                    response_status,
                    tool_call_time_str,
                    response_time_str,
                    total_time_str,
                    details,
                ]
            )

        # Define table headers
        headers = [
            "MODEL",
            "TOOL SUPPORT",
            "HANDLED",  # Shortened from "TOOL RESULT HANDLED"
            "CALL TIME",  # Shortened from "TOOL CALL TIME"
            "RESP TIME",  # Shortened from "RESPONSE TIME"
            "TOTAL TIME",
            "DETAILS",
        ]

        # Generate table with a more compact format
        return tabulate(table_data, headers=headers, tablefmt="simple")

    def summarize_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for test results.

        Args:
            results: List of test result dictionaries

        Returns:
            Dict[str, Any]: Summary statistics
        """
        total_count = len(results)
        supported_count = sum(
            1 for r in results if r.get("tool_call_support", False)
        )
        handles_response_count = sum(
            1
            for r in results
            if r.get("tool_call_support", False)
            and r.get("handles_response", False)
        )

        # Calculate average times, excluding None values
        tool_call_times = [
            r.get("response_times", {}).get("tool_call_time")
            for r in results
            if r.get("response_times", {}).get("tool_call_time") is not None
        ]

        response_times = [
            r.get("response_times", {}).get("response_processing_time")
            for r in results
            if r.get("response_times", {}).get("response_processing_time")
            is not None
        ]

        total_times = [
            r.get("response_times", {}).get("total_time")
            for r in results
            if r.get("response_times", {}).get("total_time") is not None
        ]

        # Calculate averages
        avg_tool_time = (
            sum(tool_call_times) / len(tool_call_times)
            if tool_call_times
            else 0
        )
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        avg_total_time = (
            sum(total_times) / len(total_times) if total_times else 0
        )

        # Find fastest model (by total time)
        fastest_model = None
        fastest_time = float("inf")

        for result in results:
            total_time = result.get("response_times", {}).get("total_time")
            if total_time is not None and total_time < fastest_time:
                fastest_time = total_time
                fastest_model = result["model"]

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_count": total_count,
            "supported_count": supported_count,
            "handles_response_count": handles_response_count,
            "avg_tool_time": avg_tool_time,
            "avg_response_time": avg_response_time,
            "avg_total_time": avg_total_time,
            "fastest_model": (
                {
                    "model": fastest_model,
                    "time": fastest_time,
                }
                if fastest_model
                else None
            ),
        }

    def print_summary(
        self, results: List[Dict[str, Any]], sort_key: str = "name"
    ) -> None:
        """Print a summary of test results.

        Args:
            results: List of test result dictionaries
            sort_key: Key to sort by ('name', 'tool_call_time', 'response_time', 'total_time')
        """
        # Sort results
        sorted_results = self.sort_results(results, sort_key)

        # Print table
        table = self.format_table(sorted_results)
        print("\n" + table)

        # Generate summary statistics
        summary = self.summarize_results(results)

        # Print summary
        click.echo("\n=== Summary ===")
        click.echo(f"Total models tested: {summary['total_count']}")

        tool_support_percent = (
            (summary["supported_count"] / summary["total_count"]) * 100
            if summary["total_count"] > 0
            else 0
        )
        click.echo(
            f"{summary['supported_count']} out of {summary['total_count']} models ({tool_support_percent:.1f}%) support hello_world tool calls"
        )

        response_handling_percent = (
            (summary["handles_response_count"] / summary["supported_count"])
            * 100
            if summary["supported_count"] > 0
            else 0
        )
        click.echo(
            f"{summary['handles_response_count']} out of {summary['supported_count']} models ({response_handling_percent:.1f}%) correctly handle tool responses"
        )

        click.echo("\n=== Performance ===")
        click.echo(
            f"Average tool call time: {summary['avg_tool_time']:.2f} seconds"
        )
        click.echo(
            f"Average response processing time: {summary['avg_response_time']:.2f} seconds"
        )
        click.echo(
            f"Average total time: {summary['avg_total_time']:.2f} seconds"
        )

        if summary["fastest_model"]:
            click.echo(
                f"Fastest model: {summary['fastest_model']['model']} ({summary['fastest_model']['time']:.2f} seconds)"
            )

        click.echo("\n=== Tool Support Status ===")

        # Count models by status
        full_support = sum(
            1
            for r in results
            if r.get("tool_call_support", False)
            and r.get("handles_response", False)
        )
        partial_support = sum(
            1
            for r in results
            if r.get("tool_call_support", False)
            and not r.get("handles_response", False)
        )
        no_support = sum(
            1 for r in results if not r.get("tool_call_support", False)
        )

        # Print status counts with colors
        click.secho(
            f"✅ Full support (calls tool + uses result): {full_support}",
            fg="green",
        )
        click.secho(
            f"⚠️ Partial support (calls tool but ignores result): {partial_support}",
            fg="yellow",
        )
        click.secho(
            f"❌ No support (does not call tool): {no_support}", fg="red"
        )

        click.echo("\n")

    def format_json_output(
        self, results: List[Dict[str, Any]], include_raw: bool = False
    ) -> str:
        """Format test results as JSON.

        Args:
            results: List of test result dictionaries
            include_raw: Whether to include raw API response data

        Returns:
            str: JSON string of test results
        """
        # Create a deep copy of results to avoid modifying the original
        output_results = []

        for result in results:
            output_result = {
                "model": result["model"],
                "supports_tool_calls": result["tool_call_support"],
                "handles_response": result["handles_response"],
                "details": result["details"],
                "timings": {
                    "tool_call_time_seconds": result.get(
                        "response_times", {}
                    ).get("tool_call_time"),
                    "response_processing_time_seconds": result.get(
                        "response_times", {}
                    ).get("response_processing_time"),
                    "total_time_seconds": result.get("response_times", {}).get(
                        "total_time"
                    ),
                },
                "timestamp": datetime.datetime.now().isoformat(),
            }

            # Include raw response data if requested
            if include_raw and "raw_response" in result:
                output_result["raw_response"] = result["raw_response"]

            output_results.append(output_result)

        # Add summary statistics
        summary = self.summarize_results(results)

        # Format as pretty JSON
        return json.dumps(
            {
                "results": output_results,
                "summary": summary,
            },
            indent=2,
        )
