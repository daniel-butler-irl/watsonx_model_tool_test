#!/usr/bin/env python3
"""
Result handler for WatsonX Tool Tester.

This module provides the ResultHandler class for processing, sorting,
and formatting tool test results.
"""

import datetime
import json
from typing import Any, Dict, List, Optional

import click
from tabulate import tabulate

from watsonx_tool_tester.utils.formatting import (
    format_response_success,
    format_response_time,
    format_tool_call_success,
)
from watsonx_tool_tester.utils.history_manager import HistoryManager
from watsonx_tool_tester.utils.html_generator import HTMLReportGenerator


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

        # Check if any results have reliability data
        has_reliability_data = any(
            "reliability" in result for result in results
        )

        for result in results:
            model_id = result["model"]
            supports_tool_call = result["tool_call_support"]
            handles_response = result["handles_response"]

            # Format tool call support with reliability context
            if has_reliability_data and "reliability" in result:
                reliability_info = result.get("reliability")
                is_reliable = (
                    reliability_info.get("is_reliable")
                    if reliability_info
                    else None
                )
                tool_success_rate = (
                    reliability_info.get("tool_call_success_rate", 0)
                    if reliability_info
                    else 0
                )
                iterations = (
                    reliability_info.get("iterations", 1)
                    if reliability_info
                    else 1
                )

                # Calculate actual success count from rate and iterations
                tool_successes = int(tool_success_rate * iterations)

                # Show success/total format for supported models (only consider tool calling success)
                if supports_tool_call and is_reliable is not None:
                    if tool_success_rate == 1.0:
                        tool_call_status = click.style(
                            f"✅ RELIABLE ({tool_successes}/{iterations})",
                            fg="green",
                        )
                    else:
                        tool_call_status = click.style(
                            f"⚠️ UNRELIABLE ({tool_successes}/{iterations})",
                            fg="yellow",
                        )
                elif not supports_tool_call:
                    # For unsupported models, show 0/iterations
                    tool_call_status = click.style(
                        f"❌ NOT SUPPORTED (0/{iterations})", fg="red"
                    )
                else:
                    tool_call_status = format_tool_call_success(
                        supports_tool_call
                    )
            else:
                tool_call_status = format_tool_call_success(supports_tool_call)

            # Format response handling with reliability context
            if has_reliability_data and "reliability" in result:
                reliability_info = result.get("reliability")
                is_reliable = (
                    reliability_info.get("is_reliable")
                    if reliability_info
                    else None
                )
                response_success_rate = (
                    reliability_info.get("response_handling_success_rate", 0)
                    if reliability_info
                    else 0
                )
                iterations = (
                    reliability_info.get("iterations", 1)
                    if reliability_info
                    else 1
                )

                # Calculate actual success count from rate and iterations
                # For response handling, use tool_successes as denominator since we calculated it above
                response_successes = (
                    int(response_success_rate * tool_successes)
                    if tool_successes > 0
                    else 0
                )

                # Show response handling based on model support
                if not supports_tool_call:
                    # For unsupported models, show N/A
                    response_status = click.style("N/A", fg="bright_black")
                elif supports_tool_call and tool_successes > 0:
                    # For supported models, show success/attempts format
                    if response_success_rate == 1.0:
                        response_status = click.style(
                            f"✅ CORRECT ({response_successes}/{tool_successes})",
                            fg="green",
                        )
                    elif response_success_rate > 0:
                        response_status = click.style(
                            f"⚠️ PARTIAL ({response_successes}/{tool_successes})",
                            fg="yellow",
                        )
                    else:
                        response_status = click.style(
                            f"❌ NEVER HANDLES (0/{tool_successes})", fg="red"
                        )
                else:
                    response_status = format_response_success(handles_response)
            else:
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
            if (
                len(details) > 60
            ):  # Reduced from 80 to make room for reliability column
                details = details[:57] + "..."

            # Build row data
            row_data = [
                model_id,
                tool_call_status,
                response_status,
            ]

            # Add reliability column right after HANDLED if we have reliability data
            if has_reliability_data:
                reliability_info = result.get("reliability")
                if reliability_info:
                    iterations = reliability_info.get("iterations", 1)
                    is_reliable = reliability_info.get("is_reliable")

                    if iterations > 1:
                        tool_success_rate = reliability_info.get(
                            "tool_call_success_rate", 0
                        )
                        response_success_rate = reliability_info.get(
                            "response_handling_success_rate", 0
                        )

                        # Simplified reliability display since other columns show the detail
                        if is_reliable is None:
                            reliability_str = "❌ NOT SUPPORTED"
                        elif is_reliable:
                            reliability_str = "✅ RELIABLE"
                        else:
                            reliability_str = "⚠️ UNRELIABLE"
                    else:
                        # Single iteration results
                        if not result.get("tool_call_support", False):
                            reliability_str = "❌ NOT SUPPORTED"
                        else:
                            reliability_str = "✅ SINGLE TEST"
                else:
                    reliability_str = "✅ SINGLE TEST"
                row_data.append(reliability_str)

            # Add timing columns
            row_data.extend(
                [
                    tool_call_time_str,
                    response_time_str,
                    total_time_str,
                    details,
                ]
            )

            table_data.append(row_data)

        # Define table headers
        headers = [
            "MODEL",
            "TOOL SUPPORT",
            "HANDLED",  # Shortened from "TOOL RESULT HANDLED"
        ]

        # Add reliability header right after HANDLED if we have reliability data
        if has_reliability_data:
            # Get iterations count for header
            iterations = 1
            if results:
                first_reliability = next(
                    (
                        r.get("reliability")
                        for r in results
                        if "reliability" in r
                    ),
                    None,
                )
                if first_reliability:
                    iterations = first_reliability.get("iterations", 1)

            if iterations > 1:
                headers.append(f"RELIABILITY ({iterations}x)")
            else:
                headers.append("RELIABILITY")

        # Add timing and details headers
        headers.extend(
            [
                "CALL TIME",  # Shortened from "TOOL CALL TIME"
                "RESP TIME",  # Shortened from "RESPONSE TIME"
                "TOTAL TIME",
                "DETAILS",
            ]
        )

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

        # Calculate reliability statistics
        reliability_stats = None
        # Only include models that support tool calling for reliability calculation
        reliability_results = [
            r
            for r in results
            if "reliability" in r and r.get("tool_call_support", False)
        ]
        if reliability_results:
            reliable_count = sum(
                1
                for r in reliability_results
                if r.get("reliability", {}).get("is_reliable", False)
            )
            # Only count models with is_reliable == False as unreliable
            # Models with is_reliable == None are unsupported, not unreliable
            unreliable_count = sum(
                1
                for r in reliability_results
                if r.get("reliability", {}).get("is_reliable") is False
            )

            # Calculate average success rates
            avg_tool_success_rate = (
                sum(
                    r.get("reliability", {}).get("tool_call_success_rate", 0)
                    for r in reliability_results
                )
                / len(reliability_results)
                if reliability_results
                else 0
            )

            avg_response_success_rate = (
                sum(
                    r.get("reliability", {}).get(
                        "response_handling_success_rate", 0
                    )
                    for r in reliability_results
                )
                / len(reliability_results)
                if reliability_results
                else 0
            )

            # Get test parameters
            iterations = (
                reliability_results[0]
                .get("reliability", {})
                .get("iterations", 1)
            )

            reliability_stats = {
                "total_tested": len(reliability_results),
                "reliable_count": reliable_count,
                "unreliable_count": unreliable_count,
                "avg_tool_success_rate": avg_tool_success_rate,
                "avg_response_success_rate": avg_response_success_rate,
                "iterations": iterations,
            }

        # Calculate average times, excluding None values and failed models
        # Only include models that support tool calling for performance metrics
        successful_results = [
            r for r in results if r.get("tool_call_support", False)
        ]

        tool_call_times = [
            r.get("response_times", {}).get("tool_call_time")
            for r in successful_results
            if r.get("response_times", {}).get("tool_call_time") is not None
        ]

        response_times = [
            r.get("response_times", {}).get("response_processing_time")
            for r in successful_results
            if r.get("response_times", {}).get("response_processing_time")
            is not None
        ]

        total_times = [
            r.get("response_times", {}).get("total_time")
            for r in successful_results
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

        # Find fastest model (by total time, only among successful models)
        fastest_model = None
        fastest_time = float("inf")

        for result in successful_results:
            total_time = result.get("response_times", {}).get("total_time")
            if total_time is not None and total_time < fastest_time:
                fastest_time = total_time
                fastest_model = result["model"]

        summary = {
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

        # Add reliability stats if available
        if reliability_stats:
            summary["reliability"] = reliability_stats

        return summary

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
        click.echo("(Only includes models that support tool calling)")
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

        # Print reliability information if available
        if "reliability" in summary:
            rel_stats = summary["reliability"]
            click.echo("\n=== Reliability Assessment ===")
            click.echo(f"Test iterations per model: {rel_stats['iterations']}")
            click.echo("Reliability standard: 100% consistency required")
            click.echo(
                f"Models tested for reliability: {rel_stats['total_tested']}"
            )

            if rel_stats["total_tested"] > 0:
                reliable_percent = (
                    rel_stats["reliable_count"] / rel_stats["total_tested"]
                ) * 100
                click.secho(
                    f"✅ Reliable models: {rel_stats['reliable_count']} ({reliable_percent:.1f}%)",
                    fg="green",
                )
                click.secho(
                    f"⚠️ Unreliable models: {rel_stats['unreliable_count']} ({(rel_stats['unreliable_count'] / rel_stats['total_tested'] * 100):.1f}%)",
                    fg="yellow",
                )
                click.echo(
                    f"Average tool call success rate (across all models): {rel_stats['avg_tool_success_rate']:.1%}"
                )
                click.echo(
                    "  - Tool call success: Model correctly invokes the hello_world tool with proper parameters"
                )
                click.echo(
                    f"Average response handling success rate (across all models): {rel_stats['avg_response_success_rate']:.1%}"
                )
                click.echo(
                    "  - Response handling success: Model correctly uses the tool's result in its final response"
                )

                # Add model breakdown by reliability status
                click.echo("\n=== Model Status Breakdown ===")

                # Count models by reliability status
                supported_reliable = rel_stats["reliable_count"]
                supported_unreliable = rel_stats["unreliable_count"]
                unsupported = (
                    rel_stats["total_tested"]
                    - supported_reliable
                    - supported_unreliable
                )

                click.secho(
                    f"✅ Supported & Reliable: {supported_reliable} models",
                    fg="green",
                )
                if supported_unreliable > 0:
                    click.secho(
                        f"⚠️ Supported but Unreliable: {supported_unreliable} models",
                        fg="yellow",
                    )
                click.secho(
                    f"❌ Not Supported: {unsupported} models", fg="red"
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

    def format_html_output(
        self,
        results: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format test results as HTML.

        Args:
            results: List of test result dictionaries
            config: Optional configuration information

        Returns:
            str: HTML string of test results
        """
        # Generate summary statistics
        summary = self.summarize_results(results)

        # Create HTML report generator with history manager
        history_manager = HistoryManager()
        html_generator = HTMLReportGenerator(history_manager=history_manager)

        # Generate HTML content
        html_content = html_generator.generate_html_report(
            results=results,
            summary=summary,
            config=config,
        )

        return html_content
