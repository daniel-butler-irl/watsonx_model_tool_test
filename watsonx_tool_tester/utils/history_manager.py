#!/usr/bin/env python3
"""
CSV-based History Manager for WatsonX Tool Tester.

This module provides functionality to track model performance over time
using CSV files stored in the repository, without requiring external services.
"""

import csv
import datetime
import os
import re
from typing import Any, Dict, List, Optional


class HistoryManager:
    """Manager for tracking model performance history using CSV files."""

    def __init__(self, history_dir: str = "reports/history"):
        """Initialize the history manager.

        Args:
            history_dir: Directory to store history CSV files
        """
        self.history_dir = history_dir
        self.results_file = os.path.join(history_dir, "test_results.csv")
        self.models_file = os.path.join(history_dir, "models_registry.csv")
        self.summary_file = os.path.join(history_dir, "daily_summary.csv")

        # Ensure directory exists
        os.makedirs(history_dir, exist_ok=True)

        # Initialize CSV files if they don't exist
        self._initialize_csv_files()

    def _parse_datetime_safely(
        self, date_string: str
    ) -> Optional[datetime.datetime]:
        """Parse datetime string with multiple fallback formats.

        Args:
            date_string: The datetime string to parse

        Returns:
            Parsed datetime object or None if parsing fails
        """
        if not date_string:
            return None

        # Try different formats in order of preference
        formats = [
            "%Y-%m-%d",  # Date only: 2025-07-16
            "%Y-%m-%d %H:%M:%S",  # Date with time: 2025-07-16 16:13:44
            "%Y-%m-%dT%H:%M:%S",  # ISO format: 2025-07-16T16:13:44
            "%Y-%m-%dT%H:%M:%S.%f",  # ISO with microseconds: 2025-07-16T16:13:44.132759
        ]

        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        # Try ISO format parsing as fallback
        try:
            # Remove any timezone info and normalize the format
            clean_date = re.sub(
                r"[+-]\d{2}:?\d{2}$", "", date_string
            )  # Remove timezone
            clean_date = clean_date.replace(
                "Z", ""
            )  # Remove Z timezone indicator

            # Try to extract just the date part if it's a full ISO string
            if "T" in clean_date:
                # Try parsing as ISO format
                return datetime.datetime.fromisoformat(clean_date)
            else:
                # Try parsing as date only
                return datetime.datetime.strptime(clean_date, "%Y-%m-%d")
        except (ValueError, AttributeError):
            return None

    def _initialize_csv_files(self) -> None:
        """Initialize CSV files with headers if they don't exist."""
        # Test results CSV
        if not os.path.exists(self.results_file):
            with open(self.results_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "date",
                        "model_id",
                        "tool_call_support",
                        "handles_response",
                        "is_reliable",
                        "tool_call_time",
                        "response_time",
                        "total_time",
                        "iterations",
                        "tool_success_rate",
                        "response_success_rate",
                        "details",
                        "tool_call_raw",
                        "response_raw",
                        "error_message",
                        "test_prompt",
                        "expected_result",
                        "actual_result",
                        "model_version",
                        "test_config",
                    ]
                )

        # Models registry CSV
        if not os.path.exists(self.models_file):
            with open(self.models_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model_id",
                        "first_seen",
                        "last_seen",
                        "ever_worked",
                        "ever_handled_response",
                        "total_tests",
                        "working_days",
                        "response_handling_days",
                        "display_name",
                        "last_error",
                        "avg_tool_time",
                        "avg_response_time",
                        "best_tool_time",
                        "best_response_time",
                        "consistency_score",
                    ]
                )

        # Daily summary CSV
        if not os.path.exists(self.summary_file):
            with open(self.summary_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "datetime",
                        "total_models",
                        "working_models",
                        "handling_models",
                        "reliable_models",
                        "avg_tool_time",
                        "avg_response_time",
                        "avg_total_time",
                        "fastest_model",
                        "fastest_time",
                    ]
                )

    def record_test_results(
        self, results: List[Dict[str, Any]], test_date: Optional[str] = None
    ) -> None:
        """Record test results to CSV history.

        If results already exist for the same model and date, they will be replaced
        with the new results to prevent duplicates.

        Args:
            results: List of test result dictionaries
            test_date: Date of the test (defaults to today)
        """
        if test_date is None:
            test_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        timestamp = datetime.datetime.utcnow().isoformat()

        # Read existing results and build lookup for duplicates
        existing_results = []
        model_date_indices = {}  # (model_id, date) -> row_index
        fieldnames = [
            "timestamp",
            "date",
            "model_id",
            "tool_call_support",
            "handles_response",
            "is_reliable",
            "tool_call_time",
            "response_time",
            "total_time",
            "iterations",
            "tool_success_rate",
            "response_success_rate",
            "details",
            "tool_call_raw",
            "response_raw",
            "error_message",
            "test_prompt",
            "expected_result",
            "actual_result",
            "model_version",
            "test_config",
        ]

        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                # Use the fieldnames from the file if available, otherwise use defaults
                if reader.fieldnames:
                    fieldnames = reader.fieldnames
                for i, row in enumerate(reader):
                    existing_results.append(row)
                    key = (row["model_id"], row["date"])
                    model_date_indices[key] = i

        # Process new results
        for result in results:
            # Extract reliability info
            reliability = result.get("reliability", {})
            is_reliable = reliability.get(
                "is_reliable", None
            )  # None for unsupported models
            tool_success_rate = reliability.get("tool_call_success_rate", 0.0)
            response_success_rate = reliability.get(
                "response_handling_success_rate", 0.0
            )
            iterations = reliability.get("iterations", 1)

            # Extract timing info
            times = result.get("response_times", {})
            tool_call_time = times.get("tool_call_time") or 0.0
            response_time = times.get("response_processing_time") or 0.0
            total_time = times.get("total_time") or 0.0

            # Extract detailed test information
            test_details = result.get("test_details", {})
            tool_call_raw = test_details.get("tool_call_response", "")
            response_raw = test_details.get("final_response", "")
            error_message = test_details.get("error_message", "")
            test_prompt = test_details.get("test_prompt", "")
            expected_result = test_details.get("expected_result", "")
            actual_result = test_details.get("actual_result", "")

            # Extract model metadata
            model_info = result.get("model_info", {})
            model_version = model_info.get("version", "")

            # Extract test configuration
            test_config = result.get("test_config", {})
            config_str = str(test_config) if test_config else ""

            # Create new row data
            new_row = {
                "timestamp": timestamp,
                "date": test_date,
                "model_id": result["model"],
                "tool_call_support": result.get("tool_call_support", False),
                "handles_response": result.get("handles_response", False),
                "is_reliable": is_reliable,
                "tool_call_time": tool_call_time,
                "response_time": response_time,
                "total_time": total_time,
                "iterations": iterations,
                "tool_success_rate": tool_success_rate,
                "response_success_rate": response_success_rate,
                "details": result.get("details", ""),
                "tool_call_raw": tool_call_raw,
                "response_raw": response_raw,
                "error_message": error_message,
                "test_prompt": test_prompt,
                "expected_result": expected_result,
                "actual_result": actual_result,
                "model_version": model_version,
                "test_config": config_str,
            }

            # Check if we already have results for this model+date combination
            key = (result["model"], test_date)
            if key in model_date_indices:
                # Replace existing result
                existing_index = model_date_indices[key]
                existing_results[existing_index] = new_row
            else:
                # Add new result
                existing_results.append(new_row)

        # Write all results back to file
        with open(self.results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_results:
                writer.writerow(row)

        # Update models registry
        self._update_models_registry(results, test_date)

        # Update daily summary
        self._update_daily_summary(results, test_date)

    def _update_models_registry(
        self, results: List[Dict[str, Any]], test_date: str
    ) -> None:
        """Update the models registry with new results."""
        # Load existing registry
        registry = {}
        if os.path.exists(self.models_file):
            with open(self.models_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    registry[row["model_id"]] = row

        # Update registry with new results
        for result in results:
            model_id = result["model"]
            tool_support = result.get("tool_call_support", False)
            handles_response = result.get("handles_response", False)

            # Extract performance metrics
            times = result.get("response_times", {})
            tool_time = times.get("tool_call_time", 0.0)
            response_time = times.get("response_processing_time", 0.0)

            # Extract error information
            test_details = result.get("test_details", {})
            error_message = test_details.get("error_message", "")

            if model_id not in registry:
                # New model
                registry[model_id] = {
                    "model_id": model_id,
                    "first_seen": test_date,
                    "last_seen": test_date,
                    "ever_worked": str(tool_support),
                    "ever_handled_response": str(handles_response),
                    "total_tests": "1",
                    "working_days": "1" if tool_support else "0",
                    "response_handling_days": "1" if handles_response else "0",
                    "display_name": self._extract_display_name(model_id),
                    "last_error": error_message,
                    "avg_tool_time": str(
                        tool_time if tool_time is not None else 0.0
                    ),
                    "avg_response_time": str(
                        response_time if response_time is not None else 0.0
                    ),
                    "best_tool_time": (
                        str(tool_time)
                        if tool_time is not None and tool_time > 0
                        else "0"
                    ),
                    "best_response_time": (
                        str(response_time)
                        if response_time is not None and response_time > 0
                        else "0"
                    ),
                    "consistency_score": "1.0" if tool_support else "0.0",
                }
            else:
                # Update existing model
                registry[model_id]["last_seen"] = test_date
                registry[model_id]["ever_worked"] = str(
                    registry[model_id]["ever_worked"].lower() == "true"
                    or tool_support
                )
                registry[model_id]["ever_handled_response"] = str(
                    registry[model_id]["ever_handled_response"].lower()
                    == "true"
                    or handles_response
                )

                # Update test counts
                total_tests = int(registry[model_id]["total_tests"]) + 1
                registry[model_id]["total_tests"] = str(total_tests)

                if tool_support:
                    registry[model_id]["working_days"] = str(
                        int(registry[model_id]["working_days"]) + 1
                    )
                if handles_response:
                    registry[model_id]["response_handling_days"] = str(
                        int(registry[model_id]["response_handling_days"]) + 1
                    )

                # Update error information
                if error_message:
                    registry[model_id]["last_error"] = error_message

                # Update timing averages
                if tool_time is not None and tool_time > 0:
                    old_avg = float(registry[model_id]["avg_tool_time"])
                    new_avg = (
                        old_avg * (total_tests - 1) + tool_time
                    ) / total_tests
                    registry[model_id]["avg_tool_time"] = str(new_avg)

                    old_best = float(registry[model_id]["best_tool_time"])
                    if old_best == 0 or tool_time < old_best:
                        registry[model_id]["best_tool_time"] = str(tool_time)

                if response_time is not None and response_time > 0:
                    old_avg = float(registry[model_id]["avg_response_time"])
                    new_avg = (
                        old_avg * (total_tests - 1) + response_time
                    ) / total_tests
                    registry[model_id]["avg_response_time"] = str(new_avg)

                    old_best = float(registry[model_id]["best_response_time"])
                    if old_best == 0 or response_time < old_best:
                        registry[model_id]["best_response_time"] = str(
                            response_time
                        )

                # Update consistency score
                working_days = int(registry[model_id]["working_days"])
                consistency_score = working_days / total_tests
                registry[model_id]["consistency_score"] = str(
                    consistency_score
                )

        # Save updated registry
        with open(self.models_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model_id",
                    "first_seen",
                    "last_seen",
                    "ever_worked",
                    "ever_handled_response",
                    "total_tests",
                    "working_days",
                    "response_handling_days",
                    "display_name",
                    "last_error",
                    "avg_tool_time",
                    "avg_response_time",
                    "best_tool_time",
                    "best_response_time",
                    "consistency_score",
                ],
            )
            writer.writeheader()
            for model_data in registry.values():
                writer.writerow(model_data)

    def _update_daily_summary(
        self, results: List[Dict[str, Any]], test_date: str
    ) -> None:
        """Update daily summary statistics."""
        # Calculate summary stats
        total_models = len(results)
        working_models = sum(
            1 for r in results if r.get("tool_call_support", False)
        )
        handling_models = sum(
            1 for r in results if r.get("handles_response", False)
        )
        reliable_models = sum(
            1
            for r in results
            if r.get("reliability", {}).get("is_reliable", False)
        )

        # Calculate timing averages
        times = [r.get("response_times", {}) for r in results]
        tool_times = [
            t.get("tool_call_time", 0)
            for t in times
            if t.get("tool_call_time")
        ]
        response_times = [
            t.get("response_processing_time", 0)
            for t in times
            if t.get("response_processing_time")
        ]
        total_times = [
            t.get("total_time", 0) for t in times if t.get("total_time")
        ]

        avg_tool_time = sum(tool_times) / len(tool_times) if tool_times else 0
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        avg_total_time = (
            sum(total_times) / len(total_times) if total_times else 0
        )

        # Find fastest model
        fastest_model = ""
        fastest_time = float("inf")
        for result in results:
            total_time = result.get("response_times", {}).get("total_time")
            if total_time and total_time < fastest_time:
                fastest_time = total_time
                fastest_model = result["model"]

        if fastest_time == float("inf"):
            fastest_time = 0

        # Load existing summary and update/append
        summary_data = []
        if os.path.exists(self.summary_file):
            with open(self.summary_file, "r") as f:
                reader = csv.DictReader(f)
                summary_data = [
                    row
                    for row in reader
                    if row.get("datetime", row.get("date", "")) != test_date
                ]

        # Add new summary
        summary_data.append(
            {
                "datetime": test_date,
                "total_models": total_models,
                "working_models": working_models,
                "handling_models": handling_models,
                "reliable_models": reliable_models,
                "avg_tool_time": f"{avg_tool_time:.3f}",
                "avg_response_time": f"{avg_response_time:.3f}",
                "avg_total_time": f"{avg_total_time:.3f}",
                "fastest_model": fastest_model,
                "fastest_time": f"{fastest_time:.3f}",
            }
        )

        # Save updated summary
        with open(self.summary_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "datetime",
                    "total_models",
                    "working_models",
                    "handling_models",
                    "reliable_models",
                    "avg_tool_time",
                    "avg_response_time",
                    "avg_total_time",
                    "fastest_model",
                    "fastest_time",
                ],
            )
            writer.writeheader()
            for row in summary_data:
                writer.writerow(row)

    def _extract_display_name(self, model_id: str) -> str:
        """Extract a display name from model ID."""
        # Extract model name from ID
        if "/" in model_id:
            return model_id.split("/")[-1]
        return model_id

    def get_trackable_models(self) -> List[Dict[str, Any]]:
        """Get list of models that should be tracked (have ever worked at least partially).

        Returns:
            List of model dictionaries with tracking information
        """
        trackable_models = []

        if os.path.exists(self.models_file):
            with open(self.models_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Include models that have ever worked or handled responses
                    if (
                        row["ever_worked"].lower() == "true"
                        or row["ever_handled_response"].lower() == "true"
                    ):
                        trackable_models.append(
                            {
                                "model_id": row["model_id"],
                                "display_name": row["display_name"],
                                "first_seen": row["first_seen"],
                                "last_seen": row["last_seen"],
                                "total_tests": int(row["total_tests"]),
                                "working_days": int(row["working_days"]),
                                "response_handling_days": int(
                                    row["response_handling_days"]
                                ),
                                "reliability_rate": (
                                    int(row["working_days"])
                                    / int(row["total_tests"])
                                    if int(row["total_tests"]) > 0
                                    else 0
                                ),
                                "last_error": row.get("last_error", ""),
                                "avg_tool_time": float(
                                    row.get("avg_tool_time", 0)
                                ),
                                "avg_response_time": float(
                                    row.get("avg_response_time", 0)
                                ),
                                "best_tool_time": float(
                                    row.get("best_tool_time", 0)
                                ),
                                "best_response_time": float(
                                    row.get("best_response_time", 0)
                                ),
                                "consistency_score": float(
                                    row.get("consistency_score", 0)
                                ),
                            }
                        )

        return sorted(trackable_models, key=lambda x: x["display_name"])

    def get_model_history(
        self, model_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical performance data for a specific model.

        Args:
            model_id: The model ID to get history for
            days: Number of days to look back

        Returns:
            List of historical performance data points
        """
        history = []

        if os.path.exists(self.results_file):
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
                days=days
            )

            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["model_id"] == model_id:
                        # Try to parse date with fallback for different formats
                        test_date = self._parse_datetime_safely(row["date"])
                        if test_date is None:
                            # Skip rows with unparseable dates
                            continue
                        if test_date >= cutoff_date:
                            history.append(
                                {
                                    "date": row["date"],
                                    "timestamp": row["timestamp"],
                                    "tool_call_support": row[
                                        "tool_call_support"
                                    ].lower()
                                    == "true",
                                    "handles_response": row[
                                        "handles_response"
                                    ].lower()
                                    == "true",
                                    "is_reliable": (
                                        None
                                        if not row["is_reliable"]
                                        or row["is_reliable"].lower()
                                        in ["none", ""]
                                        else (
                                            True
                                            if row["is_reliable"].lower()
                                            == "true"
                                            else False
                                        )
                                    ),
                                    "tool_call_time": (
                                        float(row["tool_call_time"])
                                        if row["tool_call_time"]
                                        else None
                                    ),
                                    "response_time": (
                                        float(row["response_time"])
                                        if row["response_time"]
                                        else None
                                    ),
                                    "total_time": (
                                        float(row["total_time"])
                                        if row["total_time"]
                                        else None
                                    ),
                                    "tool_success_rate": float(
                                        row["tool_success_rate"]
                                    ),
                                    "response_success_rate": float(
                                        row["response_success_rate"]
                                    ),
                                    "details": row["details"],
                                    "tool_call_raw": row.get(
                                        "tool_call_raw", ""
                                    ),
                                    "response_raw": row.get(
                                        "response_raw", ""
                                    ),
                                    "error_message": row.get(
                                        "error_message", ""
                                    ),
                                    "test_prompt": row.get("test_prompt", ""),
                                    "expected_result": row.get(
                                        "expected_result", ""
                                    ),
                                    "actual_result": row.get(
                                        "actual_result", ""
                                    ),
                                    "model_version": row.get(
                                        "model_version", ""
                                    ),
                                    "test_config": row.get("test_config", ""),
                                }
                            )

        return sorted(history, key=lambda x: x["date"])

    def get_daily_summary_history(
        self, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get daily summary history.

        Args:
            days: Number of days to look back

        Returns:
            List of daily summary data points
        """
        history = []

        if os.path.exists(self.summary_file):
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
                days=days
            )

            with open(self.summary_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to parse date with fallback for different formats
                    test_date = self._parse_datetime_safely(row["date"])
                    if test_date is None:
                        # Skip rows with unparseable dates
                        continue
                    if test_date >= cutoff_date:
                        history.append(
                            {
                                "date": row["date"],
                                "total_models": int(row["total_models"]),
                                "working_models": int(row["working_models"]),
                                "handling_models": int(row["handling_models"]),
                                "reliable_models": int(row["reliable_models"]),
                                "avg_tool_time": float(row["avg_tool_time"]),
                                "avg_response_time": float(
                                    row["avg_response_time"]
                                ),
                                "avg_total_time": float(row["avg_total_time"]),
                                "fastest_model": row["fastest_model"],
                                "fastest_time": float(row["fastest_time"]),
                            }
                        )

        return sorted(history, key=lambda x: x["date"])

    def get_status_matrix(
        self, days: int = 30
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get status matrix for service-outage-style visualization.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary mapping model_id to list of daily status points
        """
        trackable_models = self.get_trackable_models()
        model_ids = [model["model_id"] for model in trackable_models]

        # Get date range
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=days)
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime("%Y-%m-%d"))
            current_date += datetime.timedelta(days=1)

        # Initialize status matrix
        status_matrix = {}
        for model_id in model_ids:
            status_matrix[model_id] = []
            for date in date_range:
                status_matrix[model_id].append(
                    {
                        "date": date,
                        "status": "untested",  # untested, working, partial, broken
                        "details": "",
                    }
                )

        # Fill in actual test results
        if os.path.exists(self.results_file):
            # First, collect all test results grouped by model_id and date
            test_data_by_model_date = {}

            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_id = row["model_id"]
                    test_date = row["date"]

                    if model_id in status_matrix and test_date in date_range:
                        key = (model_id, test_date)
                        if key not in test_data_by_model_date:
                            test_data_by_model_date[key] = []
                        test_data_by_model_date[key].append(row)

            # Process each model+date combination, using the most recent result
            for (model_id, test_date), rows in test_data_by_model_date.items():
                # Sort by timestamp to get the most recent result
                sorted_rows = sorted(
                    rows, key=lambda x: x.get("timestamp", ""), reverse=True
                )
                most_recent_row = sorted_rows[0]

                # Find the status entry for this date
                for status_entry in status_matrix[model_id]:
                    if status_entry["date"] == test_date:
                        tool_support = (
                            most_recent_row["tool_call_support"].lower()
                            == "true"
                        )
                        handles_response = (
                            most_recent_row["handles_response"].lower()
                            == "true"
                        )
                        # Debug: log the actual value being processed
                        reliability_value = most_recent_row.get(
                            "is_reliable", ""
                        )
                        is_reliable = (
                            reliability_value.lower() == "true"
                            if reliability_value and reliability_value.strip()
                            else False
                        )

                        if not tool_support:
                            # Check if model worked before to differentiate broken vs not_supported
                            if self.has_previously_worked(model_id):
                                status_entry["status"] = "broken"
                                status_entry["details"] = (
                                    "Model previously worked but now fails"
                                )
                            else:
                                status_entry["status"] = "not_supported"
                                status_entry["details"] = (
                                    "Model does not support tool calling"
                                )
                        elif tool_support and handles_response and is_reliable:
                            status_entry["status"] = "working"
                            status_entry["details"] = (
                                "Full tool calling support - reliable"
                            )
                        elif (
                            tool_support
                            and handles_response
                            and not is_reliable
                        ):
                            status_entry["status"] = "unreliable"
                            status_entry["details"] = (
                                "Full tool calling support - inconsistent results"
                            )
                        elif tool_support and not handles_response:
                            status_entry["status"] = "partial"
                            status_entry["details"] = (
                                "Tool calling only - doesn't handle responses"
                            )
                        else:
                            status_entry["status"] = "broken"
                            status_entry["details"] = "Tool calling failed"
                        break

        return status_matrix

    def cleanup_old_data(self, days_to_keep: int = 90) -> None:
        """Clean up old historical data.

        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
            days=days_to_keep
        )
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        # Clean up test results
        if os.path.exists(self.results_file):
            temp_file = self.results_file + ".tmp"
            with open(self.results_file, "r") as infile, open(
                temp_file, "w", newline=""
            ) as outfile:
                reader = csv.DictReader(infile)
                if reader.fieldnames:
                    writer = csv.DictWriter(
                        outfile, fieldnames=reader.fieldnames
                    )
                    writer.writeheader()

                    for row in reader:
                        # Handle missing date field gracefully
                        if "date" in row and row["date"] >= cutoff_str:
                            writer.writerow(row)

            os.replace(temp_file, self.results_file)

        # Clean up daily summary
        if os.path.exists(self.summary_file):
            temp_file = self.summary_file + ".tmp"
            with open(self.summary_file, "r") as infile, open(
                temp_file, "w", newline=""
            ) as outfile:
                reader = csv.DictReader(infile)
                if reader.fieldnames:
                    writer = csv.DictWriter(
                        outfile, fieldnames=reader.fieldnames
                    )
                    writer.writeheader()

                    for row in reader:
                        # Handle missing date field gracefully
                        if "date" in row and row["date"] >= cutoff_str:
                            writer.writerow(row)

            os.replace(temp_file, self.summary_file)

    def get_detailed_test_results(
        self, model_id: Optional[str] = None, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get detailed test results with complete information.

        Args:
            model_id: Optional model ID to filter by
            days: Number of days to look back

        Returns:
            List of detailed test result data points
        """
        detailed_results = []

        if os.path.exists(self.results_file):
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
                days=days
            )

            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if model_id and row["model_id"] != model_id:
                        continue

                    # Try to parse date with fallback for different formats
                    test_date = self._parse_datetime_safely(row["date"])
                    if test_date is None:
                        # Skip rows with unparseable dates
                        continue
                    if test_date >= cutoff_date:
                        detailed_results.append(
                            {
                                "date": row["date"],
                                "timestamp": row["timestamp"],
                                "model_id": row["model_id"],
                                "tool_call_support": row[
                                    "tool_call_support"
                                ].lower()
                                == "true",
                                "handles_response": row[
                                    "handles_response"
                                ].lower()
                                == "true",
                                "is_reliable": row["is_reliable"].lower()
                                == "true",
                                "performance": {
                                    "tool_call_time": (
                                        float(row["tool_call_time"])
                                        if row["tool_call_time"]
                                        else 0
                                    ),
                                    "response_time": (
                                        float(row["response_time"])
                                        if row["response_time"]
                                        else 0
                                    ),
                                    "total_time": (
                                        float(row["total_time"])
                                        if row["total_time"]
                                        else 0
                                    ),
                                    "tool_success_rate": float(
                                        row["tool_success_rate"]
                                    ),
                                    "response_success_rate": float(
                                        row["response_success_rate"]
                                    ),
                                    "iterations": int(row["iterations"]),
                                },
                                "test_details": {
                                    "tool_call_raw": row.get(
                                        "tool_call_raw", ""
                                    ),
                                    "response_raw": row.get(
                                        "response_raw", ""
                                    ),
                                    "error_message": row.get(
                                        "error_message", ""
                                    ),
                                    "test_prompt": row.get("test_prompt", ""),
                                    "expected_result": row.get(
                                        "expected_result", ""
                                    ),
                                    "actual_result": row.get(
                                        "actual_result", ""
                                    ),
                                    "details": row["details"],
                                },
                                "model_info": {
                                    "version": row.get("model_version", ""),
                                    "display_name": self._extract_display_name(
                                        row["model_id"]
                                    ),
                                },
                                "test_config": row.get("test_config", ""),
                            }
                        )

        return sorted(detailed_results, key=lambda x: x["timestamp"])

    def get_error_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get analysis of errors and failure patterns.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with error analysis data
        """
        error_analysis = {
            "total_errors": 0,
            "models_with_errors": set(),
            "error_patterns": {},
            "failure_trends": {},
            "most_common_errors": [],
        }

        if os.path.exists(self.results_file):
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
                days=days
            )

            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to parse date with fallback for different formats
                    test_date = self._parse_datetime_safely(row["date"])
                    if test_date is None:
                        # Skip rows with unparseable dates
                        continue
                    if test_date >= cutoff_date:
                        error_message = row.get("error_message", "")
                        if error_message:
                            error_analysis["total_errors"] += 1
                            error_analysis["models_with_errors"].add(
                                row["model_id"]
                            )

                            # Count error patterns
                            if (
                                error_message
                                not in error_analysis["error_patterns"]
                            ):
                                error_analysis["error_patterns"][
                                    error_message
                                ] = 0
                            error_analysis["error_patterns"][
                                error_message
                            ] += 1

                            # Track failure trends by date
                            date_key = row["date"]
                            if (
                                date_key
                                not in error_analysis["failure_trends"]
                            ):
                                error_analysis["failure_trends"][date_key] = 0
                            error_analysis["failure_trends"][date_key] += 1

        # Convert set to list for JSON serialization
        error_analysis["models_with_errors"] = list(
            error_analysis["models_with_errors"]
        )

        # Get most common errors
        error_analysis["most_common_errors"] = sorted(
            error_analysis["error_patterns"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[
            :10
        ]  # Top 10 most common errors

        return error_analysis

    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with performance trend data
        """
        performance_trends = {
            "daily_averages": {},
            "model_performance": {},
            "improvement_trends": {},
            "consistency_analysis": {},
        }

        if os.path.exists(self.results_file):
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
                days=days
            )
            daily_data = {}
            model_data = {}

            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to parse date with fallback for different formats
                    test_date = self._parse_datetime_safely(row["date"])
                    if test_date is None:
                        # Skip rows with unparseable dates
                        continue
                    if test_date >= cutoff_date:
                        date_key = row["date"]
                        model_id = row["model_id"]

                        # Collect daily data
                        if date_key not in daily_data:
                            daily_data[date_key] = {
                                "tool_times": [],
                                "response_times": [],
                                "total_times": [],
                                "success_rates": [],
                                "supported_model_success_rates": [],
                            }

                        if row["tool_call_time"]:
                            daily_data[date_key]["tool_times"].append(
                                float(row["tool_call_time"])
                            )
                        if row["response_time"]:
                            daily_data[date_key]["response_times"].append(
                                float(row["response_time"])
                            )
                        if row["total_time"]:
                            daily_data[date_key]["total_times"].append(
                                float(row["total_time"])
                            )

                        daily_data[date_key]["success_rates"].append(
                            float(row["tool_success_rate"])
                        )

                        # Only include success rates for models that support tool calling
                        if row["tool_call_support"].lower() == "true":
                            daily_data[date_key][
                                "supported_model_success_rates"
                            ].append(float(row["tool_success_rate"]))

                        # Collect model data
                        if model_id not in model_data:
                            model_data[model_id] = {
                                "tests": [],
                                "consistency_scores": [],
                            }

                        model_data[model_id]["tests"].append(
                            {
                                "date": date_key,
                                "tool_time": (
                                    float(row["tool_call_time"])
                                    if row["tool_call_time"]
                                    else 0
                                ),
                                "response_time": (
                                    float(row["response_time"])
                                    if row["response_time"]
                                    else 0
                                ),
                                "total_time": (
                                    float(row["total_time"])
                                    if row["total_time"]
                                    else 0
                                ),
                                "success_rate": float(
                                    row["tool_success_rate"]
                                ),
                            }
                        )

            # Calculate daily averages
            for date_key, data in daily_data.items():
                performance_trends["daily_averages"][date_key] = {
                    "avg_tool_time": (
                        sum(data["tool_times"]) / len(data["tool_times"])
                        if data["tool_times"]
                        else 0
                    ),
                    "avg_response_time": (
                        sum(data["response_times"])
                        / len(data["response_times"])
                        if data["response_times"]
                        else 0
                    ),
                    "avg_total_time": (
                        sum(data["total_times"]) / len(data["total_times"])
                        if data["total_times"]
                        else 0
                    ),
                    "avg_success_rate": (
                        sum(data["success_rates"]) / len(data["success_rates"])
                        if data["success_rates"]
                        else 0
                    ),
                    "supported_models_success_rate": (
                        sum(data["supported_model_success_rates"])
                        / len(data["supported_model_success_rates"])
                        if data["supported_model_success_rates"]
                        else 0
                    ),
                }

            # Calculate model performance trends
            for model_id, data in model_data.items():
                if len(data["tests"]) > 1:
                    # Calculate trend (improvement/decline)
                    recent_tests = data["tests"][-5:]  # Last 5 tests
                    early_tests = data["tests"][:5]  # First 5 tests

                    if len(recent_tests) >= 2 and len(early_tests) >= 2:
                        recent_avg = sum(
                            t["success_rate"] for t in recent_tests
                        ) / len(recent_tests)
                        early_avg = sum(
                            t["success_rate"] for t in early_tests
                        ) / len(early_tests)

                        performance_trends["model_performance"][model_id] = {
                            "early_avg": early_avg,
                            "recent_avg": recent_avg,
                            "trend": recent_avg - early_avg,
                            "total_tests": len(data["tests"]),
                        }

        return performance_trends

    def has_previously_worked(self, model_id: str, days: int = 365) -> bool:
        """Check if a model has ever worked successfully in the past.

        Args:
            model_id: The model ID to check
            days: Number of days to look back in history (default: 365)

        Returns:
            bool: True if the model has ever had successful tool calling in the past
        """
        if not os.path.exists(self.results_file):
            return False

        # Calculate the cutoff date
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(
            days=days
        )

        try:
            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["model_id"] != model_id:
                        continue

                    # Parse the test date
                    test_date = self._parse_datetime_safely(row["date"])
                    if test_date is None or test_date < cutoff_date:
                        continue

                    # Check if this test showed tool calling support
                    tool_support = (
                        row.get("tool_call_support", "").lower() == "true"
                    )
                    if tool_support:
                        return True

        except Exception:
            # If there's any issue reading the file, assume no previous success
            return False

        return False

    def is_new_model(self, model_id: str) -> bool:
        """Check if a model is newly detected (within the last 2 weeks).

        Args:
            model_id: The model ID to check

        Returns:
            bool: True if the model was first seen within the last 2 weeks, False otherwise
        """
        if not os.path.exists(self.models_file):
            return False

        # Calculate the cutoff date (2 weeks ago)
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(weeks=2)

        with open(self.models_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["model_id"] == model_id:
                    # Parse the first_seen date
                    first_seen = self._parse_datetime_safely(row["first_seen"])
                    if first_seen is None:
                        return False

                    # Check if first_seen is within the last 2 weeks
                    return first_seen >= cutoff_date

        return False

    def is_previously_working(
        self, model_id: str, current_result: Dict[str, Any]
    ) -> bool:
        """Check if a model was previously working but is now broken/unreliable.

        Args:
            model_id: The model ID to check
            current_result: Current test result for the model

        Returns:
            bool: True if model was working in recent history but is currently not working
        """
        # Current model must not be fully working for this badge to apply
        current_working = (
            current_result.get("tool_call_support", False)
            and current_result.get("handles_response", False)
            and current_result.get("reliability", {}).get("is_reliable")
            is True
        )
        if current_working:
            return False

        # Check historical data for recent working status
        history = self.get_model_history(model_id, days=7)  # Check last week
        if not history:
            return False

        # Look for recent working states (excluding today)
        for entry in history:
            if (
                entry["tool_call_support"]
                and entry["handles_response"]
                and entry["is_reliable"] is True
            ):
                return True

        return False

    def is_newly_working(
        self, model_id: str, current_result: Dict[str, Any]
    ) -> bool:
        """Check if a model is newly working (wasn't working before but is now).

        Args:
            model_id: The model ID to check
            current_result: Current test result for the model

        Returns:
            bool: True if model is currently working but wasn't working in recent history
        """
        # Current model must be fully working for this badge to apply
        current_working = (
            current_result.get("tool_call_support", False)
            and current_result.get("handles_response", False)
            and current_result.get("reliability", {}).get("is_reliable")
            is True
        )
        if not current_working:
            return False

        # Check historical data
        history = self.get_model_history(model_id, days=7)  # Check last week
        if not history:
            # No history means it's newly working
            return True

        # Look for any previous working states
        for entry in history:
            if (
                entry["tool_call_support"]
                and entry["handles_response"]
                and entry["is_reliable"] is True
            ):
                # Was working before, so not newly working
                return False

        # Has history but was never fully working before
        return True

    def is_currently_broken(
        self, model_id: str, current_result: Dict[str, Any]
    ) -> bool:
        """Check if a model is currently broken (was working, now completely broken).

        Args:
            model_id: The model ID to check
            current_result: Current test result for the model

        Returns:
            bool: True if model was working but is now completely broken (no tool support)
        """
        # Current model must have no tool support for this badge to apply
        if current_result.get("tool_call_support", False):
            return False

        # Check historical data for recent working status
        history = self.get_model_history(model_id, days=7)  # Check last week
        if not history:
            return False

        # Look for recent working states
        for entry in history:
            if (
                entry["tool_call_support"]
                and entry["handles_response"]
                and entry["is_reliable"] is True
            ):
                return True

        return False
