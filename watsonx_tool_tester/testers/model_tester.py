#!/usr/bin/env python3
"""
Model tester for WatsonX Tool Tester.

This module provides the ModelTester class for testing tool call
capabilities of AI models.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Union

import click

from watsonx_tool_tester.clients.litellm import LiteLLMClient
from watsonx_tool_tester.clients.watsonx import WatsonXClient
from watsonx_tool_tester.config import ClientType, Config
from watsonx_tool_tester.testers.result_handler import ResultHandler
from watsonx_tool_tester.utils import errors
from watsonx_tool_tester.utils import logging as log_utils
from watsonx_tool_tester.utils.history_manager import HistoryManager


class ModelTester:
    """Tester for AI model tool call capabilities.

    This class coordinates the testing process, including client initialization,
    model discovery, test execution, and result processing.

    Attributes:
        config: The configuration to use for testing
        client: The API client to use for testing
        logger: Logger for the model tester
    """

    client: Union[WatsonXClient, LiteLLMClient]

    def __init__(self, config: Config):
        """Initialize the model tester.

        Args:
            config: The configuration to use for testing
        """
        self.config = config
        self.logger = logging.getLogger("watsonx_tool_tester.tester")

        # Set log level based on config
        if config.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            log_level = log_utils.get_log_level(config.log_level)
            self.logger.setLevel(log_level)

        # Initialize the appropriate client based on the client type
        if config.client_type == ClientType.WATSONX:
            client_config = {
                "base_url": config.watsonx_url,
                "api_key": config.watsonx_apikey,
                "project_id": config.watsonx_project_id,
                "region": config.watsonx_region,
                "debug": config.debug,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            self.client = WatsonXClient(client_config)
        elif config.client_type == ClientType.LITELLM:
            client_config = {
                "base_url": config.litellm_url,
                "auth_token": config.litellm_token,
                "debug": config.debug,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            self.client = LiteLLMClient(client_config)
        else:
            raise ValueError(f"Unknown client type: {config.client_type}")

        # Set up logging directory
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

    def validate_credentials(self) -> bool:
        """Validate API credentials.

        Returns:
            bool: True if the credentials are valid, False otherwise
        """
        self.logger.info("Validating API credentials...")

        try:
            valid = self.client.validate_credentials()

            if valid:
                self.logger.info("Credentials are valid")
            else:
                self.logger.error("Invalid credentials")

            return valid

        except errors.CredentialError as e:
            self.logger.error(f"Credential validation failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error during credential validation: {str(e)}"
            )
            return False

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models from the API.

        Returns:
            List[Dict[str, Any]]: List of available models
        """
        self.logger.info("Getting available models...")

        try:
            models = self.client.get_models()

            if not models:
                self.logger.warning("No models found")
            else:
                self.logger.info(f"Found {len(models)} models")

            return models

        except Exception as e:
            self.logger.error(f"Error getting models: {str(e)}")
            return []

    def test_model(self, model_id: str) -> Dict[str, Any]:
        """Test a model's tool call capabilities.

        Args:
            model_id: The ID of the model to test

        Returns:
            Dict[str, Any]: Test results with reliability metrics if multiple iterations
        """
        self.logger.info(f"Testing model {model_id}...")

        # If only one iteration, use the original simple test
        if self.config.test_iterations == 1:
            return self._test_model_single(model_id)

        # Multiple iterations for reliability testing
        return self._test_model_multiple_iterations(model_id)

    def _test_model_single(self, model_id: str) -> Dict[str, Any]:
        """Test a model once (original behavior).

        Args:
            model_id: The ID of the model to test

        Returns:
            Dict[str, Any]: Test results
        """
        try:
            # Test the model
            result = self.client.test_hello_world_tool(model_id)
            (
                tool_call_support,
                handles_response,
                details,
                response_data,
                response_times,
            ) = result

            # Format test result
            test_result = {
                "model": model_id,
                "tool_call_support": tool_call_support,
                "handles_response": handles_response,
                "details": details,
                "response_times": response_times,
                "raw_response": response_data,
            }

            # Log result
            support_str = (
                "supports" if tool_call_support else "does not support"
            )
            response_str = "handles" if handles_response else "does not handle"
            self.logger.info(
                f"Model {model_id} {support_str} tool calls and {response_str} responses"
            )

            # Save full response to log file if debug mode is enabled
            if self.config.debug and response_data:
                log_file = os.path.join(
                    self.config.log_dir,
                    f"{model_id.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json",
                )

                # The log_file will always be a string at this point since we're joining strings
                # and the config.log_dir is created in __init__ if it doesn't exist
                with open(log_file, "w") as f:
                    json.dump(response_data, f, indent=2)

                self.logger.debug(f"Saved full response to {log_file}")

            return test_result

        except Exception as e:
            self.logger.error(f"Error testing model {model_id}: {str(e)}")

            # Return result with error
            return {
                "model": model_id,
                "tool_call_support": False,
                "handles_response": False,
                "details": f"Error: {str(e)}",
                "response_times": {
                    "tool_call_time": 0,
                    "response_processing_time": 0,
                    "total_time": 0,
                },
                "raw_response": None,
            }

    def _test_model_multiple_iterations(self, model_id: str) -> Dict[str, Any]:
        """Test a model multiple times to assess reliability.

        Args:
            model_id: The ID of the model to test

        Returns:
            Dict[str, Any]: Test results with reliability metrics
        """
        iterations = self.config.test_iterations
        # Always run minimum 5 iterations unless user specifically requested fewer
        min_required_iterations = min(5, iterations)
        self.logger.info(
            f"Running {iterations} iterations for reliability testing (minimum {min_required_iterations} required)..."
        )

        # Track results across iterations
        tool_call_successes = 0
        response_handling_successes = 0
        all_results = []
        total_tool_call_time = 0
        total_response_time = 0
        total_time = 0
        error_count = 0
        error_details = []

        for i in range(iterations):
            max_retries = 3  # Number of retries for intermittent errors
            retry_count = 0

            while retry_count <= max_retries:
                try:
                    if iterations > 1:
                        self.logger.debug(
                            f"Testing model {model_id} - iteration {i+1}/{iterations}"
                        )

                    # Test the model
                    result = self.client.test_hello_world_tool(model_id)
                    (
                        tool_call_support,
                        handles_response,
                        details,
                        response_data,
                        response_times,
                    ) = result

                    # Check if this is a definitive "not supported" result
                    # Only terminate early after minimum required iterations to avoid false positives
                    if (
                        i + 1
                    ) >= min_required_iterations and not tool_call_support:
                        # Check if all attempts so far have been definitive failures
                        all_failed = True
                        definitive_failures = 0
                        for prev_result in all_results:
                            if prev_result.get("tool_call_support", False):
                                all_failed = False
                                break
                            if self._is_definitive_not_supported(
                                prev_result.get("details", "")
                            ):
                                definitive_failures += 1

                        # Check if model has historical working status - if so, be more lenient
                        has_previously_worked = self._has_previously_worked(
                            model_id
                        )

                        # Require ALL minimum iterations to be definitive failures before terminating
                        required_definitive_failures = min_required_iterations

                        if has_previously_worked:
                            self.logger.debug(
                                f"Model {model_id} has worked before - requiring all {min_required_iterations} minimum iterations to be definitive failures"
                            )

                        # Only terminate early if ALL minimum iterations failed AND ALL were definitive failures
                        if (
                            all_failed
                            and definitive_failures
                            >= required_definitive_failures
                        ):
                            self.logger.info(
                                f"Model {model_id} consistently fails tool calling after {i + 1} attempts ({definitive_failures} definitive failures) - skipping remaining iterations"
                            )
                            # Return early with current results for definitively unsupported models
                            return self._create_unsupported_result(
                                model_id, details, response_times
                            )

                    # Track successes
                    if tool_call_support:
                        tool_call_successes += 1
                    if handles_response:
                        response_handling_successes += 1

                    # Accumulate timing data
                    total_tool_call_time += (
                        response_times.get("tool_call_time", 0) or 0
                    )
                    total_response_time += (
                        response_times.get("response_processing_time", 0) or 0
                    )
                    total_time += response_times.get("total_time", 0) or 0

                    # Store individual result
                    iteration_result = {
                        "iteration": i + 1,
                        "tool_call_support": tool_call_support,
                        "handles_response": handles_response,
                        "details": details,
                        "response_times": response_times,
                    }
                    all_results.append(iteration_result)

                    # Save detailed logs for each iteration in debug mode
                    if self.config.debug and response_data:
                        log_file = os.path.join(
                            self.config.log_dir,
                            f"{model_id.replace('/', '_')}-iter{i+1}-{time.strftime('%Y%m%d-%H%M%S')}.json",
                        )
                        with open(log_file, "w") as f:
                            json.dump(response_data, f, indent=2)

                    # Early failure detection with probe iterations
                    # After completing probe iterations, check if we should continue
                    PROBE_ITERATIONS = 5
                    MIN_SUCCESS_THRESHOLD = 1

                    if (
                        i + 1 == PROBE_ITERATIONS
                        and iterations > PROBE_ITERATIONS
                    ):
                        # Check if we have any successes in probe iterations
                        probe_successes = sum(
                            1
                            for r in all_results
                            if r.get("tool_call_support", False)
                        )

                        if probe_successes < MIN_SUCCESS_THRESHOLD:
                            self.logger.info(
                                f"Model {model_id} failed probe iterations ({probe_successes}/{PROBE_ITERATIONS} successes) - skipping remaining iterations"
                            )
                            # Create early termination result with probe data
                            return self._create_probe_failure_result(
                                model_id,
                                all_results,
                                tool_call_successes,
                                response_handling_successes,
                                total_tool_call_time,
                                total_response_time,
                                total_time,
                                error_details,
                                PROBE_ITERATIONS,
                            )

                    # Break out of retry loop on success
                    break

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Iteration {i+1}: {str(e)}"

                    # Check if this is an intermittent error that should be retried
                    if (
                        retry_count <= max_retries
                        and self._is_intermittent_error(str(e))
                    ):
                        self.logger.warning(
                            f"Intermittent error in iteration {i+1} for model {model_id} (retry {retry_count}/{max_retries}): {str(e)}"
                        )
                        time.sleep(1)  # Brief pause before retry
                        continue

                    # Max retries reached or non-intermittent error
                    error_count += 1
                    error_details.append(error_msg)
                    self.logger.error(
                        f"Error in iteration {i+1} for model {model_id}: {str(e)}"
                    )

                    # Store error result
                    iteration_result = {
                        "iteration": i + 1,
                        "tool_call_support": False,
                        "handles_response": False,
                        "details": error_msg,
                        "response_times": {
                            "tool_call_time": 0,
                            "response_processing_time": 0,
                            "total_time": 0,
                        },
                    }
                    all_results.append(iteration_result)
                    break  # Exit retry loop

        # Calculate reliability metrics
        successful_iterations = iterations - error_count
        tool_call_success_rate = (
            tool_call_successes / iterations if iterations > 0 else 0
        )

        # Response handling success rate should only count iterations where tool calls succeeded
        # If tool call fails, we can't evaluate response handling for that iteration
        response_handling_success_rate = (
            response_handling_successes / tool_call_successes
            if tool_call_successes > 0
            else 0
        )

        # Determine overall support - ANY successful tool calls indicate support
        # This allows us to properly categorize models that work sometimes vs never
        overall_tool_call_support = tool_call_successes > 0
        overall_handles_response = response_handling_successes > 0

        # Calculate reliability based on consistency
        if overall_tool_call_support:
            # Models with some tool call support can be reliable or unreliable
            is_reliable = (
                tool_call_success_rate == 1.0
                and response_handling_success_rate == 1.0
            )
        else:
            # Models with no tool call support are not classified as reliable/unreliable
            is_reliable = None

        # Calculate average timings (only for successful iterations)
        avg_tool_call_time = (
            total_tool_call_time / successful_iterations
            if successful_iterations > 0
            else 0
        )
        avg_response_time = (
            total_response_time / successful_iterations
            if successful_iterations > 0
            else 0
        )
        avg_total_time = (
            total_time / successful_iterations
            if successful_iterations > 0
            else 0
        )

        # Use the most common error details, or success message if reliable
        if is_reliable is True:
            details = "Consistent success across all iterations"
        elif is_reliable is False:
            # Find the most common failure pattern for supported but unreliable models
            if error_count > 0:
                details = error_details[0]  # Show first error as primary issue
            else:
                details = (
                    f"Inconsistent results across {iterations} iterations"
                )
        else:
            # Not supported models - use the most common error
            if error_count > 0:
                details = error_details[0]
            else:
                details = "Tool calling not supported"

        # Log reliability summary
        if is_reliable is None:
            reliability_status = "NOT SUPPORTED"
        else:
            reliability_status = "RELIABLE" if is_reliable else "UNRELIABLE"

        self.logger.info(
            f"Model {model_id} - {reliability_status}: "
            f"Tool calls: {tool_call_successes}/{iterations} ({tool_call_success_rate:.1%}), "
            f"Response handling: {response_handling_successes}/{iterations} ({response_handling_success_rate:.1%})"
        )

        # Format comprehensive test result
        test_result = {
            "model": model_id,
            "tool_call_support": overall_tool_call_support,
            "handles_response": overall_handles_response,
            "details": details,
            "response_times": {
                "tool_call_time": avg_tool_call_time,
                "response_processing_time": avg_response_time,
                "total_time": avg_total_time,
            },
            # Reliability metrics
            "reliability": {
                "iterations": iterations,
                "tool_call_success_rate": tool_call_success_rate,
                "response_handling_success_rate": response_handling_success_rate,
                "is_reliable": is_reliable,
                "error_count": error_count,
            },
            "raw_response": None,  # Don't store raw responses for multiple iterations
            "iteration_results": all_results,  # Store individual iteration results
        }

        return test_result

    def test_all_models(
        self, filter_model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Test all available models, or a specific model if filter_model is provided.

        Args:
            filter_model: Optional model ID to test only that model

        Returns:
            List[Dict[str, Any]]: Test results for all tested models
        """
        # Set up logging
        log_utils.setup_logger(
            debug=self.config.debug,
            log_dir=self.config.log_dir,
            log_level=self.config.log_level,
            file_logging=False,  # Disable file logging by default
        )

        # Validate credentials
        if not self.validate_credentials():
            raise errors.CredentialError("Invalid API credentials")

        # Get available models
        all_models = self.get_available_models()

        if not all_models:
            raise errors.ClientError("No models found")

        # Filter models if requested
        if filter_model:
            self.logger.info(f"Filtering for model: {filter_model}")
            models_to_test = [
                m
                for m in all_models
                if filter_model.lower() in m["id"].lower()
            ]

            if not models_to_test:
                available_models = ", ".join(m["id"] for m in all_models)
                raise errors.ConfigurationError(
                    f"Model {filter_model} not found. Available models: {available_models}"
                )
        else:
            models_to_test = all_models

        # Exclude models based on exclusion patterns
        models_to_test = self._filter_excluded_models(models_to_test)

        # Test models
        self.logger.info(f"Testing {len(models_to_test)} models...")
        results = []

        with click.progressbar(
            models_to_test,
            label="Testing models",
            item_show_func=lambda m: m["id"] if m else "",
        ) as bar:
            for model in bar:
                model_id = model["id"]
                result = self.test_model(model_id)
                results.append(result)

        # Sort results
        handler = ResultHandler()
        results = handler.sort_results(results, self.config.sort_key)

        # Save results to file if specified
        if self.config.output_file:
            self._save_results_to_file(results)

        return results

    def _save_results_to_file(self, results: List[Dict[str, Any]]) -> None:
        """Save test results to a file.

        Args:
            results: Test results to save
        """
        try:
            # Ensure we have a valid output file path
            if not self.config.output_file:
                self.logger.error("No output file specified")
                return

            self.logger.info(f"Saving results to {self.config.output_file}")

            # CRITICAL FIX: Save to CSV BEFORE generating HTML report
            # This ensures the HTML generator can find today's data when checking the timeline
            if self.config.save_history:
                self.logger.info(
                    "Saving results to CSV history before generating HTML report"
                )
                history_manager = HistoryManager()
                history_manager.record_test_results(results)
                self.logger.info("CSV history saved successfully")

            # Ensure the directory for the output file exists
            output_dir = os.path.dirname(self.config.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            handler = ResultHandler()

            # Generate output based on format preference
            if self.config.html_output:
                # Create config info for HTML report
                config_info = {
                    "iterations": self.config.test_iterations,
                    "client_type": self.config.client_type.value,
                    "total_models": len(results),
                }

                # Generate HTML output
                html_output = handler.format_html_output(results, config_info)

                # Ensure output file has .html extension
                output_file = self.config.output_file
                if not output_file.endswith(".html"):
                    output_file = output_file.rsplit(".", 1)[0] + ".html"

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(html_output)

                self.logger.info(f"HTML report saved to {output_file}")
            else:
                # Generate JSON output
                json_output = handler.format_json_output(results)

                with open(self.config.output_file, "w") as f:
                    f.write(json_output)

                self.logger.info(
                    f"JSON results saved to {self.config.output_file}"
                )

        except Exception as e:
            self.logger.error(f"Error saving results to file: {str(e)}")

    def _load_exclusion_patterns(self) -> Set[str]:
        """Load model exclusion patterns from config and exclusion file.

        Returns:
            Set[str]: Set of model exclusion patterns
        """
        exclusion_patterns = set(self.config.exclude_models)

        # Load patterns from exclusion file if specified
        if self.config.exclude_file and os.path.exists(
            self.config.exclude_file
        ):
            try:
                self.logger.info(
                    f"Loading exclusion patterns from {self.config.exclude_file}"
                )
                with open(self.config.exclude_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if not line or line.startswith("#"):
                            continue
                        exclusion_patterns.add(line)
                self.logger.info(
                    f"Loaded {len(exclusion_patterns)} exclusion patterns"
                )
            except Exception as e:
                self.logger.error(f"Error loading exclusion file: {str(e)}")

        return exclusion_patterns

    def _should_exclude_model(self, model_id: str, patterns: Set[str]) -> bool:
        """Determine if a model should be excluded based on patterns.

        Args:
            model_id: The ID of the model to check
            patterns: Set of exclusion patterns (direct names or regex patterns)

        Returns:
            bool: True if the model should be excluded, False otherwise
        """
        # Direct name match
        if model_id in patterns:
            return True

        # Try regex patterns
        for pattern in patterns:
            try:
                if re.search(pattern, model_id):
                    return True
            except re.error:
                # Not a valid regex, treat as literal string match
                pass

        return False

    def _filter_excluded_models(
        self, models: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out models based on exclusion patterns.

        Args:
            models: List of model dictionaries

        Returns:
            List[Dict[str, Any]]: Filtered list of models
        """
        if not self.config.exclude_models and not self.config.exclude_file:
            return models

        exclusion_patterns = self._load_exclusion_patterns()
        if not exclusion_patterns:
            return models

        original_count = len(models)
        filtered_models = [
            model
            for model in models
            if not self._should_exclude_model(model["id"], exclusion_patterns)
        ]

        excluded_count = original_count - len(filtered_models)
        if excluded_count > 0:
            self.logger.info(
                f"Excluded {excluded_count} models based on exclusion patterns"
            )

        return filtered_models

    def _is_definitive_not_supported(self, details: str) -> bool:
        """Check if error details indicate definitive lack of tool calling support.

        Args:
            details: Error details string

        Returns:
            bool: True if this is a definitive "not supported" error
        """
        # Only API-level errors that definitively indicate lack of tool calling support
        # These are server-side rejections, not response-level issues
        definitive_not_supported_patterns = [
            "unsupported parameter: tools",
            "unsupported parameter: functions",
            "invalid parameter: tools",
            "invalid parameter: functions",
            "tool calling not supported by this model",
            "function calling not supported by this model",
        ]

        # Patterns that suggest the model tried to respond but failed - these are intermittent
        # Should NOT be treated as definitive lack of support
        intermittent_patterns = [
            "does not support tool calling",  # Often followed by "Finish reason: stop" - model attempted response
            "does not support function calling",  # Same as above
            "did not use hello_world tool",  # Model confusion, not lack of support
            "no tool calls in response",  # Could be temporary failure
            "finish reason: stop",  # Model gave a response, just not proper tool call
            "tools are not supported",  # Could be response-level confusion
            "tool calls are not supported",  # Could be response-level confusion
            "function calls are not supported",  # Could be response-level confusion
        ]

        details_lower = details.lower()

        # First check if this matches intermittent patterns - if so, it's NOT definitive
        for pattern in intermittent_patterns:
            if pattern.lower() in details_lower:
                return False

        # Check for definitive patterns
        for pattern in definitive_not_supported_patterns:
            if pattern.lower() in details_lower:
                return True

        # Check for HTTP error codes that definitively indicate lack of support
        # But only if they contain "unsupported" or "invalid", not general tool/function mentions
        if (
            "400" in details_lower
            and "unsupported parameter" in details_lower
            and ("tool" in details_lower or "function" in details_lower)
        ):
            return True
        if (
            "422" in details_lower
            and "unsupported parameter" in details_lower
            and ("tool" in details_lower or "function" in details_lower)
        ):
            return True

        # Default to not definitive - be conservative about early termination
        return False

    def _has_previously_worked(self, model_id: str) -> bool:
        """Check if a model has previously worked with tool calling.

        Args:
            model_id: The model ID to check

        Returns:
            bool: True if model has successful tool calling history
        """
        try:
            from watsonx_tool_tester.utils.history_manager import (
                HistoryManager,
            )

            history_manager = HistoryManager()

            # Check last 14 days for any successful tool calling
            history = history_manager.get_model_history(model_id, days=14)

            for entry in history:
                if (
                    entry.get("tool_call_support", False)
                    and entry.get("handles_response", False)
                    and entry.get("is_reliable")
                    is not None  # Was actually tested, not just assumed
                ):
                    return True

            return False

        except Exception as e:
            # If we can't check history, be conservative and assume no history
            self.logger.debug(f"Could not check history for {model_id}: {e}")
            return False

    def _is_intermittent_error(self, error_msg: str) -> bool:
        """Check if error might be intermittent and worth retrying.

        Args:
            error_msg: Error message string

        Returns:
            bool: True if this might be an intermittent error
        """
        # Common patterns that indicate intermittent/network issues
        intermittent_patterns = [
            "timeout",
            "connection",
            "network",
            "502 Bad Gateway",
            "503 Service Unavailable",
            "504 Gateway Timeout",
            "500 Internal Server Error",
            "ConnectionError",
            "ReadTimeout",
            "HTTPError",
            "RequestException",
            "too many requests",
            "rate limit",
            "temporarily unavailable",
            "service unavailable",
        ]

        error_lower = error_msg.lower()
        return any(
            pattern.lower() in error_lower for pattern in intermittent_patterns
        )

    def _create_unsupported_result(
        self,
        model_id: str,
        details: str,
        response_times: Dict[str, Optional[float]],
    ) -> Dict[str, Any]:
        """Create a result for an unsupported model.

        Args:
            model_id: Model identifier
            details: Error details
            response_times: Response timing information

        Returns:
            Dict[str, Any]: Test result for unsupported model
        """
        return {
            "model": model_id,
            "tool_call_support": False,
            "handles_response": False,
            "details": details,
            "response_times": {
                "tool_call_time": response_times.get("tool_call_time", 0),
                "response_processing_time": response_times.get(
                    "response_processing_time", 0
                ),
                "total_time": response_times.get("total_time", 0),
            },
            # Single iteration result for unsupported models
            "reliability": {
                "iterations": 1,
                "tool_call_success_rate": 0.0,
                "response_handling_success_rate": 0.0,
                "is_reliable": None,  # None indicates not supported
                "error_count": 0,
            },
            "raw_response": None,
        }

    def _create_probe_failure_result(
        self,
        model_id: str,
        all_results: List[Dict[str, Any]],
        tool_call_successes: int,
        response_handling_successes: int,
        total_tool_call_time: float,
        total_response_time: float,
        total_time: float,
        error_details: List[str],
        probe_iterations: int,
    ) -> Dict[str, Any]:
        """Create a result for a model that failed probe iterations.

        Args:
            model_id: Model identifier
            all_results: Results from probe iterations
            tool_call_successes: Number of successful tool calls
            response_handling_successes: Number of successful response handlings
            total_tool_call_time: Total tool call time from probe iterations
            total_response_time: Total response processing time from probe iterations
            total_time: Total time from probe iterations
            error_details: List of error messages
            probe_iterations: Number of probe iterations completed

        Returns:
            Dict[str, Any]: Test result for probe failure
        """
        # Calculate rates based on probe iterations
        tool_call_success_rate = (
            tool_call_successes / probe_iterations
            if probe_iterations > 0
            else 0
        )
        response_handling_success_rate = (
            response_handling_successes / tool_call_successes
            if tool_call_successes > 0
            else 0
        )

        # Calculate average timings
        successful_iterations = len(
            [r for r in all_results if r.get("tool_call_support", False)]
        )
        avg_tool_call_time = (
            total_tool_call_time / successful_iterations
            if successful_iterations > 0
            else 0
        )
        avg_response_time = (
            total_response_time / successful_iterations
            if successful_iterations > 0
            else 0
        )
        avg_total_time = (
            total_time / successful_iterations
            if successful_iterations > 0
            else 0
        )

        # Determine overall support
        overall_tool_call_support = tool_call_successes > 0
        overall_handles_response = response_handling_successes > 0

        # Create details message
        details = f"Failed probe iterations ({tool_call_successes}/{probe_iterations} successes)"
        if error_details:
            # Use the most common error as primary detail
            from collections import Counter

            most_common_error = Counter(error_details).most_common(1)[0][0]
            details = most_common_error

        return {
            "model": model_id,
            "tool_call_support": overall_tool_call_support,
            "handles_response": overall_handles_response,
            "details": details,
            "response_times": {
                "tool_call_time": avg_tool_call_time,
                "response_processing_time": avg_response_time,
                "total_time": avg_total_time,
            },
            # Probe iteration reliability data
            "reliability": {
                "iterations": probe_iterations,
                "tool_call_success_rate": tool_call_success_rate,
                "response_handling_success_rate": response_handling_success_rate,
                "is_reliable": (
                    False if overall_tool_call_support else None
                ),  # Not reliable if mostly failing
                "error_count": len(error_details),
            },
            "raw_response": None,
            "iteration_results": all_results,  # Include probe results for analysis
        }
