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
            Dict[str, Any]: Test results
        """
        self.logger.info(f"Testing model {model_id}...")

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
                "response_times": {"total_time": None},
                "raw_response": None,
            }

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

            # Ensure the directory for the output file exists
            output_dir = os.path.dirname(self.config.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            handler = ResultHandler()
            json_output = handler.format_json_output(results)

            with open(self.config.output_file, "w") as f:
                f.write(json_output)

            self.logger.info(f"Results saved to {self.config.output_file}")

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
