#!/usr/bin/env python3
"""
Configuration module for WatsonX Tool Tester.

This module provides configuration classes and functions for setting up
the tool tester with the appropriate clients and parameters.
"""

import enum
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("watsonx_tool_tester")

# Reliability threshold for model evaluation
# Models with both tool_success_rate and response_success_rate >= this threshold are considered reliable
# Set to 0.9 (90%) to allow for occasional network issues or API timeouts while maintaining high quality
RELIABILITY_THRESHOLD = 0.9


def load_dotenv(dotenv_path=None):
    """Load environment variables from .env file.

    Args:
        dotenv_path: Optional path to .env file. If None, looks in current directory.

    Returns:
        bool: True if .env file was loaded, False otherwise
    """
    if dotenv_path is None:
        dotenv_path = Path(".env")

    if not dotenv_path.exists():
        return False

    try:
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse key-value pairs
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Don't override existing environment variables
                    if key not in os.environ:
                        os.environ[key] = value

        return True
    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")
        return False


class ClientType(enum.Enum):
    """Enumeration of supported client types."""

    WATSONX = "watsonx"
    LITELLM = "litellm"


@dataclass
class Config:
    """Configuration for the WatsonX Tool Tester.

    This class holds the configuration parameters for testing tool calls
    against different AI models.

    Attributes:
        client_type: The type of client to use (watsonx or litellm)
        watsonx_url: URL for WatsonX API
        watsonx_apikey: API key for WatsonX API
        watsonx_project_id: Project ID for WatsonX API
        watsonx_region: Region for WatsonX API
        watsonx_api_version: Version parameter for WatsonX API calls
        watsonx_model_spec_version: Version parameter for WatsonX model specs API
        litellm_url: URL for LiteLLM proxy API
        litellm_token: Authentication token for LiteLLM proxy
        model: Optional model ID to test a specific model
        exclude_models: List of models or regex patterns to exclude from testing
        exclude_file: Optional path to a file containing model names/patterns to exclude
        temperature: Temperature parameter for model generation
        max_tokens: Maximum tokens for model generation
        sort_key: How to sort the test results
        debug: Whether to enable debug logging
        log_dir: Directory for storing log files
        output_file: Optional file path to save results to
        html_output: Whether to generate HTML output format
        log_level: Log level for the application
    """

    client_type: ClientType = ClientType.WATSONX
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"
    watsonx_apikey: Optional[str] = None
    watsonx_project_id: Optional[str] = None
    watsonx_region: str = "us-south"
    watsonx_api_version: str = "2023-05-29"
    watsonx_model_spec_version: str = "2025-04-16"
    litellm_url: str = "http://localhost:8000"
    litellm_token: Optional[str] = None
    model: Optional[str] = None
    exclude_models: List[str] = field(
        default_factory=list
    )  # List of models or patterns to exclude
    exclude_file: Optional[str] = None  # File containing models to exclude
    temperature: float = 0.0  # Default to 0 for deterministic output
    max_tokens: int = 500  # Default to 500 tokens
    sort_key: str = "name"
    debug: bool = False
    log_dir: str = "tool_test_logs"
    output_file: Optional[str] = None
    html_output: bool = False
    log_level: str = "INFO"
    # Reliability testing parameters
    test_iterations: int = (
        5  # Number of times to test each model (default to 5 for reliability)
    )
    # History tracking parameters
    save_history: bool = False  # Whether to save test results to CSV files


def load_config_from_env() -> Config:
    """Load configuration from environment variables.

    First tries to load variables from .env file, then uses environment variables.

    Returns:
        Config: Configuration with values from environment variables
    """
    # Try to load .env file first
    dotenv_loaded = load_dotenv()
    if dotenv_loaded:
        logger.debug("Loaded environment variables from .env file")

    config = Config()

    # Client type
    client_env = os.environ.get("WATSONX_TOOL_CLIENT", "").lower()
    if client_env in ["watsonx", "litellm"]:
        config.client_type = ClientType(client_env)

    # WatsonX config
    if "WATSONX_URL" in os.environ:
        config.watsonx_url = os.environ["WATSONX_URL"]
    if "WATSONX_APIKEY" in os.environ:
        config.watsonx_apikey = os.environ["WATSONX_APIKEY"]
    if "WATSONX_PROJECT_ID" in os.environ:
        config.watsonx_project_id = os.environ["WATSONX_PROJECT_ID"]
    if "WATSONX_REGION" in os.environ:
        config.watsonx_region = os.environ["WATSONX_REGION"]
    if "WATSONX_API_VERSION" in os.environ:
        config.watsonx_api_version = os.environ["WATSONX_API_VERSION"]

    # LiteLLM config
    if "LITELLM_URL" in os.environ:
        config.litellm_url = os.environ["LITELLM_URL"]
    if "LITELLM_TOKEN" in os.environ:
        config.litellm_token = os.environ["LITELLM_TOKEN"]

    # Other config
    if "WATSONX_TOOL_MODEL" in os.environ:
        config.model = os.environ["WATSONX_TOOL_MODEL"]
    if "WATSONX_TOOL_EXCLUDE" in os.environ:
        # Parse comma-separated list of excluded models/patterns
        config.exclude_models = [
            m.strip()
            for m in os.environ["WATSONX_TOOL_EXCLUDE"].split(",")
            if m.strip()
        ]
    if "WATSONX_TOOL_EXCLUDE_FILE" in os.environ:
        config.exclude_file = os.environ["WATSONX_TOOL_EXCLUDE_FILE"]
    if "WATSONX_TOOL_SORT" in os.environ:
        config.sort_key = os.environ["WATSONX_TOOL_SORT"]
    if "WATSONX_TOOL_DEBUG" in os.environ:
        config.debug = os.environ["WATSONX_TOOL_DEBUG"].lower() in [
            "true",
            "yes",
            "1",
            "on",
        ]
    if "WATSONX_TOOL_LOG_DIR" in os.environ:
        config.log_dir = os.environ["WATSONX_TOOL_LOG_DIR"]
    if "WATSONX_TOOL_OUTPUT" in os.environ:
        config.output_file = os.environ["WATSONX_TOOL_OUTPUT"]
    if "WATSONX_TOOL_HTML_OUTPUT" in os.environ:
        config.html_output = os.environ[
            "WATSONX_TOOL_HTML_OUTPUT"
        ].lower() in [
            "true",
            "yes",
            "1",
            "on",
        ]
    if "LOG_LEVEL" in os.environ:
        config.log_level = os.environ["LOG_LEVEL"]
    if "WATSONX_TOOL_ITERATIONS" in os.environ:
        try:
            config.test_iterations = int(os.environ["WATSONX_TOOL_ITERATIONS"])
        except ValueError:
            logger.warning(
                f"Invalid WATSONX_TOOL_ITERATIONS value: {os.environ['WATSONX_TOOL_ITERATIONS']}"
            )

    return config


def update_config_from_args(config: Config, args: Dict[str, Any]) -> Config:
    """Update configuration with command-line arguments.

    Args:
        config: Base configuration to update
        args: Dictionary of command-line arguments

    Returns:
        Config: Updated configuration
    """
    # Client type
    if args.get("client"):
        config.client_type = ClientType(args["client"])

    # WatsonX config
    if args.get("watsonx_url"):
        config.watsonx_url = args["watsonx_url"]
    if args.get("watsonx_apikey"):
        config.watsonx_apikey = args["watsonx_apikey"]
    if args.get("watsonx_project_id"):
        config.watsonx_project_id = args["watsonx_project_id"]
    if args.get("watsonx_region"):
        config.watsonx_region = args["watsonx_region"]
    if args.get("watsonx_api_version"):
        config.watsonx_api_version = args["watsonx_api_version"]

    # LiteLLM config
    if args.get("litellm_url"):
        config.litellm_url = args["litellm_url"]
    if args.get("litellm_token"):
        config.litellm_token = args["litellm_token"]

    # Other config
    if args.get("model"):
        config.model = args["model"]
    if args.get("exclude"):
        config.exclude_models = args["exclude"]
    if args.get("exclude_file"):
        config.exclude_file = args["exclude_file"]
    if args.get("sort"):
        config.sort_key = args["sort"]
    if "debug" in args:
        config.debug = args["debug"]
    if args.get("log_dir"):
        config.log_dir = args["log_dir"]
    if args.get("output"):
        config.output_file = args["output"]
    if "html_output" in args:
        config.html_output = args["html_output"]
    if args.get("log_level"):
        config.log_level = args["log_level"]
    if args.get("iterations"):
        config.test_iterations = args["iterations"]
    if "save_history" in args:
        config.save_history = args["save_history"]

    return config


def validate_config(config: Config) -> Tuple[bool, Optional[str]]:
    """Validate the configuration.

    Args:
        config: The configuration to validate

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if config.client_type == ClientType.WATSONX:
        # Check required WatsonX parameters
        if not config.watsonx_apikey:
            return (
                False,
                "WatsonX API key is required. Set WATSONX_APIKEY environment variable or use --watsonx-apikey.",
            )
        if not config.watsonx_project_id:
            return (
                False,
                "WatsonX Project ID is required. Set WATSONX_PROJECT_ID environment variable or use --watsonx-project-id.",
            )

    elif config.client_type == ClientType.LITELLM:
        # Check required LiteLLM parameters
        if not config.litellm_url:
            return (
                False,
                "LiteLLM URL is required. Set LITELLM_URL environment variable or use --litellm-url.",
            )

    # Validate sort key
    valid_sort_keys = ["name", "tool_call_time", "response_time", "total_time"]
    if config.sort_key not in valid_sort_keys:
        return (
            False,
            f"Invalid sort key '{config.sort_key}'. Valid options are: {', '.join(valid_sort_keys)}",
        )

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.log_level.upper() not in valid_log_levels:
        return (
            False,
            f"Invalid log level '{config.log_level}'. Valid options are: {', '.join(valid_log_levels)}",
        )

    return True, None
