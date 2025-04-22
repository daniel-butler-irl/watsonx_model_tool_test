#!/usr/bin/env python3
"""
Tests for the configuration module.

This module contains tests for configuration loading, validation, and updating.
"""

import os
from unittest import mock

from watsonx_tool_tester.config import (
    ClientType,
    Config,
    load_config_from_env,
    update_config_from_args,
    validate_config,
)


def test_default_config():
    """Test that the default configuration has the expected values."""
    config = Config()
    assert config.client_type == ClientType.WATSONX
    assert config.watsonx_url == "https://us-south.ml.cloud.ibm.com"
    assert config.watsonx_apikey is None
    assert config.watsonx_project_id is None
    assert config.watsonx_region == "us-south"
    assert config.litellm_url == "http://localhost:8000"
    assert config.litellm_token is None
    assert config.model is None
    assert config.sort_key == "name"
    assert config.debug is False
    assert config.log_dir == "tool_test_logs"
    assert config.output_file is None


@mock.patch.dict(
    os.environ,
    {
        "WATSONX_TOOL_CLIENT": "litellm",
        "WATSONX_URL": "https://eu-de.ml.cloud.ibm.com",
        "WATSONX_APIKEY": "test-api-key",
        "WATSONX_PROJECT_ID": "test-project-id",
        "WATSONX_REGION": "eu-de",
        "LITELLM_URL": "http://localhost:9000",
        "LITELLM_TOKEN": "test-token",
        "WATSONX_TOOL_MODEL": "test-model",
        "WATSONX_TOOL_SORT": "total_time",
        "WATSONX_TOOL_DEBUG": "true",
        "WATSONX_TOOL_LOG_DIR": "custom-logs",
        "WATSONX_TOOL_OUTPUT": "output.json",
    },
)
def test_load_config_from_env():
    """Test loading configuration from environment variables."""
    config = load_config_from_env()

    assert config.client_type == ClientType.LITELLM
    assert config.watsonx_url == "https://eu-de.ml.cloud.ibm.com"
    assert config.watsonx_apikey == "test-api-key"
    assert config.watsonx_project_id == "test-project-id"
    assert config.watsonx_region == "eu-de"
    assert config.litellm_url == "http://localhost:9000"
    assert config.litellm_token == "test-token"
    assert config.model == "test-model"
    assert config.sort_key == "total_time"
    assert config.debug is True
    assert config.log_dir == "custom-logs"
    assert config.output_file == "output.json"


def test_update_config_from_args():
    """Test updating configuration with command-line arguments."""
    config = Config()
    args = {
        "client": "litellm",
        "watsonx_url": "https://test-url.com",
        "watsonx_apikey": "cli-api-key",
        "watsonx_project_id": "cli-project-id",
        "watsonx_region": "test-region",
        "litellm_url": "http://test:8080",
        "litellm_token": "cli-token",
        "model": "cli-model",
        "sort": "response_time",
        "debug": True,
        "log_dir": "cli-logs",
        "output": "cli-output.json",
    }

    updated_config = update_config_from_args(config, args)

    assert updated_config.client_type == ClientType.LITELLM
    assert updated_config.watsonx_url == "https://test-url.com"
    assert updated_config.watsonx_apikey == "cli-api-key"
    assert updated_config.watsonx_project_id == "cli-project-id"
    assert updated_config.watsonx_region == "test-region"
    assert updated_config.litellm_url == "http://test:8080"
    assert updated_config.litellm_token == "cli-token"
    assert updated_config.model == "cli-model"
    assert updated_config.sort_key == "response_time"
    assert updated_config.debug is True
    assert updated_config.log_dir == "cli-logs"
    assert updated_config.output_file == "cli-output.json"


def test_update_config_with_partial_args():
    """Test updating configuration with partial command-line arguments."""
    config = Config(
        watsonx_apikey="original-key",
        watsonx_project_id="original-project",
    )
    args = {
        "debug": True,
        "model": "partial-model",
    }

    updated_config = update_config_from_args(config, args)

    assert updated_config.client_type == ClientType.WATSONX
    assert updated_config.watsonx_apikey == "original-key"
    assert updated_config.watsonx_project_id == "original-project"
    assert updated_config.model == "partial-model"
    assert updated_config.debug is True


def test_validate_config_watsonx_valid():
    """Test validation of a valid WatsonX configuration."""
    config = Config(
        client_type=ClientType.WATSONX,
        watsonx_apikey="test-key",
        watsonx_project_id="test-project",
    )

    is_valid, message = validate_config(config)

    assert is_valid is True
    assert message is None


def test_validate_config_watsonx_invalid():
    """Test validation of an invalid WatsonX configuration."""
    config = Config(
        client_type=ClientType.WATSONX,
        watsonx_apikey=None,  # Missing API key
        watsonx_project_id="test-project",
    )

    is_valid, message = validate_config(config)

    assert is_valid is False
    assert "API key" in message


def test_validate_config_litellm_valid():
    """Test validation of a valid LiteLLM configuration."""
    config = Config(
        client_type=ClientType.LITELLM,
        litellm_url="http://test:8000",
    )

    is_valid, message = validate_config(config)

    assert is_valid is True
    assert message is None


def test_validate_config_invalid_sort():
    """Test validation with invalid sort key."""
    config = Config(
        client_type=ClientType.WATSONX,
        watsonx_apikey="test-key",
        watsonx_project_id="test-project",
        sort_key="invalid",
    )

    is_valid, message = validate_config(config)

    assert is_valid is False
    assert "sort key" in message.lower()
