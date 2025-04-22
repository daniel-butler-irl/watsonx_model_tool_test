#!/usr/bin/env python3
"""
Tests for the ModelTester class.

This module contains tests for the ModelTester class functionality.
"""

from unittest import mock

import pytest

from tests.mocks.mock_client import MockClient
from watsonx_tool_tester.config import ClientType, Config
from watsonx_tool_tester.testers.model_tester import ModelTester
from watsonx_tool_tester.utils.errors import (
    ClientError,
    ConfigurationError,
    CredentialError,
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        client_type=ClientType.WATSONX,
        watsonx_url="https://mock-api.test",
        watsonx_apikey="mock-key",
        watsonx_project_id="mock-project",
        watsonx_region="us-south",
        debug=True,
        log_dir="mock_logs",
        output_file="mock_output.json",
    )


@pytest.fixture
def mock_config_with_exclusions():
    """Create a mock configuration with model exclusions for testing."""
    return Config(
        client_type=ClientType.WATSONX,
        watsonx_url="https://mock-api.test",
        watsonx_apikey="mock-key",
        watsonx_project_id="mock-project",
        watsonx_region="us-south",
        debug=True,
        log_dir="mock_logs",
        output_file="mock_output.json",
        exclude_models=["mock/model-2", "nonexistent-model"],
    )


@pytest.fixture
def model_tester(mock_config):
    """Create a ModelTester instance with mocked client."""
    with mock.patch(
        "watsonx_tool_tester.testers.model_tester.WatsonXClient",
        return_value=MockClient(),
    ):
        with mock.patch(
            "watsonx_tool_tester.testers.model_tester.log_utils.setup_logger"
        ):
            tester = ModelTester(mock_config)
            yield tester


@pytest.fixture
def model_tester_with_exclusions(mock_config_with_exclusions):
    """Create a ModelTester instance with mocked client and exclusion config."""
    with mock.patch(
        "watsonx_tool_tester.testers.model_tester.WatsonXClient",
        return_value=MockClient(),
    ):
        with mock.patch(
            "watsonx_tool_tester.testers.model_tester.log_utils.setup_logger"
        ):
            tester = ModelTester(mock_config_with_exclusions)
            yield tester


def test_initialization(mock_config):
    """Test initialization of the ModelTester."""
    # Test with WatsonX client
    with mock.patch(
        "watsonx_tool_tester.testers.model_tester.WatsonXClient"
    ) as mock_watsonx:
        with mock.patch(
            "watsonx_tool_tester.testers.model_tester.os.makedirs"
        ):
            tester = ModelTester(mock_config)
            mock_watsonx.assert_called_once()
            assert tester.config == mock_config

    # Test with LiteLLM client
    mock_config.client_type = ClientType.LITELLM
    mock_config.litellm_url = "https://mock-litellm.test"
    mock_config.litellm_token = "mock-token"

    with mock.patch(
        "watsonx_tool_tester.testers.model_tester.LiteLLMClient"
    ) as mock_litellm:
        with mock.patch(
            "watsonx_tool_tester.testers.model_tester.os.makedirs"
        ):
            tester = ModelTester(mock_config)
            mock_litellm.assert_called_once()

    # Test with invalid client type
    mock_config.client_type = "invalid"
    with pytest.raises(ValueError):
        with mock.patch(
            "watsonx_tool_tester.testers.model_tester.os.makedirs"
        ):
            ModelTester(mock_config)


def test_validate_credentials(model_tester):
    """Test credential validation."""
    # Test successful validation
    assert model_tester.validate_credentials() is True

    # Test failed validation
    model_tester.client.validation_success = False
    assert model_tester.validate_credentials() is False

    # Test exception handling
    model_tester.client.validate_credentials = mock.MagicMock(
        side_effect=CredentialError("Test error")
    )
    assert model_tester.validate_credentials() is False

    model_tester.client.validate_credentials = mock.MagicMock(
        side_effect=Exception("Test error")
    )
    assert model_tester.validate_credentials() is False


def test_get_available_models(model_tester):
    """Test getting available models."""
    # Test successful retrieval
    models = model_tester.get_available_models()
    assert len(models) == 3
    assert models[0]["id"] == "mock/model-1"

    # Test empty model list
    model_tester.client.models = []
    models = model_tester.get_available_models()
    assert len(models) == 0

    # Test exception handling
    model_tester.client.get_models = mock.MagicMock(
        side_effect=Exception("Test error")
    )
    models = model_tester.get_available_models()
    assert len(models) == 0


def test_test_model(model_tester):
    """Test testing a single model."""
    # Test successful test
    result = model_tester.test_model("mock/model-1")
    assert result["model"] == "mock/model-1"
    assert result["tool_call_support"] is True
    assert result["handles_response"] is True

    # Test model without tool call support
    result = model_tester.test_model("mock/model-3")
    assert result["model"] == "mock/model-3"
    assert result["tool_call_support"] is False
    assert result["handles_response"] is False

    # Test exception handling
    model_tester.client.test_hello_world_tool = mock.MagicMock(
        side_effect=Exception("Test error")
    )
    result = model_tester.test_model("mock/model-1")
    assert result["model"] == "mock/model-1"
    assert result["tool_call_support"] is False
    assert result["handles_response"] is False
    assert "Error:" in result["details"]


def test_test_all_models(model_tester):
    """Test testing all models."""
    # Mock setup_logger to avoid actual logging setup
    with mock.patch(
        "watsonx_tool_tester.testers.model_tester.log_utils.setup_logger"
    ):
        # Mock save_results_to_file to avoid file operations
        model_tester._save_results_to_file = mock.MagicMock()

        # Test all models
        results = model_tester.test_all_models()
        assert len(results) == 3

        # Test with filter
        results = model_tester.test_all_models(filter_model="model-1")
        assert len(results) == 1
        assert results[0]["model"] == "mock/model-1"

        # Test with invalid filter
        with pytest.raises(ConfigurationError):
            model_tester.test_all_models(filter_model="nonexistent")

        # Test with credential validation failure
        model_tester.validate_credentials = mock.MagicMock(return_value=False)
        with pytest.raises(CredentialError):
            model_tester.test_all_models()

        # Test with no models found
        model_tester.validate_credentials = mock.MagicMock(return_value=True)
        model_tester.get_available_models = mock.MagicMock(return_value=[])
        with pytest.raises(ClientError):
            model_tester.test_all_models()


def test_save_results_to_file(model_tester, tmp_path):
    """Test saving results to file."""
    # Create test results
    results = [
        {
            "model": "test-model",
            "tool_call_support": True,
            "handles_response": True,
            "details": "Success",
            "response_times": {
                "tool_call_time": 1.0,
                "response_processing_time": 0.5,
                "total_time": 1.5,
            },
        }
    ]

    # Set up output file
    output_file = str(tmp_path / "test_results.json")
    model_tester.config.output_file = output_file

    # Mock the file operations
    with mock.patch(
        "builtins.open", mock.mock_open()
    ) as mock_file, mock.patch(
        "os.path.exists", return_value=True
    ), mock.patch(
        "os.makedirs"
    ):

        # Test saving results
        model_tester._save_results_to_file(results)

        # Verify that the file was "opened" for writing
        mock_file.assert_called_once_with(output_file, "w")

        # Verify that write was called (content was written to the file)
        mock_file().write.assert_called()

    # Test exception handling
    with mock.patch(
        "builtins.open", mock.MagicMock(side_effect=Exception("Test error"))
    ), mock.patch("os.path.exists", return_value=True), mock.patch(
        "os.makedirs"
    ):
        # This should not raise an exception
        model_tester._save_results_to_file(results)  # Test exception handling


def test_load_exclusion_patterns(model_tester_with_exclusions, tmp_path):
    """Test loading exclusion patterns."""
    # Test direct exclusions from config
    patterns = model_tester_with_exclusions._load_exclusion_patterns()
    assert "mock/model-2" in patterns
    assert "nonexistent-model" in patterns

    # Create mock file content
    mock_file_content = "# Test comment\nmock/model-3\nregex/.*-test\n\n"

    # Set new exclude file path
    exclude_file = str(tmp_path / "exclude.txt")
    model_tester_with_exclusions.config.exclude_file = exclude_file

    # Mock both the file existence check and the file open operation
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open", mock.mock_open(read_data=mock_file_content)
    ) as mock_file:

        # Now load patterns again - should include patterns from both config and file
        patterns = model_tester_with_exclusions._load_exclusion_patterns()

        # Verify mock file was "opened"
        mock_file.assert_called_once_with(exclude_file, "r")

        # Check patterns
        assert "mock/model-2" in patterns  # From config
        assert "nonexistent-model" in patterns  # From config
        assert "mock/model-3" in patterns  # From file
        assert "regex/.*-test" in patterns  # From file
        assert len(patterns) == 4  # Total unique patterns


def test_should_exclude_model(model_tester_with_exclusions):
    """Test model exclusion logic."""
    # Test direct name match
    assert model_tester_with_exclusions._should_exclude_model(
        "mock/model-2", {"mock/model-2"}
    )

    # Test regex pattern match
    patterns = {"mock/.*-2", "test/model"}
    assert model_tester_with_exclusions._should_exclude_model(
        "mock/model-2", patterns
    )
    assert not model_tester_with_exclusions._should_exclude_model(
        "mock/model-1", patterns
    )

    # Test invalid regex handling
    patterns = {"[invalid regex", "mock/model-3"}
    assert not model_tester_with_exclusions._should_exclude_model(
        "mock/model-2", patterns
    )
    assert model_tester_with_exclusions._should_exclude_model(
        "mock/model-3", patterns
    )


def test_filter_excluded_models(model_tester_with_exclusions):
    """Test filtering models based on exclusion patterns."""
    models = [
        {"id": "mock/model-1"},
        {"id": "mock/model-2"},
        {"id": "mock/model-3"},
    ]

    # Test with direct config exclusions
    filtered = model_tester_with_exclusions._filter_excluded_models(models)
    assert len(filtered) == 2
    assert filtered[0]["id"] == "mock/model-1"
    assert filtered[1]["id"] == "mock/model-3"

    # Test with regex pattern
    model_tester_with_exclusions.config.exclude_models = ["mock/model-.*"]
    filtered = model_tester_with_exclusions._filter_excluded_models(models)
    assert len(filtered) == 0  # All models excluded by regex


def test_test_all_models_with_exclusions(model_tester_with_exclusions):
    """Test that excluded models are not tested."""
    # Mock setup_logger to avoid actual logging setup
    with mock.patch(
        "watsonx_tool_tester.testers.model_tester.log_utils.setup_logger"
    ):
        # Mock save_results_to_file to avoid file operations
        model_tester_with_exclusions._save_results_to_file = mock.MagicMock()

        # Test all models with exclusions
        results = model_tester_with_exclusions.test_all_models()

        # Should only test models 1 and 3, excluding model 2
        assert len(results) == 2
        model_ids = [result["model"] for result in results]
        assert "mock/model-1" in model_ids
        assert "mock/model-3" in model_ids
        assert "mock/model-2" not in model_ids
