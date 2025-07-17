#!/usr/bin/env python3
"""
Tests for the CLI module.

This module contains tests for the CLI functionality.
"""

import sys
from unittest import mock

import pytest
from click.testing import CliRunner

from watsonx_tool_tester.cli import cli


@pytest.fixture
def runner():
    """Create an isolated CliRunner for testing Click commands."""
    return CliRunner(mix_stderr=False, env={"PYTHONPATH": ":".join(sys.path)})


# These tests skip the full import process for better isolation
def test_test_command_with_invalid_client(runner):
    """Test the 'test' command with an invalid client type."""
    result = runner.invoke(cli, ["test", "--client", "invalid"])

    assert (
        result.exit_code == 2
    )  # Click returns 2 for parameter validation errors
    assert "Invalid value for '--client'" in result.stderr


def test_test_command_with_missing_credentials(runner):
    """Test the 'test' command with missing credentials."""
    # No API key or project ID
    result = runner.invoke(cli, ["test", "--client", "watsonx"])

    assert result.exit_code == 1
    # Check for the actual error message that appears in the output
    assert (
        "Invalid credentials" in result.stdout
        or "Invalid API credentials" in result.stdout
    )


# For the remaining tests, we'll need to monkeypatch low-level functions
# to avoid any real API calls
class TestCLIWithMocks:
    """Tests that require mocked APIs."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Set up mocks for all API-calling methods"""
        # Create mock model data to return
        self.test_models = [
            {"id": "mock-test-model-1", "name": "Mock Test Model 1"},
            {"id": "mock-test-model-2", "name": "Mock Test Model 2"},
        ]

        self.test_results = [
            {
                "model": "mock-test-model-1",
                "tool_call_support": True,
                "handles_response": True,
                "details": "Successfully called tool",
                "response_times": {
                    "tool_call_time": 0.5,
                    "response_time": 1.0,
                    "total_time": 1.5,
                },
            },
            {
                "model": "mock-test-model-2",
                "tool_call_support": False,
                "handles_response": False,
                "details": "Failed to call tool",
                "response_times": {
                    "tool_call_time": 0.0,
                    "response_time": 0.0,
                    "total_time": 0.5,
                },
            },
        ]

        # Patch WatsonXClient methods
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.watsonx.WatsonXClient.validate_credentials",
            mock.MagicMock(return_value=True),
        )
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.watsonx.WatsonXClient.get_models",
            mock.MagicMock(return_value=self.test_models),
        )
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.watsonx.WatsonXClient.test_hello_world_tool",
            mock.MagicMock(
                return_value=(True, True, "Success", {}, {"total_time": 1.0})
            ),
        )

        # Patch LiteLLMClient methods
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.litellm.LiteLLMClient.validate_credentials",
            mock.MagicMock(return_value=True),
        )
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.litellm.LiteLLMClient.get_models",
            mock.MagicMock(return_value=self.test_models),
        )
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.litellm.LiteLLMClient.test_hello_world_tool",
            mock.MagicMock(
                return_value=(True, True, "Success", {}, {"total_time": 1.0})
            ),
        )

        # To test failures, we'll need to override these in specific tests
        self.mock_validate_credentials = mock.MagicMock(return_value=True)
        self.mock_get_models = mock.MagicMock(return_value=self.test_models)

    def test_test_command_basic(self, runner):
        """Test the basic 'test' command."""
        result = runner.invoke(
            cli,
            [
                "test",
                "--client",
                "watsonx",
                "--watsonx-apikey",
                "test-key",
                "--watsonx-project-id",
                "test-project",
            ],
        )

        assert result.exit_code == 0
        assert "MODEL" in result.stdout
        assert "TOOL SUPPORT" in result.stdout

    def test_test_command_with_model_filter(self, runner):
        """Test the 'test' command with a model filter."""
        result = runner.invoke(
            cli,
            [
                "test",
                "--client",
                "watsonx",
                "--watsonx-apikey",
                "test-key",
                "--watsonx-project-id",
                "test-project",
                "--model",
                "mock-test-model-1",
            ],
        )

        assert result.exit_code == 0

    def test_test_command_with_output_file(self, runner, tmp_path):
        """Test the 'test' command with an output file."""
        output_file = str(tmp_path / "test-results.json")

        result = runner.invoke(
            cli,
            [
                "test",
                "--client",
                "watsonx",
                "--watsonx-apikey",
                "test-key",
                "--watsonx-project-id",
                "test-project",
                "--output",
                output_file,
            ],
        )

        assert result.exit_code == 0

    def test_list_models_command(self, runner):
        """Test the 'list-models' command."""
        result = runner.invoke(
            cli,
            [
                "list-models",
                "--client",
                "watsonx",
                "--watsonx-apikey",
                "test-key",
                "--watsonx-project-id",
                "test-project",
            ],
        )

        assert result.exit_code == 0
        assert "Available models" in result.stdout
        assert "mock-test-model-1" in result.stdout
        assert "mock-test-model-2" in result.stdout

    def test_list_models_command_with_validation_failure(
        self, runner, monkeypatch
    ):
        """Test the 'list-models' command when validation fails."""
        # Override the validation method to return False
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.watsonx.WatsonXClient.validate_credentials",
            mock.MagicMock(return_value=False),
        )

        result = runner.invoke(
            cli,
            [
                "list-models",
                "--client",
                "watsonx",
                "--watsonx-apikey",
                "test-key",
                "--watsonx-project-id",
                "test-project",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid API credentials" in result.stdout

    def test_list_models_command_with_no_models(self, runner, monkeypatch):
        """Test the 'list-models' command when no models are found."""
        # Override the get_models method to return an empty list
        monkeypatch.setattr(
            "watsonx_tool_tester.clients.watsonx.WatsonXClient.get_models",
            mock.MagicMock(return_value=[]),
        )

        result = runner.invoke(
            cli,
            [
                "list-models",
                "--client",
                "watsonx",
                "--watsonx-apikey",
                "test-key",
                "--watsonx-project-id",
                "test-project",
            ],
        )

        assert result.exit_code == 0
        assert "No models found" in result.stdout

    def test_debug_flag(self, runner):
        """Test the debug flag."""
        result = runner.invoke(
            cli,
            [
                "test",
                "--client",
                "watsonx",
                "--watsonx-apikey",
                "test-key",
                "--watsonx-project-id",
                "test-project",
                "--debug",
            ],
        )

        assert result.exit_code == 0
