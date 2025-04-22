#!/usr/bin/env python3
"""
Tests for the LiteLLM client.

This module contains tests for the LiteLLM client functionality.
"""

import json
from unittest import mock

import pytest
import requests

from watsonx_tool_tester.clients.litellm import LiteLLMClient
from watsonx_tool_tester.tools.hello_world import HelloWorldTool


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    mock_resp = mock.Mock(spec=requests.Response)
    mock_resp.status_code = 200
    return mock_resp


@pytest.fixture
def litellm_client():
    """Create a LiteLLMClient instance for testing."""
    config = {
        "base_url": "https://mock-litellm.test",
        "auth_token": "mock-token",
        "model": "mock-model",
        "debug": True,
    }
    return LiteLLMClient(config)


def test_initialization():
    """Test initialization of the LiteLLMClient."""
    config = {
        "base_url": "https://test-url.com",
        "auth_token": "test-token",
        "model": "test-model",
        "debug": True,
        "temperature": 0.5,
        "max_tokens": 100,
    }
    client = LiteLLMClient(config)

    assert client.base_url == "https://test-url.com"
    assert client.auth_token == "test-token"
    assert client.model == "test-model"
    assert client.debug is True
    assert client.temperature == 0.5
    assert client.max_tokens == 100
    assert client.headers == {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }


def test_extract_error_details():
    """Test the extract_error_details method for handling WatsonX errors in LiteLLM responses."""
    client = LiteLLMClient(
        {
            "base_url": "https://mock-litellm.test",
            "auth_token": "mock-token",
        }
    )

    # Test watsonxException error extraction
    watsonx_error_json = json.dumps(
        {
            "error": {
                "message": 'watsonxException: {"errors":[{"code":"invalid_parameter","message":"Invalid parameter value"}]}'
            }
        }
    )
    result = client.extract_error_details(watsonx_error_json)
    assert "invalid_parameter: Invalid parameter value" in result

    # Test regex fallback extraction
    regex_error_json = json.dumps(
        {
            "error": {
                "message": 'watsonxException: Error processing request, code":"model_not_found", message":"Model not available"'
            }
        }
    )
    result = client.extract_error_details(regex_error_json)
    assert "model_not_found: Model not available" in result

    # Test standard LiteLLM error
    standard_error_json = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid request parameters",
            }
        }
    )
    result = client.extract_error_details(standard_error_json)
    assert "Error: Invalid request parameters" in result


def test_get_tool_schema():
    """Test that the tool schema is correctly formatted for LiteLLM/OpenAI compatibility."""

    # Create a hello world tool
    hello_tool = HelloWorldTool()

    # Get the schema
    schema = hello_tool.get_definition("openai")

    # Verify the schema format matches OpenAI requirements
    assert schema["type"] == "function"
    assert "function" in schema
    assert "name" in schema["function"]
    assert "description" in schema["function"]
    assert "parameters" in schema["function"]

    # Verify the parameters are correctly structured (not nested with additional properties/type fields)
    parameters = schema["function"]["parameters"]
    assert "type" in parameters
    assert "properties" in parameters

    # Check parameters are properly formatted with expected properties
    properties = parameters["properties"]
    assert "name" in properties
    assert "language" in properties

    # Ensure the name property has the correct structure
    assert properties["name"]["type"] == "string"
    assert "description" in properties["name"]

    # Ensure the language property has the correct structure with enum values
    assert "enum" in properties["language"]
    assert "spanish" in properties["language"]["enum"]


def test_generate_response_with_tool_call(litellm_client, mock_response):
    """Test generating a response with a tool call."""
    # Mock the requests.post to return a successful response with a tool call
    tool_call_response = {
        "id": "test-id",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "hello_world",
                                "arguments": '{"name":"Daniel","language":"spanish"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    mock_response.json.return_value = tool_call_response

    with mock.patch("requests.post", return_value=mock_response):
        # Call generate_response with a tool
        hello_tool = HelloWorldTool()
        response = litellm_client.generate_response(
            "Please greet Daniel in Spanish", tools=[hello_tool]
        )

        # Verify the response contains the tool call
        assert (
            response["choices"][0]["message"]["tool_calls"][0]["function"][
                "name"
            ]
            == "hello_world"
        )


def test_validate_config():
    """Test LiteLLMClient config validation."""
    # Valid config
    client = LiteLLMClient(
        {
            "base_url": "https://mock-litellm.test",
            "auth_token": "mock-token",
        }
    )
    errors = client.validate_config()
    assert len(errors) == 0
    assert client.base_url == "https://mock-litellm.test"

    # Missing auth token
    invalid_client = LiteLLMClient(
        {
            "base_url": "https://mock-litellm.test",
        }
    )
    errors = invalid_client.validate_config()
    assert len(errors) > 0
    assert any("Authentication token" in error for error in errors)

    # Missing base URL
    invalid_client = LiteLLMClient(
        {
            "auth_token": "mock-token",
            "base_url": "",  # Empty string to trigger validation failure
        }
    )
    errors = invalid_client.validate_config()
    assert len(errors) > 0
    assert any("Base URL" in error for error in errors)
