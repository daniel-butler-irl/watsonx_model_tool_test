#!/usr/bin/env python3
"""
Mock client for testing.

This module provides a mock client implementation that can be used
for testing without making actual API calls.
"""

from typing import Any, Dict, List, Optional, Tuple

from watsonx_tool_tester.clients.base import BaseClient
from watsonx_tool_tester.tools.base import BaseTool


class MockClient(BaseClient):
    """A mock client for testing the ModelTester class.

    This client simulates the behavior of a real API client without making
    actual API calls.

    Attributes:
        models: List of mock models to return
        tool_supported_models: Set of model IDs that should support tool calls
        response_handling_models: Set of model IDs that should handle responses correctly
        validation_success: Whether credential validation should succeed
        error_message: Error message to return when validation fails
    """

    def __init__(
        self,
        base_url: str = "https://mock-api.test",
        debug: bool = False,
        models: Optional[List[Dict[str, Any]]] = None,
        tool_supported_models: Optional[List[str]] = None,
        response_handling_models: Optional[List[str]] = None,
        validation_success: bool = True,
        error_message: str = "Mock validation error",
    ):
        """Initialize the mock client.

        Args:
            base_url: The base URL for the mock API
            debug: Whether to enable debug logging
            models: List of mock models to return
            tool_supported_models: List of model IDs that should support tool calls
            response_handling_models: List of model IDs that should handle responses
                correctly
            validation_success: Whether credential validation should succeed
            error_message: Error message to return when validation fails
        """
        super().__init__({"model": None, "debug": debug})

        # Default mock models if none provided
        self.models = models or [
            {"id": "mock/model-1"},
            {"id": "mock/model-2"},
            {"id": "mock/model-3"},
        ]

        # Default supported models if none provided
        self.tool_supported_models = set(
            tool_supported_models or ["mock/model-1", "mock/model-2"]
        )

        # Default models that handle responses correctly if none provided
        self.response_handling_models = set(
            response_handling_models or ["mock/model-1"]
        )

        self.validation_success = validation_success
        self.error_message = error_message

    @property
    def client_type(self) -> str:
        """Get the type of client.

        Returns:
            str: The type of the client ('mock')
        """
        return "mock"

    @property
    def available_models(self) -> List[str]:
        """Get the list of available models for this client.

        Returns:
            List[str]: List of available model names
        """
        return [model["id"] for model in self.models]

    def validate_config(self) -> List[str]:
        """Validate the client configuration.

        Returns:
            List[str]: A list of validation errors, empty if valid
        """
        return [] if self.validation_success else [self.error_message]

    def validate_credentials(self) -> bool:
        """Validate if the provided API credentials are valid.

        Returns:
            bool: Always returns True for the mock client
        """
        return self.validation_success

    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of mock models.

        Returns:
            List[Dict[str, Any]]: List of mock models
        """
        return self.models

    def generate_response(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the mock model.

        Args:
            prompt: The prompt to send to the model
            tools: Optional list of tools to make available to the model
            tool_choice: Optional specification for which tool to use

        Returns:
            Dict[str, Any]: A mock response
        """
        return {
            "id": "mock-response-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": self.model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    def test_hello_world_tool(
        self, model_id: str
    ) -> Tuple[bool, bool, str, Optional[Dict[str, Any]], Dict[str, float]]:
        """Test if the mock model can use the hello_world tool.

        Args:
            model_id: The ID of the mock model to test

        Returns:
            Tuple containing simulated test results
        """
        # Determine if the model supports tool calls
        supports_tool_call = model_id in self.tool_supported_models

        # Determine if the model handles responses correctly
        handles_response = model_id in self.response_handling_models

        # Generate detail message based on the results
        if supports_tool_call and handles_response:
            details = "Successfully called tool and used its result"
        elif supports_tool_call:
            details = "Called tool but did not use its result correctly"
        else:
            details = "Did not call the tool"

        # Simulate response times
        response_times = {
            "tool_call_time": 1.5,
            "response_processing_time": 0.8 if supports_tool_call else None,
            "total_time": 2.3 if supports_tool_call else 1.5,
        }

        # Simulate API response data
        response_data = (
            {
                "id": "mock-response",
                "model": model_id,
                "tool_call_supported": supports_tool_call,
                "response_handling_supported": handles_response,
            }
            if supports_tool_call
            else None
        )

        return (
            supports_tool_call,
            handles_response,
            details,
            response_data,
            response_times,
        )
