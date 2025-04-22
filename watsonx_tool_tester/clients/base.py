#!/usr/bin/env python3
"""
Base client class for the WatsonX Tool Tester.

This module defines the base class for AI model clients,
which provide a common interface for different AI model APIs.
"""

import abc
import time
from typing import Any, Dict, List, Optional, Tuple

from watsonx_tool_tester.tools.base import BaseTool
from watsonx_tool_tester.utils.errors import ClientError


class BaseClient(abc.ABC):
    """Base class for all AI model clients in the WatsonX Tool Tester.

    This abstract base class defines the interface that all model clients must implement.
    Clients handle communication with different AI model APIs (WatsonX, LiteLLM, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the base client with configuration.

        Args:
            config: Configuration dictionary for the client
        """
        self.config = config
        self.model = config.get("model")
        self.temperature = float(config.get("temperature", 0))
        self.max_tokens = int(config.get("max_tokens", 1024))
        self.verbose = bool(config.get("verbose", False))

    @property
    @abc.abstractmethod
    def client_type(self) -> str:
        """Get the type of client (e.g., 'watsonx', 'litellm').

        Returns:
            str: The type of the client
        """
        pass

    @property
    @abc.abstractmethod
    def available_models(self) -> List[str]:
        """Get the list of available models for this client.

        Returns:
            List[str]: List of available model names
        """
        pass

    @abc.abstractmethod
    def validate_config(self) -> List[str]:
        """Validate the client configuration.

        Returns:
            List[str]: A list of validation errors, empty if valid
        """
        pass

    @abc.abstractmethod
    def generate_response(
        self,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the AI model.

        Args:
            prompt: The prompt to send to the model
            tools: Optional list of tools to make available to the model
            tool_choice: Optional specification for which tool to use

        Returns:
            Dict[str, Any]: The model's response, including any tool calls
        """
        pass

    def test_tool_call(
        self,
        tool: BaseTool,
        prompt: str,
        format: Optional[str] = None,
        tool_choice: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Test if the model can call the provided tool.

        Args:
            tool: The tool to test
            prompt: The prompt instructing the model to use the tool
            format: Optional format to use for the tool schema
            tool_choice: Optional specification for which tool to use

        Returns:
            Tuple[Dict[str, Any], float]: The model response and the time taken
        """
        start_time = time.time()

        try:
            response = self.generate_response(prompt, [tool], tool_choice)
            elapsed_time = time.time() - start_time
            return response, elapsed_time
        except Exception as e:
            elapsed_time = time.time() - start_time
            raise ClientError(f"Error testing tool call: {str(e)}") from e

    def run_tool_call(
        self,
        tool: BaseTool,
        prompt: str,
        format: Optional[str] = None,
        tool_choice: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], float]:
        """Run a full tool call test including tool execution.

        Args:
            tool: The tool to test
            prompt: The prompt instructing the model to use the tool
            format: Optional format to use for the tool schema
            tool_choice: Optional specification for which tool to use

        Returns:
            Tuple[Dict[str, Any], Optional[Dict[str, Any]], float]:
                The model response, tool execution result, and time taken
        """
        response, elapsed_time = self.test_tool_call(
            tool, prompt, format, tool_choice
        )

        # Extract tool call arguments from the response
        tool_call_args = self._extract_tool_call_args(response, tool.name)

        if tool_call_args:
            # Execute the tool with the extracted arguments
            tool_result = tool.execute(**tool_call_args)
        else:
            tool_result = None

        return response, tool_result, elapsed_time

    def _extract_tool_call_args(
        self, response: Dict[str, Any], tool_name: str
    ) -> Optional[Dict[str, Any]]:
        """Extract tool call arguments from a model response.

        Args:
            response: The model response
            tool_name: The name of the tool to extract arguments for

        Returns:
            Optional[Dict[str, Any]]: The tool call arguments or None if not found
        """
        # Implementation depends on the response format, will be overridden by subclasses
        return None
