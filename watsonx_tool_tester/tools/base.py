#!/usr/bin/env python3
"""
Base tool class for the WatsonX Tool Tester.

This module defines the base class for tools that can be called
by AI models through their function calling capabilities.
"""

import abc
from typing import Any, Dict, List, Optional


class BaseTool(abc.ABC):
    """Abstract base class for all tool implementations.

    This class defines the interface that all tools must implement to be
    compatible with the WatsonX Tool Tester framework.

    Attributes:
        name: The name of the tool as it will be presented to the model.
        description: A description of what the tool does.
        parameters: The parameters the tool accepts in JSON Schema format.
        required_parameters: List of parameters that are required.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required_parameters: Optional[List[str]] = None,
    ):
        """Initialize the tool with its metadata.

        Args:
            name: The name of the tool as it will be presented to the model.
            description: A description of what the tool does.
            parameters: The parameters the tool accepts in JSON Schema format.
            required_parameters: List of parameters that are required.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required_parameters = required_parameters or []

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: The parameters to pass to the tool.

        Returns:
            The result of executing the tool.
        """
        pass

    def get_definition(self, format_type: str = "openai") -> Dict[str, Any]:
        """Get the tool definition in the specified format.

        Args:
            format_type: The format to return the definition in.
                      Supported formats: "openai", "watsonx", "anthropic"

        Returns:
            A dictionary containing the tool definition in the specified format.

        Raises:
            ValueError: If the requested format is not supported.
        """
        if format_type == "openai":
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }
        elif format_type == "watsonx":
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_parameters,
                },
            }
        elif format_type == "anthropic":
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_parameters,
                },
            }
        else:
            msg = f"Unsupported tool definition format: {format_type}"
            raise ValueError(msg)

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema in a format compatible with API requests.

        This is an alias for get_definition with the default OpenAI format.

        Returns:
            A dictionary containing the tool definition in OpenAI format.
        """
        return self.get_definition(format_type="openai")
