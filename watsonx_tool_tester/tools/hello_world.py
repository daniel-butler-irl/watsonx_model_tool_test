#!/usr/bin/env python3
"""
Hello World tool for the WatsonX Tool Tester.

This module implements a simple hello world tool that can be used
to test AI models' tool calling capabilities.
"""

from typing import Dict

from watsonx_tool_tester.tools.base import BaseTool


class HelloWorldTool(BaseTool):
    """A simple hello world tool for testing AI model function calling.

    This tool takes a name parameter and returns a greeting message.
    It is meant to be a simple test for AI models' function calling
    capabilities.
    """

    def __init__(self):
        """Initialize the HelloWorldTool with its metadata."""
        super().__init__(
            name="hello_world",
            description="A simple tool that returns a greeting message for a given name",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person to greet",
                    },
                    "language": {
                        "type": "string",
                        "enum": [
                            "english",
                            "spanish",
                            "french",
                            "german",
                            "japanese",
                        ],
                        "description": "The language to use for the greeting",
                    },
                },
                "required": ["name"],
            },
            required_parameters=["name"],
        )

    def execute(self, **kwargs) -> Dict[str, str]:
        """Execute the hello world tool with the provided parameters.

        Args:
            **kwargs: Parameters passed to the tool, expected to contain 'name'
                     and optionally 'language'

        Returns:
            Dict[str, str]: A dictionary containing the greeting message
        """
        name = kwargs.get("name", "World")
        language = kwargs.get("language", "english").lower()

        greetings = {
            "english": f"Hello, {name}!",
            "spanish": f"¡Hola, {name}!",
            "french": f"Bonjour, {name}!",
            "german": f"Hallo, {name}!",
            "japanese": f"こんにちは, {name}さん!",
        }

        greeting = greetings.get(language, greetings["english"])

        return {
            "greeting": greeting,
            "language": language,
            "name": name,
            "message": f"The hello_world tool was successfully called with name='{name}' and language='{language}'",
        }
