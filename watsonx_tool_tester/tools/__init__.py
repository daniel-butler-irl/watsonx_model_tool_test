#!/usr/bin/env python3
"""
Tools module for WatsonX Tool Tester.

This module handles tool registration, discovery, and schema generation.
"""

from typing import Any, Dict, List

from watsonx_tool_tester.tools.base import BaseTool
from watsonx_tool_tester.tools.hello_world import HelloWorldTool

# Register all available tools
_TOOLS = {
    "hello_world": HelloWorldTool(),
}


def get_tool(name: str) -> BaseTool:
    """Get a tool instance by name.

    Args:
        name: The name of the tool to get

    Returns:
        BaseTool: The requested tool instance

    Raises:
        ValueError: If the tool is not found
    """
    if name not in _TOOLS:
        available_tools = ", ".join(_TOOLS.keys())
        raise ValueError(
            f"Tool '{name}' not found. Available tools: {available_tools}"
        )

    return _TOOLS[name]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Get OpenAI-compatible function schemas for all available tools.

    Returns:
        List[Dict[str, Any]]: List of function schemas
    """
    return [tool.get_definition() for tool in _TOOLS.values()]
