"""
WatsonX Tool Tester package.

This package provides tools for testing the tool call capabilities
of AI models in IBM WatsonX and via LiteLLM proxies.
"""

__version__ = "0.1.0"

from watsonx_tool_tester.cli import cli
from watsonx_tool_tester.clients.base import BaseClient
from watsonx_tool_tester.clients.litellm import LiteLLMClient
from watsonx_tool_tester.clients.watsonx import WatsonXClient
from watsonx_tool_tester.config import ClientType, Config
from watsonx_tool_tester.testers.model_tester import ModelTester
from watsonx_tool_tester.testers.result_handler import ResultHandler
from watsonx_tool_tester.tools.base import BaseTool
from watsonx_tool_tester.tools.hello_world import HelloWorldTool

__all__ = [
    "cli",
    "Config",
    "ClientType",
    "BaseClient",
    "WatsonXClient",
    "LiteLLMClient",
    "ModelTester",
    "ResultHandler",
    "BaseTool",
    "HelloWorldTool",
]
