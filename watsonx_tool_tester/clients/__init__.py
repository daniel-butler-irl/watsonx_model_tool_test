#!/usr/bin/env python3
"""
Clients module for WatsonX Tool Tester.

This module provides client implementations for interacting with different
AI model providers to test tool call capabilities.
"""

from watsonx_tool_tester.clients.base import BaseClient  # Fixed class name
from watsonx_tool_tester.clients.litellm import LiteLLMClient
from watsonx_tool_tester.clients.watsonx import WatsonXClient

__all__ = ["BaseClient", "LiteLLMClient", "WatsonXClient"]  # Updated to match
