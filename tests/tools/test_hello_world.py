#!/usr/bin/env python3
"""
Tests for the HelloWorldTool class.

This module contains tests for the functionality of the HelloWorldTool.
"""

import pytest

from watsonx_tool_tester.tools.hello_world import HelloWorldTool


@pytest.fixture
def hello_tool():
    """Create a HelloWorldTool instance."""
    return HelloWorldTool()


def test_name(hello_tool):
    """Test that the tool name is correct."""
    assert hello_tool.name == "hello_world"


def test_description(hello_tool):
    """Test that the tool description is correct."""
    assert "simple tool" in hello_tool.description.lower()
    assert "greeting" in hello_tool.description.lower()


def test_parameters(hello_tool):
    """Test that the tool parameters are correct."""
    params = hello_tool.parameters

    # Check parameter structure
    assert params["type"] == "object"
    assert "properties" in params
    assert "name" in params["properties"]
    assert params["properties"]["name"]["type"] == "string"
    assert "required" in params
    assert "name" in params["required"]


def test_execute_with_name(hello_tool):
    """Test tool execution with a name parameter."""
    result = hello_tool.execute(name="Test User")

    assert result["greeting"] == "Hello, Test User!"
    assert "successfully called" in result["message"]


def test_execute_default(hello_tool):
    """Test tool execution with no parameters (should use default)."""
    result = hello_tool.execute()

    assert result["greeting"] == "Hello, World!"
    assert "successfully called" in result["message"]
