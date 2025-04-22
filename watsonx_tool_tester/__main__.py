#!/usr/bin/env python3
"""
Main entry point for the WatsonX Tool Tester package.

This module allows the package to be run directly with
`python -m watsonx_tool_tester`.
"""

import sys

from watsonx_tool_tester.cli import cli

if __name__ == "__main__":
    sys.exit(cli())
