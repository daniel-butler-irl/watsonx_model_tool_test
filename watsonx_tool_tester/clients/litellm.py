#!/usr/bin/env python3
"""
LiteLLM proxy client for model testing.

This module provides a client for interacting with a LiteLLM proxy server
to test tool call capabilities of models.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from watsonx_tool_tester.clients.base import BaseClient
from watsonx_tool_tester.tools import get_tool_schemas
from watsonx_tool_tester.tools.base import BaseTool
from watsonx_tool_tester.utils import logging as log_utils

logger = log_utils.get_logger("clients.litellm")


class LiteLLMClient(BaseClient):  # Changed from ModelClient to BaseClient
    """Client for interacting with a LiteLLM proxy server.

    This client handles authentication and making requests to a LiteLLM proxy
    to test tool call capabilities.

    Attributes:
        base_url: The base URL for the LiteLLM proxy server
        auth_token: The authentication token for the LiteLLM proxy
        headers: HTTP headers to use for API requests
        debug: Whether to enable debug output
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Initialize the LiteLLM client.

        Args:
            config: Configuration dictionary containing:
                - base_url: The base URL for the LiteLLM proxy server
                - auth_token: The authentication token for the LiteLLM proxy
                - debug: Whether to enable debug output (defaults to False)
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8000")
        self.auth_token = (
            config.get("auth_token", "").strip()
            if config.get("auth_token")
            else None
        )
        self.debug = config.get("debug", False)

        self.headers = {
            "Authorization": (
                f"Bearer {self.auth_token}" if self.auth_token else None
            ),
            "Content-Type": "application/json",
        }

        # Remove None values from headers
        self.headers = {k: v for k, v in self.headers.items() if v is not None}

    def validate_credentials(self) -> bool:
        """Validate if the provided auth token is valid.

        Returns:
            bool: True if credentials are valid, False otherwise
        """
        try:
            logger.info("Validating LiteLLM credentials...")

            # Try to call an API endpoint that requires authentication
            response = requests.get(
                f"{self.base_url}/v1/models", headers=self.headers, timeout=30
            )

            if response.status_code == 200:
                logger.info("✅ LiteLLM credentials validated successfully.")
                return True
            else:
                logger.error(
                    f"❌ LiteLLM authentication failed: HTTP {response.status_code} - {response.text}"
                )
                return False

        except requests.exceptions.RequestException as err:
            logger.error(f"❌ Error validating LiteLLM credentials: {err}")
            return False
        except Exception as err:
            logger.error(f"❌ Unexpected error during validation: {str(err)}")
            return False

    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from the LiteLLM proxy.

        Returns:
            List[Dict[str, Any]]: List of models
        """
        try:
            # Get models from LiteLLM proxy
            endpoint = f"{self.base_url}/v1/models"

            if self.debug:
                logger.info(f"Fetching models from LiteLLM proxy: {endpoint}")

            response = requests.get(endpoint, headers=self.headers, timeout=30)
            response.raise_for_status()

            models_data = response.json()

            if self.debug:
                logger.info(
                    f"Models API Response: {json.dumps(models_data, indent=2)}"
                )

            # Extract and format models information
            models = []
            if "data" in models_data:
                models = models_data["data"]
                logger.info(
                    f"Successfully retrieved {len(models)} models from LiteLLM proxy"
                )

            return models

        except requests.exceptions.RequestException as err:
            logger.error(f"Error getting models from LiteLLM proxy: {err}")
            return []

        except Exception as err:
            logger.error(f"Unexpected error fetching models: {str(err)}")
            return []

    def test_hello_world_tool(
        self, model_id: str
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Test if the model can use the hello_world tool and process its response.

        Args:
            model_id: The ID of the model to test

        Returns:
            Tuple containing:
                bool: Whether the model supports tool calls
                bool: Whether the model handles tool response correctly
                str: Details about the test result
                Optional[Dict[str, Any]]: Full response data if available
                Dict[str, Optional[float]]: Response times for different parts of the test
        """
        prompt = (
            "Please greet Daniel in Spanish using the hello_world function."
        )

        # Get the hello_world tool schema
        hello_world_tool = get_tool_schemas()[0]

        # Prepare payload for the OpenAI-compatible LiteLLM endpoint
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that uses tools when needed.",
                },
                {"role": "user", "content": prompt},
            ],
            "tools": [hello_world_tool],
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 500,
        }

        if self.debug:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"TESTING HELLO WORLD TOOL WITH MODEL: {model_id}")
            logger.info(f"{'=' * 50}")
            logger.info(f"Request Payload: {json.dumps(payload, indent=2)}")

        response_times: Dict[str, Optional[float]] = {
            "tool_call_time": None,
            "response_processing_time": None,
            "total_time": None,
        }

        try:
            endpoint = f"{self.base_url}/v1/chat/completions"

            if self.debug:
                logger.info(f"Sending request to {endpoint}...")

            start_time = time.time()
            response = requests.post(
                endpoint, headers=self.headers, json=payload, timeout=60
            )
            tool_call_time = time.time() - start_time
            response_times["tool_call_time"] = tool_call_time

            if self.debug:
                logger.info(
                    f"Request completed in {tool_call_time:.2f} seconds"
                )
                logger.info(f"Response Status Code: {response.status_code}")

            if response.status_code != 200:
                if self.debug:
                    logger.error(f"Full Error Response: {response.text}")
                error_details = self.extract_error_details(response.text)
                response_times["total_time"] = tool_call_time
                return (
                    False,
                    False,
                    error_details,
                    None,
                    response_times,
                )

            # Parse the successful response
            response_data = response.json()
            if self.debug:
                logger.info(
                    f"Full Response: {json.dumps(response_data, indent=2)}"
                )

            # Process the response to check for tool calls
            message = response_data["choices"][0]["message"]
            if self.debug:
                logger.info(
                    f"Message Content: {json.dumps(message, indent=2)}"
                )

            # Check if the model used tool calls
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                if self.debug:
                    logger.info(
                        f"✅ SUCCESS: {model_id} supports hello_world tool calls"
                    )
                    logger.info(
                        f"Tool call details: {json.dumps(tool_calls, indent=2)}"
                    )

                # Validate the tool call parameters
                is_valid, validation_details = self.validate_tool_call_params(
                    tool_calls
                )
                if is_valid:
                    if self.debug:
                        logger.info(
                            "✅ CORRECT: Hello world tool call parameters are valid"
                        )

                    # Execute the tool call and get a response
                    tool_call = tool_calls[0]
                    function_name = tool_call.get("function", {}).get("name")
                    arguments = json.loads(
                        tool_call.get("function", {}).get("arguments", "{}")
                    )

                    tool_result = self.execute_tool_call(
                        function_name, arguments
                    )
                    if self.debug:
                        logger.info(
                            f"Tool execution result: {json.dumps(tool_result, indent=2)}"
                        )

                    # Test if the model can use the tool result
                    second_prompt = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that uses tools when needed.",
                        },
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": tool_calls[0]["id"],
                        },
                    ]

                    second_payload = {
                        "model": model_id,
                        "messages": second_prompt,
                        "temperature": 0.3,
                        "max_tokens": 500,
                    }

                    if self.debug:
                        logger.info(
                            "Sending second request to test tool result handling..."
                        )
                        logger.info(
                            f"Second Request Payload: {json.dumps(second_payload, indent=2)}"
                        )

                    second_start_time = time.time()
                    second_response = requests.post(
                        endpoint,
                        headers=self.headers,
                        json=second_payload,
                        timeout=60,
                    )
                    response_processing_time = time.time() - second_start_time
                    response_times["response_processing_time"] = (
                        response_processing_time
                    )
                    response_times["total_time"] = (
                        tool_call_time + response_processing_time
                    )

                    if self.debug:
                        logger.info(
                            f"Second request completed in {response_processing_time:.2f} seconds"
                        )
                        logger.info(
                            f"Total processing time: {response_times['total_time']:.2f} seconds"
                        )

                    if second_response.status_code == 200:
                        second_response_data = second_response.json()
                        if self.debug:
                            logger.info(
                                f"Second Response: {json.dumps(second_response_data, indent=2)}"
                            )

                        second_message = second_response_data["choices"][0][
                            "message"
                        ]
                        content = second_message.get("content", "")

                        # Check if the response contains the greeting in Spanish
                        if (
                            "¡Hola, Daniel!" in content
                            or "Hola, Daniel" in content
                        ):
                            if self.debug:
                                logger.info(
                                    "✅ SUCCESS: Model correctly used the tool result"
                                )
                            return (
                                True,
                                True,
                                "Successfully handled tool result",
                                second_response_data,
                                response_times,
                            )
                        else:
                            if self.debug:
                                logger.warning(
                                    "⚠️ PARTIAL: Model used tool but did not properly incorporate result"
                                )
                                logger.info(f"Response content: {content}")
                            return (
                                True,
                                False,
                                "Tool called but result not properly used",
                                second_response_data,
                                response_times,
                            )
                    else:
                        if self.debug:
                            logger.error(
                                f"❌ ERROR: Second request failed: {second_response.text}"
                            )
                        return (
                            True,
                            False,
                            f"Tool called but failed to process result: {self.extract_error_details(second_response.text)}",
                            None,
                            response_times,
                        )
                else:
                    if self.debug:
                        logger.warning(f"⚠️ INCORRECT: {validation_details}")
                    response_times["total_time"] = tool_call_time
                    return (
                        True,
                        False,
                        validation_details,
                        None,
                        response_times,
                    )

            # If no tool calls found
            if self.debug:
                logger.error(
                    f"❌ FAILED: {model_id} did not use hello_world tool"
                )
                logger.info(
                    f"Response content: {json.dumps(message.get('content', ''), indent=2)}"
                )

            response_times["total_time"] = tool_call_time
            return (
                False,
                False,
                "No tool calls in response",
                None,
                response_times,
            )

        except requests.exceptions.RequestException as err:
            if self.debug:
                logger.error(f"❌ ERROR: Request failed: {err}")
            if response_times["tool_call_time"]:
                response_times["total_time"] = response_times["tool_call_time"]
            return False, False, str(err), None, response_times

        except Exception as err:
            if self.debug:
                logger.error(f"❌ ERROR: Unexpected error: {err}")
            if response_times["tool_call_time"]:
                response_times["total_time"] = response_times["tool_call_time"]
            return (
                False,
                False,
                f"Unexpected error: {str(err)}",
                None,
                response_times,
            )

    @property
    def client_type(self) -> str:
        """Get the type of client.

        Returns:
            str: The client type ('litellm')
        """
        return "litellm"

    @property
    def available_models(self) -> List[str]:
        """Get the list of available models for this client.

        Returns:
            List[str]: List of available model names
        """
        models = self.get_models()
        model_ids = []
        for model in models:
            model_id = model.get("id")
            if model_id is not None:
                model_ids.append(model_id)
        return model_ids

    def validate_config(self) -> List[str]:
        """Validate the client configuration.

        Returns:
            List[str]: A list of validation errors, empty if valid
        """
        errors = []

        if not self.base_url:
            errors.append("Base URL is required")

        if not self.auth_token:
            errors.append("Authentication token is required")

        return errors

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
        if not self.model:
            raise ValueError("Model must be specified in config")

        # Convert tools to OpenAI format if provided
        openai_tools = None
        if tools:
            openai_tools = [tool.get_schema() for tool in tools]

        # Create payload following OpenAI format
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that uses tools when needed.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add tools if provided
        if openai_tools:
            payload["tools"] = openai_tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
            else:
                payload["tool_choice"] = "auto"

        # Send request to LiteLLM proxy
        try:
            endpoint = f"{self.base_url}/v1/chat/completions"
            response = requests.post(
                endpoint, headers=self.headers, json=payload, timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_message = f"Error from LiteLLM proxy: {response.status_code} - {response.text}"
                raise ValueError(error_message)

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {str(e)}")

    def extract_error_details(self, response_text: str) -> str:
        """Extract error details from an error response and format for display.

        Args:
            response_text: The error response text

        Returns:
            str: Formatted error message suitable for display in results table
        """
        try:
            error_data = json.loads(response_text)

            # Handle specific error formats
            if "error" in error_data:
                if isinstance(error_data["error"], dict):
                    # OpenAI-style errors with message field
                    if "message" in error_data["error"]:
                        error_message = error_data["error"]["message"]

                        # Handle nested WatsonX errors in LiteLLM proxy responses
                        if "watsonxException" in error_message:
                            # Try to extract the nested JSON error object
                            nested_json_match = re.search(
                                r"{\"errors\":\[.*?\]}", error_message
                            )
                            if nested_json_match:
                                try:
                                    nested_error = json.loads(
                                        nested_json_match.group()
                                    )
                                    if (
                                        "errors" in nested_error
                                        and len(nested_error["errors"]) > 0
                                    ):
                                        # Get the first error message
                                        first_error = nested_error["errors"][0]
                                        return f"{first_error.get('code', 'ERROR')}: {first_error.get('message', 'Unknown error')}"
                                except json.JSONDecodeError:
                                    pass

                            # If nested parsing fails, extract with regex
                            code_match = re.search(
                                r"code\":\"([^\"]+)\"", error_message
                            )
                            msg_match = re.search(
                                r"message\":\"([^\"]+)\"", error_message
                            )

                            if code_match and msg_match:
                                return f"{code_match.group(1)}: {msg_match.group(1)}"

                        return f"Error: {error_message}"
                    # Handle errors with type field
                    elif (
                        "type" in error_data["error"]
                        and "message" in error_data["error"]
                    ):
                        error_type = error_data["error"]["type"]
                        error_msg = error_data["error"].get(
                            "message", "Unknown error"
                        )
                        return f"{error_type}: {error_msg}"
                    # Fall back to first key-value pair
                    else:
                        first_key = next(iter(error_data["error"]))
                        return f"{first_key}: {error_data['error'][first_key]}"
                else:
                    # Handle string error
                    return f"Error: {error_data['error']}"

            # Handle Watson.ai direct API error format
            if (
                "errors" in error_data
                and isinstance(error_data["errors"], list)
                and len(error_data["errors"]) > 0
            ):
                first_error = error_data["errors"][0]
                if "code" in first_error and "message" in first_error:
                    return f"{first_error['code']}: {first_error['message']}"

            # Handle errors in other common formats
            if "message" in error_data:
                return f"Error: {error_data['message']}"

            # If we have a trace ID, include it
            if "trace" in error_data:
                return f"API Error (trace: {error_data['trace']})"

            # Limit to first 100 characters if it's a long raw JSON
            json_str = json.dumps(error_data)
            if len(json_str) > 100:
                return f"Error response: {json_str[:97]}..."
            return f"Error response: {json_str}"

        except json.JSONDecodeError:
            # Not valid JSON, might be HTML or plain text
            if isinstance(response_text, str):
                # Check for HTML content
                if (
                    response_text.strip().startswith(
                        ("<html", "<!DOCTYPE", "<HTML")
                    )
                    or "<html>" in response_text
                ):
                    if "404 Not Found" in response_text:
                        return "Error: 404 Not Found - Endpoint unavailable"
                    elif "403 Forbidden" in response_text:
                        return "Error: 403 Forbidden - Access denied"
                    else:
                        return "Error: Server returned HTML response"

                # Plain text - truncate if too long
                if len(response_text) > 100:
                    return f"Error: {response_text[:97]}..."
                return f"Error: {response_text}"

            return "Unknown error (response could not be parsed)"

    def execute_tool_call(
        self, function_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool call with the given arguments.

        Args:
            function_name: Name of the function to execute
            arguments: Arguments to pass to the function

        Returns:
            Dict[str, Any]: Result of the tool execution
        """
        if function_name == "hello_world":
            name = arguments.get("name", "User")
            language = arguments.get("language", "english").lower()

            greeting = f"Hello, {name}!"
            if language == "spanish":
                greeting = f"¡Hola, {name}!"
            elif language == "french":
                greeting = f"Bonjour, {name}!"
            elif language == "german":
                greeting = f"Hallo, {name}!"
            elif language == "italian":
                greeting = f"Ciao, {name}!"

            return {"greeting": greeting}
        else:
            return {"error": f"Unknown function: {function_name}"}

    def validate_tool_call_params(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Validate that the tool call parameters are correct.

        Args:
            tool_calls: List of tool calls from the model response

        Returns:
            Tuple[bool, str]: (is_valid, validation_details)
        """
        if not tool_calls or len(tool_calls) == 0:
            return False, "No tool calls found"

        tool_call = tool_calls[0]
        function = tool_call.get("function", {})
        function_name = function.get("name")

        if function_name != "hello_world":
            return False, f"Expected 'hello_world' but got '{function_name}'"

        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except json.JSONDecodeError:
            return False, "Invalid JSON in arguments"

        # Verify name parameter exists
        if "name" not in arguments:
            return False, "Missing required parameter 'name'"

        # Language parameter is optional but should be valid if present
        if "language" in arguments:
            valid_languages = [
                "english",
                "spanish",
                "french",
                "german",
                "italian",
            ]
            if arguments["language"].lower() not in valid_languages:
                return False, f"Invalid language: {arguments['language']}"

        return True, "Valid tool call parameters"
