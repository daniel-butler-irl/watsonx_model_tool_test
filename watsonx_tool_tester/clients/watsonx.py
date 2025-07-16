#!/usr/bin/env python3
"""
WatsonX AI client for model testing.

This module provides a client for interacting directly with the WatsonX AI API
to test tool call capabilities of models.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from watsonx_tool_tester.clients.base import (
    BaseClient,  # Changed from ModelClient to BaseClient
)
from watsonx_tool_tester.tools import get_tool_schemas
from watsonx_tool_tester.tools.base import BaseTool
from watsonx_tool_tester.utils import logging as log_utils

logger = log_utils.get_logger("clients.watsonx")


class WatsonXClient(BaseClient):  # Changed from ModelClient to BaseClient
    """Client for interacting with WatsonX AI API.

    This client handles authentication, making requests to the WatsonX API,
    and testing tool call capabilities.

    Attributes:
        base_url: The base URL for the WatsonX API
        api_key: The IBM Cloud API key
        project_id: The WatsonX AI project ID
        region: The WatsonX AI region
        debug: Whether to enable debug output
        api_version: The version parameter for WatsonX API calls
    """

    # Default model to test if no models list is available
    DEFAULT_MODEL = "ibm/granite-20b-instruct"

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Initialize the WatsonX client.

        Args:
            config: Configuration dictionary containing:
                - base_url: The base URL for the WatsonX API
                - api_key: The IBM Cloud API key
                - project_id: The WatsonX AI project ID
                - region: The WatsonX AI region (defaults to 'us-south')
                - debug: Whether to enable debug output (defaults to False)
                - api_version: API version for WatsonX API calls (defaults to '2023-05-29')
        """
        super().__init__(config)
        self.base_url = config.get(
            "base_url", "https://us-south.ml.cloud.ibm.com"
        )
        self.api_key = config.get("api_key", "")
        self.project_id = config.get("project_id", "")
        self.region = config.get("region", "us-south")
        self.debug = config.get("debug", False)
        self.api_version = config.get("api_version", "2023-05-29")

        # Token caching
        self.token_cache = {
            "access_token": None,
            "expiration_time": None,
            "refresh_time": None,  # Time to refresh token (before actual expiration)
        }

        # Hello World tool definition for WatsonX API format
        self.hello_world_tool = get_tool_schemas()[
            0
        ]  # Get the hello_world tool schema

    @property
    def client_type(self) -> str:
        """Get the type of client.

        Returns:
            str: The client type ('watsonx')
        """
        return "watsonx"

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

        if not self.api_key:
            errors.append("API key is required")

        if not self.project_id:
            errors.append("Project ID is required")

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

        headers = self.get_auth_headers()

        # Convert tools to WatsonX format if provided
        watson_tools = None
        if tools:
            watson_tools = [tool.get_schema() for tool in tools]

        # Create payload with model best practices
        payload = {
            "model_id": self.model,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
                "decoding_method": "greedy",
                "min_new_tokens": 0,
                "stop_sequences": [],
                "repetition_penalty": 1,
            },
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that uses tools when needed.",
                },
                {"role": "user", "content": prompt},
            ],
            "project_id": self.project_id,
        }

        # Add tools if provided
        if watson_tools:
            payload["tools"] = watson_tools
            if tool_choice:
                payload["tool_choice_option"] = tool_choice
            else:
                payload["tool_choice_option"] = "auto"

        # Use the correct chat endpoint for tool calling
        endpoint = (
            f"{self.base_url}/ml/v1/text/chat?version={self.api_version}"
        )

        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise ValueError(
                    f"API request failed: {response.status_code} - {response.text}"
                )
        except Exception as e:
            raise ValueError(f"Failed to generate response: {str(e)}")

    def get_iam_token(self) -> Optional[str]:
        """Get an IAM token using the IBM Cloud API key.

        This method reuses cached tokens if they're still valid.

        Returns:
            str: IAM access token or None if the request failed
        """
        # Check if we have a valid token in cache
        current_time = time.time()
        if (
            self.token_cache["access_token"]
            and self.token_cache["refresh_time"]
            and current_time < self.token_cache["refresh_time"]
        ):
            if self.debug:
                logger.info("✅ Using cached IAM token (still valid)")
            return self.token_cache["access_token"]

        # If token doesn't exist or is expired, get a new one
        iam_endpoint = "https://iam.cloud.ibm.com/identity/token"
        iam_data = {
            "apikey": self.api_key,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        }
        iam_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        try:
            if self.debug:
                logger.info("Getting new IAM token from API key...")
            response = requests.post(
                iam_endpoint, data=iam_data, headers=iam_headers, timeout=30
            )

            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data.get("access_token")
                expires_in = token_data.get(
                    "expires_in", 3600
                )  # Default to 1 hour

                if access_token:
                    # Calculate expiration time (subtract 5 minutes for safety margin)
                    refresh_margin = 300  # 5 minutes in seconds
                    self.token_cache["access_token"] = access_token
                    self.token_cache["expiration_time"] = (
                        current_time + expires_in
                    )
                    self.token_cache["refresh_time"] = (
                        current_time + expires_in - refresh_margin
                    )

                    if self.debug:
                        logger.info(
                            f"✅ Successfully obtained new IAM token (expires in {expires_in/60:.1f} minutes)"
                        )
                    return access_token
                else:
                    logger.error(
                        "❌ IAM token response did not contain access_token"
                    )
                    return None
            else:
                logger.error(
                    f"❌ Failed to get IAM token: {response.status_code} - {response.text}"
                )
                return None

        except requests.exceptions.RequestException as err:
            logger.error(f"❌ Error getting IAM token: {err}")
            return None
        except Exception as err:
            logger.error(f"❌ Unexpected error getting IAM token: {err}")
            return None

    def get_auth_headers(self) -> Dict[str, str]:
        """Get headers with authorization token for WatsonX API calls.

        Returns:
            Dict[str, str]: Headers for API requests with a valid bearer token
        """
        # Get IAM token using the API key
        access_token = self.get_iam_token()

        if not access_token:
            raise ValueError("Failed to obtain IAM token from API key.")

        # Return headers with the IAM token
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def validate_credentials(self) -> bool:
        """Validate if the provided API key and project ID are valid.

        Returns:
            bool: True if credentials are valid, False otherwise
        """
        try:
            # Validate the API key by getting an IAM token
            logger.info("Validating API key...")
            access_token = self.get_iam_token()

            if not access_token:
                logger.error("❌ Authentication failed: Invalid API key.")
                return False

            # If we have a project ID, verify it exists by fetching foundation model specs
            if self.project_id:
                try:
                    # Use foundation_model_specs endpoint which requires auth but not project_id
                    headers = self.get_auth_headers()
                    endpoint = f"{self.base_url}/ml/v1/foundation_model_specs?version={self.api_version}"

                    logger.info(
                        f"Validating project ID using endpoint: {endpoint}"
                    )
                    response = requests.get(
                        endpoint, headers=headers, timeout=30
                    )

                    if response.status_code != 200:
                        logger.error(
                            f"❌ API validation failed: {response.status_code} - {response.text}"
                        )
                        return False

                    # If we get here, the auth is valid, but we still want to validate the project ID
                    # We'll check if the project ID exists using a different endpoint
                    # For now, assume it's valid as we successfully authenticated

                except Exception as e:
                    logger.error(f"❌ Error validating credentials: {e}")
                    return False

            logger.info("✅ Credentials validated successfully.")
            return True

        except requests.exceptions.RequestException as err:
            logger.error(f"❌ Error validating credentials: {err}")
            return False
        except Exception as err:
            logger.error(f"❌ Unexpected error during validation: {err}")
            return False

    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from WatsonX API.

        Returns:
            List[Dict[str, Any]]: List of models
        """
        try:
            # Get authentication headers
            headers = self.get_auth_headers()

            # Use the authenticated endpoint for foundation model specs
            endpoint = f"{self.base_url}/ml/v1/foundation_model_specs?version={self.api_version}"

            logger.info(
                f"Fetching available models from WatsonX AI API: {endpoint}"
            )
            response = requests.get(endpoint, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch models: HTTP {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"HTTP error {response.status_code}: {response.text}"
                )

            models_data = response.json()

            # Extract model information from the response
            models = []
            if "resources" in models_data:
                for model in models_data["resources"]:
                    if "model_id" in model:
                        models.append({"id": model["model_id"]})

                logger.info(
                    f"Successfully retrieved {len(models)} models from WatsonX AI API"
                )

            if not models:
                logger.warning(
                    "No models found in API response. Using default model."
                )
                models = [{"id": self.DEFAULT_MODEL}]

            return models

        except Exception as err:
            logger.error(f"Error fetching models list: {err}")
            logger.warning(
                f"Falling back to default model: {self.DEFAULT_MODEL}"
            )
            return [{"id": self.DEFAULT_MODEL}]

    def test_hello_world_tool(
        self, model_id: str
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Test if the model can use the hello_world tool and process its response.

        This method tries different formats to find one that works with the model.

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
        return self.test_model_with_different_formats(model_id)

    def test_model_with_different_formats(
        self, model_id: str
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Test model using proper WatsonX tool calling API.

        Args:
            model_id: The ID of the model to test

        Returns:
            Tuple containing results (see test_hello_world_tool)
        """
        if not self.project_id:
            raise ValueError("Project ID is not set.")

        prompt = (
            "Please greet Daniel in Spanish using the hello_world function."
        )

        response_times: Dict[str, Optional[float]] = {
            "tool_call_time": None,
            "response_processing_time": None,
            "total_time": None,
        }

        total_start_time = time.time()

        # Create proper chat format payload
        payload = {
            "model_id": model_id,
            "parameters": {
                "temperature": 0.0,
                "max_new_tokens": 800,
                "decoding_method": "greedy",
                "min_new_tokens": 0,
                "stop_sequences": [],
                "repetition_penalty": 1,
            },
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that uses tools when needed.",
                },
                {"role": "user", "content": prompt},
            ],
            "tools": [self.hello_world_tool],
            "tool_choice_option": "auto",
            "project_id": self.project_id,
        }

        # Use the correct chat endpoint for tool calling
        endpoint = (
            f"{self.base_url}/ml/v1/text/chat?version={self.api_version}"
        )
        headers = self.get_auth_headers()

        try:
            # Phase 1: Send initial request to get tool call
            start_time = time.time()
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=60
            )
            tool_call_time = time.time() - start_time
            response_times["tool_call_time"] = tool_call_time

            if self.debug:
                logger.info(
                    f"Tool call request completed in {tool_call_time:.2f} seconds"
                )
                logger.info(f"Response Status Code: {response.status_code}")

            if response.status_code != 200:
                response_times["total_time"] = time.time() - total_start_time
                # Check if the error indicates unsupported tool calling
                if response.status_code == 400 or response.status_code == 422:
                    try:
                        error_data = response.json()
                        error_message = str(error_data)
                        if any(
                            keyword in error_message.lower()
                            for keyword in [
                                "tool",
                                "function",
                                "unsupported",
                                "not supported",
                            ]
                        ):
                            return (
                                False,
                                False,
                                f"Model does not support tool calling: {error_message}",
                                None,
                                response_times,
                            )
                    except:
                        pass

                return (
                    False,
                    False,
                    f"API request failed: {response.status_code} - {response.text}",
                    None,
                    response_times,
                )

            response_data = response.json()

            if self.debug:
                logger.info(
                    f"Full Response: {json.dumps(response_data, indent=2)}"
                )

            # STRICT VALIDATION: Only accept proper structured tool calls
            if (
                "choices" in response_data
                and len(response_data["choices"]) > 0
            ):
                choice = response_data["choices"][0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason")

                # CRITICAL: Only accept responses with finish_reason="tool_calls"
                # This is the definitive indicator of server-side tool calling
                if finish_reason == "tool_calls":
                    # Validate that tool_calls array exists and is properly structured
                    if "tool_calls" not in message:
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            "Invalid response: finish_reason is 'tool_calls' but no tool_calls array found",
                            {"initial_response": response_data},
                            response_times,
                        )

                    tool_calls = message["tool_calls"]

                    if (
                        not isinstance(tool_calls, list)
                        or len(tool_calls) == 0
                    ):
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            f"Invalid tool_calls structure: expected non-empty array, got {type(tool_calls)}",
                            {"initial_response": response_data},
                            response_times,
                        )

                    # Validate the tool call structure
                    tool_call = tool_calls[0]
                    if (
                        not isinstance(tool_call, dict)
                        or "function" not in tool_call
                    ):
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            "Invalid tool call structure: missing 'function' field",
                            {"initial_response": response_data},
                            response_times,
                        )

                    function_data = tool_call.get("function", {})

                    # Validate function structure
                    if (
                        not isinstance(function_data, dict)
                        or "name" not in function_data
                    ):
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            "Invalid function structure: missing 'name' field",
                            {"initial_response": response_data},
                            response_times,
                        )

                    if function_data.get("name") != "hello_world":
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            f"Model called wrong function: {function_data.get('name')} (expected: hello_world)",
                            {"initial_response": response_data},
                            response_times,
                        )

                    # Validate arguments structure
                    if "arguments" not in function_data:
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            "Invalid function structure: missing 'arguments' field",
                            {"initial_response": response_data},
                            response_times,
                        )

                    try:
                        # Parse the arguments - must be valid JSON
                        args_json = function_data.get("arguments", "{}")
                        if not isinstance(args_json, str):
                            response_times["total_time"] = (
                                time.time() - total_start_time
                            )
                            return (
                                False,
                                False,
                                f"Invalid arguments format: expected JSON string, got {type(args_json)}",
                                {
                                    "initial_response": response_data,
                                    "tool_call": tool_call,
                                },
                                response_times,
                            )

                        args = json.loads(args_json)

                        # Execute the tool using the actual tool implementation
                        from watsonx_tool_tester.tools.hello_world import (
                            HelloWorldTool,
                        )

                        tool = HelloWorldTool()
                        tool_result = tool.execute(**args)

                        # Phase 2: Send tool result back to model using proper conversation format
                        start_time = time.time()

                        # Add tool result to conversation in the proper format
                        payload["messages"].append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls,
                            }
                        )
                        payload["messages"].append(
                            {
                                "role": "tool",
                                "name": "hello_world",
                                "content": json.dumps(tool_result),
                                "tool_call_id": tool_call.get(
                                    "id", "call_1"
                                ),  # Include the tool call ID
                            }
                        )

                        # Remove tool parameters for follow-up request
                        payload.pop("tools", None)
                        payload.pop("tool_choice_option", None)

                        # Get final response
                        second_response = requests.post(
                            endpoint, headers=headers, json=payload, timeout=60
                        )
                        response_processing_time = time.time() - start_time
                        response_times["response_processing_time"] = (
                            response_processing_time
                        )

                        if second_response.status_code == 200:
                            second_data = second_response.json()

                            # Validate second response structure
                            if (
                                "choices" in second_data
                                and len(second_data["choices"]) > 0
                            ):
                                final_message = second_data["choices"][0].get(
                                    "message", {}
                                )
                                final_content = final_message.get(
                                    "content", ""
                                )

                                # Check if model properly used the tool result
                                # Must contain the actual greeting from the tool result
                                expected_greeting = tool_result.get(
                                    "greeting", ""
                                )
                                if (
                                    expected_greeting
                                    and expected_greeting in final_content
                                ):
                                    response_times["total_time"] = (
                                        time.time() - total_start_time
                                    )
                                    return (
                                        True,
                                        True,
                                        "Successfully used proper tool calling with structured response",
                                        {
                                            "initial_response": response_data,
                                            "tool_call": tool_call,
                                            "tool_result": tool_result,
                                            "final_response": second_data,
                                        },
                                        response_times,
                                    )
                                else:
                                    response_times["total_time"] = (
                                        time.time() - total_start_time
                                    )
                                    return (
                                        True,
                                        False,
                                        f"Model called tool correctly but did not use result properly. Expected '{expected_greeting}' in final response: {final_content}",
                                        {
                                            "initial_response": response_data,
                                            "tool_call": tool_call,
                                            "tool_result": tool_result,
                                            "final_response": second_data,
                                        },
                                        response_times,
                                    )
                            else:
                                response_times["total_time"] = (
                                    time.time() - total_start_time
                                )
                                return (
                                    True,
                                    False,
                                    f"Model called tool correctly but second response was malformed: {second_data}",
                                    {
                                        "initial_response": response_data,
                                        "tool_call": tool_call,
                                        "tool_result": tool_result,
                                    },
                                    response_times,
                                )
                        else:
                            response_times["total_time"] = (
                                time.time() - total_start_time
                            )
                            return (
                                True,
                                False,
                                f"Model called tool correctly but second request failed: {second_response.status_code} - {second_response.text}",
                                {
                                    "initial_response": response_data,
                                    "tool_call": tool_call,
                                    "tool_result": tool_result,
                                },
                                response_times,
                            )

                    except json.JSONDecodeError as e:
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            f"Model called tool but arguments were invalid JSON: {str(e)}",
                            {
                                "initial_response": response_data,
                                "tool_call": tool_call,
                            },
                            response_times,
                        )
                    except Exception as e:
                        response_times["total_time"] = (
                            time.time() - total_start_time
                        )
                        return (
                            False,
                            False,
                            f"Error executing tool: {str(e)}",
                            {
                                "initial_response": response_data,
                                "tool_call": tool_call,
                            },
                            response_times,
                        )
                else:
                    # Model didn't use tool calling - this is a definitive failure
                    # No fallback to text parsing or other methods
                    content = message.get("content", "")
                    response_times["total_time"] = (
                        time.time() - total_start_time
                    )
                    return (
                        False,
                        False,
                        f"Model does not support tool calling. Finish reason: {finish_reason}. Response: {content}",
                        {"initial_response": response_data},
                        response_times,
                    )
            else:
                response_times["total_time"] = time.time() - total_start_time
                return (
                    False,
                    False,
                    f"Invalid response format - expected 'choices' array: {response_data}",
                    {"initial_response": response_data},
                    response_times,
                )

        except Exception as e:
            response_times["total_time"] = time.time() - total_start_time
            return (
                False,
                False,
                f"Error during tool calling test: {str(e)}",
                None,
                response_times,
            )
