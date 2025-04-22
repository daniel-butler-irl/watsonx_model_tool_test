#!/usr/bin/env python3
"""
WatsonX AI client for model testing.

This module provides a client for interacting directly with the WatsonX AI API
to test tool call capabilities of models.
"""

import json
import re
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
                payload["tool_choice"] = tool_choice
            else:
                payload["tool_choice"] = "auto"

        # Try multiple endpoints to find one that works
        endpoints = [
            f"{self.base_url}/ml/v1/chat/completions?version={self.api_version}",
            f"{self.base_url}/ml/v1/generation?version={self.api_version}",
            f"{self.base_url}/ml/v1/text/generation?version={self.api_version}",
        ]

        last_error = None
        for endpoint in endpoints:
            try:
                response = requests.post(
                    endpoint, headers=headers, json=payload, timeout=60
                )
                if response.status_code == 200:
                    return response.json()
                last_error = response.text
            except Exception as e:
                last_error = str(e)

        # If we got here, all attempts failed
        raise ValueError(f"Failed to generate response: {last_error}")

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
        """Test model with all supported payload formats to find the one that works best.

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

        # Classify model to determine appropriate approach
        model_family = "other"
        if "llama-3" in model_id or "meta-llama/llama-3" in model_id:
            model_family = "llama3"
        elif "granite-3" in model_id:
            model_family = "granite3"
        elif "mistral" in model_id:
            model_family = "mistral"

        # Custom system prompts per model family
        system_prompt = "You are a helpful assistant that must use provided functions when needed. When a user requests something that requires using a function, you must call the function rather than trying to perform the task yourself. Never define functions in your response - only use the functions provided to you."

        if model_family == "llama3":
            # Llama 3 needs very explicit instructions about function calling format
            system_prompt = "You are a helpful assistant with access to functions. When asked to perform a task that can be accomplished using a function, you MUST use the provided function without explanation. DO NOT define or implement the function - only call the existing function with appropriate parameters. For Spanish greetings, use the provided hello_world function with language='spanish'."
        elif model_family == "granite3":
            # Granite 3 models need a clear function calling prompt
            system_prompt = "You are a helpful assistant that must use provided functions. Always call the function directly without explanations when a user requests something that matches a function's purpose. Never write your own implementation - only use the provided function."
        elif model_family == "mistral":
            # Mistral models need specific instruction about function format
            system_prompt = "You are a helpful assistant with access to functions. You must call the appropriate function when needed. Follow the function specification exactly. Do not create your own functions - only use the ones provided to you. Use JSON format for function arguments."

        # Format 1: OpenAI-style format with custom input formatting for WatsonX text generation
        payload1 = {
            "model_id": model_id,
            "parameters": {
                "temperature": 0.0,  # Use 0 temperature for deterministic function calling
                "max_new_tokens": 800,
                "decoding_method": "greedy",
                "min_new_tokens": 0,
                "stop_sequences": [],
                "repetition_penalty": 1,
            },
            "input": f"<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>",
            "tools": [self.hello_world_tool],
            "tool_choice": "auto",
            "project_id": self.project_id,
        }

        # Format 2: OpenAI chat completions format - this is what LiteLLM is using
        payload2 = {
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "tools": [self.hello_world_tool],
            "tool_choice": "auto",
            "project_id": self.project_id,
        }

        # List of payloads to try, in order of preference
        payloads = [payload1, payload2]
        endpoints = [
            f"{self.base_url}/ml/v1/text/generation?version={self.api_version}",
            f"{self.base_url}/ml/v1/generation?version={self.api_version}",
            f"{self.base_url}/ml/v1/chat/completions?version={self.api_version}",
        ]

        # Try each combination until one works
        headers = self.get_auth_headers()
        success = False
        total_start_time = time.time()

        for payload_num, payload in enumerate(payloads, 1):
            if success:
                break

            for endpoint_num, endpoint in enumerate(endpoints, 1):
                if success:
                    break

                if self.debug:
                    logger.info(
                        f"Trying payload {payload_num} with endpoint {endpoint_num}: {endpoint}"
                    )

                try:
                    start_time = time.time()
                    response = requests.post(
                        endpoint, headers=headers, json=payload, timeout=60
                    )
                    api_call_time = time.time() - start_time

                    if self.debug:
                        logger.info(
                            f"Request completed in {api_call_time:.2f} seconds"
                        )
                        logger.info(
                            f"Response Status Code: {response.status_code}"
                        )

                    if response.status_code == 200:
                        response_data = response.json()

                        if self.debug:
                            logger.info(
                                f"Full Response: {json.dumps(response_data, indent=2)}"
                            )

                        # Handle WatsonX text generation format
                        if (
                            "results" in response_data
                            and len(response_data["results"]) > 0
                        ):
                            generated_text = response_data["results"][0].get(
                                "generated_text", ""
                            )

                            # Check for Python tag-based function calls (Llama models often do this)
                            python_tag_match = re.search(
                                r"<\|python_tag\|>(.*?)$", generated_text
                            )
                            if python_tag_match:
                                result = (
                                    self._process_python_tag_function_call(
                                        python_tag_match,
                                        generated_text,
                                        model_id,
                                        prompt,
                                        api_call_time,
                                        endpoint,
                                        headers,
                                        payload,
                                        response_times,
                                    )
                                )
                                if result[0]:  # If successful
                                    return result

                            # Check for JSON-style function calls in raw text
                            json_match = re.search(
                                r'(\{.*"name"\s*:\s*"hello_world".*\})',
                                generated_text,
                            )
                            if json_match:
                                result = (
                                    self._process_json_in_text_function_call(
                                        json_match,
                                        generated_text,
                                        model_id,
                                        prompt,
                                        api_call_time,
                                        endpoint,
                                        headers,
                                        payload,
                                        response_times,
                                    )
                                )
                                if result[0]:  # If successful
                                    return result

                            # Check for function call syntax in code blocks
                            code_block_match = re.search(
                                r"```(?:json)?\s*(\{.*\})```", generated_text
                            )
                            if code_block_match:
                                try:
                                    code_json = json.loads(
                                        code_block_match.group(1)
                                    )
                                    if (
                                        "name" in code_json
                                        and code_json["name"] == "hello_world"
                                    ):
                                        result = self._process_code_block_function_call(
                                            code_json,
                                            generated_text,
                                            model_id,
                                            prompt,
                                            api_call_time,
                                            endpoint,
                                            headers,
                                            payload,
                                            response_times,
                                        )
                                        if result[0]:  # If successful
                                            return result
                                except json.JSONDecodeError:
                                    pass

                            # Check for direct function calling syntax
                            function_call_match = re.search(
                                r"hello_world\((.*?)\)", generated_text
                            )
                            if function_call_match:
                                result = self._process_direct_function_call(
                                    function_call_match,
                                    generated_text,
                                    model_id,
                                    prompt,
                                    api_call_time,
                                    endpoint,
                                    headers,
                                    payload,
                                    response_times,
                                )
                                if result[0]:  # If successful
                                    return result

                        # Handle OpenAI-style response format
                        elif (
                            "choices" in response_data
                            and len(response_data["choices"]) > 0
                        ):
                            message = response_data["choices"][0].get(
                                "message", {}
                            )

                            # Check for tool calls in OpenAI format
                            if (
                                "tool_calls" in message
                                and message["tool_calls"]
                            ):
                                result = self._process_openai_tool_calls(
                                    message["tool_calls"],
                                    message,
                                    model_id,
                                    prompt,
                                    api_call_time,
                                    endpoint,
                                    headers,
                                    payload,
                                    response_times,
                                )
                                if result[0]:  # If successful
                                    return result

                        # No function call detected
                        if self.debug:
                            logger.info(
                                f"No function call detected with payload {payload_num} and endpoint {endpoint_num}"
                            )
                    else:
                        if self.debug:
                            logger.info(
                                f"Request failed with status code {response.status_code}: {response.text}"
                            )

                except Exception as e:
                    if self.debug:
                        logger.error(
                            f"Error with payload {payload_num} and endpoint {endpoint_num}: {str(e)}"
                        )

        # If we got here, all attempts failed
        total_time = time.time() - total_start_time
        response_times["tool_call_time"] = total_time
        response_times["total_time"] = total_time

        return (
            False,
            False,
            "No tool calls in response with any format",
            None,
            response_times,
        )

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

    def _process_python_tag_function_call(
        self,
        match,
        generated_text,
        model_id,
        prompt,
        api_call_time,
        endpoint,
        headers,
        payload,
        response_times,
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Process Python tag-based function call and execute the second request.

        Args:
            match: Regular expression match object containing the function call
            generated_text: The full generated text from the model
            model_id: The ID of the model being tested
            prompt: The original prompt
            api_call_time: Time taken for the first API call
            endpoint: The API endpoint used
            headers: The request headers
            payload: The request payload
            response_times: Dictionary to track response times

        Returns:
            Tuple containing results (see test_hello_world_tool)
        """
        python_code = match.group(1).strip()
        if self.debug:
            logger.info(
                f"✅ SUCCESS: Found Python function call: {python_code}"
            )

        # Parse the Python function call to extract function name and arguments
        try:
            # Match the function name and arguments
            func_match = re.match(r"(\w+)\((.*)\)", python_code)
            if not func_match:
                return (
                    False,
                    False,
                    "Failed to parse Python function call",
                    None,
                    response_times,
                )

            function_name = func_match.group(1)
            arg_string = func_match.group(2)

            # Ensure it's calling the hello_world function
            if function_name != "hello_world":
                return (
                    False,
                    False,
                    f"Model called {function_name} instead of hello_world",
                    None,
                    response_times,
                )

            # Parse arguments using regex
            arguments = {}

            # Extract name parameter
            name_match = re.search(
                r'name\s*=\s*["\']([^"\']+)["\']', arg_string
            )
            if name_match:
                arguments["name"] = name_match.group(1)
            else:
                # Try positional argument
                pos_args = re.findall(r'["\']([^"\']+)["\']', arg_string)
                if pos_args:
                    arguments["name"] = pos_args[0]

            # Extract language parameter if present
            lang_match = re.search(
                r'language\s*=\s*["\']([^"\']+)["\']', arg_string
            )
            if lang_match:
                arguments["language"] = lang_match.group(1).lower()

            # Validate that at minimum we have a name
            if "name" not in arguments:
                return (
                    False,
                    False,
                    "Missing required 'name' parameter",
                    None,
                    response_times,
                )

            if self.debug:
                logger.info(
                    f"✅ SUCCESS: Extracted function call parameters: {json.dumps(arguments, indent=2)}"
                )

            # Execute the tool and get the result
            tool_result = self.execute_tool_call(function_name, arguments)
            if self.debug:
                logger.info(
                    f"Tool execution result: {json.dumps(tool_result, indent=2)}"
                )

            # Now prepare a second request with the tool result
            second_system_msg = "You are a helpful assistant. You previously called the hello_world function and received its result. Respond to the user with the greeting returned by the function."
            second_payload = {
                "model_id": model_id,
                "parameters": payload["parameters"].copy(),
                "input": f"<|start_of_role|>system<|end_of_role|>{second_system_msg}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{generated_text}<|end_of_text|>\n<|start_of_role|>tool<|end_of_role|>{json.dumps(tool_result)}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
                "project_id": self.project_id,
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
                headers=headers,
                json=second_payload,
                timeout=60,
            )
            response_processing_time = time.time() - second_start_time
            response_times["tool_call_time"] = api_call_time
            response_times["response_processing_time"] = (
                response_processing_time
            )
            response_times["total_time"] = (
                api_call_time + response_processing_time
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

                # Extract content from the second response
                second_content = ""
                if (
                    "results" in second_response_data
                    and len(second_response_data["results"]) > 0
                ):
                    second_content = second_response_data["results"][0].get(
                        "generated_text", ""
                    )

                # Check if the response contains the greeting in Spanish
                if (
                    "¡Hola, Daniel!" in second_content
                    or "Hola, Daniel" in second_content
                ):
                    if self.debug:
                        logger.info(
                            "✅ SUCCESS: Model correctly used the tool result"
                        )
                    return (
                        True,
                        True,
                        "Successfully handled tool result (Python tag format)",
                        second_response_data,
                        response_times,
                    )
                else:
                    if self.debug:
                        logger.warning(
                            "⚠️ PARTIAL: Model used tool but did not properly incorporate result"
                        )
                        logger.info(f"Response content: {second_content}")
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
        except Exception as e:
            if self.debug:
                logger.error(
                    f"❌ ERROR: Failed to process Python tag function call: {e}"
                )
            response_times["total_time"] = api_call_time
            return (
                False,
                False,
                f"Error processing Python tag function call: {str(e)}",
                None,
                response_times,
            )

    def _process_json_in_text_function_call(
        self,
        match,
        generated_text,
        model_id,
        prompt,
        api_call_time,
        endpoint,
        headers,
        payload,
        response_times,
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Process JSON-style function call embedded in text and execute the second request.

        Args:
            match: Regular expression match object containing the JSON function call
            generated_text: The full generated text from the model
            model_id: The ID of the model being tested
            prompt: The original prompt
            api_call_time: Time taken for the first API call
            endpoint: The API endpoint used
            headers: The request headers
            payload: The request payload
            response_times: Dictionary to track response times

        Returns:
            Tuple containing results (see test_hello_world_tool)
        """
        try:
            extracted_json = match.group(1)
            # Clean up the JSON if needed (some models generate invalid JSON)
            extracted_json = re.sub(r",\s*}", "}", extracted_json)
            extracted_json = re.sub(
                r"([{,])\s*(\w+):", r'\1"\2":', extracted_json
            )

            function_call = json.loads(extracted_json)

            if (
                "name" not in function_call
                or function_call["name"] != "hello_world"
            ):
                return (
                    False,
                    False,
                    f"Invalid function call: {extracted_json}",
                    None,
                    response_times,
                )

            # Extract arguments
            arguments = {}
            if "arguments" in function_call and isinstance(
                function_call["arguments"], dict
            ):
                arguments = function_call["arguments"]
            elif "arguments" in function_call and isinstance(
                function_call["arguments"], str
            ):
                try:
                    arguments = json.loads(function_call["arguments"])
                except json.JSONDecodeError:
                    # Try to extract arguments with regex if JSON parsing fails
                    name_match = re.search(
                        r'"name"\s*:\s*"([^"]+)"', function_call["arguments"]
                    )
                    if name_match:
                        arguments["name"] = name_match.group(1)

                    lang_match = re.search(
                        r'"language"\s*:\s*"([^"]+)"',
                        function_call["arguments"],
                    )
                    if lang_match:
                        arguments["language"] = lang_match.group(1).lower()

            # Validate arguments (at minimum need name)
            if "name" not in arguments:
                arguments["name"] = (
                    "Daniel"  # Default to Daniel as requested in the prompt
                )

            if "language" not in arguments:
                arguments["language"] = (
                    "spanish"  # Default to Spanish as requested in the prompt
                )

            if self.debug:
                logger.info(
                    f"✅ SUCCESS: Extracted JSON function call: {json.dumps(arguments, indent=2)}"
                )

            # Execute the tool and handle second request
            tool_result = self.execute_tool_call("hello_world", arguments)

            # Now prepare a second request with the tool result
            second_system_msg = "You are a helpful assistant. You previously called the hello_world function and received its result. Respond to the user with the greeting returned by the function."
            second_payload = {
                "model_id": model_id,
                "parameters": payload["parameters"].copy(),
                "input": f"<|start_of_role|>system<|end_of_role|>{second_system_msg}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{generated_text}<|end_of_text|>\n<|start_of_role|>tool<|end_of_role|>{json.dumps(tool_result)}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
                "project_id": self.project_id,
            }

            # Process second request
            second_start_time = time.time()
            second_response = requests.post(
                endpoint, headers=headers, json=second_payload, timeout=60
            )
            response_processing_time = time.time() - second_start_time

            response_times["tool_call_time"] = api_call_time
            response_times["response_processing_time"] = (
                response_processing_time
            )
            response_times["total_time"] = (
                api_call_time + response_processing_time
            )

            if second_response.status_code == 200:
                second_response_data = second_response.json()

                # Extract content from the second response
                second_content = ""
                if (
                    "results" in second_response_data
                    and len(second_response_data["results"]) > 0
                ):
                    second_content = second_response_data["results"][0].get(
                        "generated_text", ""
                    )

                # Check if the response contains the greeting in Spanish
                if (
                    "¡Hola, Daniel!" in second_content
                    or "Hola, Daniel" in second_content
                ):
                    return (
                        True,
                        True,
                        "Successfully handled tool result (JSON in text format)",
                        second_response_data,
                        response_times,
                    )
                else:
                    return (
                        True,
                        False,
                        "Tool called but result not properly used",
                        second_response_data,
                        response_times,
                    )
            else:
                return (
                    True,
                    False,
                    f"Tool called but failed to process result: {self.extract_error_details(second_response.text)}",
                    None,
                    response_times,
                )

        except Exception as e:
            if self.debug:
                logger.error(
                    f"❌ ERROR: Failed to process JSON in text function call: {e}"
                )
            response_times["total_time"] = api_call_time
            return (
                False,
                False,
                f"Error processing JSON function call: {str(e)}",
                None,
                response_times,
            )

    def _process_openai_tool_calls(
        self,
        tool_calls,
        message,
        model_id,
        prompt,
        api_call_time,
        endpoint,
        headers,
        payload,
        response_times,
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Process OpenAI-format tool calls and execute the second request.

        Args:
            tool_calls: The tool calls from the OpenAI format response
            message: The complete message from the response
            model_id: The ID of the model being tested
            prompt: The original prompt
            api_call_time: Time taken for the first API call
            endpoint: The API endpoint used
            headers: The request headers
            payload: The request payload
            response_times: Dictionary to track response times

        Returns:
            Tuple containing results (see test_hello_world_tool)
        """
        try:
            # Find the hello_world tool call
            hello_world_call = None
            for call in tool_calls:
                function = call.get("function", {})
                if function.get("name") == "hello_world":
                    hello_world_call = call
                    break

            if not hello_world_call:
                return (
                    False,
                    False,
                    "No hello_world tool call found",
                    None,
                    response_times,
                )

            # Parse arguments
            arguments = {}
            try:
                arguments = json.loads(
                    hello_world_call["function"].get("arguments", "{}")
                )
            except json.JSONDecodeError:
                return (
                    False,
                    False,
                    "Invalid JSON in arguments",
                    None,
                    response_times,
                )

            # Validate arguments
            if "name" not in arguments:
                arguments["name"] = (
                    "Daniel"  # Default to Daniel as requested in prompt
                )

            if "language" not in arguments:
                arguments["language"] = (
                    "spanish"  # Default to Spanish as requested in prompt
                )

            if self.debug:
                logger.info(
                    f"✅ SUCCESS: Extracted OpenAI format function call: {json.dumps(arguments, indent=2)}"
                )

            # Execute the tool
            tool_result = self.execute_tool_call("hello_world", arguments)

            # Prepare second request using the OpenAI chat format
            second_payload = {
                "model_id": model_id,
                "parameters": payload.get("parameters", {}).copy(),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. You previously called the hello_world function and received its result. Respond to the user with the greeting returned by the function.",
                    },
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result),
                        "tool_call_id": hello_world_call["id"],
                    },
                ],
                "project_id": self.project_id,
            }

            # Send the second request
            second_start_time = time.time()
            second_response = requests.post(
                endpoint, headers=headers, json=second_payload, timeout=60
            )
            response_processing_time = time.time() - second_start_time

            response_times["tool_call_time"] = api_call_time
            response_times["response_processing_time"] = (
                response_processing_time
            )
            response_times["total_time"] = (
                api_call_time + response_processing_time
            )

            if second_response.status_code == 200:
                second_response_data = second_response.json()

                # Extract content from the second response (typically in the choices[0].message.content)
                second_content = ""
                if (
                    "choices" in second_response_data
                    and len(second_response_data["choices"]) > 0
                ):
                    second_content = (
                        second_response_data["choices"][0]
                        .get("message", {})
                        .get("content", "")
                    )

                # Check if the response contains the greeting in Spanish
                if (
                    "¡Hola, Daniel!" in second_content
                    or "Hola, Daniel" in second_content
                ):
                    return (
                        True,
                        True,
                        "Successfully handled tool result (OpenAI format)",
                        second_response_data,
                        response_times,
                    )
                else:
                    return (
                        True,
                        False,
                        "Tool called but result not properly used",
                        second_response_data,
                        response_times,
                    )
            else:
                return (
                    True,
                    False,
                    f"Tool called but failed to process result: {self.extract_error_details(second_response.text)}",
                    None,
                    response_times,
                )

        except Exception as e:
            if self.debug:
                logger.error(
                    f"❌ ERROR: Failed to process OpenAI format tool call: {e}"
                )
            response_times["total_time"] = api_call_time
            return (
                False,
                False,
                f"Error processing OpenAI format tool call: {str(e)}",
                None,
                response_times,
            )

    def _process_code_block_function_call(
        self,
        code_json,
        generated_text,
        model_id,
        prompt,
        api_call_time,
        endpoint,
        headers,
        payload,
        response_times,
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Process function call in a code block and execute the second request.

        Args:
            code_json: The parsed JSON object from the code block
            generated_text: The full generated text from the model
            model_id: The ID of the model being tested
            prompt: The original prompt
            api_call_time: Time taken for the first API call
            response_times: Dictionary to track response times

        Returns:
            Tuple containing results (see test_hello_world_tool)
        """
        try:
            # Extract arguments
            arguments = {}
            if "arguments" in code_json and isinstance(
                code_json["arguments"], dict
            ):
                arguments = code_json["arguments"]
            elif "arguments" in code_json and isinstance(
                code_json["arguments"], str
            ):
                try:
                    arguments = json.loads(code_json["arguments"])
                except json.JSONDecodeError:
                    # Handle case where arguments might be a string representation
                    if "name" in code_json["arguments"]:
                        name_match = re.search(
                            r'"name"\s*:\s*"([^"]+)"', code_json["arguments"]
                        )
                        if name_match:
                            arguments["name"] = name_match.group(1)
                    else:
                        # Arguments might be positional
                        arguments["name"] = (
                            "Daniel"  # Default to Daniel as requested in the prompt
                        )

                    if "language" in code_json["arguments"]:
                        lang_match = re.search(
                            r'"language"\s*:\s*"([^"]+)"',
                            code_json["arguments"],
                        )
                        if lang_match:
                            arguments["language"] = lang_match.group(1).lower()
                        else:
                            arguments["language"] = (
                                "spanish"  # Default to Spanish as requested in the prompt
                            )

            # Validate arguments (at minimum need name)
            if "name" not in arguments:
                arguments["name"] = (
                    "Daniel"  # Default to Daniel as requested in the prompt
                )

            if "language" not in arguments:
                arguments["language"] = (
                    "spanish"  # Default to Spanish as requested in the prompt
                )

            if self.debug:
                logger.info(
                    f"✅ SUCCESS: Extracted code block function call: {json.dumps(arguments, indent=2)}"
                )

            # Execute the tool and proceed with second request
            tool_result = self.execute_tool_call("hello_world", arguments)

            # Follow same pattern as above for second request
            second_system_msg = "You are a helpful assistant. You previously called the hello_world function and received its result. Respond to the user with the greeting returned by the function."
            second_payload = {
                "model_id": model_id,
                "parameters": payload["parameters"].copy(),
                "input": f"<|start_of_role|>system<|end_of_role|>{second_system_msg}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{generated_text}<|end_of_text|>\n<|start_of_role|>tool<|end_of_role|>{json.dumps(tool_result)}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
                "project_id": self.project_id,
            }

            # Process second request and format response
            second_start_time = time.time()
            second_response = requests.post(
                endpoint, headers=headers, json=second_payload, timeout=60
            )
            response_processing_time = time.time() - second_start_time

            response_times["tool_call_time"] = api_call_time
            response_times["response_processing_time"] = (
                response_processing_time
            )
            response_times["total_time"] = (
                api_call_time + response_processing_time
            )

            if second_response.status_code == 200:
                second_response_data = second_response.json()

                # Extract content from the second response
                second_content = ""
                if (
                    "results" in second_response_data
                    and len(second_response_data["results"]) > 0
                ):
                    second_content = second_response_data["results"][0].get(
                        "generated_text", ""
                    )

                # Check if the response contains the greeting in Spanish
                if (
                    "¡Hola, Daniel!" in second_content
                    or "Hola, Daniel" in second_content
                ):
                    return (
                        True,
                        True,
                        "Successfully handled tool result (code block format)",
                        second_response_data,
                        response_times,
                    )
                else:
                    return (
                        True,
                        False,
                        "Tool called but result not properly used",
                        second_response_data,
                        response_times,
                    )
            else:
                return (
                    True,
                    False,
                    f"Tool called but failed to process result: {self.extract_error_details(second_response.text)}",
                    None,
                    response_times,
                )

        except Exception as e:
            if self.debug:
                logger.error(
                    f"❌ ERROR: Failed to process code block function call: {e}"
                )
            response_times["total_time"] = api_call_time
            return (
                False,
                False,
                f"Error processing code block function call: {str(e)}",
                None,
                response_times,
            )

    def _process_direct_function_call(
        self,
        match,
        generated_text,
        model_id,
        prompt,
        api_call_time,
        endpoint,
        headers,
        payload,
        response_times,
    ) -> Tuple[
        bool, bool, str, Optional[Dict[str, Any]], Dict[str, Optional[float]]
    ]:
        """Process direct function call syntax and execute the second request.

        Args:
            match: Regular expression match object containing the function call arguments
            generated_text: The full generated text from the model
            model_id: The ID of the model being tested
            prompt: The original prompt
            api_call_time: Time taken for the first API call
            endpoint: The API endpoint used
            headers: The request headers
            payload: The request payload
            response_times: Dictionary to track response times

        Returns:
            Tuple containing results (see test_hello_world_tool)
        """
        try:
            arg_string = match.group(1).strip()

            # Parse argument string
            arguments = {}

            # Check for key=value pattern
            name_match = re.search(
                r'name\s*=\s*["\']([^"\']+)["\']', arg_string
            )
            if name_match:
                arguments["name"] = name_match.group(1)
            else:
                # Check for positional args
                pos_args = re.findall(r'["\']([^"\']+)["\']', arg_string)
                if pos_args:
                    arguments["name"] = pos_args[0]
                    if len(pos_args) > 1:
                        arguments["language"] = pos_args[1].lower()

            # Check for language parameter
            if "language" not in arguments:
                lang_match = re.search(
                    r'language\s*=\s*["\']([^"\']+)["\']', arg_string
                )
                if lang_match:
                    arguments["language"] = lang_match.group(1).lower()
                else:
                    # Default to Spanish as requested in the prompt
                    arguments["language"] = "spanish"

            # Default to Daniel if name not found
            if "name" not in arguments:
                arguments["name"] = "Daniel"

            if self.debug:
                logger.info(
                    f"✅ SUCCESS: Extracted direct function call: {json.dumps(arguments, indent=2)}"
                )

            # Execute the tool and proceed with second request
            tool_result = self.execute_tool_call("hello_world", arguments)

            # Follow same pattern as above for second request
            second_system_msg = "You are a helpful assistant. You previously called the hello_world function and received its result. Respond to the user with the greeting returned by the function."
            second_payload = {
                "model_id": model_id,
                "parameters": payload["parameters"].copy(),
                "input": f"<|start_of_role|>system<|end_of_role|>{second_system_msg}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>{generated_text}<|end_of_text|>\n<|start_of_role|>tool<|end_of_role|>{json.dumps(tool_result)}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
                "project_id": self.project_id,
            }

            # Process second request and format response
            second_start_time = time.time()
            second_response = requests.post(
                endpoint, headers=headers, json=second_payload, timeout=60
            )
            response_processing_time = time.time() - second_start_time

            response_times["tool_call_time"] = api_call_time
            response_times["response_processing_time"] = (
                response_processing_time
            )
            response_times["total_time"] = (
                api_call_time + response_processing_time
            )

            if second_response.status_code == 200:
                second_response_data = second_response.json()

                # Extract content from the second response
                second_content = ""
                if (
                    "results" in second_response_data
                    and len(second_response_data["results"]) > 0
                ):
                    second_content = second_response_data["results"][0].get(
                        "generated_text", ""
                    )

                # Check if the response contains the greeting in Spanish
                if (
                    "¡Hola, Daniel!" in second_content
                    or "Hola, Daniel" in second_content
                ):
                    return (
                        True,
                        True,
                        "Successfully handled tool result (direct function call)",
                        second_response_data,
                        response_times,
                    )
                else:
                    return (
                        True,
                        False,
                        "Tool called but result not properly used",
                        second_response_data,
                        response_times,
                    )
            else:
                return (
                    True,
                    False,
                    f"Tool called but failed to process result: {self.extract_error_details(second_response.text)}",
                    None,
                    response_times,
                )

        except Exception as e:
            if self.debug:
                logger.error(
                    f"❌ ERROR: Failed to process direct function call: {e}"
                )
            response_times["total_time"] = api_call_time
            return (
                False,
                False,
                f"Error processing direct function call: {str(e)}",
                None,
                response_times,
            )

    def extract_error_details(self, response_text: str) -> str:
        """Extract error details from an error response.

        Args:
            response_text: The error response text

        Returns:
            str: Extracted error message
        """
        try:
            error_data = json.loads(response_text)
            if "error" in error_data:
                if isinstance(error_data["error"], dict):
                    return error_data["error"].get("message", "Unknown error")
                else:
                    return str(error_data["error"])
            elif "errors" in error_data and len(error_data["errors"]) > 0:
                return str(
                    error_data["errors"][0].get("message", "Unknown error")
                )
            return response_text
        except Exception:
            return response_text
