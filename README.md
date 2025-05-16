# WatsonX Tool Tester

A Python tool for testing call capabilities of AI models in IBM WatsonX and via LiteLLM proxies.
This is a simple tool to test whether AI models can properly invoke a simple tool when prompted and process the tool response and use the results correctly. It is a brute-force test that tries to call the tool in multiple ways and checks if the model can handle the result. By default it runs against all available models and does not attempt to determine if the model should support the tool or not.

## Overview

WatsonX Tool Tester provides a standardized way to test whether AI models can properly:
1. Invoke a simple tool when prompted
2. Process the tool response and use the results correctly

The package supports direct testing against WatsonX.ai API or through a LiteLLM proxy server.

![Example Tool Tester Output](./images/sample_report.png)

## How Tests Work

The tool testing process evaluates two key capabilities:

1. **Tool Call Support**: Tests if a model correctly identifies when to use a tool and properly invokes it with the right parameters.
2. **Result Handling**: Tests if a model can correctly process and utilize the results returned by the tool.

### The Hello World Tool Test

Models are tested using a simple `hello_world` tool that:
- Takes a "name" parameter (required) and a "language" parameter (optional)
- Returns a greeting in the specified language

The test works as follows:
1. The model is asked to greet someone in a specific language using the tool
2. If the model correctly invokes the tool, that counts as "TOOL SUPPORT"
3. The test then feeds the tool's response back to the model
4. If the model correctly incorporates the greeting from the tool in its response, that counts as "TOOL RESULT HANDLED"

### Supported Tool Call Formats

The tool tester recognizes multiple formats that models can use to invoke functions:

| Format | Description | Success Message |
|--------|-------------|----------------|
| **OpenAI Format** | Uses the standard OpenAI tool calling schema with properly formatted tool_calls | "Successfully handled tool result (OpenAI format)" |
| **Direct Function Call** | Uses programming-style function calling syntax (e.g., `hello_world("Daniel", "spanish")`) | "Successfully handled tool result (direct function call)" |
| **JSON in Text** | Embeds a JSON object with function name and arguments in regular text | "Successfully handled tool result (JSON in text format)" |
| **Code Block Format** | Places a JSON function call object inside code blocks | "Successfully handled tool result (code block format)" |
| **Python Tag Format** | Uses special <\|python_tag\|> wrapper for function calls (common in some models) | "Successfully handled tool result (Python tag format)" |

The test will attempt each format when testing a model, reporting the specific format that worked when successful.

### Success Criteria

Success in these tests is measured at two levels:

- **Partial Success**: The model correctly identifies when to use the tool and invokes it with valid parameters.
- **Full Success**: The model not only invokes the tool correctly but also properly uses the results returned by the tool.

Only models that achieve full success are considered in the performance metrics (average times and fastest model determination).

## Installation

```bash
# Clone the repository
git clone git@github.com:daniel-butler-irl/watsonx_model_tool_test.git
cd watsonx_MODEL_tool_test

# Install the package
pip install .

# Or install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

The package provides a command-line interface for running tests:

```bash
# Test all models in WatsonX
watsonx-tool-tester test --client watsonx \
    --watsonx-apikey YOUR_API_KEY \
    --watsonx-project-id YOUR_PROJECT_ID

# Test a specific model
watsonx-tool-tester test --client watsonx \
    --watsonx-apikey YOUR_API_KEY \
    --watsonx-project-id YOUR_PROJECT_ID \
    --model ibm/granite-20b-instruct

# Exclude specific models from testing
watsonx-tool-tester test --client watsonx \
    --watsonx-apikey YOUR_API_KEY \
    --watsonx-project-id YOUR_PROJECT_ID \
    --exclude "mixtral" --exclude "claude"

# Exclude models using a file with model names/patterns
watsonx-tool-tester test --client watsonx \
    --watsonx-apikey YOUR_API_KEY \
    --watsonx-project-id YOUR_PROJECT_ID \
    --exclude-file excludes.txt

# List available models
watsonx-tool-tester list-models --client watsonx \
    --watsonx-apikey YOUR_API_KEY \
    --watsonx-project-id YOUR_PROJECT_ID

# Test models via LiteLLM proxy
watsonx-tool-tester test --client litellm \
    --litellm-url http://localhost:8000 \
    --litellm-token YOUR_TOKEN
```

### Using environment variables

You can also use environment variables to configure the tool:

```bash
export WATSONX_APIKEY=YOUR_API_KEY
export WATSONX_PROJECT_ID=YOUR_PROJECT_ID
export WATSONX_TOOL_CLIENT=watsonx  # or litellm
export WATSONX_TOOL_EXCLUDE="mixtral,llama"  # Comma-separated list of models to exclude
export WATSONX_TOOL_EXCLUDE_FILE="./excludes.txt"   # File with models to exclude

watsonx-tool-tester test
```

For LiteLLM:

```bash
export LITELLM_URL=http://localhost:8000
export LITELLM_TOKEN=YOUR_TOKEN
export WATSONX_TOOL_CLIENT=litellm

watsonx-tool-tester test
```

### Python API

You can also use the package programmatically in your Python code:

```python
from watsonx_tool_tester import Config, ClientType, ModelTester

# Configure the tester
config = Config(
    client_type=ClientType.WATSONX,
    watsonx_apikey="YOUR_API_KEY",
    watsonx_project_id="YOUR_PROJECT_ID",
    debug=True,  # Enable verbose logging
)

# Create the tester
tester = ModelTester(config)

# Test all models
results = tester.test_all_models()

# Test a specific model
results = tester.test_all_models(filter_model="ibm/granite-20b-instruct")

# Exclude specific models
config.exclude_models = ["model1", "model2"]
results = tester.test_all_models()
```

## Test Results Explained

The test results are presented in a table format with the following columns:

| Column | Description |
|--------|-------------|
| MODEL | The model ID being tested |
| TOOL SUPPORT | Whether the model correctly invokes the tool (✅ or ❌) |
| HANDLED | Whether the model correctly uses the results returned by the tool (✅ or ❌) |
| CALL TIME | Time taken for the model to generate the initial tool call |
| RESP TIME | Time taken for the model to process the tool response |
| TOTAL TIME | Sum of tool call time and response time |
| DETAILS | Additional information about test results or errors |

### Results Summary

After the table, a summary section provides:

1. **Basic Statistics**:
   - Total models tested
   - Number of models that support tool calls
   - Number of models that correctly handle tool responses

2. **Performance Metrics** (only for fully successful models):
   - Average tool call time
   - Average response processing time
   - Average total time
   - Fastest model (based on total time)

3. **Support Categories**:
   - Full support: Models that can call the tool and use its result
   - Partial support: Models that can call the tool but ignore its result
   - No support: Models that cannot call the tool at all

## Available Configuration Options

| CLI Option               | Environment Variable      | Description                            |
|--------------------------|---------------------------|----------------------------------------|
| `--client`               | `WATSONX_TOOL_CLIENT`     | Client type (`watsonx` or `litellm`)   |
| `--watsonx-url`          | `WATSONX_URL`             | URL for WatsonX API                    |
| `--watsonx-apikey`       | `WATSONX_APIKEY`          | API key for WatsonX API                |
| `--watsonx-project-id`   | `WATSONX_PROJECT_ID`      | Project ID for WatsonX API             |
| `--watsonx-region`       | `WATSONX_REGION`          | Region for WatsonX API                 |
| `--watsonx-api-version`  | `WATSONX_API_VERSION`     | API version for WatsonX API            |
| `--litellm-url`          | `LITELLM_URL`             | URL for LiteLLM proxy API              |
| `--litellm-token`        | `LITELLM_TOKEN`           | Authentication token for LiteLLM proxy  |
| `--model`, `-m`          | `WATSONX_TOOL_MODEL`      | Specific model to test                 |
| `--exclude`              | `WATSONX_TOOL_EXCLUDE`    | Models to exclude from testing         |
| `--exclude-file`         | `WATSONX_TOOL_EXCLUDE_FILE` | File with models to exclude            |
| `--sort`                 | `WATSONX_TOOL_SORT`       | How to sort results                    |
| `--debug`, `-d`          | `WATSONX_TOOL_DEBUG`      | Enable debug logging                   |
| `--log-dir`              | `WATSONX_TOOL_LOG_DIR`    | Directory for log files                |
| `--output`, `-o`         | `WATSONX_TOOL_OUTPUT`     | File to save results                   |

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/ibm/watsonx-tool-tester.git
cd watsonx-tool-tester

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Use make targets for common tasks
make test           # Run all tests
make lint           # Run linter checks
make format         # Format code
make check          # Run all checks

# Or run pytest directly
pytest
pytest --cov=watsonx_tool_tester
pytest tests/tools/test_hello_world.py
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
