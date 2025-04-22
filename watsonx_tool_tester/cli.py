#!/usr/bin/env python3
"""
Command-line interface for WatsonX Tool Tester.

This module provides the CLI entry point for the WatsonX Tool Tester package,
allowing users to test AI model tool call capabilities from the command line.
"""

import sys
from typing import Dict, Optional

import click

from watsonx_tool_tester.config import (
    load_config_from_env,
    update_config_from_args,
    validate_config,
)
from watsonx_tool_tester.testers.model_tester import ModelTester
from watsonx_tool_tester.testers.result_handler import ResultHandler
from watsonx_tool_tester.utils import errors


@click.group()
@click.version_option()
def cli():
    """Test tool call capabilities of AI models in IBM WatsonX and via LiteLLM proxies."""
    pass


@cli.command()
@click.option(
    "--client",
    type=click.Choice(["watsonx", "litellm"]),
    help="Client type to use (watsonx or litellm)",
)
@click.option("--watsonx-url", help="Base URL for WatsonX API")
@click.option("--watsonx-apikey", help="API key for WatsonX API")
@click.option("--watsonx-project-id", help="Project ID for WatsonX API")
@click.option("--watsonx-region", help="Region for WatsonX API")
@click.option(
    "--watsonx-api-version",
    default="2023-05-29",
    help="API version for WatsonX API (default: 2023-05-29)",
)
@click.option("--litellm-url", help="URL for LiteLLM proxy")
@click.option("--litellm-token", help="Authentication token for LiteLLM proxy")
@click.option("--model", "-m", help="Specific model to test")
@click.option(
    "--exclude",
    multiple=True,
    help="Model name or regex pattern to exclude from testing (can be used multiple times)",
)
@click.option(
    "--exclude-file",
    type=click.Path(
        exists=True, readable=True, file_okay=True, dir_okay=False
    ),
    help="File containing model names or regex patterns to exclude (one per line)",
)
@click.option(
    "--sort",
    type=click.Choice(
        ["name", "tool_call_time", "response_time", "total_time"]
    ),
    help="How to sort the results",
)
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
@click.option("--log-dir", help="Directory for log files")
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set logging level (default: INFO, overridden by --debug)",
)
@click.option("--output", "-o", help="File to save results in JSON format")
def test(
    client: Optional[str],
    watsonx_url: Optional[str],
    watsonx_apikey: Optional[str],
    watsonx_project_id: Optional[str],
    watsonx_region: Optional[str],
    watsonx_api_version: Optional[str],
    litellm_url: Optional[str],
    litellm_token: Optional[str],
    model: Optional[str],
    exclude: Optional[tuple],
    exclude_file: Optional[str],
    sort: Optional[str],
    debug: bool,
    log_dir: Optional[str],
    log_level: Optional[str],
    output: Optional[str],
):
    """Test AI models for tool call capabilities.

    This command tests if AI models can properly call tools and use the results.
    If no specific model is provided, all available models will be tested.

    Use --exclude or --exclude-file options to skip specific models or model patterns.
    """
    # Load config from environment and update with CLI arguments
    config = load_config_from_env()

    args: Dict = {
        "client": client,
        "watsonx_url": watsonx_url,
        "watsonx_apikey": watsonx_apikey,
        "watsonx_project_id": watsonx_project_id,
        "watsonx_region": watsonx_region,
        "watsonx_api_version": watsonx_api_version,
        "litellm_url": litellm_url,
        "litellm_token": litellm_token,
        "model": model,
        "exclude": list(exclude) if exclude else None,
        "exclude_file": exclude_file,
        "sort": sort,
        "debug": debug,
        "log_dir": log_dir,
        "log_level": log_level,
        "output": output,
    }

    # Filter out None values
    args = {k: v for k, v in args.items() if v is not None}

    config = update_config_from_args(config, args)

    # Validate config
    is_valid, error_message = validate_config(config)
    if not is_valid:
        click.secho(f"Error: {error_message}", fg="red")
        sys.exit(1)

    try:
        # Initialize and run the tester
        tester = ModelTester(config)
        results = tester.test_all_models(filter_model=config.model)

        # Print summary
        handler = ResultHandler()
        handler.print_summary(results, config.sort_key)

    except errors.CredentialError as e:
        click.secho(f"Credential error: {str(e)}", fg="red")
        sys.exit(1)
    except errors.ConfigurationError as e:
        click.secho(f"Configuration error: {str(e)}", fg="red")
        sys.exit(1)
    except errors.ClientError as e:
        click.secho(f"Client error: {str(e)}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")
        if config.debug:
            click.echo("Debug trace:")
            import traceback

            click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--client",
    type=click.Choice(["watsonx", "litellm"]),
    help="Client type to use (watsonx or litellm)",
)
@click.option("--watsonx-url", help="Base URL for WatsonX API")
@click.option("--watsonx-apikey", help="API key for WatsonX API")
@click.option("--watsonx-project-id", help="Project ID for WatsonX API")
@click.option("--watsonx-region", help="Region for WatsonX API")
@click.option(
    "--watsonx-api-version",
    default="2023-05-29",
    help="API version for WatsonX API (default: 2023-05-29)",
)
@click.option("--litellm-url", help="URL for LiteLLM proxy")
@click.option("--litellm-token", help="Authentication token for LiteLLM proxy")
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set logging level (default: INFO, overridden by --debug)",
)
def list_models(
    client: Optional[str],
    watsonx_url: Optional[str],
    watsonx_apikey: Optional[str],
    watsonx_project_id: Optional[str],
    watsonx_region: Optional[str],
    watsonx_api_version: Optional[str],
    litellm_url: Optional[str],
    litellm_token: Optional[str],
    debug: bool,
    log_level: Optional[str],
):
    """List available models.

    This command lists all available models from the API.
    """
    # Load config from environment and update with CLI arguments
    config = load_config_from_env()

    args: Dict = {
        "client": client,
        "watsonx_url": watsonx_url,
        "watsonx_apikey": watsonx_apikey,
        "watsonx_project_id": watsonx_project_id,
        "watsonx_region": watsonx_region,
        "watsonx_api_version": watsonx_api_version,
        "litellm_url": litellm_url,
        "litellm_token": litellm_token,
        "debug": debug,
        "log_level": log_level,
    }

    # Filter out None values
    args = {k: v for k, v in args.items() if v is not None}

    config = update_config_from_args(config, args)

    # Validate config
    is_valid, error_message = validate_config(config)
    if not is_valid:
        click.secho(f"Error: {error_message}", fg="red")
        sys.exit(1)

    try:
        # Initialize the tester and get available models
        tester = ModelTester(config)

        if not tester.validate_credentials():
            click.secho("Invalid API credentials", fg="red")
            sys.exit(1)

        models = tester.get_available_models()

        if not models:
            click.secho("No models found", fg="yellow")
            sys.exit(0)

        # Display models
        click.secho(f"\nAvailable models ({len(models)}):", fg="green")
        for model in sorted(models, key=lambda m: m["id"]):
            model_id = model["id"]
            display_name = model.get("name", model_id)
            click.echo(f"- {model_id} ({display_name})")

        click.echo("")

    except errors.CredentialError as e:
        click.secho(f"Credential error: {str(e)}", fg="red")
        sys.exit(1)
    except errors.ConfigurationError as e:
        click.secho(f"Configuration error: {str(e)}", fg="red")
        sys.exit(1)
    except errors.ClientError as e:
        click.secho(f"Client error: {str(e)}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red")
        if config.debug:
            click.echo("Debug trace:")
            import traceback

            click.echo(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    cli()
