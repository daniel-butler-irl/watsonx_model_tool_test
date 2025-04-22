.PHONY: clean install develop test lint format check verify all venv venv-clean setup-dev

# Default target
all: check test

# Virtual environment
VENV_DIR = venv
PYTHON = python3
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip

# Create virtual environment
venv:
	@echo "Creating virtual environment at $(VENV_DIR)"
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

# Remove virtual environment
venv-clean:
	@echo "Removing virtual environment"
	rm -rf $(VENV_DIR)

# Comprehensive development setup
setup-dev: venv
	@echo "Installing development dependencies..."
	$(VENV_PIP) install -e ".[dev]"
	@echo "Running code formatting..."
	$(VENV_PYTHON) -m isort watsonx_tool_tester tests
	$(VENV_PYTHON) -m black watsonx_tool_tester tests
	@echo "Running lint checks..."
	$(VENV_PYTHON) -m flake8 watsonx_tool_tester tests
	$(VENV_PYTHON) -m mypy watsonx_tool_tester
	@echo "Setup complete! Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment."

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install package
install:
	pip install .

# Install package in development mode
develop:
	pip install -e ".[dev]"

# Run tests
test:
	$(VENV_PYTHON) -m pytest

# Run tests with coverage
coverage:
	$(VENV_PYTHON) -m pytest --cov=watsonx_tool_tester --cov-report=term --cov-report=html

# Run lint checks
lint:
	$(VENV_PYTHON) -m flake8 watsonx_tool_tester tests
	$(VENV_PYTHON) -m mypy watsonx_tool_tester

# Format code
format:
	$(VENV_PYTHON) -m isort watsonx_tool_tester tests
	$(VENV_PYTHON) -m black watsonx_tool_tester tests

# Run code style checks without modifying files
check:
	$(VENV_PYTHON) -m isort --check watsonx_tool_tester tests
	$(VENV_PYTHON) -m black --check watsonx_tool_tester tests
	$(VENV_PYTHON) -m flake8 watsonx_tool_tester tests

# Verify all quality checks
verify: check lint test

# Build package
build:
	$(VENV_PYTHON) -m build

# Run the CLI tool with environment variables from .env file
run:
	$(VENV_PYTHON) -m watsonx_tool_tester