.PHONY: help install lock test test-fast parity lint format typecheck check clean

PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/python ## Create the venv and install the package with dev extras
	$(BIN)/pip install -e ".[api,dev]"

lock: install ## Freeze the resolved environment
	$(BIN)/pip freeze --exclude-editable > requirements.lock

test: ## Run the full test suite
	$(BIN)/pytest

test-fast: ## Run everything except the slow MediaPipe/model tests
	$(BIN)/pytest -m "not slow"

parity: ## Run only the parity suite against the research reference
	$(BIN)/pytest -m parity

lint: ## Check formatting and lint rules
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Apply formatting and autofixable lint rules
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: ## Type check the package and service
	$(BIN)/mypy aced api

check: lint typecheck test ## Everything CI runs

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
