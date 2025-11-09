.PHONY: help install lint test typecheck clean

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies (production + dev)
        python -m pip install --upgrade pip setuptools wheel --break-system-packages
        pip install -r requirements.txt --break-system-packages
        pip install -r dev-requirements.txt --break-system-packages

lint: ## Run static analysis (Ruff + Black)
        ruff check .
        black --check .

lint-fix: ## Run linting and formatting with automatic fixes
        black .
        ruff check --fix .

typecheck: ## Run mypy type checks
        mypy .

test: ## Run tests (excluding e2e and integration)
        pytest -q

test-all: ## Run all tests including integration
	pytest -m ""

test-e2e: ## Run e2e tests only
	pytest -m e2e

clean: ## Clean up Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
