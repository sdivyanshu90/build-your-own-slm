# =============================================================================
# BYO-SLM developer task runner. Run `make help` for the full list.
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON ?= python3
PIP := $(PYTHON) -m pip

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install the package with dev extras (editable)
	$(PIP) install -e ".[dev]"

.PHONY: hooks
hooks: ## Install git pre-commit hooks
	pre-commit install

.PHONY: lint
lint: ## Run ruff lint checks
	ruff check src tests

.PHONY: format
format: ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

.PHONY: typecheck
typecheck: ## Run mypy static type checks
	mypy src

.PHONY: test
test: ## Run the full test suite with coverage
	pytest --cov=slm --cov-report=term-missing --cov-report=xml

.PHONY: test-fast
test-fast: ## Run tests excluding slow markers
	pytest -m "not slow"

.PHONY: check
check: lint typecheck test ## Run lint + typecheck + tests (CI parity)

.PHONY: prepare
prepare: ## Build the demo dataset + tokenizer (configs/tiny.yaml)
	$(PYTHON) -m slm.cli prepare-data --config configs/tiny.yaml

.PHONY: train
train: ## Train the tiny demo model
	$(PYTHON) -m slm.cli train --config configs/tiny.yaml

.PHONY: generate
generate: ## Sample from the trained tiny model
	$(PYTHON) -m slm.cli generate --model-dir checkpoints/tiny --prompt "Once upon a time"

.PHONY: serve
serve: ## Run the API server (reload enabled)
	$(PYTHON) -m slm.cli serve --reload

.PHONY: docker-build
docker-build: ## Build the production container image
	docker build -t byo-slm:latest .

.PHONY: docker-up
docker-up: ## Start the full stack via docker compose
	docker compose up --build

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov \
		.coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
