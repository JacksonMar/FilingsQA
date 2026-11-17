.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-integration up down ingest serve eval eval-smoke

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install deps and git hooks
	uv sync
	uv run pre-commit install

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy (strict) on src/
	uv run mypy

test: ## Unit tests (no network, LLM mocked)
	uv run pytest -m "not integration and not network"

test-integration: ## Tests that need Qdrant running (make up)
	uv run pytest -m integration

up: ## Start Qdrant
	docker compose up -d --wait

down: ## Stop Qdrant
	docker compose down

ingest: ## Download filings, chunk, index (Phase 1)
	@echo "Not implemented yet (Phase 1)"; exit 1

serve: ## Run FastAPI server (Phase 2)
	@echo "Not implemented yet (Phase 2)"; exit 1

eval: ## Full evaluation run (Phase 4)
	@echo "Not implemented yet (Phase 4)"; exit 1

eval-smoke: ## Small eval subset used as CI quality gate (Phase 4)
	@echo "Not implemented yet (Phase 4)"; exit 1
