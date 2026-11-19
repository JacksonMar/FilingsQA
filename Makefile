.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-integration test-diff up down manifest ingest dataset snapshot serve eval eval-smoke

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install deps and git hooks
	uv sync
	uv run pre-commit install

lint: ## Ruff lint + format check + GitHub workflow lint
	uv run ruff check .
	uv run ruff format --check .
	uv run actionlint .github/workflows/*.yml

format: ## Auto-format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy (strict) on src/
	uv run mypy

test: ## Unit tests (no network, LLM mocked)
	uv run pytest -m "not integration and not network and not differential"

test-integration: ## Tests that need Qdrant running (make up)
	uv run pytest -m integration

test-diff: ## Differential test: our Item parser vs edgartools on real filings
	uv run pytest -m differential -v

up: ## Start Qdrant
	docker compose up -d --wait

down: ## Stop Qdrant
	docker compose down

AS_OF ?= 2026-11-17
VERSION ?= v1
manifest: ## Pin latest 10-K + two 10-Q per company as of AS_OF (commit the result)
	uv run filingsqa build-manifest --as-of $(AS_OF)

ingest: ## Download pinned filings + XBRL facts, parse, chunk and index into Qdrant
	uv run filingsqa ingest
	uv run filingsqa index

dataset: ## Build golden dataset VERSION (needs make ingest) and validate it
	uv run filingsqa dataset build --version $(VERSION)
	uv run filingsqa dataset validate --version $(VERSION)

snapshot: ## Refresh the committed chunk snapshot used by the CI smoke gate
	uv run filingsqa dataset snapshot --version $(VERSION)

serve: ## Run the API on http://localhost:8000 (docs at /docs)
	uv run uvicorn filingsqa.api.app:create_app --factory --port 8000

eval: ## Full eval (deterministic + LLM judge) on golden VERSION; report in eval/reports/
	uv run filingsqa eval run --version $(VERSION) --subset full --judge

eval-smoke: ## ~20-item stratified subset, deterministic metrics only: the CI quality gate
	uv run filingsqa eval run --version $(VERSION) --subset smoke --no-judge
