.PHONY: help sync test clean simulations figures monte-carlo verify

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Install Python dependencies (including dev tools)
	uv sync --extra dev

test:  ## Run pytest
	uv run pytest -v

clean:  ## Remove build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

simulations:  ## Run all Monte Carlo simulations
	uv run python scripts/simulations/run_information_rent.py
	uv run python scripts/simulations/run_evt_domains.py
	uv run python scripts/simulations/run_coverage.py
	uv run python scripts/simulations/run_cluster_variance.py

figures:  ## Generate all figures and tables from simulation data
	uv run python scripts/figures/generate_all.py

monte-carlo: simulations figures  ## Run simulations and generate figures
	@echo "Monte Carlo analysis complete"

verify:  ## Run symbolic and numerical theorem verification suites
	uv run python scripts/verify_symbolic.py
	uv run python scripts/verify_numerical.py
