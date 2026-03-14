.PHONY: setup lint type test run-mcp seed refine bench baseline

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .[dev]

lint:
	$(RUFF) check .

type:
	$(MYPY) ace

test:
	$(PYTEST) -q

run-mcp:
	$(PYTHON) -m ace_mcp_server

seed:
	$(PYTHON) scripts/seed.py

refine:
	$(PYTHON) -m ace.refine.run --threshold 0.90

bench:
	$(PYTEST) -o "addopts=-v" -m benchmark eval/test_reflection_bench.py eval/test_retrieval_bench.py

baseline:
	$(PYTHON) -m ace.cli eval run --suite retrieval --write-baseline eval/baseline.json
