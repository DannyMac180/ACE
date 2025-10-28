.PHONY: setup lint type test run-mcp seed refine bench

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]

lint:
	ruff check .

type:
	mypy ace

test:
	pytest -q

run-mcp:
	python3 -m ace_mcp_server

seed:
	python3 scripts/seed.py

refine:
	python3 -m ace.refine.run --threshold 0.90

bench:
	pytest -q eval/test_bench_smoke.py -q
