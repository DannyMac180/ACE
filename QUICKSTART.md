# ACE Quickstart

This is the shortest path from clone to a working local ACE loop.

## Prerequisites

- Python 3.11+
- `git`
- an activated virtual environment, or willingness to use `.venv/`

## 1. Install

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## 2. Verify The Repo

Run the default checks before making changes:

```bash
make test
```

Optional:

```bash
make lint
make type
```

## 3. Seed A Local Playbook

Create the default local SQLite playbook:

```bash
make seed
```

Check that ACE can read it:

```bash
ace stats --json
ace retrieve "hybrid retrieval for code agents" --top-k 5
```

## 4. Run One Adaptation Loop

Prepare a task document shaped like:

```json
{
  "query": "Fix flaky retrieval tests",
  "retrieved_bullet_ids": [],
  "code_diff": "",
  "test_output": "FAIL tests/test_retrieval.py::test_timeout",
  "logs": "ERROR: retrieval timed out after 5s",
  "env_meta": {
    "ci": true
  }
}
```

The fastest offline path uses the deterministic demo reflector that already ships in the repo:

```bash
python scripts/demo_reflect.py task.json > reflection.json
ace curate --reflection reflection.json --json > delta.json
ace commit --delta delta.json --json
ace stats --json
```

If you want to use a live model instead, replace the first command with:

```bash
ace reflect --doc task.json --json > reflection.json
```

If you want to preview the delta before writing:

```bash
ace commit --delta delta.json --dry-run
```

## 5. Start A Server

MCP:

```bash
python -m ace_mcp_server
```

HTTP:

```bash
ace serve --host 127.0.0.1 --port 8000
```

Useful HTTP endpoints:

- `GET /health`
- `POST /retrieve`
- `POST /feedback`
- `GET /stats`
- `GET /playbook/version`

## 6. Common Environment Overrides

ACE reads `configs/default.toml` and lets environment variables override it.

```bash
export ACE_DB_URL="sqlite:///ace.db"
export ACE_RETRIEVAL_TOPK=24
export ACE_REFINE_THRESHOLD=0.90
export ACE_LOG_LEVEL=INFO
export MCP_TRANSPORT=stdio
```

If you are using a hosted LLM provider, set its credentials in the environment before running `ace reflect`, `ace pipeline`, or any server flow that performs reflection.

## 7. Next Steps

- Read [README.md](README.md) for the full surface area.
- Read [docs/configuration.md](docs/configuration.md) for config details.
- Read [docs/getting-started-example.md](docs/getting-started-example.md) for a more guided end-to-end example.
