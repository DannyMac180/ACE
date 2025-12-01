# ACE: Agentic Context Engineering

ACE is a configuration-driven toolkit for building agentic systems with high-quality context:
- Hybrid retrieval (vector + lexical) with tunable weights
- Content deduplication and refinement (cosine and MinHash thresholds)
- Pluggable embeddings and LLMs
- Structured logging
- MCP transport options (stdio, http, sse)

## Quick start

1) Clone and install
- Create a Python virtual environment and install project dependencies.
  - Typical options:
    - `pip install -r requirements.txt`
    - or `pip install -e .`
- Ensure your runtime can import ace.core.config.

2) Configure
- Edit [configs/default.toml](configs/default.toml) (recommended), or
- Override specific settings with environment variables (ACE_*; MCP_* for MCP transport/port).

3) Use the CLI or in code

### CLI Usage

The `ace` command provides full access to the ACE workflow:

```bash
# Retrieve bullets matching a query
ace retrieve "authentication patterns" --top-k 10

# View playbook statistics
ace stats --json

# Run the full evolution pipeline (reflect → curate → commit)
ace evolve --doc task.json --print-delta --apply

# Tag a bullet as helpful or harmful
ace tag strat-00091 --helpful

# Deduplicate and consolidate bullets
ace refine --threshold 0.90 --dry-run

# Dump full playbook
ace playbook dump --out playbook.json

# Import playbook from JSON file
ace playbook import --file playbook.json
```

Available commands:
- `retrieve` - Retrieve bullets matching a query
- `reflect` - Generate reflection from task execution data
- `curate` - Convert reflection to delta operations
- `commit` - Apply delta operations to playbook
- `evolve` - Run full reflect→curate→commit pipeline
- `playbook dump` - Export full playbook JSON
- `playbook import` - Import playbook from JSON file
- `tag` - Tag a bullet as helpful or harmful
- `refine` - Deduplicate and consolidate bullets
- `stats` - Show playbook statistics
- `serve` - Start online server for test-time sequential adaptation
- `train` - Run multi-epoch offline adaptation training

### Online Serving (Test-Time Adaptation)

Start the HTTP server for real-time adaptation with execution feedback:

```bash
# Basic cold start
ace serve --host 127.0.0.1 --port 8000

# Warm start with pre-loaded playbook (recommended for production)
ace serve --warmup playbook.json

# Disable automatic adaptation (retrieve-only mode)
ace serve --no-adapt
```

The `--warmup` option supports offline warmup as described in the ACE paper (Table 3),
where pre-training the playbook offline before online adaptation improves performance.

Use `ace <command> --help` for detailed usage of each command.

### Python API

```python
from ace.core.config import get_config

cfg = get_config()  # loads configs/default.toml + env overrides and validates
# ... use cfg.database.url, cfg.retrieval.top_k, cfg.llm.model, etc.
```

## Trajectory Schema

External agents (MCP clients, CI systems, IDEs) can record task execution data using the `TrajectoryDoc` schema. This is the standard format for feeding execution context to the ACE reflector.

### TrajectoryDoc Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | **Yes** | The task or query that was executed |
| `retrieved_bullet_ids` | string[] | No | IDs of playbook bullets retrieved and used |
| `code_diff` | string | No | Code changes made during execution |
| `test_output` | string | No | Test results or output |
| `logs` | string | No | Execution logs, errors, stack traces |
| `env_meta` | object | No | Environment metadata (final_status, tool versions, etc.) |
| `tools_used` | string[] | No | List of tools or actions invoked |

### Example Payload

```json
{
  "query": "Fix the authentication bug in login.py",
  "retrieved_bullet_ids": ["strat-00091", "trbl-00022"],
  "code_diff": "--- a/login.py\n+++ b/login.py\n@@ -15,3 +15,5 @@\n+    if not token:\n+        raise AuthError('Missing token')",
  "test_output": "PASSED test_login_with_valid_token\nFAILED test_login_without_token - AssertionError",
  "logs": "2024-01-15 10:23:45 ERROR: AuthError raised during login attempt",
  "env_meta": {
    "final_status": "partial",
    "python_version": "3.11.5",
    "test_framework": "pytest"
  },
  "tools_used": ["read_file", "edit_file", "run_tests"]
}
```

### MCP Integration

Use the MCP tools to record trajectories and generate reflections:

```bash
# 1. Record a trajectory (returns trajectory_id)
ace.record_trajectory(
  query="Fix authentication bug",
  code_diff="...",
  test_output="...",
  logs="..."
)
# Returns: {"trajectory_id": "traj-abc123def456"}

# 2. Generate reflection from trajectory
ace.reflect(trajectory_id="traj-abc123def456")
# Returns: {
#   "error_identification": "...",
#   "root_cause_analysis": "...",
#   "candidate_bullets": [...]
# }

# 3. Commit the delta to update playbook
ace.commit(delta={"ops": [...]})
```

### Python API

```python
from ace.generator import TrajectoryDoc
from ace.reflector import Reflector

# Create a TrajectoryDoc
doc = TrajectoryDoc(
    query="Implement caching for API responses",
    retrieved_bullet_ids=["strat-00042"],
    code_diff="...",
    test_output="All 15 tests passed",
    env_meta={"final_status": "success"},
)

# Reflect directly on the doc
reflector = Reflector()
reflection = reflector.reflect(doc)
```

## MCP Quickstart

ACE provides an MCP (Model Context Protocol) server for integrating with Claude and other MCP-compatible clients.

### Running the MCP Server

```bash
# Start the MCP server (stdio transport by default)
python -m ace_mcp_server

# Or use the Makefile
make run-mcp
```

### Claude Desktop Configuration

Add ACE to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["-m", "ace_mcp_server"],
      "cwd": "/path/to/ACE",
      "env": {
        "ACE_DB_URL": "sqlite:///ace.db"
      }
    }
  }
}
```

### Using with uv (recommended)

If you're using `uv` for Python package management:

```json
{
  "mcpServers": {
    "ace": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ACE", "python", "-m", "ace_mcp_server"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `ace_retrieve` | Retrieve relevant playbook bullets for a query |
| `ace_reflect` | Generate a reflection from task execution data |
| `ace_curate` | Convert a reflection into delta operations |
| `ace_commit` | Apply delta operations to the playbook |
| `ace_refine` | Deduplicate and consolidate bullets |
| `ace_stats` | Get playbook statistics |
| `ace_record_trajectory` | Record a task execution trajectory |
| `ace_pipeline` | Run the full ACE pipeline |

### MCP Resource

The server exposes the full playbook via the `ace://playbook.json` resource, which returns the same structure as `ace playbook dump`.

### Example Usage in Claude

Once configured, you can interact with ACE directly in Claude:

```
User: What strategies do you have for retrieval?

Claude: [Uses ace_retrieve tool with query "retrieval strategies"]
Found 3 relevant bullets:
- strat-001: Prefer hybrid retrieval: BM25 + embedding for better recall
- ...
```

## Configuration

See the full configuration guide with defaults, env var mappings, validation rules, and examples:
- [docs/configuration.md](docs/configuration.md)

---

If you need a different config file per environment, you can also load a specific path programmatically:
```python
from pathlib import Path
from ace.core.config import load_config

cfg = load_config(Path("configs/prod.toml"))
```
