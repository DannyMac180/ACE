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
