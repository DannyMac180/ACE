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

3) Use in code
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
