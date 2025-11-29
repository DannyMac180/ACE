# ACE Configuration

ACE uses a simple, predictable configuration system:
- Base file: TOML at configs/default.toml
- Overrides: environment variables (ACE_ prefix for most keys; MCP_* for MCP transport/port)
- Validation: types and ranges are enforced at load time; invalid values raise ValueError

Configuration is loaded once and cached:
- get_config() returns a global ACEConfig instance (lazy-loaded)
- load_config(custom_path) lets you point to a different TOML file programmatically

Example (Python):
```python
from ace.core.config import get_config, load_config

cfg = get_config()  # loads configs/default.toml + env overrides, validates once

# Or, load a specific file:
# cfg = load_config(Path("configs/prod.toml"))
```

## Default configuration (configs/default.toml)
```toml
# ACE Configuration File
# All values can be overridden by environment variables

[database]
# SQLite for local; postgres://user:pass@host/db for pgvector
url = "sqlite:///ace.db"

[embeddings]
# Embedding model for vector retrieval
model = "bge-small"

[retrieval]
# Number of bullets to retrieve per query
top_k = 24

# Lexical weight for hybrid search (0.0-1.0)
lexical_weight = 0.5

[refine]
# Cosine similarity threshold for deduplication (0.0-1.0)
threshold = 0.90

# Minhash Jaccard threshold for near-duplicate detection
minhash_threshold = 0.85

[logging]
# Log level: DEBUG, INFO, WARNING, ERROR
level = "INFO"

# Log format: json or text
format = "json"

[mcp]
# MCP transport: stdio, http, sse
transport = "stdio"

# HTTP port (only if transport=http)
port = 8000

[llm]
# Default LLM provider (openai, anthropic, etc.)
provider = "openai"

# Model name
model = "gpt-4o-mini"

# Temperature for reflection/curation
temperature = 0.0

# Max tokens for LLM responses
max_tokens = 2000
```

## Configuration reference

- database
  - url (string) — Database connection string.
    - Default: sqlite:///ace.db
    - Examples: sqlite:///path/to/ace.db, postgres://user:pass@host:5432/dbname

- embeddings
  - model (string) — Embedding model identifier used for vector retrieval.
    - Default: bge-small

- retrieval
  - top_k (int) — Number of items to retrieve per query.
    - Default: 24
  - lexical_weight (float in [0.0, 1.0]) — Weight for lexical (BM25) component in hybrid search.
    - Default: 0.5

- refine
  - threshold (float in [0.0, 1.0]) — Cosine similarity threshold for deduplication.
    - Default: 0.90
  - minhash_threshold (float in [0.0, 1.0]) — Jaccard threshold for near-duplicate detection using MinHash.
    - Default: 0.85

- logging
  - level (string; one of DEBUG, INFO, WARNING, ERROR, CRITICAL) — Minimum log level.
    - Default: INFO
  - format (string; json or text) — Log output format.
    - Default: json

- mcp
  - transport (string; one of stdio, http, sse) — MCP transport.
    - Default: stdio
  - port (int in [1, 65535]) — Port for HTTP transport; used when transport=http.
    - Default: 8000

- llm
  - provider (string) — Default LLM provider identifier (e.g., openai, anthropic).
    - Default: openai
  - model (string) — LLM model name.
    - Default: gpt-4o-mini
  - temperature (float in [0.0, 2.0]) — Decoding temperature.
    - Default: 0.0
  - max_tokens (int >= 1) — Maximum tokens for responses.
    - Default: 2000

## Environment variable overrides

Precedence: environment variables override TOML values.

Note: MCP settings use MCP_TRANSPORT and MCP_PORT (no ACE_ prefix).

- ACE_DB_URL → [database].url
- ACE_EMBEDDINGS → [embeddings].model
- ACE_RETRIEVAL_TOPK → [retrieval].top_k
- ACE_RETRIEVAL_LEXICAL_WEIGHT → [retrieval].lexical_weight
- ACE_REFINE_THRESHOLD → [refine].threshold
- ACE_REFINE_MINHASH_THRESHOLD → [refine].minhash_threshold
- ACE_LOG_LEVEL → [logging].level
- ACE_LOG_FORMAT → [logging].format
- MCP_TRANSPORT → [mcp].transport
- MCP_PORT → [mcp].port
- ACE_LLM_PROVIDER → [llm].provider
- ACE_LLM_MODEL → [llm].model
- ACE_LLM_TEMPERATURE → [llm].temperature
- ACE_LLM_MAX_TOKENS → [llm].max_tokens

Examples (bash):
```bash
# Database: switch to Postgres (e.g., for pgvector-backed retrieval)
export ACE_DB_URL="postgres://user:pass@localhost:5432/ace"

# Retrieval tuning
export ACE_RETRIEVAL_TOPK=50
export ACE_RETRIEVAL_LEXICAL_WEIGHT=0.3

# Deduplication thresholds
export ACE_REFINE_THRESHOLD=0.88
export ACE_REFINE_MINHASH_THRESHOLD=0.80

# Logging
export ACE_LOG_LEVEL=DEBUG
export ACE_LOG_FORMAT=text

# MCP over HTTP
export MCP_TRANSPORT=http
export MCP_PORT=9000

# LLM provider/model
export ACE_LLM_PROVIDER=anthropic
export ACE_LLM_MODEL=claude-3-5-sonnet-20241022
export ACE_LLM_TEMPERATURE=0.1
export ACE_LLM_MAX_TOKENS=4000
```

Windows (PowerShell) equivalents:
```powershell
$env:ACE_DB_URL = "postgres://user:pass@localhost:5432/ace"
$env:ACE_RETRIEVAL_TOPK = "50"
$env:ACE_RETRIEVAL_LEXICAL_WEIGHT = "0.3"
$env:ACE_LOG_LEVEL = "DEBUG"
$env:MCP_TRANSPORT = "http"
$env:MCP_PORT = "9000"
```

## Validation rules

These constraints are enforced during load_config(); invalid values raise ValueError:

- retrieval.top_k >= 1
- retrieval.lexical_weight ∈ [0.0, 1.0]
- refine.threshold ∈ [0.0, 1.0]
- refine.minhash_threshold ∈ [0.0, 1.0]
- logging.level ∈ {DEBUG, INFO, WARNING, ERROR, CRITICAL}
- mcp.transport ∈ {stdio, http, sse}
- mcp.port ∈ [1, 65535]
- llm.temperature ∈ [0.0, 2.0]
- llm.max_tokens >= 1

Type expectations:
- Integers: top_k, mcp.port, llm.max_tokens
- Floats: lexical_weight, refine.threshold, refine.minhash_threshold, llm.temperature
- Strings: all others

## Examples

1) Minimal local setup (defaults)
- No changes required; uses SQLite (sqlite:///ace.db), bge-small embeddings, INFO/json logging, stdio MCP, OpenAI gpt-4o-mini with temperature 0.0.

2) Production-like TOML
```toml
[database]
url = "postgres://user:pass@db.example.com:5432/ace"

[embeddings]
model = "bge-large"

[retrieval]
top_k = 48
lexical_weight = 0.4

[refine]
threshold = 0.88
minhash_threshold = 0.80

[logging]
level = "WARNING"
format = "json"

[mcp]
transport = "http"
port = 9000

[llm]
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
temperature = 0.1
max_tokens = 4000
```

3) All via environment variables (no file edits)
```bash
export ACE_DB_URL="postgres://user:pass@db.example.com:5432/ace"
export ACE_EMBEDDINGS="bge-large"
export ACE_RETRIEVAL_TOPK=48
export ACE_RETRIEVAL_LEXICAL_WEIGHT=0.4
export ACE_REFINE_THRESHOLD=0.88
export ACE_REFINE_MINHASH_THRESHOLD=0.80
export ACE_LOG_LEVEL=WARNING
export ACE_LOG_FORMAT=json
export MCP_TRANSPORT=http
export MCP_PORT=9000
export ACE_LLM_PROVIDER=anthropic
export ACE_LLM_MODEL=claude-3-5-sonnet-20241022
export ACE_LLM_TEMPERATURE=0.1
export ACE_LLM_MAX_TOKENS=4000
```

## Notes and tips

- Default file location: configs/default.toml (relative to project root).
- Programmatic overrides: use load_config(Path(...)) to load a custom file; environment variables still override that file.
- Be sure numeric environment variable values are valid; they are parsed with int()/float() and must fit the validation ranges above.
- logging.level supports CRITICAL even if the comment in default TOML only lists up to ERROR.
- mcp.port is only used when transport=http.

## Training Data Format

ACE supports two training modes based on data availability:

### Labeled Mode (Offline Evaluation)

Use when you have ground truth for computing retrieval metrics (MRR, Precision@k, Recall@k).

**File:** `ace/eval/fixtures/labeled_samples.jsonl`

```jsonl
{"query": "How do I implement hybrid retrieval?", "input": {"task_id": "001"}, "ground_truth": {"relevant_bullet_ids": ["strat-00001", "strat-00003"], "expected_output": "Use BM25 + embeddings"}, "feedback": {"code_diff": "", "test_output": "PASS", "success": true}}
```

**Fields:**
- `query` (required): The task query or prompt
- `input` (optional): Structured input data for the task
- `ground_truth` (required for labeled): Expected output for metric computation
  - `relevant_bullet_ids`: List of bullet IDs that should be retrieved
  - `expected_output`: Expected text/structured output
- `feedback` (optional): Execution feedback signals

### Unlabeled Mode (Online Adaptation)

Use for feedback-only signals when ground truth is unavailable. The system learns from execution outcomes.

**File:** `ace/eval/fixtures/unlabeled_samples.jsonl`

```jsonl
{"query": "Fix the flaky test", "feedback": {"code_diff": "-time.sleep(0.1)\n+await asyncio.sleep(0.1)", "test_output": "FAIL then PASS", "logs": "Retry logic added", "env_meta": {"ci": true}, "success": false}}
```

**Fields:**
- `query` (required): The task query or prompt
- `feedback` (required for unlabeled): Execution feedback matching Reflection inputs
  - `code_diff`: Code changes made
  - `test_output`: Test execution results
  - `logs`: Execution logs
  - `env_meta`: Environment metadata (CI, platform, etc.)
  - `success`: Whether the task succeeded

### TrainSample Schema

```python
class TrainSample(BaseModel):
    query: str                           # Required
    input: dict | None = None            # Optional structured input
    ground_truth: str | dict | None      # Present = labeled mode
    feedback: dict | None                # Execution signals
```

### Usage in TrainingRunner

```python
from ace.train.runner import TrainingRunner

runner = TrainingRunner(retrieval_k=10)

# Load labeled samples for evaluation
labeled = runner.load_train_samples("ace/eval/fixtures/labeled_samples.jsonl")

# Load unlabeled samples for online adaptation
unlabeled = runner.load_train_samples("ace/eval/fixtures/unlabeled_samples.jsonl")

# Compute metrics for labeled samples
metrics = runner.compute_retrieval_metrics(labeled, retrieved_results)
# Returns: {"mrr": 0.85, "recall@10": 0.72, "precision@10": 0.45}
```

### ACE Paper Mapping

| ACE Paper Concept | Training Data Field |
|-------------------|---------------------|
| Offline adaptation | `ground_truth` present |
| Online adaptation | `feedback` only |
| Task execution trajectory | `feedback.code_diff`, `feedback.test_output`, `feedback.logs` |
| Environment feedback | `feedback.env_meta` |
| Success signal | `feedback.success` |
