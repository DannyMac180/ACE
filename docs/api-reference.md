# ACE API Reference

This document provides comprehensive API documentation for all core modules in the ACE (Agentic Context Engineering) system.

## Table of Contents

- [Core Modules](#core-modules)
  - [Schema](#schema)
  - [Store](#store)
  - [Retriever](#retriever)
  - [Merge](#merge)
  - [PlaybookManager](#playbookmanager)
  - [Config](#config)
- [Reflector](#reflector)
- [Curator](#curator)
- [Refine](#refine)
- [Generator](#generator)
- [Serve (Online Adaptation)](#serve-online-adaptation)
- [LLM Clients](#llm-clients)
- [Evaluation](#evaluation)

---

## Core Modules

### Schema

**Module:** `ace.core.schema`

Core data models for the ACE playbook system using Pydantic.

#### Types

```python
Section = Literal[
    "strategies_and_hard_rules",
    "code_snippets_and_templates",
    "troubleshooting_and_pitfalls",
    "domain_facts_and_references",
]
```

**Section Descriptions:**
- `strategies_and_hard_rules`: High-level tactics, policies, and invariants
- `code_snippets_and_templates`: Reusable code patterns and templates
- `troubleshooting_and_pitfalls`: Common errors and how to avoid them
- `domain_facts_and_references`: Domain knowledge and reference information

#### Classes

##### `Bullet`

Represents a single reusable piece of knowledge in the playbook.

```python
class Bullet(BaseModel):
    id: str                      # Unique stable identifier (e.g., "strat-00091")
    section: Section             # Category of the bullet
    content: str                 # Short, reusable, domain-rich content
    tags: list[str] = Field(default_factory=list)  # Tags for retrieval
    helpful: int = 0             # Count of times marked helpful
    harmful: int = 0             # Count of times marked harmful
    last_used: datetime | None   # Last time this bullet was retrieved
    added_at: datetime           # When this bullet was created
```

##### `Reflection`

Simple reflection model (distinct from `ace.reflector.schema.Reflection`).

```python
class Reflection(BaseModel):
    summary: str
    critique: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

##### `DeltaBullet`

Bullet data for delta operations (used within `DeltaOp.new_bullet`).

```python
class DeltaBullet(BaseModel):
    section: Section
    content: str
    tags: list[str] = Field(default_factory=list)
    id: str | None = None        # Optional: for idempotent replay
```

##### `Playbook`

Container for the versioned collection of bullets.

```python
class Playbook(BaseModel):
    version: int                 # Playbook version number
    bullets: list[Bullet] = Field(default_factory=list)
```

##### `DeltaOp`

Represents a single operation to modify the playbook.

```python
class DeltaOp(BaseModel):
    op: str                      # Operation type: ADD, PATCH, INCR_HELPFUL, INCR_HARMFUL, DEPRECATE
    target_id: str | None        # ID of bullet to modify (for PATCH, INCR_*, DEPRECATE)
    new_bullet: dict | None      # New bullet data (for ADD)
    patch: str | None            # New content (for PATCH)
```

##### `Delta` (Pydantic)

Collection of operations to apply to a playbook. This is the **Pydantic model** used by `ace.curator.curate()`.

> **Note:** There is also `ace.core.merge.Delta`, a plain Python class used by `apply_delta()`. See [Merge](#merge) for details on converting between them.

```python
class Delta(BaseModel):
    ops: list[DeltaOp] = Field(default_factory=list)
```

##### `RefineOp`

Operations produced by the refine pipeline.

```python
class RefineOp(BaseModel):
    op: str                      # Operation type: MERGE, ARCHIVE
    survivor_id: str | None      # ID of bullet to keep (for MERGE)
    target_ids: list[str] = Field(default_factory=list)
```

##### `RefineResult`

Result summary from the refinement pipeline.

```python
class RefineResult(BaseModel):
    merged: int                  # Number of bullets merged
    archived: int                # Number of bullets archived
    ops: list[RefineOp] = Field(default_factory=list)
```

---

### Store

**Module:** `ace.core.storage.store_adapter`

Unified storage interface for ACE playbook data, backed by SQLite and FAISS.

#### `Store`

Main storage class providing persistence and retrieval.

```python
class Store:
    def __init__(self, db_path: str = "ace.db") -> None
```

**Parameters:**
- `db_path`: Path to SQLite database file (default: `"ace.db"`)

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `save_bullet(bullet)` | Save or update a bullet | `None` |
| `get_bullet(bullet_id)` | Retrieve a single bullet by ID | `Bullet \| None` |
| `get_bullets()` | Retrieve all bullets | `list[Bullet]` |
| `get_all_bullets()` | Alias for `get_bullets()` | `list[Bullet]` |
| `get_version()` | Get current playbook version | `int` |
| `set_version(version)` | Set playbook version | `None` |
| `load_playbook()` | Load entire playbook from database | `Playbook` |
| `load_playbook_data(playbook)` | Import playbook data (replaces existing) | `None` |
| `close()` | Close connections and save indices | `None` |

**Example:**

```python
from ace.core.storage.store_adapter import Store

store = Store("my_playbook.db")
playbook = store.load_playbook()
print(f"Playbook v{playbook.version} with {len(playbook.bullets)} bullets")
store.close()
```

---

### Retriever

**Module:** `ace.core.retrieve`

Hybrid retrieval system combining BM25 full-text search with vector similarity.

#### `Retriever`

```python
class Retriever:
    def __init__(self, store: Store) -> None
```

**Parameters:**
- `store`: A `Store` instance for accessing bullets

**Methods:**

##### `retrieve(query, top_k=24) -> list[Bullet]`

Perform hybrid retrieval to find relevant bullets.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Maximum number of bullets to return (default: 24)

**Returns:** List of `Bullet` objects ranked by relevance

**Algorithm:**
1. Fetch 2×top_k candidates from FTS (BM25-like)
2. Fetch 2×top_k candidates from vector search
3. Combine unique candidates
4. Rerank by lexical overlap with query terms
5. Return top_k results

**Example:**

```python
from ace.core.storage.store_adapter import Store
from ace.core.retrieve import Retriever

store = Store()
retriever = Retriever(store)
bullets = retriever.retrieve("how to handle database errors", top_k=10)
for bullet in bullets:
    print(f"[{bullet.section}] {bullet.content}")
```

---

### Merge

**Module:** `ace.core.merge`

Deterministic delta application to modify playbook state.

#### `Delta`

```python
class Delta:
    def __init__(self, ops: list[DeltaOp]) -> None
    
    @classmethod
    def from_dict(cls, d: dict) -> Delta
```

#### `DeltaOp`

```python
class DeltaOp:
    def __init__(self, op_type: str, **kwargs) -> None
    
    @classmethod
    def from_dict(cls, d: dict) -> DeltaOp
```

#### `apply_delta(playbook, delta, store) -> Playbook`

Apply delta operations to a playbook.

**Parameters:**
- `playbook`: Current `Playbook` state
- `delta`: `Delta` object containing operations
- `store`: `Store` instance for persistence

**Returns:** New `Playbook` with incremented version

**Supported Operations:**
- `ADD`: Add a new bullet
- `PATCH`: Update bullet content
- `INCR_HELPFUL`: Increment helpful counter
- `INCR_HARMFUL`: Increment harmful counter
- `DEPRECATE`: Increments harmful counter (does **not** remove or flag the bullet; true deprecation is not yet implemented)

> **Note on Delta classes:** `ace.core.merge.Delta` is a plain Python class with `op_type` attribute, while `ace.core.schema.Delta` is a Pydantic model with `op` attribute. Use `Delta.from_dict(pydantic_delta.model_dump())` to convert from Pydantic to the merge class.

**Example:**

```python
from ace.core.merge import Delta, DeltaOp, apply_delta

delta = Delta.from_dict({
    "ops": [
        {"op": "ADD", "new_bullet": {
            "id": "strat-001",
            "section": "strategies",
            "content": "Always validate input before processing",
            "tags": ["topic:validation"]
        }},
        {"op": "INCR_HELPFUL", "target_id": "strat-002"}
    ]
})

new_playbook = apply_delta(playbook, delta, store)
```

---

### PlaybookManager

**Module:** `ace.core.manager`

In-memory playbook management with delta application.

```python
class PlaybookManager:
    def __init__(self) -> None
```

**Methods:**

| Method | Description |
|--------|-------------|
| `load_playbook(path)` | Load playbook from disk (not implemented) |
| `save_playbook(path)` | Save playbook to disk (not implemented) |
| `apply_delta(delta)` | Apply a `DeltaOp` to the playbook |

**Delta Operations:**
- `ADD`: Creates new bullet with auto-generated ID if not provided
- `PATCH`: Updates content of existing bullet
- `INCR_HELPFUL`: Increments helpful counter and updates `last_used`
- `INCR_HARMFUL`: Increments harmful counter
- `DEPRECATE`: Removes bullet from playbook

---

### Config

**Module:** `ace.core.config`

Configuration management with TOML file loading and environment variable overrides.

#### Configuration Dataclasses

```python
@dataclass
class ACEConfig:
    database: DatabaseConfig     # Database settings
    embeddings: EmbeddingsConfig # Embedding model settings
    retrieval: RetrievalConfig   # Retrieval parameters
    refine: RefineConfig         # Refinement thresholds
    logging: LoggingConfig       # Logging configuration
    mcp: MCPConfig               # MCP server settings
    llm: LLMConfig               # LLM provider settings
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_DB_URL` | `sqlite:///ace.db` | Database URL |
| `ACE_EMBEDDINGS` | `bge-small` | Embedding model name |
| `ACE_RETRIEVAL_TOPK` | `24` | Default retrieval limit |
| `ACE_RETRIEVAL_LEXICAL_WEIGHT` | `0.5` | Lexical weight for hybrid search (0.0-1.0) |
| `ACE_REFINE_THRESHOLD` | `0.90` | Cosine threshold for dedup |
| `ACE_REFINE_MINHASH_THRESHOLD` | `0.85` | MinHash Jaccard threshold for near-duplicate detection |
| `ACE_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `ACE_LOG_FORMAT` | `json` | Log format (`json` or `text`) |
| `MCP_TRANSPORT` | `stdio` | MCP transport (stdio/http/sse) |
| `MCP_PORT` | `8000` | HTTP port (only if transport=http) |
| `ACE_LLM_PROVIDER` | `openai` | LLM provider (openai, anthropic, etc.) |
| `ACE_LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `ACE_LLM_TEMPERATURE` | `0.0` | LLM temperature for reflection/curation |
| `ACE_LLM_MAX_TOKENS` | `2000` | Max tokens for LLM responses |

#### Functions

##### `load_config(config_path=None) -> ACEConfig`

Load configuration from TOML file with environment overrides.

**Parameters:**
- `config_path`: Optional path to TOML file (defaults to `configs/default.toml`)

**Returns:** Validated `ACEConfig` instance

##### `get_config() -> ACEConfig`

Get global config singleton, loading if necessary.

---

## Reflector

**Module:** `ace.reflector`

Generates structured reflections from task execution outcomes.

### Schema

**Module:** `ace.reflector.schema`

```python
@dataclass
class BulletTag:
    id: str                                    # Bullet ID
    tag: Literal["helpful", "harmful"]         # Feedback tag

@dataclass
class CandidateBullet:
    section: Section                           # Target section
    content: str                               # Bullet content
    tags: list[str] = field(default_factory=list)  # Metadata tags

@dataclass
class Reflection:
    error_identification: str | None           # What went wrong
    root_cause_analysis: str | None            # Why it went wrong
    correct_approach: str | None               # How to fix it
    key_insight: str | None                    # Reusable learning
    bullet_tags: list[BulletTag] = field(default_factory=list)
    candidate_bullets: list[CandidateBullet] = field(default_factory=list)
```

### Reflector Class

**Module:** `ace.reflector.reflector`

```python
class Reflector:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        temperature: float = 0.3
    ) -> None
```

**Parameters:**
- `model`: OpenAI model to use
- `max_retries`: Maximum retry attempts on parse errors
- `temperature`: LLM temperature (lower = more deterministic)

**Environment Variables:**
- `OPENAI_API_KEY`: **Required.** The Reflector uses the OpenAI SDK directly and will fail on instantiation if this variable is not set.

#### `reflect(...) -> Reflection`

Generate a reflection from task execution data.

**Parameters:**
- `query` (str): The task or query that was executed
- `retrieved_bullet_ids` (list[str]): IDs of bullets used
- `code_diff` (str): Code changes made
- `test_output` (str): Test results
- `logs` (str): Execution logs
- `env_meta` (dict | None): Additional metadata

**Returns:** `Reflection` object

**Raises:** `ReflectionParseError` if parsing fails after max_retries

**Example:**

```python
from ace.reflector import Reflector

reflector = Reflector()
reflection = reflector.reflect(
    query="Fix authentication bug",
    retrieved_bullet_ids=["strat-001", "tmpl-002"],
    test_output="FAILED: test_auth_flow",
    logs="AuthError: Invalid token format"
)
print(reflection.root_cause_analysis)
```

---

## Curator

**Module:** `ace.curator`

Converts reflections into playbook delta operations.

### `curate(reflection) -> Delta`

Convert a `Reflection` object into a `Delta` object.

**Parameters:**
- `reflection`: A `Reflection` object from the Reflector

**Returns:** `Delta` object containing operations to update the Playbook

**Logic:**
- For each `BulletTag` with `tag="helpful"` → `INCR_HELPFUL` operation
- For each `BulletTag` with `tag="harmful"` → `INCR_HARMFUL` operation
- For each `CandidateBullet` → `ADD` operation

**Example:**

```python
from ace.curator import curate
from ace.reflector.schema import Reflection, CandidateBullet

reflection = Reflection(
    key_insight="Input validation prevents auth errors",
    candidate_bullets=[
        CandidateBullet(
            section="strategies",
            content="Validate JWT format before parsing",
            tags=["topic:auth", "topic:validation"]
        )
    ]
)

delta = curate(reflection)
print(f"Generated {len(delta.ops)} operations")
```

---

## Refine

**Module:** `ace.refine`

Deduplication, consolidation, and archival pipeline for playbook maintenance.

### RefineRunner Class

**Module:** `ace.refine.runner`

```python
class RefineRunner:
    def __init__(
        self,
        playbook: Playbook,
        threshold: float = 0.90,
        archive_ratio: float = 0.75
    ) -> None
```

**Parameters:**
- `playbook`: Current playbook to refine against
- `threshold`: Cosine similarity threshold for deduplication (default: 0.90)
- `archive_ratio`: Harmful ratio threshold for archival (default: 0.75)

#### `run(reflection) -> RefineResult`

Execute the refinement pipeline.

**Pipeline Stages:**
1. **Curator**: Convert reflection to delta operations (produces `ADD`/`INCR_*` ops)
2. **Deduplication**: Find near-duplicate bullets using embedding cosine + MinHash Jaccard
3. **Consolidation**: Transfer counters from merged bullets to survivors (modifies in-memory playbook only)
4. **Archival**: Remove low-utility bullets from in-memory playbook (harmful_ratio > archive_ratio)

> **Implementation Note:** The current implementation operates on the **in-memory playbook** passed at construction. Candidate bullets from `ADD` operations are checked for duplicates but are **not** actually added to the playbook or persisted to the store. To commit new bullets, you must separately call `apply_delta()` with the curator's output. The `_consolidate()` and `_archive()` methods modify `self.playbook.bullets` directly but do not persist changes.

**Parameters:**
- `reflection`: The `Reflection` object to process

**Returns:** `RefineResult` with merge/archive counts and operations

#### `deduplicate(delta) -> list[RefineOp]`

Find near-duplicate bullets using dual similarity metrics.

**Duplicate Detection:**
- Cosine similarity > threshold (default 0.90), OR
- MinHash Jaccard similarity > 0.85

### Module Function

```python
def refine(
    reflection: Reflection,
    playbook: Playbook,
    threshold: float = 0.90,
    archive_ratio: float = 0.75
) -> RefineResult
```

Main entry point for the refinement pipeline.

> **Current Limitations:**
> - Does not persist changes to the `Store`; operates on in-memory `Playbook` only.
> - Candidate bullets are deduplicated but not added to the playbook.
> - Use `apply_delta()` separately to commit new bullets after dedup filtering.

---

## Generator

**Module:** `ace.generator`

Task execution with trajectory capture using ReAct-style loops.

### Schema

**Module:** `ace.generator.schemas`

```python
class Step(BaseModel):
    action: str          # Action taken
    observation: str     # Result from action
    thought: str         # Reasoning for this step
    timestamp: datetime  # When step occurred

class Trajectory(BaseModel):
    steps: list[Step]                           # Sequence of steps
    initial_goal: str                           # Original task
    final_status: Literal["success", "failure", "partial"]
    total_steps: int
    started_at: datetime
    completed_at: datetime | None
```

### Generator Class

**Module:** `ace.generator.generator`

```python
class Generator:
    def __init__(
        self,
        max_steps: int = 10,
        tool_executor: Callable[[str], str] | None = None
    ) -> None
```

**Parameters:**
- `max_steps`: Maximum steps before stopping (default: 10)
- `tool_executor`: Optional callable to execute actions (defaults to mock executor)

#### `run(goal) -> Trajectory`

Execute a task with full trajectory capture.

**ReAct Loop:**
1. **Reason**: Generate thought about current state
2. **Act**: Decide on action based on reasoning
3. **Observe**: Execute action and capture result
4. **Log**: Record complete step
5. **Repeat**: Continue until goal achieved or max steps

**Example:**

```python
from ace.generator import Generator

generator = Generator(max_steps=5)
trajectory = generator.run("Analyze the authentication module")

print(f"Completed in {trajectory.total_steps} steps")
for step in trajectory.steps:
    print(f"  Action: {step.action}")
    print(f"  Result: {step.observation}")
```

---

## Serve (Online Adaptation)

**Module:** `ace.serve`

HTTP server for test-time sequential adaptation using execution feedback (no ground-truth labels).

Implements the ACE paper's online mode where the reflector relies on execution feedback (test outputs, logs, errors) to derive insights. Supports warm-start by preloading a playbook before accepting queries.

### Schema

**Module:** `ace.serve.schema`

#### `WarmupSource`

Source of playbook data at startup.

```python
class WarmupSource(str, Enum):
    NONE = "none"        # Cold start, no preloaded playbook
    FILE = "file"        # Preloaded from a JSON file (--warmup option)
    DATABASE = "database" # Started with existing database playbook
```

#### `OnlineStats`

Statistics for online serving session.

```python
class OnlineStats(BaseModel):
    session_id: str
    started_at: datetime
    requests_processed: int = 0
    total_ops_applied: int = 0
    helpful_feedback_count: int = 0
    harmful_feedback_count: int = 0
    avg_adaptation_ms: float = 0.0
    warmup_source: WarmupSource = WarmupSource.NONE
    warmup_bullets_loaded: int = 0
    warmup_playbook_version: int = 0
```

### OnlineServer Class

**Module:** `ace.serve.runner`

```python
class OnlineServer:
    def __init__(
        self,
        store: Store | None = None,
        reflector: Reflector | None = None,
        retriever: Retriever | None = None,
        auto_adapt: bool = True,
        warmup_path: str | Path | None = None,
    ) -> None
```

**Parameters:**
- `store`: Playbook store (loads from config if None)
- `reflector`: Reflector instance (creates default if None)
- `retriever`: Retriever instance (creates default if None)
- `auto_adapt`: Whether to auto-adapt on each feedback (default True)
- `warmup_path`: Path to a playbook JSON file for warm-start

#### Warm-Start Behavior

The server tracks warmup source for metrics:
- If `warmup_path` is provided, loads playbook from file (`WarmupSource.FILE`)
- If database has existing bullets, uses those (`WarmupSource.DATABASE`)
- Otherwise, starts cold (`WarmupSource.NONE`)

Paper Table 3 shows 'ReAct + ACE + offline warmup' beats cold-start online adaptation.

### Functions

#### `create_app(auto_adapt, store, warmup_path) -> FastAPI`

Create FastAPI application for online serving.

```python
def create_app(
    auto_adapt: bool = True,
    store: Store | None = None,
    warmup_path: str | Path | None = None,
) -> FastAPI
```

#### `run_server(host, port, auto_adapt, reload, warmup_path) -> None`

Run the online server with uvicorn.

```python
def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    auto_adapt: bool = True,
    reload: bool = False,
    warmup_path: str | Path | None = None,
) -> None
```

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns `{"status": "ok", "mode": "online"}` |
| `/stats` | GET | Session statistics including warmup info |
| `/playbook/version` | GET | Current playbook version |
| `/retrieve` | POST | Retrieve bullets for a query |
| `/feedback` | POST | Process execution feedback and adapt playbook |

### CLI Usage

```bash
# Cold start
ace serve --host 127.0.0.1 --port 8000

# Warm start with pre-loaded playbook
ace serve --warmup playbook.json

# Retrieve-only mode (no automatic adaptation)
ace serve --no-adapt

# Development mode with hot reload
ace serve --reload
```

---

## LLM Clients

**Module:** `ace.llm`

Abstraction layer for LLM providers.

### Schema

**Module:** `ace.llm.schemas`

```python
class Message(BaseModel):
    role: str      # 'user', 'assistant', or 'system'
    content: str   # Message content

class CompletionResponse(BaseModel):
    text: str      # Generated response
```

### LLMClient (Abstract)

**Module:** `ace.llm.client`

```python
class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        """Generate completion from messages."""
```

### MockLLMClient

Mock client for testing.

```python
class MockLLMClient(LLMClient):
    def __init__(self, response_prefix: str = "Mock response:") -> None
```

### OpenRouterClient

Production client for OpenRouter API.

```python
class OpenRouterClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,           # Defaults to OPENROUTER_API_KEY env
        model: str = "openai/gpt-5",
        site_url: str | None = None,
        app_name: str | None = None,
        default_max_tokens: int | None = None,
        default_temperature: float = 0.7,
        reasoning_effort: str | None = "medium"
    ) -> None
```

**Environment Variables:**
- `OPENROUTER_API_KEY`: Required API key

**Example:**

```python
from ace.llm import OpenRouterClient, Message

client = OpenRouterClient(model="openai/gpt-4o")
response = client.complete([
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Explain ACE in one sentence.")
])
print(response.text)
```

---

## Evaluation

**Module:** `ace.eval`

Benchmarking and evaluation harness for ACE components.

### Metrics

**Module:** `ace.eval.metrics`

#### `mean_reciprocal_rank(ranked_results, relevant_ids) -> float`

Calculate MRR for retrieval results.

**Parameters:**
- `ranked_results` (list[list[str]]): Ranked result lists per query
- `relevant_ids` (list[set[str]]): Sets of relevant IDs per query

**Returns:** MRR score between 0 and 1

#### `recall_at_k(ranked_results, relevant_ids, k) -> float`

Calculate Recall@k for retrieval results.

#### `precision_at_k(ranked_results, relevant_ids, k) -> float`

Calculate Precision@k for retrieval results.

**Example:**

```python
from ace.eval.metrics import mean_reciprocal_rank, recall_at_k

ranked = [["doc1", "doc2", "doc3"], ["doc4", "doc5"]]
relevant = [{"doc2"}, {"doc4"}]

mrr = mean_reciprocal_rank(ranked, relevant)
recall = recall_at_k(ranked, relevant, k=2)
print(f"MRR: {mrr:.3f}, Recall@2: {recall:.3f}")
```

### EvalRunner

**Module:** `ace.eval.harness`

```python
class EvalRunner:
    def __init__(self) -> None
```

#### `run_suite(suite="all", baseline_path=None, fail_on_regression=False) -> dict`

Run evaluation benchmark suite.

**Parameters:**
- `suite` (str): Which suite to run ('retrieval', 'reflection', 'e2e', 'all'). Default: `"all"`
- `baseline_path` (str | None): Path to baseline JSON for regression detection
- `fail_on_regression` (bool): Raise error on regression if True

**Returns:** Dictionary with evaluation results

> **Implementation Status:**
> - `retrieval`: **Implemented.** Loads fixtures from `ace/eval/fixtures/retrieval_cases.json` and runs retrieval benchmarks.
> - `reflection`: **Not implemented.** Returns `{"status": "not_implemented"}`.
> - `e2e`: **Not implemented.** Returns `{"status": "not_implemented"}`.
> - `all`: Runs all suites; unimplemented suites return placeholder status.

#### `format_markdown(results) -> str`

Format results as markdown report.

#### `print_results(results) -> None`

Print results in human-readable format.

**Example:**

```python
from ace.eval.harness import EvalRunner

runner = EvalRunner()
results = runner.run_suite(
    suite="retrieval",
    baseline_path="baselines/v1.json",
    fail_on_regression=True
)
print(runner.format_markdown(results))
```

---

## Quick Start

```python
from ace.core.storage.store_adapter import Store
from ace.core.retrieve import Retriever
from ace.reflector import Reflector
from ace.curator import curate
from ace.core.merge import apply_delta
from ace.core.merge import Delta as MergeDelta  # Plain class for apply_delta

# Initialize storage and retrieval
store = Store()
retriever = Retriever(store)

# Retrieve relevant bullets for a task
bullets = retriever.retrieve("handle database connection errors")
bullet_ids = [b.id for b in bullets]

# After task execution, reflect on the outcome
# Note: Requires OPENAI_API_KEY environment variable
reflector = Reflector()
reflection = reflector.reflect(
    query="handle database connection errors",
    retrieved_bullet_ids=bullet_ids,
    test_output="All tests passed",
    logs=""
)

# Convert reflection to playbook updates
# curate() returns ace.core.schema.Delta (Pydantic model)
pydantic_delta = curate(reflection)

# Apply changes to playbook
# apply_delta() expects ace.core.merge.Delta (plain class)
# Convert using Delta.from_dict()
playbook = store.load_playbook()
merge_delta = MergeDelta.from_dict(pydantic_delta.model_dump())
new_playbook = apply_delta(playbook, merge_delta, store)

print(f"Playbook updated to v{new_playbook.version}")
store.close()
```

> **Note on Delta classes:** The codebase has two `Delta` classes:
> - `ace.core.schema.Delta` — Pydantic model returned by `curate()`
> - `ace.core.merge.Delta` — Plain Python class expected by `apply_delta()`
>
> Use `.model_dump()` and `Delta.from_dict()` to convert between them as shown above.
