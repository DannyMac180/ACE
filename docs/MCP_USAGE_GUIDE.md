# ACE MCP Server — Usage Guide

## What this provides

An MCP server exposing ACE tools so LLM clients (e.g., Claude Desktop) can retrieve, reflect on, curate, and refine a Playbook of reusable bullets (strategies, templates, troubleshooting tips, etc.) via a Generator → Reflector → Curator → Merge loop.

Deterministic merges and refinement; LLMs only propose.

## 1) Installation & Setup

### Prerequisites

- Python 3.11+
- For local storage: SQLite (default). For pgvector, provide a postgres URL and ensure your DB has pgvector installed.
- Network access for embedding models if using remote providers.

### Install

From the repo root:
```bash
# Editable install (dev)
pip install -e .

# Regular install
pip install .
```

Or as a package (if published):
```bash
pip install ace
```

### Run as an MCP server

The server exposes stdio MCP by default via FastMCP. Use your MCP client to launch it (recommended). If you want to run it manually for testing:

```bash
python -m ace_mcp_server
```

The process will wait for MCP stdio. Typically you let the client spawn it.

### Verify

If your client can list MCP tools, you should see:
- Tools: `ace_retrieve`, `ace_record_trajectory`, `ace_reflect`, `ace_curate`, `ace_commit`, `ace_refine`, `ace_stats`
- Resource: `ace://playbook.json`

## 2) Available Tools

### Common data shapes

**Bullet** (`ace.core.schema.Bullet`):
```json
{
  "id": "string",
  "section": "strategies" | "templates" | "troubleshooting" | "code_snippets" | "facts",
  "content": "string",
  "tags": ["string"],
  "helpful": 0,
  "harmful": 0,
  "last_used": "ISO datetime | null",
  "added_at": "ISO datetime"
}
```

**Playbook** (`ace.core.schema.Playbook`):
```json
{
  "version": 1,
  "bullets": [/* Bullet objects */]
}
```

**Reflection** (`ace.reflector.schema.Reflection`):
```json
{
  "error_identification": "string|null",
  "root_cause_analysis": "string|null",
  "correct_approach": "string|null",
  "key_insight": "string|null",
  "bullet_tags": [{"id": "string", "tag": "helpful"|"harmful"}],
  "candidate_bullets": [
    {
      "section": "string",
      "content": "string",
      "tags": ["string"]
    }
  ]
}
```

**Delta** (`ace.core.schema.Delta`):
```json
{
  "ops": [
    {"op": "ADD", "new_bullet": {"section": "strategies", "content": "...", "tags": ["..."]}},
    {"op": "PATCH", "target_id": "strat-00091", "patch": "revised bullet content"},
    {"op": "INCR_HELPFUL", "target_id": "strat-00091"},
    {"op": "INCR_HARMFUL", "target_id": "tmpl-00022"},
    {"op": "DEPRECATE", "target_id": "trbl-00007"}
  ]
}
```

### Tool: `ace_retrieve`

**Purpose**: Hybrid retrieval (lexical + vector + rerank) for relevant bullets.

**Parameters**:
- `query` (str): free-text query or task description
- `top_k` (int, default=24): number of bullets to return

**Returns**: List of Bullet objects (as dicts)

**Example**:
```json
{
  "tool_name": "ace_retrieve",
  "arguments": {
    "query": "improve retrieval quality for bm25 + embeddings",
    "top_k": 12
  }
}
```

Response:
```json
[
  {
    "id": "strat-00001",
    "section": "strategies",
    "content": "Prefer hybrid retrieval: BM25 + embedding; rerank by lexical overlap with query terms; default top_k=24.",
    "tags": ["topic:retrieval", "stack:python"],
    "helpful": 3,
    "harmful": 0,
    "last_used": "2025-01-02T12:34:56.000000",
    "added_at": "2025-01-01T00:00:00.000000"
  }
]
```

### Tool: `ace_record_trajectory`

**Purpose**: Persist a generator/agent trajectory for later reflection and stats.

**Parameters**:
- `doc` (dict): must match `ace.generator.schemas.Trajectory`
  - Typical fields: attempt/task id, query, steps/tool calls, outcome, errors, timestamps, metadata

**Returns**: trajectory_id (string)

**Example**:
```json
{
  "tool_name": "ace_record_trajectory",
  "arguments": {
    "doc": {
      "id": "attempt-123",
      "query": "add refine duplicates feature",
      "steps": [
        {
          "tool": "retrieve",
          "args": {"query": "deduplicate bullets"},
          "result_ids": ["strat-00010"]
        }
      ],
      "outcome": "passed",
      "created_at": "2025-01-02T12:00:00Z"
    }
  }
}
```

Response: `"attempt-123"`

### Tool: `ace_reflect`

**Purpose**: Generate a Reflection from an attempt/trajectory, including candidate bullets and tags for helpful/harmful.

**Parameters**:
- `doc` (dict): The input context for reflection (e.g., query, retrieved_bullet_ids, logs, code_diff, test_output, env)

**Returns**: Reflection dict

**Example**:
```json
{
  "tool_name": "ace_reflect",
  "arguments": {
    "doc": {
      "query": "tests failing due to flaky network in integration step",
      "retrieved_bullet_ids": ["trbl-00017", "strat-00033"],
      "logs": "TimeoutError on CI step 'deploy test db'",
      "env": {"ci": "github", "repo": "ace"}
    }
  }
}
```

Response:
```json
{
  "error_identification": "Network flakiness during external dependency setup",
  "root_cause_analysis": "Unreliable network and missing retries/exponential backoff",
  "correct_approach": "Add retries with jitter around external calls; mark non-critical steps as soft-fail",
  "key_insight": "Stabilize CI by isolating network-dependent steps and caching artifacts",
  "bullet_tags": [{"id": "trbl-00017", "tag": "helpful"}],
  "candidate_bullets": [
    {
      "section": "troubleshooting",
      "content": "Wrap network calls in retries with jitter; fail open for non-critical steps in CI.",
      "tags": ["topic:ci", "tool:requests"]
    }
  ]
}
```

### Tool: `ace_curate`

**Purpose**: Apply a Reflection to the current playbook: add/patch bullets and update helpful/harmful counters. This implementation persists changes immediately.

**Parameters**:
- `reflection_data` (dict): Reflection dict, as emitted by ace_reflect

**Returns**: `{"merged": int, "archived": int}`

**Example**:
```json
{
  "tool_name": "ace_curate",
  "arguments": {
    "reflection_data": {
      "bullet_tags": [{"id": "trbl-00017", "tag": "helpful"}],
      "candidate_bullets": [
        {
          "section": "troubleshooting",
          "content": "Wrap network calls in retries with jitter; fail open for non-critical steps in CI.",
          "tags": ["topic:ci", "tool:requests"]
        }
      ]
    }
  }
}
```

Response: `{"merged": 1, "archived": 0}`

**Note**: Unlike a traditional Curator that returns a Delta for a separate commit, this server integrates curation with storage. Use `ace_commit` only when you have an explicit Delta to apply.

### Tool: `ace_commit`

**Purpose**: Deterministically apply a Delta (ADD, PATCH, INCR_*, DEPRECATE) and increment playbook version.

**Parameters**:
- `delta` (dict): Delta dict

**Returns**: `{"version": int}`

**Example**:
```json
{
  "tool_name": "ace_commit",
  "arguments": {
    "delta": {
      "ops": [
        {
          "op": "ADD",
          "new_bullet": {
            "section": "strategies",
            "content": "Never rewrite the whole playbook; only ADD/PATCH/DEPRECATE bullets.",
            "tags": ["topic:curation", "policy"]
          }
        },
        {"op": "INCR_HELPFUL", "target_id": "strat-00001"}
      ]
    }
  }
}
```

Response: `{"version": 42}`

### Tool: `ace_refine`

**Purpose**: Deduplicate near-duplicate bullets, consolidate, and archive low-utility bullets. Persists changes.

**Parameters**:
- `threshold` (float, default=0.90): cosine similarity threshold for dedup

**Returns**: `{"merged": int, "archived": int}`

**Example**:
```json
{
  "tool_name": "ace_refine",
  "arguments": {"threshold": 0.92}
}
```

Response: `{"merged": 3, "archived": 1}`

### Tool: `ace_stats`

**Purpose**: Quick health stats for the playbook.

**Parameters**: none

**Returns**: `{"num_bullets": int, "helpful_ratio": float}`
- `num_bullets`: total bullets
- `helpful_ratio`: average helpful counter across bullets (sum(helpful)/N; 0 if empty)

**Example response**: `{"num_bullets": 87, "helpful_ratio": 1.74}`

## 3) Resource: `ace://playbook.json`

**Purpose**: Fetch the entire current playbook as JSON.

**Returns**:
```json
{
  "version": 42,
  "bullets": [
    {
      "id": "strat-00001",
      "section": "strategies",
      "content": "...",
      "tags": ["topic:..."],
      "helpful": 2,
      "harmful": 0,
      "last_used": "2025-01-02T12:34:56.000000",
      "added_at": "2025-01-01T00:00:00.000000"
    }
  ]
}
```

## 4) Workflow Examples

### A. Basic reflection → curation → commit cycle

1. Retrieve context (optional, for your own LLM prompt):
   - `ace_retrieve` with your task query
2. Reflect on an attempt:
   - `ace_reflect` with inputs: query, retrieved_bullet_ids, logs/test_output, env, etc.
3. Curate:
   - `ace_curate` with the Reflection. This persists adds/patches/counter updates immediately.
4. Commit (only if you generate an explicit Delta yourself):
   - `ace_commit` with a Delta if you have manual edits or want to enforce a deterministic change.

Example:
```
call ace_reflect {doc: {...}} -> reflection
call ace_curate {reflection_data: reflection} -> {"merged": X, "archived": Y}
call ace_stats -> {"num_bullets": N, "helpful_ratio": R}
```

### B. Run refinement to deduplicate bullets

When the playbook starts accumulating near-duplicates or noise:

```
call ace_refine {threshold: 0.90} -> {"merged": 3, "archived": 1}
```

Optional: tighten to 0.92–0.95 if you see too-aggressive merges at 0.90.

### C. Retrieve relevant bullets for a task

Use descriptive queries and tags to bias retrieval:

```
call ace_retrieve {
  query: "repo:ace topic:retrieval bm25 rerank embeddings",
  top_k: 12
}
```

Incorporate returned bullets into your LLM prompt/context.

### D. Record a trajectory for later analysis

After a task run, save the trajectory:

```
call ace_record_trajectory {doc: {...}} -> "attempt-123"
```

Later, you can feed it to `ace_reflect`.

## 5) Integration with MCP clients

### Claude Desktop

Add the server to your MCP configuration so Claude spawns it on demand.

Typical config file: `~/.config/claude/config.json` (Linux), `%APPDATA%\Claude\config.json` (Windows), or per Claude's docs.

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["-m", "ace_mcp_server"],
      "env": {
        "ACE_DB_URL": "sqlite:///ace.db",
        "ACE_EMBEDDINGS": "bge-small",
        "ACE_REFINE_THRESHOLD": "0.90",
        "ACE_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Restart Claude Desktop. You should see tools `ace_*` available and the `ace://playbook.json` resource.

### Other MCP-compatible clients

Configure to launch:
- `command`: `python`
- `args`: `["-m", "ace_mcp_server"]`
- Transport is stdio by default; most clients will use stdio.

### Local testing

You can run `python -m ace_mcp_server` from a terminal and connect with any MCP client that supports manual stdio attach.

## 6) Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_DB_URL` | `sqlite:///ace.db` | Use `postgres://...` for pgvector-backed storage |
| `ACE_EMBEDDINGS` | `bge-small` | Any local/remote embedding model your install supports |
| `ACE_RETRIEVAL_TOPK` | `24` | Default top_k for retrieval |
| `ACE_REFINE_THRESHOLD` | `0.90` | Default cosine threshold used by refine |
| `ACE_LOG_LEVEL` | `INFO` | `DEBUG` for more verbose logs |
| `MCP_TRANSPORT` | `stdio` | Advanced; leave as stdio for Claude |

**Notes**:
- The server's `curate` and `refine` operations persist changes immediately to the Store.
- `commit` is only needed if you explicitly build and want to apply a Delta.
- Retrieval uses hybrid lexical + vector with reranking; add tags like `repo:*`, `topic:*` to improve match quality.

## 7) Practical Tips and Guardrails

- Keep bullets short and reusable; prefer one clear tactic over paragraphs
- Always tag bullets (repo/service/topic/tool) to improve retrieval
- Run `refine` periodically (e.g., weekly or after N curations) to control duplication
- If LLM JSON parsing is flaky, retry with schema-guided prompts; Reflection/Curator must emit strict JSON
- Never include secrets or full chain-of-thought in bullets; use placeholders and short tactics
- If you need deterministic, reviewable changes, use `ace_commit` with a Delta instead of relying on integrated curation

## 8) Troubleshooting

**Tools not visible in client**:
- Check config path and `mcpServers` entry; ensure command is `python` and args is `-m ace_mcp_server`
- Verify Python 3.11+ and that ace is installed in the same environment the client uses

**Reflection/curation errors**:
- Ensure `reflection_data` matches the Reflection schema. Missing fields can be null or omitted; `candidate_bullets` and `bullet_tags` are key to changes

**Retrieval seems off**:
- Use richer tags in queries (e.g., `topic:refine repo:ace`). Tweak `ACE_RETRIEVAL_TOPK` if needed

**Too many duplicates**:
- Run `ace_refine` and consider raising threshold to 0.92–0.95

## 9) Minimal end-to-end example

```python
# Retrieve helpful context
bullets = ace_retrieve(query="reduce CI flakiness due to network")

# Reflect on a failed CI attempt
reflection = ace_reflect(
    doc={
        "query": "CI step failing with TimeoutError",
        "retrieved_bullet_ids": [b["id"] for b in bullets],
        "logs": "...",
        "env": {"ci": "github", "repo": "ace"}
    }
)

# Curate (persist)
result = ace_curate(reflection_data=reflection)
# {"merged": 1, "archived": 0}

# Periodic refine
refine_result = ace_refine(threshold=0.90)
# {"merged": 2, "archived": 1}

# Check stats
stats = ace_stats()
# {"num_bullets": 88, "helpful_ratio": 1.8}
```

That's it—add ACE as an MCP server, call reflect → curate during your dev loop, run refine periodically, and use retrieve to pull the best bullets into your prompts.
