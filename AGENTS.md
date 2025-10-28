This document is the single source of truth for how our coding agent (Amp) should work on the ACE (Agentic Context Engineering) project. It defines roles, interfaces, prompts, file layout, commands, CI rules, guardrails, and success metrics. Hand this file to Amp at project start and keep it updated.

For project issue tracking use the bd tool.

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
bd ready --json
```

**Create new issues:**
```bash
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**
```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**
```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### MCP Server (Recommended)

If using Claude or MCP-compatible clients, install the beads MCP server:

```bash
pip install beads-mcp
```

Add to MCP config (e.g., `~/.config/claude/config.json`):
```json
{
  "beads": {
    "command": "beads-mcp",
    "args": []
  }
}
```

Then use `mcp__beads__*` functions instead of CLI commands.

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and QUICKSTART.md.

0) Project TL;DR

Goal: Implement ACE as a small Python library + optional MCP server so any LLM client can evolve a Playbook (a set of itemized “bullets”) via a Generator → Reflector → Curator → Merge loop.

The paper that this project is based on is available here: /Users/danielmcateer/Desktop/dev/ACE/docs/Agentic Context Engineering.pdf

If asked to validate against the paper, use the paper as the source of truth.

Key idea: Never rewrite the whole prompt. We add/patch/deprecate small bullets (deltas), and periodically grow-and-refine (dedup/merge/retire) to prevent context bloat or collapse.

Primary deliverables:

ace/ Python package (playbook store, retrieval, merge, reflector+curator prompts/parsers).

ace_mcp_server/ exposing tools (retrieve, reflect, curate, commit, refine, stats) over MCP.

Tiny eval harness (unit tests + smoke benchmarks).

Dev ergonomics (scripts, Makefile, CI, telemetry).

Amp should use the workflows below to implement features, add tests, and harden the system incrementally.

1) Repo Layout & Conventions
.
├─ ace/                         # Python library
│  ├─ core/                     # schemas, store, retrieval, merge/apply_delta
│  │  ├─ schema.py              # Bullet, Playbook
│  │  ├─ store.py               # SQLite/pgvector adapters
│  │  ├─ retrieve.py            # hybrid lexical+vector retrieval
│  │  └─ merge.py               # deterministic delta application
│  ├─ generator/                # task runner (ReAct-style optional), trajectory log
│  ├─ reflector/                # prompts + JSON parser (Reflection)
│  ├─ curator/                  # prompts + JSON parser (Delta)
│  └─ refine/                   # dedup, consolidation, archiving
├─ ace_mcp_server/              # MCP server wrapper (FastMCP or equivalent)
│  ├─ server.py                 # tools/resources exposed
│  └─ __main__.py               # entrypoint
├─ eval/                        # smoke tests + mini-bench harness
│  ├─ data/                     # fixtures
│  └─ test_*.py                 # unit/integration tests
├─ scripts/                     # CLI utilities (seed, run loop, bench)
├─ configs/                     # model & retrieval configs
├─ tests/                       # library unit tests
├─ pyproject.toml               # poetry/pip build
├─ Makefile                     # dev commands
├─ .github/workflows/ci.yml     # CI (lint, typecheck, tests)
└─ AGENTS.md                    # THIS FILE


Language/Tooling defaults

Python ≥3.11, ruff for lint, mypy for typing, pytest for tests.

Vector store: faiss-cpu by default; optional pgvector.

MCP: Python FastMCP (or equivalent) over stdio.

2) ACE Core Schemas (Contracts)
2.1 Bullet & Playbook
# ace/core/schema.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, List

Section = Literal["strategies", "templates", "troubleshooting", "code_snippets", "facts"]

@dataclass
class Bullet:
    id: str                      # unique, stable (e.g., "strat-00091")
    section: Section
    content: str                 # short, reusable, domain-rich
    tags: List[str] = field(default_factory=list)   # e.g., ["repo:ace","topic:retrieval","db:pg"]
    helpful: int = 0
    harmful: int = 0
    last_used: Optional[datetime] = None
    added_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Playbook:
    version: int
    bullets: List[Bullet]

2.2 Reflection (output of Reflector)
{
  "error_identification": "string|null",
  "root_cause_analysis": "string|null",
  "correct_approach": "string|null",
  "key_insight": "string|null",
  "bullet_tags": [
    {"id": "strat-00091", "tag": "helpful" | "harmful"}
  ],
  "candidate_bullets": [
    {"section": "strategies", "content": "short tactic", "tags": ["topic:...","tool:..."]}
  ]
}

2.3 Delta (input to Curator merge)
{
  "ops": [
    {"op":"ADD","new_bullet":{"section":"strategies","content":"...", "tags":["..."]}},
    {"op":"PATCH","target_id":"strat-00091","patch":"revised bullet content"},
    {"op":"INCR_HELPFUL","target_id":"strat-00091"},
    {"op":"INCR_HARMFUL","target_id":"tmpl-00022"},
    {"op":"DEPRECATE","target_id":"trbl-00007"}
  ]
}


Rule: All merges are deterministic, non-LLM code. LLMs propose; code merges.

3) Retrieval & Update Policy

Retrieval: hybrid lexical + vector with rerank; default top_k=24 bullets. Prefer bullets matching repo:*, topic:*, stack:*.

Update: On each task or failed attempt, run reflect → curate → commit. Only add small, reusable bullets; avoid verbose prose.

Grow & Refine: Run periodically (or when context tokens exceed a threshold):

dedup near duplicates (embedding + minhash),

consolidate (keep the clearest content; transfer counters),

archive low-utility bullets (e.g., harmful/helpful ratio > 2 for 3+ occurrences).

4) MCP Server Tools (if using MCP)

Expose these tool contracts (names are stable):

ace.retrieve(query: str, top_k: int=24) -> Bullet[]

ace.record_trajectory(doc: dict) -> str (id)

ace.reflect(doc: dict) -> Reflection

ace.curate(reflection: Reflection) -> Delta

ace.commit(delta: Delta) -> {"version": int}

ace.refine(threshold: float=0.90) -> {"merged": int, "archived": int}

ace.stats() -> {"num_bullets": int, "helpful_ratio": float, ...}

Resource: ace://playbook.json (entire current playbook)

If MCP is not used, mirror these as REST endpoints.

5) Amp Workflows (Do This)

Amp: treat each workflow as a checklist. If a step fails, capture artifacts (logs, diffs) and run the reflect → curate → commit loop to evolve the playbook.

5.1 Bootstrap the project

Create package skeleton under ace/ and ace_mcp_server/ using the layout above.

Implement ace/core/schema.py, store.py, merge.py, retrieve.py minimal versions.

Add seed bullets (see §7) and scripts/seed.py.

Implement ace_mcp_server/server.py exposing tool stubs.

Add Makefile, pyproject.toml, ruff, mypy, pytest.

Add CI (.github/workflows/ci.yml): lint → typecheck → tests.

5.2 Implement a feature (green-to-green loop)

Plan: create issue; list acceptance criteria + tests.

Retrieve: call ace.retrieve for topic tags (e.g., topic:retrieval, db:faiss).

Code: implement feature in small PRs; write/adjust tests first.

Run tests: on failures, capture logs/output as environment feedback.

Reflect: generate a Reflection JSON proposing bullets from the failure/insight.

Curate → Commit: produce a Delta (ADD/PATCH or counters) and apply.

Refine: if playbook grows > threshold, run ace.refine.

PR: open PR with Playbook delta summary in the description (auto-generated).

5.3 Fix CI failures (doctor mode)

Parse CI logs, cluster signature (Timeout detox iOS-17 60s).

reflect to produce a troubleshooting bullet (short, verifiable).

curate → commit.

Re-run CI; if green, increment helpful on related bullets.

5.4 Migration (pattern)

Create topic:migration:<target> bullets (codemods, known breakages, order).

For each failure, add small bullets (e.g., “replace import X with Y”).

After migration, refine to merge duplicates.

5.5 Incident runbook

Add minimal bullets: detection → hypothesis → command → rollback.

Store source links; prefer commands that are copy-pastable.

Tag with source:incident and service tags.

6) Commands, Scripts, and Makefile

Makefile (baseline)

.PHONY: setup lint type test run-mcp seed refine bench

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev]

lint:
\truff check .

type:
\tmypy ace

test:
\tpytest -q

run-mcp:
\tpython -m ace_mcp_server

seed:
\tpython scripts/seed.py

refine:
\tpython -m ace.refine.run --threshold 0.90

bench:
\tpytest -q eval/test_bench_smoke.py -q


Expected npm/pnpm (if JS fixtures exist)
Not required for core, but if present use pnpm lint:fix, pnpm typecheck, pnpm test.

7) Seed Bullets (give the agent a starting brain)

Add a few high-leverage, short bullets:

Retrieval hygiene

section: strategies
content: "Prefer hybrid retrieval: BM25 + embedding; rerank by lexical overlap with query terms; default top_k=24."
tags: ["topic:retrieval","stack:python"]


Delta discipline

section: strategies
content: "Never rewrite the whole playbook. Only ADD/PATCH/DEPRECATE bullets; run refine for dedup."
tags: ["topic:curation","policy"]


Dedup rule

section: strategies
content: "Consider bullets near-duplicate if cosine>0.90 OR minhash Jaccard>0.85; keep clearer text; transfer counters."
tags: ["topic:refine","retrieval"]


JSON strictness

section: troubleshooting
content: "Reflector/Curator must emit valid JSON without markdown fencing; reject and retry on parse errors."
tags: ["topic:parsing","robustness"]


Test-first

section: templates
content: "Unit test template for merge: apply Delta ops and assert version increment + idempotency."
tags: ["topic:testing"]


MCP tool shape

section: code_snippets
content: "Expose MCP tools 'ace.retrieve|reflect|curate|commit|refine|stats'; resource 'ace://playbook.json'."
tags: ["topic:mcp"]

8) Prompts (Reflector & Curator) — Guidance

Output strict JSON matching §2.2 / §2.3.

Keep insights actionable and reusable. Avoid narrative or chain-of-thought.

Prefer one small bullet over a paragraph of advice.

Always tag bullets (repo/service/topic/tool) to improve retrieval.

Reflector input fields (suggested):

query, retrieved_bullet_ids[], code_diff, test_output, logs, env(meta)


Curator policy:

Map bullet_tags → INCR_HELPFUL/HARMFUL.

For candidate_bullets, run duplicate check; emit ADD or PATCH.

Never emit both ADD and PATCH for the same semantic content.

9) CI / PR Policy

CI stages: lint → type → unit tests → integration (optional) → refine (dry-run)

PR Checklist (auto comment):

 Tests added/updated and green

 Playbook delta posted (ops count, new ids)

 No JSON parser errors in Reflect/Curate

 Refine dry-run shows ≤ 10% near-dups

Branch protections: block merge on failing checklist.

10) Observability (What to Log)

Log as JSON lines (structlog or stdlib):

attempt_id, query, retrieved_ids[], tokens_in/out

trajectory summary (tool calls, errors)

reflection (hash + schema version)

delta ops (ADD/PATCH/DEPRECATE/INCR_*)

playbook_version before/after

timings: adaptation_ms, merge_ms, refine_ms

Redact secrets; do not log raw credentials.

11) Config & Secrets

Use environment variables; default values in configs/default.toml.

Var	Default	Notes
ACE_DB_URL	sqlite:///ace.db	SQLite for local; postgres://… for pgvector
ACE_EMBEDDINGS	bge-small	Any local embedding model works
ACE_RETRIEVAL_TOPK	24	Retrieve bullets per query
ACE_REFINE_THRESHOLD	0.90	Embedding cosine for dedup
ACE_LOG_LEVEL	INFO	DEBUG in local dev
MCP_TRANSPORT	stdio	Or http, sse

Secrets (API keys) are never written to playbook content.

12) Security & Safety Guardrails

No full prompt rewrites. Only deltas.

No secrets in bullets. Replace with placeholders.

No chain-of-thought in stored bullets. Use short tactics.

License compliance: Include licenses for third-party code snippets if any.

PII: Do not store live user data in bullets, only patterns.

13) Success Metrics (Track Weekly)

Dev loop: PR open → CI green (median minutes) ↓

Review cycles: requested changes per PR ↓

CI failure recurrence: identical signature repeats ↓

Adoption: % tasks that retrieve ≥1 helpful bullet ↑

Delta hit-rate: candidate bullets later marked helpful ↑

Cost/latency: tokens & seconds per adaptation ↓

14) Quickstarts for Amp
14.1 Scaffold

Create folders per §1.

Implement schema.py, merge.py with Delta ops and tests.

Add store.py (SQLite + FAISS index build).

Write retrieve.py (BM25 + embeddings + simple rerank).

Add seed bullets (§7) via scripts/seed.py.

Implement MCP server.py with tool stubs calling library.

Wire Makefile, ci.yml.

Run: make setup && make test && make run-mcp.

14.2 Add a feature: “Refine duplicates”

Code ace/refine/runner.py with cosine+minhash dedup + consolidation.

Tests: synthesize near-dup bullets; assert one survivor + counter transfer.

Add bullet about dedup policy if missing.

14.3 Doctor a failing test

Reproduce failure in pytest.

Reflect → Curate → Commit a troubleshooting bullet.

Re-run tests; increment helpful tag.

15) Example PR Template
## What
Implement <feature>. Adds/patches the following ACE components: <files>.

## Playbook Delta
```json
{ "ops": [ {"op":"ADD","new_bullet":{"section":"strategies","content":"...", "tags":["topic:..."]}} ] }

Tests

 Unit tests for merge/refine

 Integration smoke

Notes

Retrieval top_k kept at 24; refine threshold 0.90.


---

## 16) Non-MCP Option (if we skip MCP)

Expose REST endpoints that mirror the MCP tools:

- `POST /retrieve` `{query, top_k}` → `Bullet[]`
- `POST /reflect` `{doc}` → `Reflection`
- `POST /curate` `{reflection}` → `Delta`
- `POST /commit` `{delta}` → `{"version":int}`
- `POST /refine` `{threshold}` → stats
- `GET /playbook` → full playbook JSON

---

## 17) Known Gotchas (Treat as tests)

- JSON parse errors from LLMs → **retry with schema-guided prompting**.  
- Duplicate bullets creep in → refine weekly or after > N deltas.  
- Bullets get too generic → reviewers mark **harmful**; curator deprecates.  
- Retrieval starves long but relevant bullets → increase lexical weight; rerank.

---

## 18) Contact Points

- **Owner:** Dan (Field Engineer)  
- **Agent:** Amp (coding agent)  
- **Decision log:** in PR descriptions under “Playbook Delta”  
- **Status board:** GitHub Projects (Backlog / In-Progress / Review / Done)

---

## 19) Project Documentation

- Project documentation is located in the `docs` directory.
- All architecture diagrams are located in the `docs/arch-diagrams` directory.

**Final note for Amp:** Always prefer *small, verifiable improvements* (tests + short bullets) over large rewrites. Every task should make the codebase **and the playbook** a little better.
::contentReference[oaicite:0]{index=0}