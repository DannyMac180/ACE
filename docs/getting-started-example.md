# ACE Getting Started Example

This is the shortest path from curiosity to first value in ACE.

You will:

1. seed a clean local playbook
2. retrieve the bullets ACE thinks will help
3. reflect on a concrete run
4. curate that reflection into delta ops
5. commit the delta and confirm the playbook changed

The example uses a deterministic reflection helper so you can run it without configuring an external LLM first. After that, you can swap in `ace reflect` with your own provider.

## Prerequisites

- cloned repo
- project venv available at `.venv/`
- about one minute for the first embedding model load

## Run The Example

From the repo root:

```bash
REPO_ROOT=$(pwd)
WORKDIR=$(mktemp -d)
cd "$WORKDIR"

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/seed.py"
ACE_DB_URL=ace.db "$REPO_ROOT/.venv/bin/python" -m ace.cli retrieve "hybrid retrieval" --top-k 3
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/demo_reflect.py" "$REPO_ROOT/docs/assets/ace-proof-demo/demo-task.json" > reflection.json
ACE_DB_URL=ace.db "$REPO_ROOT/.venv/bin/python" -m ace.cli curate --reflection reflection.json --json > delta.json
ACE_DB_URL=ace.db "$REPO_ROOT/.venv/bin/python" -m ace.cli commit --delta delta.json --json
ACE_DB_URL=ace.db "$REPO_ROOT/.venv/bin/python" -m ace.cli stats --json
```

## What You Should See

After `seed.py`:

```text
✓ Seeded 11 initial bullets
  Version: 0
  Total bullets in store: 11
```

After `retrieve`:

```text
Found 3 bullets:
[strat-00001] (strategies_and_hard_rules)
  Prefer hybrid retrieval: BM25 + embedding; rerank by lexical overlap with query terms; default top_k=24.
```

After `commit`:

```json
{
  "version": 1
}
```

After `stats`:

```json
{
  "version": 1,
  "total_bullets": 12,
  "helpful": 2,
  "harmful": 0,
  "helpful_ratio": 1.0
}
```

## Why This Is Useful

This example shows the part that matters:

- ACE starts with a versioned playbook, not a giant static prompt.
- retrieval gives the agent concrete tactics to use on the task at hand
- reflection records what worked and what should be added next
- commit updates the playbook with a small deterministic delta

In one loop, the playbook grows by one reusable bullet and records two helpful signals without rewriting everything.

## CLI-First Note

The flow above is the fastest local path. Once you want to use a live model, replace the deterministic helper with:

```bash
ACE_LLM_PROVIDER=openrouter \
OPENROUTER_API_KEY=... \
ACE_DB_URL=ace.db \
"$REPO_ROOT/.venv/bin/python" -m ace.cli reflect --doc "$REPO_ROOT/docs/assets/ace-proof-demo/demo-task.json" --json
```

## MCP-First Note

If your main workflow is an IDE or coding agent, start the server:

```bash
"$REPO_ROOT/.venv/bin/python" -m ace_mcp_server
```

Then run the same loop through MCP:

- `ace_retrieve` with query `hybrid retrieval`
- `ace_reflect` or the published demo payload in [`docs/assets/ace-proof-demo/reflection.json`](docs/assets/ace-proof-demo/reflection.json)
- `ace_commit` with [`docs/assets/ace-proof-demo/delta.json`](docs/assets/ace-proof-demo/delta.json)
- `ace_stats` to confirm the updated version and bullet count

## Next Step

Replace `demo-task.json` with a real task from your own repo, run the same loop, and then inspect whether the new bullet should be kept, patched, or refined.

If you want the exact published transcript and still image used for launch proof posts, see [`docs/ace-proof-demo.md`](docs/ace-proof-demo.md).
