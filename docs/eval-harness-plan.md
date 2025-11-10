# Evaluation Harness Implementation Plan

## Current State

**Existing test structure:**
- `tests/` - 10 unit test files covering core modules:
  - test_config.py
  - test_curator.py
  - test_eval_harness.py
  - test_llm_client.py
  - test_llm_integration.py
  - test_merge.py
  - test_refiner.py
  - test_reflector.py
  - test_schema.py
  - test_store.py

- `ace/eval/` - Evaluation harness module (basic implementation):
  - harness.py (EvalRunner with stub methods)
  - fixtures/ (directory for test data)

## Proposed ace/eval/ Module Structure

```
ace/eval/
├── __init__.py                 # Public API exports
├── harness.py                  # Main evaluation runner
├── metrics.py                  # Metric calculators (precision, recall, etc.)
├── fixtures/                   # Test data and scenarios
│   ├── __init__.py
│   ├── retrieval_cases.json    # Known query→bullet pairs
│   ├── reflection_cases.json   # Task data → expected reflections
│   └── delta_cases.json        # Reflection → expected deltas
├── benchmarks/                 # Performance and quality benchmarks
│   ├── __init__.py
│   ├── retrieval_bench.py      # Retrieval quality (MRR, NDCG)
│   ├── reflection_bench.py     # Reflection quality
│   └── e2e_bench.py            # Full pipeline tests
└── reports/                    # Generated evaluation outputs
    └── .gitkeep
```

## CLI Interface Design

### New `ace eval` Command

```bash
# Run all evaluation benchmarks
ace eval run

# Run specific benchmark suite
ace eval run --suite retrieval
ace eval run --suite reflection
ace eval run --suite e2e

# Output format options
ace eval run --json
ace eval run --format markdown --out results.md

# Continuous evaluation (for CI)
ace eval run --baseline baseline.json --fail-on-regression
```

## Changes to ace/cli.py

Add new subparser under main parser:

```python
eval_parser = subparsers.add_parser("eval", help="Run evaluation benchmarks")
eval_subparsers = eval_parser.add_subparsers(dest="eval_cmd", required=True)

eval_run = eval_subparsers.add_parser("run", help="Run evaluation suite")
eval_run.add_argument("--suite", choices=["retrieval", "reflection", "e2e", "all"], default="all")
eval_run.add_argument("--json", action="store_true")
eval_run.add_argument("--format", choices=["text", "json", "markdown"], default="text")
eval_run.add_argument("--out", help="Output file path")
eval_run.add_argument("--baseline", help="Baseline JSON for regression detection")
eval_run.add_argument("--fail-on-regression", action="store_true")
eval_run.set_defaults(func=cmd_eval_run)
```

## Metrics to Track

### Retrieval Quality
- **MRR (Mean Reciprocal Rank)**: Average position of first relevant bullet
- **NDCG@k**: Normalized discounted cumulative gain
- **Recall@k**: Proportion of relevant bullets in top-k

### Reflection Quality
- **Schema compliance**: % valid JSON outputs
- **Candidate relevance**: Human-labeled quality of proposed bullets
- **Tag accuracy**: Correct helpful/harmful assignments

### End-to-End Quality
- **Delta applicability**: % deltas that apply without errors
- **Dedup effectiveness**: Reduction in near-duplicates after refine
- **Context growth rate**: Bullets added vs. archived over time

### Performance
- **Latency**: Time for retrieve, reflect, curate, commit operations
- **Token usage**: LLM tokens consumed per operation

## Implementation Phases

### Phase 1: Basic Harness (Week 1)
- [ ] Create `ace/eval/harness.py` with EvalRunner class
- [ ] Implement simple fixture loading from JSON
- [ ] Add `cmd_eval_run` to cli.py
- [ ] Create 3-5 golden retrieval test cases

### Phase 2: Retrieval Benchmarks (Week 1-2)
- [ ] Implement MRR, Recall@k metrics in metrics.py
- [ ] Create retrieval_bench.py with automated tests
- [ ] Generate baseline.json from current system
- [ ] Add CI check for retrieval regression

### Phase 3: Reflection Benchmarks (Week 2)
- [ ] Create reflection test fixtures (task → expected output)
- [ ] Implement schema validation metrics
- [ ] Add reflection_bench.py with quality checks

### Phase 4: E2E Benchmarks (Week 3)
- [ ] Full pipeline tests (retrieve → reflect → curate → commit)
- [ ] Delta applicability and idempotency tests
- [ ] Performance profiling and token tracking

### Phase 5: Reporting & CI Integration (Week 3-4)
- [ ] Markdown report generator
- [ ] Regression detection logic
- [ ] Update .github/workflows/ci.yml with eval stage
- [ ] Dashboard/visualization (optional)

## Fixture Format Examples

### retrieval_cases.json
```json
{
  "cases": [
    {
      "query": "How to handle PostgreSQL connection pooling?",
      "relevant_bullet_ids": ["strat-00045", "code-00123"],
      "irrelevant_bullet_ids": ["tmpl-00099"]
    }
  ]
}
```

### reflection_cases.json
```json
{
  "cases": [
    {
      "task_data": {
        "query": "Add retry logic to API calls",
        "test_output": "FAILED: test_api_timeout",
        "code_diff": "..."
      },
      "expected": {
        "should_contain_insight": true,
        "should_tag_helpful": [],
        "min_candidate_bullets": 1
      }
    }
  ]
}
```

## Success Criteria

- ✅ `ace eval run` executes without errors
- ✅ At least 10 golden test cases per benchmark type
- ✅ CI fails on >10% regression in MRR or Recall@10
- ✅ Eval runtime < 60 seconds for full suite
- ✅ Documentation shows how to add new test cases

## Open Questions

1. Should we use human-labeled data or synthetic test cases initially?
2. What threshold constitutes a "regression" (10%, 15%)?
3. Should eval fixtures live in git or external storage?
4. Do we need separate "fast" vs. "comprehensive" eval modes?
