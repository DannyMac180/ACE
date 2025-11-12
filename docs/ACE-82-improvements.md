# ACE-82 Code Review Improvements

## Summary
Addressed all code review findings for ACE-82 (Create 3-5 golden retrieval test cases).

## Changes Made

### 1. Fixed pytest discovery (High Priority)
**Problem**: Tests in `retrieval_bench.py` were never collected by pytest because pytest only collects files named `test_*.py`.

**Solution**: Renamed `eval/retrieval_bench.py` → `eval/test_retrieval_bench.py`

**Verification**: `pytest --collect-only` now shows all 13 tests (8 regular + 5 golden)

### 2. Full hybrid retrieval coverage (High Priority)
**Problem**: Tests used `MockEmbeddingStore` which only did lexical token counting, completely bypassing FAISS embeddings, vector similarity, and hybrid retrieval.

**Solution**: 
- Created `HybridStore` class in `test_helpers.py` that uses real embeddings and FAISS
- Created `golden_store` pytest fixture that populates a `HybridStore` with golden bullets
- All golden tests now exercise the complete retrieval pipeline: BM25 FTS + vector embeddings + lexical reranking

**Files Modified**:
- `eval/test_helpers.py`: Added `HybridStore` class
- `eval/test_retrieval_bench.py`: Added `golden_store` fixture, updated all golden tests to use it

### 3. Fixed helpful score test (Medium Priority)
**Problem**: `test_golden_retrieval_helpful_score_ranking` claimed to test helpful score ranking, but the `Retriever` only uses lexical overlap (not helpful/harmful scores).

**Solution**: 
- Renamed test to `test_golden_retrieval_lexical_rerank_precision`
- Updated docstring and assertions to match actual retriever behavior
- Test now validates lexical overlap reranking (which is what the code actually does)

### 4. DRY fixture with guaranteed cleanup (Low Priority)
**Problem**: Each golden test manually created/populated/closed the store, risking resource leaks on errors.

**Solution**: 
- Created `golden_store` pytest fixture using `yield` pattern
- Fixture guarantees cleanup via pytest's fixture teardown mechanism
- Eliminates duplicate setup/teardown code across 5 tests

## Test Results
All 13 tests pass ✅:
- 8 regular retrieval tests (using `LightweightStore` for speed)
- 5 golden tests (using `HybridStore` for full coverage)

```bash
$ pytest eval/test_retrieval_bench.py -v
======================== 13 passed, 93 warnings in 2.74s ========================
```

## Golden Test Coverage

1. **test_golden_retrieval_pgvector_query**: Pgvector error troubleshooting
2. **test_golden_retrieval_delta_curation_policy**: Curation delta policy retrieval
3. **test_golden_retrieval_mcp_json_error**: MCP JSON parse error troubleshooting
4. **test_golden_retrieval_multi_tag_filtering**: Multi-tag domain overlap
5. **test_golden_retrieval_lexical_rerank_precision**: Lexical overlap ranking

## Impact
- ✅ Tests now run in CI (pytest discovery fixed)
- ✅ Tests now exercise full hybrid retrieval pipeline (FAISS + BM25 + reranking)
- ✅ Tests accurately document actual retriever behavior
- ✅ Resource leaks prevented via fixture teardown
- ✅ Code is DRY (no duplicate setup/teardown)
