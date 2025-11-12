# Retrieval Benchmark Improvements

## Summary

Implemented comprehensive fixes to `eval/retrieval_bench.py` based on code review findings. All changes ensure hermetic, fast, and maintainable tests.

## Changes Made

### 1. Added pytest-benchmark Dependency (High Priority)
- **File**: `pyproject.toml`
- **Change**: Added `pytest-benchmark>=4.0.0` to dev dependencies
- **Rationale**: Benchmark tests require this package for proper test collection and execution

### 2. Created Lightweight Test Store (High Priority)
- **File**: `eval/test_helpers.py` (new)
- **Changes**:
  - Created `MockEmbeddingStore` that uses simple lexical matching instead of FAISS
  - Created `LightweightStore` that avoids loading SentenceTransformer models
  - Eliminates dependency on shared `faiss_index.idx` file
  - Makes tests hermetic and fast (~30µs per test vs potential seconds with real embeddings)
- **Rationale**: Avoids heavyweight FAISS and embedding model overhead for unit tests

### 3. Fixed Store Fixture Lifecycle (Medium Priority)
- **File**: `eval/retrieval_bench.py`
- **Changes**:
  - Changed fixture from `return` to `yield` pattern
  - Added explicit `store.close()` cleanup
  - Each test gets isolated temporary DB and index paths
- **Rationale**: Prevents SQLite/FAISS handle leaks and ensures test isolation

### 4. Strengthened Test Assertions (Low Priority)
- **File**: `eval/retrieval_bench.py`
- **Changes**:
  - `test_retrieval_empty_query`: Added length assertions (`<= 5`)
  - `test_retrieval_no_matches`: Added overlap validation for nonsense queries
  - Updated docstrings to match actual behavior
- **Rationale**: Makes tests more meaningful and catches regressions

### 5. CI and Documentation Updates (Low Priority)
- **Files**: `.github/workflows/ci.yml`, `pyproject.toml`, `eval/README.md` (new)
- **Changes**:
  - CI now excludes benchmarks by default (`-m "not benchmark"`)
  - Added default pytest option to skip benchmarks
  - Created `eval/README.md` documenting how to run benchmarks
  - Developers can opt-in with `pytest -m benchmark`
- **Rationale**: Fast CI runs while preserving benchmark capability for local performance testing

## Test Results

All 123 tests passing (8 retrieval tests + 115 existing):
```bash
$ pytest
===== 123 passed, 1 skipped in 3.16s =====

$ pytest -m benchmark
===== 1 passed in 2.55s =====
```

## Performance

- **Before**: Tests loaded FAISS + SentenceTransformer (multi-second overhead)
- **After**: ~30µs per test with lightweight mocks
- **Benchmark test**: Still available for opt-in performance measurement

## Files Changed

1. `pyproject.toml` - Added pytest-benchmark dependency, configured markers
2. `eval/test_helpers.py` - New lightweight test fixtures
3. `eval/retrieval_bench.py` - Updated to use LightweightStore, yield fixture, stronger assertions
4. `.github/workflows/ci.yml` - Exclude benchmarks from CI
5. `eval/README.md` - Documentation for running tests and benchmarks

## Verification

```bash
# Run non-benchmark tests (default)
make test

# Run only benchmarks
pytest -m benchmark

# Run everything
pytest -m ""
```
