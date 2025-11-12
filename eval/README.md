# Evaluation Harness

This directory contains automated tests and benchmarks for the ACE system.

## Running Tests

### Standard Unit Tests

By default, benchmark tests are excluded for faster CI/local runs:

```bash
# Run all non-benchmark tests (default)
make test
# or
pytest

# Run specific test file
pytest eval/retrieval_bench.py
```

### Performance Benchmarks

To run performance benchmarks (opt-in):

```bash
# Run only benchmark tests
pytest -m benchmark

# Run all tests including benchmarks
pytest -m ""

# Run benchmarks with detailed output
pytest -m benchmark -v --benchmark-only
```

### Test Organization

- `retrieval_bench.py` - Retrieval system tests (hybrid search, reranking)
- `test_helpers.py` - Lightweight test fixtures (no FAISS/embedding overhead)

## Test Design

All tests use `LightweightStore` from `test_helpers.py` which:
- Avoids loading FAISS and SentenceTransformer models
- Uses simple lexical matching instead of embeddings
- Keeps tests fast and hermetic
- Isolates each test run (no shared state)

## CI Behavior

CI automatically excludes benchmark tests to keep runs fast. See `.github/workflows/ci.yml`.
