# Regression Detection

ACE's regression detection system tracks benchmark metrics over time and alerts when performance degrades beyond configurable thresholds.

## Overview

The regression detector:
- Records benchmark results to `.benchmarks/history.jsonl`
- Compares current metrics against baseline (median of recent runs)
- Detects regressions using configurable thresholds
- Supports both "higher is better" (success rate, precision) and "lower is better" (latency) metrics

## Usage

### Basic Usage

```python
from ace.core.regression import get_detector

detector = get_detector()

# Record a benchmark result
detector.record_result(
    benchmark_name="retrieval_golden_tests",
    metric_name="precision",
    value=0.92,
    metadata={"top_k": 5, "query_type": "pgvector"},
)

# Check for regression
report = detector.detect_regression(
    benchmark_name="retrieval_golden_tests",
    metric_name="precision",
    current_value=0.85,
    higher_is_better=True,
)

if report.detected:
    print(f"⚠️  {report.message}")
    print(f"Severity: {report.severity}")
```

### Custom Thresholds

```python
from ace.core.regression import RegressionDetector, RegressionThresholds

thresholds = RegressionThresholds(
    success_rate_drop=0.10,  # 10% drop in success rate triggers warning
    retrieval_precision_drop=0.15,  # 15% drop in retrieval metrics
    performance_increase=0.25,  # 25% increase in latency triggers warning
    absolute_failure_threshold=0.75,  # Critical if success rate < 75%
)

detector = RegressionDetector(thresholds)
```

## Metrics and Thresholds

### Default Thresholds

| Metric Type | Threshold | Example Metrics |
|-------------|-----------|-----------------|
| Success Rate | 5% drop | `success_rate`, `precision`, `recall` |
| Retrieval | 10% drop | `retrieval_precision`, `retrieval_recall` |
| Performance | 20% increase | `latency_ms`, `duration`, `time` |

### Severity Levels

- **critical**: Success rate dropped below absolute threshold (default 80%)
- **warning**: Metric exceeded threshold but not critical
- **info**: No regression detected or first run (no baseline)

## Baseline Calculation

Baselines use the **median** of the last N samples (default 5) to avoid outliers:

```python
# Records: [0.85, 0.90, 0.88, 0.92, 0.87]
# Baseline: 0.88 (median)
baseline = detector.get_baseline("bench_name", "metric_name", n_samples=5)
```

Requires at least 2 historical samples to establish a baseline.

## Integration with Golden Tests

Use regression detection in pytest golden tests:

```python
import pytest
from ace.core.regression import get_detector

@pytest.mark.retrieval_regression
def test_golden_retrieval_pgvector():
    detector = get_detector()
    
    # Run retrieval test
    results = retriever.retrieve("pgvector extension missing", top_k=3)
    precision = calculate_precision(results, expected_ids=["trbl-golden-001"])
    
    # Record result
    detector.record_result(
        benchmark_name="golden_retrieval_pgvector",
        metric_name="precision",
        value=precision,
    )
    
    # Check for regression
    report = detector.detect_regression(
        benchmark_name="golden_retrieval_pgvector",
        metric_name="precision",
        current_value=precision,
        higher_is_better=True,
    )
    
    # Assert test passes AND no regression
    assert "trbl-golden-001" in [b.id for b in results]
    assert not report.detected, f"Regression detected: {report.message}"
```

## CI Integration

Add regression check to CI pipeline:

```yaml
- name: Run benchmarks with regression detection
  run: |
    pytest -m retrieval_regression --json-report --json-report-file=report.json
    python scripts/check_regressions.py report.json
```

Example `scripts/check_regressions.py`:

```python
#!/usr/bin/env python
import json
import sys
from ace.core.regression import get_detector

detector = get_detector()

# Load benchmark results from pytest
with open(sys.argv[1]) as f:
    report = json.load(f)

regressions = []
for test in report["tests"]:
    if test.get("regression_report"):
        rep = test["regression_report"]
        if rep["detected"] and rep["severity"] in ["warning", "critical"]:
            regressions.append(rep)

if regressions:
    print("❌ Regressions detected:")
    for reg in regressions:
        print(f"  - {reg['benchmark_name']}.{reg['metric_name']}: {reg['message']}")
    sys.exit(1)
else:
    print("✅ No regressions detected")
```

## File Format

Results are stored in `.benchmarks/history.jsonl`:

```json
{"timestamp": "2025-11-17T12:00:00Z", "benchmark_name": "retrieval", "metric_name": "precision", "value": 0.92, "metadata": {"top_k": 5}}
{"timestamp": "2025-11-17T13:00:00Z", "benchmark_name": "retrieval", "metric_name": "precision", "value": 0.90, "metadata": {"top_k": 5}}
```

## Best Practices

1. **Tag benchmarks clearly**: Use descriptive names like `golden_retrieval_pgvector` rather than `test_1`
2. **Record metadata**: Include context (top_k, model version, etc.) for debugging
3. **Run regularly**: Regression detection requires historical data
4. **Start conservative**: Use stricter thresholds (e.g., 3-5% drops) for critical metrics
5. **Monitor trends**: Check `.benchmarks/history.jsonl` for gradual degradation
6. **Fail fast**: Set `absolute_failure_threshold` for critical success rates
