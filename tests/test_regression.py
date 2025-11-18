# tests/test_regression.py
"""Tests for regression detection."""
from datetime import datetime, timedelta, timezone

import pytest

from ace.core.regression import (
    BenchmarkResult,
    RegressionDetector,
    RegressionThresholds,
)


@pytest.fixture
def detector():
    """Fresh detector for each test."""
    det = RegressionDetector()
    det.reset()
    return det


@pytest.fixture
def custom_thresholds():
    """Custom thresholds for testing."""
    return RegressionThresholds(
        success_rate_drop=0.10,  # 10% drop
        retrieval_precision_drop=0.15,  # 15% drop
        performance_increase=0.25,  # 25% increase
        absolute_failure_threshold=0.75,
    )


def test_record_result(detector):
    """Can record benchmark results."""
    detector.record_result(
        benchmark_name="test_retrieval",
        metric_name="precision",
        value=0.95,
        metadata={"top_k": 5},
    )

    assert len(detector.results) == 1
    result = detector.results[0]
    assert result.benchmark_name == "test_retrieval"
    assert result.metric_name == "precision"
    assert result.value == 0.95
    assert result.metadata["top_k"] == 5


def test_get_baseline_no_history(detector):
    """Baseline is None with no history."""
    baseline = detector.get_baseline("test_bench", "metric")
    assert baseline is None


def test_get_baseline_insufficient_samples(detector):
    """Baseline is None with only one sample."""
    detector.record_result("test_bench", "metric", 0.90)
    baseline = detector.get_baseline("test_bench", "metric")
    assert baseline is None


def test_get_baseline_median_calculation(detector):
    """Baseline uses median of recent samples."""
    values = [0.85, 0.90, 0.88, 0.92, 0.87]
    for val in values:
        detector.record_result("test_bench", "success_rate", val)

    baseline = detector.get_baseline("test_bench", "success_rate", n_samples=5)
    assert baseline == 0.88  # median of [0.85, 0.87, 0.88, 0.90, 0.92]


def test_get_baseline_recent_samples(detector):
    """Baseline uses only N most recent samples."""
    now = datetime.now(timezone.utc)
    for i, val in enumerate([0.70, 0.75, 0.80, 0.85, 0.90]):
        result = BenchmarkResult(
            timestamp=now - timedelta(days=5-i),
            benchmark_name="bench",
            metric_name="metric",
            value=val,
            metadata={},
        )
        detector.results.append(result)

    baseline = detector.get_baseline("bench", "metric", n_samples=3)
    assert baseline == 0.85  # median of [0.80, 0.85, 0.90]


def test_detect_no_regression_first_run(detector):
    """No regression on first run (no baseline)."""
    report = detector.detect_regression(
        benchmark_name="test_bench",
        metric_name="precision",
        current_value=0.90,
    )

    assert not report.detected
    assert report.severity == "info"
    assert "No baseline" in report.message


def test_detect_no_regression_within_threshold(detector):
    """No regression when within threshold."""
    for val in [0.90, 0.92, 0.91, 0.89, 0.90]:
        detector.record_result("bench", "precision", val)

    report = detector.detect_regression(
        benchmark_name="bench",
        metric_name="precision",
        current_value=0.87,  # ~4% drop from baseline 0.90
        higher_is_better=True,
    )

    assert not report.detected
    assert report.severity == "info"
    assert "OK:" in report.message


def test_detect_regression_success_rate_drop(detector):
    """Detect regression when success rate drops below threshold."""
    for val in [0.95, 0.96, 0.95, 0.94, 0.95]:
        detector.record_result("reflection_parse", "success_rate", val)

    report = detector.detect_regression(
        benchmark_name="reflection_parse",
        metric_name="success_rate",
        current_value=0.88,  # ~7% drop from baseline 0.95
        higher_is_better=True,
    )

    assert report.detected
    assert report.severity == "warning"
    assert "REGRESSION" in report.message
    assert report.baseline_value == pytest.approx(0.95, abs=0.01)
    assert report.current_value == 0.88


def test_detect_critical_regression_absolute_threshold(detector, custom_thresholds):
    """Critical severity when success rate drops below absolute threshold."""
    det = RegressionDetector(custom_thresholds)
    det.reset()

    for val in [0.90, 0.91, 0.90, 0.89, 0.90]:
        det.record_result("parse", "success_rate", val)

    report = det.detect_regression(
        benchmark_name="parse",
        metric_name="success_rate",
        current_value=0.70,  # Below absolute threshold of 0.75
        higher_is_better=True,
    )

    assert report.detected
    assert report.severity == "critical"
    assert report.current_value < custom_thresholds.absolute_failure_threshold


def test_detect_regression_performance_increase(detector):
    """Detect regression when latency increases (lower is better)."""
    for val in [100.0, 105.0, 102.0, 98.0, 100.0]:
        detector.record_result("retrieval", "latency_ms", val)

    report = detector.detect_regression(
        benchmark_name="retrieval",
        metric_name="latency_ms",
        current_value=125.0,  # ~23% increase from baseline 101.0
        higher_is_better=False,  # Lower latency is better
    )

    assert report.detected
    assert report.severity == "warning"
    assert "increased" in report.message


def test_detect_regression_retrieval_precision(detector):
    """Detect regression for retrieval-specific metrics."""
    for val in [0.80, 0.82, 0.81, 0.79, 0.80]:
        detector.record_result("golden_retrieval", "retrieval_precision", val)

    report = detector.detect_regression(
        benchmark_name="golden_retrieval",
        metric_name="retrieval_precision",
        current_value=0.70,  # ~12% drop from baseline 0.80
        higher_is_better=True,
    )

    assert report.detected
    assert "REGRESSION" in report.message


def test_custom_thresholds(custom_thresholds):
    """Custom thresholds are respected."""
    det = RegressionDetector(custom_thresholds)
    det.reset()

    for val in [0.90, 0.91, 0.90, 0.89, 0.90]:
        det.record_result("bench", "success_rate", val)

    # 8% drop - would fail default (5%) but pass custom (10%)
    report = det.detect_regression(
        benchmark_name="bench",
        metric_name="success_rate",
        current_value=0.83,
        higher_is_better=True,
    )

    assert not report.detected  # Within 10% custom threshold


def test_threshold_selection_by_metric_name(detector):
    """Different metrics use appropriate thresholds."""
    for val in [100.0] * 5:
        detector.record_result("bench", "latency_ms", val)

    # Performance metrics use performance_increase threshold (20%)
    threshold = detector._get_threshold("latency_ms")
    assert threshold == detector.thresholds.performance_increase

    # Success rate uses success_rate_drop threshold (5%)
    threshold = detector._get_threshold("success_rate")
    assert threshold == detector.thresholds.success_rate_drop


def test_static_baseline(detector):
    """Static baseline overrides historical baseline."""
    # Historical baseline
    for val in [0.90] * 5:
        detector.record_result("bench", "metric", val)
    
    # Should use static baseline (0.80) instead of historical (0.90)
    # Current 0.75 is a 6% drop from 0.80, which is > 5% default threshold
    report = detector.detect_regression(
        benchmark_name="bench",
        metric_name="metric",
        current_value=0.75,
        static_baseline=0.80
    )

    assert report.detected
    assert report.baseline_value == 0.80
    
    # Using historical (0.90), 0.75 would be ~16% drop (detected)
    # But let's test a case where static passes but historical fails
    # Historical 0.90. Current 0.88. Drop 2%.
    # Static 0.85. Current 0.88. Increase.
    
    report2 = detector.detect_regression(
        benchmark_name="bench",
        metric_name="metric",
        current_value=0.88,
        static_baseline=0.85
    )
    
    assert not report2.detected
    assert report2.baseline_value == 0.85


def test_persistence_and_reload(detector, tmp_path):
    """Results persist across detector instances."""
    # Use instance attribute to override class attribute
    fpath = tmp_path / "benchmarks.jsonl"
    detector._benchmarks_file = fpath
    detector._ensure_benchmarks_dir()

    detector.record_result("bench", "metric", 0.95)
    detector.record_result("bench", "metric", 0.93)

    # New instance
    det2 = RegressionDetector()
    det2._benchmarks_file = fpath
    # Manually trigger load since __init__ ran before we set the file path
    det2._load_history()

    baseline = det2.get_baseline("bench", "metric")
    assert baseline == 0.94  # median of [0.93, 0.95]


def test_reset_clears_history(detector):
    """Reset clears all results."""
    detector.record_result("bench", "metric", 0.90)
    detector.record_result("bench", "metric", 0.92)

    assert len(detector.results) == 2

    detector.reset()

    assert len(detector.results) == 0
    baseline = detector.get_baseline("bench", "metric")
    assert baseline is None


def test_change_percentage_calculation(detector):
    """Change percentage is calculated correctly."""
    for val in [100.0] * 5:
        detector.record_result("bench", "metric", val)

    report = detector.detect_regression(
        benchmark_name="bench",
        metric_name="metric",
        current_value=120.0,
        higher_is_better=False,
    )

    assert report.change_pct == pytest.approx(0.20, abs=0.01)  # 20% increase


def test_zero_baseline_handling(detector):
    """Handles zero baseline gracefully."""
    for val in [0.0] * 5:
        detector.record_result("bench", "metric", val)

    report = detector.detect_regression(
        benchmark_name="bench",
        metric_name="metric",
        current_value=0.5,
        higher_is_better=True,
    )

    assert report.change_pct == 0.0
    assert not report.detected
