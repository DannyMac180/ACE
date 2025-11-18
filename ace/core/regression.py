# ace/core/regression.py
"""Regression detection for ACE benchmarks and metrics."""
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


@dataclass
class BenchmarkResult:
    """Single benchmark result record."""
    timestamp: datetime
    benchmark_name: str
    metric_name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionThresholds:
    """Configurable regression detection thresholds."""
    success_rate_drop: float = 0.05  # 5% drop in success rate
    retrieval_precision_drop: float = 0.10  # 10% drop in retrieval precision
    performance_increase: float = 0.20  # 20% increase in latency/time
    absolute_failure_threshold: float = 0.80  # Fail if success rate < 80%


@dataclass
class RegressionReport:
    """Regression detection report."""
    detected: bool
    baseline_value: float
    current_value: float
    change_pct: float
    threshold: float
    metric_name: str
    benchmark_name: str
    severity: Literal["critical", "warning", "info"]
    message: str


class RegressionDetector:
    """Detects regressions in benchmark metrics using baseline comparison."""

    _benchmarks_file: Path = Path(".benchmarks") / "history.jsonl"

    def __init__(self, thresholds: RegressionThresholds | None = None):
        self.thresholds = thresholds or RegressionThresholds()
        self.results: list[BenchmarkResult] = []
        self._ensure_benchmarks_dir()
        self._load_history()

    def _ensure_benchmarks_dir(self):
        """Create .benchmarks directory if it doesn't exist."""
        self._benchmarks_file.parent.mkdir(exist_ok=True)

    def _load_history(self):
        """Load historical benchmark results from disk."""
        if not self._benchmarks_file.exists():
            return

        try:
            with open(self._benchmarks_file) as f:
                for line in f:
                    data = json.loads(line)
                    result = BenchmarkResult(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        benchmark_name=data["benchmark_name"],
                        metric_name=data["metric_name"],
                        value=data["value"],
                        metadata=data.get("metadata", {}),
                    )
                    self.results.append(result)
        except Exception:
            pass

    def record_result(
        self,
        benchmark_name: str,
        metric_name: str,
        value: float,
        metadata: dict | None = None,
    ):
        """Record a benchmark result."""
        result = BenchmarkResult(
            timestamp=datetime.now(UTC),
            benchmark_name=benchmark_name,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {},
        )
        self.results.append(result)
        self._persist_result(result)

    def _persist_result(self, result: BenchmarkResult):
        """Append result to JSONL file."""
        try:
            with open(self._benchmarks_file, "a") as f:
                data = {
                    "timestamp": result.timestamp.isoformat(),
                    "benchmark_name": result.benchmark_name,
                    "metric_name": result.metric_name,
                    "value": result.value,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass

    def get_baseline(
        self,
        benchmark_name: str,
        metric_name: str,
        n_samples: int = 5,
    ) -> float | None:
        """Get baseline value for a benchmark metric.

        Uses median of last N successful runs to avoid outliers.
        """
        matching = [
            r for r in self.results
            if r.benchmark_name == benchmark_name and r.metric_name == metric_name
        ]

        if len(matching) < 2:
            return None

        recent = sorted(matching, key=lambda r: r.timestamp, reverse=True)[:n_samples]
        values = [r.value for r in recent]

        values.sort()
        mid = len(values) // 2
        if len(values) % 2 == 0:
            return (values[mid - 1] + values[mid]) / 2.0
        return values[mid]

    def detect_regression(
        self,
        benchmark_name: str,
        metric_name: str,
        current_value: float,
        higher_is_better: bool = True,
        static_baseline: float | None = None,
    ) -> RegressionReport:
        """Detect if current value represents a regression from baseline.

        Args:
            benchmark_name: Name of the benchmark
            metric_name: Name of the metric (e.g., "success_rate", "precision")
            current_value: Current measured value
            higher_is_better: If True, lower values are regressions;
                if False, higher values are
            static_baseline: Optional fixed baseline value to compare against
                (overrides historical baseline)

        Returns:
            RegressionReport with detection results
        """
        baseline: float | None
        if static_baseline is not None:
            baseline = static_baseline
        else:
            baseline = self.get_baseline(benchmark_name, metric_name)

        if baseline is None:
            return RegressionReport(
                detected=False,
                baseline_value=0.0,
                current_value=current_value,
                change_pct=0.0,
                threshold=0.0,
                metric_name=metric_name,
                benchmark_name=benchmark_name,
                severity="info",
                message=f"No baseline for {benchmark_name}.{metric_name} - recording first result",
            )

        if baseline == 0:
            change_pct = 0.0
        else:
            change_pct = (current_value - baseline) / baseline

        threshold = self._get_threshold(metric_name)

        is_regression = False
        severity: Literal["critical", "warning", "info"] = "info"

        if higher_is_better:
            drop_pct = -change_pct
            if drop_pct >= threshold:
                is_regression = True
                abs_threshold = self.thresholds.absolute_failure_threshold
                if metric_name == "success_rate" and current_value < abs_threshold:
                    severity = "critical"
                else:
                    severity = "warning"
        else:
            increase_pct = change_pct
            if increase_pct >= threshold:
                is_regression = True
                severity = "warning"

        if is_regression:
            direction = "dropped" if higher_is_better else "increased"
            message = (
                f"REGRESSION: {benchmark_name}.{metric_name} {direction} "
                f"{abs(change_pct)*100:.1f}% "
                f"(baseline={baseline:.3f}, current={current_value:.3f}, "
                f"threshold={threshold*100:.1f}%)"
            )
        else:
            message = (
                f"OK: {benchmark_name}.{metric_name} within threshold "
                f"(baseline={baseline:.3f}, current={current_value:.3f}, "
                f"change={change_pct*100:+.1f}%)"
            )

        return RegressionReport(
            detected=is_regression,
            baseline_value=baseline,
            current_value=current_value,
            change_pct=change_pct,
            threshold=threshold,
            metric_name=metric_name,
            benchmark_name=benchmark_name,
            severity=severity,
            message=message,
        )

    def _get_threshold(self, metric_name: str) -> float:
        """Get regression threshold for a metric."""
        # More robust matching logic
        name = metric_name.lower()
        if "latency" in name or "time" in name or "duration" in name:
            return self.thresholds.performance_increase
        elif "retrieval" in name and ("precision" in name or "recall" in name or "mrr" in name):
            return self.thresholds.retrieval_precision_drop
        elif "success_rate" in name or "precision" in name or "recall" in name:
            return self.thresholds.success_rate_drop
        return self.thresholds.success_rate_drop

    def reset(self):
        """Clear all results (for testing)."""
        self.results.clear()
        if self._benchmarks_file.exists():
            self._benchmarks_file.unlink()


def get_detector(thresholds: RegressionThresholds | None = None) -> RegressionDetector:
    """Get the global regression detector instance."""
    return RegressionDetector(thresholds)
