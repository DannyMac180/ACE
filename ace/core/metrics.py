# ace/core/metrics.py
"""Schema validation metrics tracking for ACE."""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ValidationAttempt:
    """Single validation attempt record."""
    timestamp: datetime
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    schema_type: Literal["reflection", "delta"] = "reflection"


@dataclass
class ValidationMetrics:
    """Aggregated validation metrics."""
    total_attempts: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    json_decode_errors: int = 0
    schema_validation_errors: int = 0
    markdown_fence_removals: int = 0
    error_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_parses / self.total_attempts

    def to_dict(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "success_rate": round(self.success_rate, 3),
            "json_decode_errors": self.json_decode_errors,
            "schema_validation_errors": self.schema_validation_errors,
            "markdown_fence_removals": self.markdown_fence_removals,
            "error_breakdown": self.error_breakdown,
        }


class MetricsTracker:
    """Singleton metrics tracker for schema validation."""

    _instance: Optional["MetricsTracker"] = None
    _metrics_file: Path = Path("ace.db").parent / ".validation_metrics.jsonl"
    _initialized: bool

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.attempts: list[ValidationAttempt] = []
        self._load_history()

    def _load_history(self):
        """Load historical metrics from disk."""
        if not self._metrics_file.exists():
            return

        try:
            with open(self._metrics_file) as f:
                for line in f:
                    data = json.loads(line)
                    attempt = ValidationAttempt(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        success=data["success"],
                        error_type=data.get("error_type"),
                        error_message=data.get("error_message"),
                        schema_type=data.get("schema_type", "reflection"),
                    )
                    self.attempts.append(attempt)
        except Exception:
            pass

    def record_attempt(
        self,
        success: bool,
        error_type: str | None = None,
        error_message: str | None = None,
        schema_type: Literal["reflection", "delta"] = "reflection",
    ):
        """Record a validation attempt."""
        attempt = ValidationAttempt(
            timestamp=datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
            success=success,
            error_type=error_type,
            error_message=error_message,
            schema_type=schema_type,
        )
        self.attempts.append(attempt)
        self._persist_attempt(attempt)

    def _persist_attempt(self, attempt: ValidationAttempt):
        """Append attempt to JSONL file."""
        try:
            with open(self._metrics_file, "a") as f:
                data = {
                    "timestamp": attempt.timestamp.isoformat(),
                    "success": attempt.success,
                    "error_type": attempt.error_type,
                    "error_message": attempt.error_message,
                    "schema_type": attempt.schema_type,
                }
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass

    def get_metrics(
        self, schema_type: Literal["reflection", "delta"] | None = None
    ) -> ValidationMetrics:
        """Compute aggregated metrics."""
        attempts = self.attempts
        if schema_type:
            attempts = [a for a in attempts if a.schema_type == schema_type]

        metrics = ValidationMetrics()
        metrics.total_attempts = len(attempts)

        for attempt in attempts:
            if attempt.success:
                metrics.successful_parses += 1
            else:
                metrics.failed_parses += 1

                if attempt.error_type:
                    if "JSONDecodeError" in attempt.error_type or "Invalid JSON" in str(
                        attempt.error_message
                    ):
                        metrics.json_decode_errors += 1
                    else:
                        metrics.schema_validation_errors += 1

                    error_key = attempt.error_type or "unknown"
                    metrics.error_breakdown[error_key] = (
                        metrics.error_breakdown.get(error_key, 0) + 1
                    )

        return metrics

    def reset(self):
        """Clear all metrics (for testing)."""
        self.attempts.clear()
        if self._metrics_file.exists():
            self._metrics_file.unlink()


def get_tracker() -> MetricsTracker:
    """Get the global metrics tracker instance."""
    return MetricsTracker()
