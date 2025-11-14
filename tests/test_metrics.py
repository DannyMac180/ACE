# tests/test_metrics.py
"""Tests for schema validation metrics tracking."""
import json

import pytest

from ace.core.metrics import ValidationMetrics, get_tracker
from ace.reflector.parser import ReflectionParseError, parse_reflection


class TestMetricsTracker:
    """Test metrics tracker functionality."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker = get_tracker()
        tracker.reset()

    def test_singleton_pattern(self):
        """MetricsTracker should be a singleton."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()
        assert tracker1 is tracker2

    def test_record_successful_parse(self):
        """Track successful parse attempts."""
        tracker = get_tracker()
        tracker.record_attempt(success=True, schema_type="reflection")

        metrics = tracker.get_metrics()
        assert metrics.total_attempts == 1
        assert metrics.successful_parses == 1
        assert metrics.failed_parses == 0
        assert metrics.success_rate == 1.0

    def test_record_json_decode_error(self):
        """Track JSON decode errors."""
        tracker = get_tracker()
        tracker.record_attempt(
            success=False,
            error_type="JSONDecodeError",
            error_message="Expecting value: line 1 column 1 (char 0)",
            schema_type="reflection",
        )

        metrics = tracker.get_metrics()
        assert metrics.total_attempts == 1
        assert metrics.successful_parses == 0
        assert metrics.failed_parses == 1
        assert metrics.json_decode_errors == 1
        assert metrics.schema_validation_errors == 0
        assert "JSONDecodeError" in metrics.error_breakdown

    def test_record_schema_validation_error(self):
        """Track schema validation errors."""
        tracker = get_tracker()
        tracker.record_attempt(
            success=False,
            error_type="SchemaValidationError",
            error_message="bullet_tags must be a list",
            schema_type="reflection",
        )

        metrics = tracker.get_metrics()
        assert metrics.schema_validation_errors == 1
        assert metrics.json_decode_errors == 0
        assert "SchemaValidationError" in metrics.error_breakdown

    def test_success_rate_calculation(self):
        """Calculate success rate correctly."""
        tracker = get_tracker()
        tracker.record_attempt(success=True, schema_type="reflection")
        tracker.record_attempt(success=True, schema_type="reflection")
        tracker.record_attempt(
            success=False,
            error_type="JSONDecodeError",
            schema_type="reflection",
        )

        metrics = tracker.get_metrics()
        assert metrics.total_attempts == 3
        assert metrics.successful_parses == 2
        assert metrics.failed_parses == 1
        assert abs(metrics.success_rate - 0.667) < 0.01

    def test_filter_by_schema_type(self):
        """Filter metrics by schema type."""
        tracker = get_tracker()
        tracker.record_attempt(success=True, schema_type="reflection")
        tracker.record_attempt(success=True, schema_type="delta")
        tracker.record_attempt(
            success=False,
            error_type="JSONDecodeError",
            schema_type="reflection",
        )

        reflection_metrics = tracker.get_metrics(schema_type="reflection")
        assert reflection_metrics.total_attempts == 2
        assert reflection_metrics.successful_parses == 1

        delta_metrics = tracker.get_metrics(schema_type="delta")
        assert delta_metrics.total_attempts == 1
        assert delta_metrics.successful_parses == 1


class TestParserIntegration:
    """Test metrics integration with parser."""

    def setup_method(self):
        """Reset tracker before each test."""
        tracker = get_tracker()
        tracker.reset()

    def test_successful_parse_records_metric(self):
        """Successful parse should record metric."""
        valid_json = json.dumps({
            "error_identification": "Test error",
            "bullet_tags": [{"id": "strat-001", "tag": "helpful"}],
            "candidate_bullets": [
                {"section": "strategies", "content": "test", "tags": ["test"]}
            ]
        })

        parse_reflection(valid_json)

        tracker = get_tracker()
        metrics = tracker.get_metrics()
        assert metrics.total_attempts == 1
        assert metrics.successful_parses == 1

    def test_json_decode_error_records_metric(self):
        """JSON decode error should record metric."""
        invalid_json = "not json at all"

        with pytest.raises(ReflectionParseError):
            parse_reflection(invalid_json)

        tracker = get_tracker()
        metrics = tracker.get_metrics()
        assert metrics.total_attempts == 1
        assert metrics.failed_parses == 1
        assert metrics.json_decode_errors == 1

    def test_schema_validation_error_records_metric(self):
        """Schema validation error should record metric."""
        invalid_schema = json.dumps({
            "bullet_tags": "not a list"
        })

        with pytest.raises(ReflectionParseError):
            parse_reflection(invalid_schema)

        tracker = get_tracker()
        metrics = tracker.get_metrics()
        assert metrics.total_attempts == 1
        assert metrics.failed_parses == 1
        assert metrics.schema_validation_errors == 1

    def test_invalid_tag_value_records_metric(self):
        """Invalid tag value should record metric."""
        invalid_tag = json.dumps({
            "bullet_tags": [{"id": "strat-001", "tag": "invalid_value"}]
        })

        with pytest.raises(ReflectionParseError):
            parse_reflection(invalid_tag)

        tracker = get_tracker()
        metrics = tracker.get_metrics()
        assert metrics.schema_validation_errors == 1

    def test_invalid_section_records_metric(self):
        """Invalid section should record metric."""
        invalid_section = json.dumps({
            "candidate_bullets": [{"section": "invalid_section", "content": "test"}]
        })

        with pytest.raises(ReflectionParseError):
            parse_reflection(invalid_section)

        tracker = get_tracker()
        metrics = tracker.get_metrics()
        assert metrics.schema_validation_errors == 1


class TestValidationMetrics:
    """Test ValidationMetrics dataclass."""

    def test_to_dict(self):
        """ValidationMetrics should convert to dict properly."""
        metrics = ValidationMetrics(
            total_attempts=10,
            successful_parses=8,
            failed_parses=2,
            json_decode_errors=1,
            schema_validation_errors=1,
            error_breakdown={"JSONDecodeError": 1, "SchemaValidationError": 1},
        )

        result = metrics.to_dict()
        assert result["total_attempts"] == 10
        assert result["successful_parses"] == 8
        assert result["failed_parses"] == 2
        assert result["success_rate"] == 0.8
        assert result["json_decode_errors"] == 1
        assert result["schema_validation_errors"] == 1
        assert result["error_breakdown"] == {"JSONDecodeError": 1, "SchemaValidationError": 1}

    def test_zero_attempts_success_rate(self):
        """Success rate should be 0 when no attempts."""
        metrics = ValidationMetrics()
        assert metrics.success_rate == 0.0
