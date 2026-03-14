"""Tests for logging secret redaction."""

import io
import json
import logging

from ace.core.logging_utils import (
    REDACTED,
    SecretRedactionFilter,
    configure_logging,
    redact_secrets,
)


def test_redact_secrets_handles_nested_values():
    payload = {
        "api_key": "sk-secret-value",
        "nested": [
            "Authorization: Bearer jwt-token-value",
            "OPENROUTER_API_KEY=sk-or-v1-secret-value",
            "postgres://ace_user:super-secret@db.example.com/ace",
            '{"token":"secret-token"}',
        ],
    }

    redacted = redact_secrets(payload)
    serialized = json.dumps(redacted)

    assert "sk-secret-value" not in serialized
    assert "jwt-token-value" not in serialized
    assert "super-secret" not in serialized
    assert "secret-token" not in serialized
    assert redacted["api_key"] == REDACTED
    assert redacted["nested"][0] == f"Authorization: Bearer {REDACTED}"
    assert redacted["nested"][1] == f"OPENROUTER_API_KEY={REDACTED}"
    assert redacted["nested"][2] == f"postgres://ace_user:{REDACTED}@db.example.com/ace"
    assert redacted["nested"][3] == f'{{"token":"{REDACTED}"}}'


def test_secret_redaction_filter_redacts_message_args():
    record = logging.LogRecord(
        name="ace.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Authorization: Bearer %s db=%s",
        args=("jwt-token-value", "postgres://ace_user:super-secret@db.example.com/ace"),
        exc_info=None,
    )

    log_filter = SecretRedactionFilter()
    assert log_filter.filter(record) is True

    rendered = record.getMessage()

    assert "jwt-token-value" not in rendered
    assert "super-secret" not in rendered
    assert f"Authorization: Bearer {REDACTED}" in rendered
    assert f"db=postgres://ace_user:{REDACTED}@db.example.com/ace" in rendered


def test_configure_logging_redacts_output_and_does_not_duplicate_filters():
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    try:
        root_logger.handlers = [handler]
        configure_logging(level="INFO", fmt="json")
        configure_logging(level="INFO", fmt="json")

        root_logger.info("OPENROUTER_API_KEY=%s", "sk-or-v1-secret-value")

        output = stream.getvalue()
        assert "sk-or-v1-secret-value" not in output
        assert REDACTED in output
        assert output.strip().startswith("{")
        assert sum(
            isinstance(log_filter, SecretRedactionFilter)
            for log_filter in handler.filters
        ) == 1
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
