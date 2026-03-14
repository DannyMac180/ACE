"""Logging helpers with secret redaction for ACE."""

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_JSON_SECRET_RE = re.compile(
    r'("(?:(?:[A-Za-z0-9_]*_)?(?:api_key|access_token|refresh_token|auth_token|secret|password|token))"\s*:\s*")([^"]+)(")',
    re.IGNORECASE,
)
_KV_SECRET_RE = re.compile(
    r"\b([A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN|AUTH[_-]?TOKEN|SECRET|PASSWORD))\b(\s*[:=]\s*)([^\s,;]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\b(authorization\s*[:=]\s*bearer\s+)([^\s,;]+)", re.IGNORECASE)
_URL_PASSWORD_RE = re.compile(
    r"([a-z][a-z0-9+.-]*://[^:/\s]+:)([^@/\s]+)(@)",
    re.IGNORECASE,
)
_SECRET_PREFIX_RE = re.compile(r"\b(sk(?:-or-v1)?-[A-Za-z0-9_-]+)\b")


def _redact_string(value: str) -> str:
    redacted = _JSON_SECRET_RE.sub(rf"\1{REDACTED}\3", value)
    redacted = _KV_SECRET_RE.sub(rf"\1\2{REDACTED}", redacted)
    redacted = _BEARER_RE.sub(rf"\1{REDACTED}", redacted)
    redacted = _URL_PASSWORD_RE.sub(rf"\1{REDACTED}\3", redacted)
    return _SECRET_PREFIX_RE.sub(REDACTED, redacted)


def redact_secrets(value: Any) -> Any:
    """Recursively redact common secrets from loggable values."""
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        return {key: redact_secrets(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, set):
        return {redact_secrets(item) for item in value}
    return value


class SecretRedactionFilter(logging.Filter):
    """Log filter that redacts common secrets before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            try:
                record.msg = redact_secrets(str(record.msg) % record.args)
                record.args = ()
            except (TypeError, ValueError):
                record.msg = redact_secrets(record.msg)
                record.args = redact_secrets(record.args)
        else:
            record.msg = redact_secrets(record.msg)
        return True


class ACEJsonFormatter(logging.Formatter):
    """Minimal JSON formatter for ACE entrypoints."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _has_redaction_filter(handler: logging.Handler) -> bool:
    return any(isinstance(log_filter, SecretRedactionFilter) for log_filter in handler.filters)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure root logging and install secret redaction."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler())

    formatter: logging.Formatter
    if fmt.lower() == "json":
        formatter = ACEJsonFormatter()
    else:
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")

    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        if not _has_redaction_filter(handler):
            handler.addFilter(SecretRedactionFilter())
