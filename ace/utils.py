import hashlib
import json
import logging
import secrets
import sys
from datetime import UTC, datetime
from typing import Any


def generate_bullet_id(section: str, counter: int) -> str:
    """Generate stable bullet ID in format: {section_prefix}-{counter:05d}"""
    prefix_map = {
        "strategies": "strat",
        "templates": "tmpl",
        "troubleshooting": "trbl",
        "code_snippets": "snip",
        "facts": "fact",
    }
    prefix = prefix_map.get(section, "misc")
    return f"{prefix}-{counter:05d}"


def generate_trajectory_id() -> str:
    """Generate unique trajectory ID using timestamp + random suffix"""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    rand = secrets.token_hex(4)
    return f"traj-{ts}-{rand}"


def content_hash(text: str) -> str:
    """Generate stable SHA-256 hash of text content"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def minhash_jaccard(text_a: str, text_b: str, num_hashes: int = 128) -> float:
    """Estimate Jaccard similarity using MinHash with n-grams"""
    def get_ngrams(text: str, n: int = 3) -> set:
        words = text.lower().split()
        ngrams = set()
        for word in words:
            for i in range(len(word) - n + 1):
                ngrams.add(word[i:i+n])
        return ngrams

    ngrams_a = get_ngrams(text_a)
    ngrams_b = get_ngrams(text_b)

    if not ngrams_a or not ngrams_b:
        return 0.0

    signature_a = []
    signature_b = []

    for seed in range(num_hashes):
        min_a = min((hash((ng, seed)) for ng in ngrams_a), default=sys.maxsize)
        min_b = min((hash((ng, seed)) for ng in ngrams_b), default=sys.maxsize)
        signature_a.append(min_a)
        signature_b.append(min_b)

    matches = sum(1 for a, b in zip(signature_a, signature_b, strict=True) if a == b)
    return matches / num_hashes


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure structured JSON logging"""
    log_level = getattr(logging, level.upper(), logging.INFO)

    if json_format:
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_obj = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)
                if hasattr(record, "extra"):
                    log_obj.update(record.extra)
                return json.dumps(log_obj)

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(log_level)


def log_event(event_type: str, data: dict[str, Any]) -> None:
    """Log structured event with metadata"""
    logger = logging.getLogger("ace.events")
    extra_data = {"event_type": event_type, **data}
    logger.info(event_type, extra=extra_data)
