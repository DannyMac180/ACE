import logging
import pytest
from ace.utils import (
    generate_bullet_id,
    generate_trajectory_id,
    content_hash,
    minhash_jaccard,
    setup_logging,
    log_event,
)


def test_generate_bullet_id():
    assert generate_bullet_id("strategies", 91) == "strat-00091"
    assert generate_bullet_id("templates", 22) == "tmpl-00022"
    assert generate_bullet_id("troubleshooting", 7) == "trbl-00007"
    assert generate_bullet_id("code_snippets", 100) == "snip-00100"
    assert generate_bullet_id("facts", 1) == "fact-00001"
    assert generate_bullet_id("unknown", 5) == "misc-00005"


def test_generate_trajectory_id():
    traj_id = generate_trajectory_id()
    assert traj_id.startswith("traj-")
    parts = traj_id.split("-")
    assert len(parts) == 4
    assert len(parts[3]) == 8


def test_content_hash():
    text = "Prefer hybrid retrieval: BM25 + embedding"
    hash1 = content_hash(text)
    hash2 = content_hash(text)
    assert hash1 == hash2
    assert len(hash1) == 16
    
    different_text = "Different content"
    hash3 = content_hash(different_text)
    assert hash3 != hash1


def test_minhash_jaccard_identical():
    text = "This is a test sentence for minhash"
    similarity = minhash_jaccard(text, text)
    assert similarity == 1.0


def test_minhash_jaccard_similar():
    text_a = "Prefer hybrid retrieval using BM25 and embeddings"
    text_b = "Prefer hybrid retrieval with BM25 plus embeddings"
    similarity = minhash_jaccard(text_a, text_b)
    assert similarity > 0.5


def test_minhash_jaccard_different():
    text_a = "Completely different text about database"
    text_b = "Totally unrelated content about networks"
    similarity = minhash_jaccard(text_a, text_b)
    assert similarity < 0.3


def test_minhash_jaccard_empty():
    assert minhash_jaccard("", "test") == 0.0
    assert minhash_jaccard("test", "") == 0.0
    assert minhash_jaccard("", "") == 0.0


def test_setup_logging_json(caplog):
    setup_logging(level="INFO", json_format=True)
    logger = logging.getLogger("test")
    logger.info("Test message")


def test_setup_logging_text(caplog):
    setup_logging(level="DEBUG", json_format=False)
    logger = logging.getLogger("test")
    logger.debug("Debug message")


def test_log_event():
    log_event("test_event", {"key": "value", "count": 42})
