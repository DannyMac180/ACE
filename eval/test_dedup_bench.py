"""
Dedup quality benchmark for ACE refinement system.

Tests the deduplication logic (Cosine + MinHash) to ensure:
- Exact duplicates are caught
- Near-duplicates (semantically similar) are caught
- Distinct bullets are preserved
- Thresholds are effective
"""

import hashlib
from unittest.mock import patch

import numpy as np
import pytest

from ace.core.schema import Bullet, Delta, DeltaOp, Playbook
from ace.refine.runner import RefineRunner


def deterministic_embedding(text: str) -> np.ndarray:
    """
    Generate a deterministic fake embedding based on text content.
    Returns a 384-dim vector (same as MiniLM) derived from the hash.
    """
    # Create a deterministic seed from the text
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    # Generate random vector
    vector = rng.random(384).astype(np.float32)
    # Normalize
    return vector / np.linalg.norm(vector)


@pytest.fixture
def mock_embedding():
    """Patch the generate_embedding function to use deterministic fake embeddings."""
    with patch("ace.refine.runner.generate_embedding", side_effect=deterministic_embedding) as mock:
        yield mock


@pytest.fixture
def base_playbook() -> Playbook:
    """Create a base playbook with some canonical bullets."""
    bullets = [
        Bullet(
            id="strat-001",
            section="strategies",
            content="Use hybrid retrieval with BM25 and embeddings for best results",
            tags=["topic:retrieval", "stack:python"],
        ),
        Bullet(
            id="strat-002",
            section="strategies",
            content="Always validate JSON output from LLM reflections to prevent parse errors",
            tags=["topic:parsing", "robustness"],
        ),
        Bullet(
            id="tmpl-001",
            section="templates",
            content=(
                "Unit test template for merge operations: apply Delta and assert version increment"
            ),
            tags=["topic:testing", "merge"],
        ),
    ]
    return Playbook(version=1, bullets=bullets)


@pytest.fixture
def runner(base_playbook) -> RefineRunner:
    return RefineRunner(playbook=base_playbook, threshold=0.90)


def test_dedup_exact_match(runner, mock_embedding):
    """Test that exact text matches are identified as duplicates."""
    # Candidate is identical to strat-001
    candidate = {
        "section": "strategies",
        "content": "Use hybrid retrieval with BM25 and embeddings for best results",
        "tags": ["topic:retrieval"],
    }

    op = DeltaOp(op="ADD", new_bullet=candidate)
    delta = Delta(ops=[op])

    merge_ops = runner.deduplicate(delta)

    assert len(merge_ops) == 1
    assert merge_ops[0].op == "MERGE"
    assert merge_ops[0].survivor_id == "strat-001"
    # Verify target_ids
    assert merge_ops[0].target_ids == ["candidate-0"]


def test_dedup_near_match_cosine(runner, mock_embedding):
    """Test that semantically similar text is identified (Cosine)."""
    # For the mock embedding (seeded by hash), we need a text that produces a
    # similar vector. Since hash is chaotic, it's hard to find a "similar" text
    # that generates a similar hash-seeded vector without brute forcing.
    # Instead, we can patch generate_embedding to return a close vector for a specific input.

    original_text = runner.playbook.bullets[0].content
    original_vec = deterministic_embedding(original_text)

    # Create a vector that is 0.95 similar to original
    # v2 = 0.95 * v1 + sqrt(1-0.95^2) * random_orthogonal
    # But simpler: just patch the side effect to return a specific vector for the candidate

    candidate_text = (
        "For optimal results, utilize hybrid retrieval combining BM25 and vector embeddings"
    )

    # Create a "similar" vector manually
    rng = np.random.default_rng(42)
    noise = rng.random(384).astype(np.float32) * 0.05  # Small noise
    similar_vec = original_vec + noise
    similar_vec = similar_vec / np.linalg.norm(similar_vec)

    def side_effect(text):
        if text == candidate_text:
            return similar_vec
        return deterministic_embedding(text)

    with patch("ace.refine.runner.generate_embedding", side_effect=side_effect):
        candidate = {
            "section": "strategies",
            "content": candidate_text,
            "tags": ["topic:retrieval"],
        }

        op = DeltaOp(op="ADD", new_bullet=candidate)
        delta = Delta(ops=[op])

        merge_ops = runner.deduplicate(delta)

        assert len(merge_ops) == 1
        assert merge_ops[0].op == "MERGE"
        assert merge_ops[0].survivor_id == "strat-001"
        assert merge_ops[0].target_ids == ["candidate-0"]


def test_dedup_near_match_minhash(runner, mock_embedding):
    """Test that text with high word overlap is identified (MinHash)."""
    # Candidate is similar to tmpl-001 (word reordering/minor change)
    # MinHash doesn't use embeddings, so deterministic_embedding mock is fine
    # (it just prevents heavy model load).
    candidate = {
        "section": "templates",
        "content": (
            "Unit test template for merge operations: assert version increment and apply Delta"
        ),
        "tags": ["topic:testing"],
    }

    op = DeltaOp(op="ADD", new_bullet=candidate)
    delta = Delta(ops=[op])

    merge_ops = runner.deduplicate(delta)

    assert len(merge_ops) == 1
    assert merge_ops[0].op == "MERGE"
    assert merge_ops[0].survivor_id == "tmpl-001"
    assert merge_ops[0].target_ids == ["candidate-0"]


def test_dedup_distinct(runner, mock_embedding):
    """Test that distinct content is NOT merged."""
    candidate = {
        "section": "strategies",
        "content": "Something completely different about deployment pipelines",
        "tags": ["topic:deployment"],
    }

    op = DeltaOp(op="ADD", new_bullet=candidate)
    delta = Delta(ops=[op])

    merge_ops = runner.deduplicate(delta)

    assert len(merge_ops) == 0


def test_dedup_near_threshold_mismatch(runner, mock_embedding):
    """Test that content just below similarity threshold is NOT merged."""
    # Need to craft two vectors with similarity slightly < 0.90
    # Let's use a dedicated test setup for precise control

    original_text = runner.playbook.bullets[0].content
    original_vec = deterministic_embedding(original_text)

    # Construct vector with similarity ~0.85
    # If we take v2 = cos(theta)*v1 + sin(theta)*v_orth
    # cos(theta) = 0.85
    rng = np.random.default_rng(123)
    random_vec = rng.random(384).astype(np.float32)
    # Make orthogonal to original
    v_orth = random_vec - np.dot(random_vec, original_vec) * original_vec
    v_orth = v_orth / np.linalg.norm(v_orth)

    target_sim = 0.85
    candidate_vec = target_sim * original_vec + np.sqrt(1 - target_sim**2) * v_orth

    candidate_text = "Somewhat related but distinct retrieval strategy"

    def side_effect(text):
        if text == candidate_text:
            return candidate_vec
        return deterministic_embedding(text)

    with patch("ace.refine.runner.generate_embedding", side_effect=side_effect):
        candidate = {
            "section": "strategies",
            "content": candidate_text,
            "tags": ["topic:retrieval"],
        }

        op = DeltaOp(op="ADD", new_bullet=candidate)
        delta = Delta(ops=[op])

        merge_ops = runner.deduplicate(delta)

        # Should NOT match because cosine < 0.90
        # Also need to ensure MinHash doesn't match (text is very different)
        assert len(merge_ops) == 0


@pytest.mark.benchmark
def test_dedup_performance(benchmark, base_playbook, mock_embedding):
    """Benchmark deduplication performance (with mocked embeddings)."""
    runner = RefineRunner(playbook=base_playbook)

    candidate = {
        "section": "strategies",
        "content": "Use hybrid retrieval with BM25 and embeddings for best results",
        "tags": ["topic:retrieval"],
    }
    op = DeltaOp(op="ADD", new_bullet=candidate)
    delta = Delta(ops=[op])

    # Benchmark the deduplicate method
    result = benchmark(runner.deduplicate, delta)

    assert len(result) == 1
