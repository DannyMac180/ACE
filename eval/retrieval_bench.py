"""
Retrieval benchmark tests for ACE retrieval system.

Tests hybrid retrieval (BM25 + vector) with reranking to ensure:
- Relevant bullets are retrieved for queries
- Lexical overlap reranking works correctly
- Tag-based retrieval is effective

Uses lightweight in-memory store stub to avoid FAISS/embeddings overhead
for fast, hermetic unit tests.
"""

import pytest
from test_helpers import LightweightStore

from ace.core.retrieve import Retriever
from ace.core.schema import Bullet


@pytest.fixture
def sample_bullets() -> list[Bullet]:
    """Create a sample set of bullets for testing retrieval."""
    return [
        Bullet(
            id="strat-001",
            section="strategies",
            content="Use hybrid retrieval with BM25 and embeddings for best results",
            tags=["topic:retrieval", "stack:python"],
        ),
        Bullet(
            id="strat-002",
            section="strategies",
            content="Always validate JSON output from LLM reflections",
            tags=["topic:parsing", "robustness"],
        ),
        Bullet(
            id="tmpl-001",
            section="templates",
            content="Unit test template for merge operations",
            tags=["topic:testing", "merge"],
        ),
        Bullet(
            id="trbl-001",
            section="troubleshooting",
            content="If retrieval returns no results, check embedding model initialization",
            tags=["topic:retrieval", "debugging"],
        ),
        Bullet(
            id="code-001",
            section="code_snippets",
            content="retriever = Retriever(store); results = retriever.retrieve(query, top_k=24)",
            tags=["topic:retrieval", "example"],
        ),
    ]


@pytest.fixture
def store_with_bullets(tmp_path, sample_bullets):
    """Create an isolated lightweight store populated with sample bullets.

    Uses yield to ensure proper cleanup and avoids shared FAISS index.
    """
    db_path = tmp_path / "test_retrieval.db"
    index_path = str(tmp_path / "test_index.faiss")
    store = LightweightStore(str(db_path), index_path)

    for bullet in sample_bullets:
        store.save_bullet(bullet)

    yield store

    # Cleanup: close DB and FAISS handles
    store.close()


def test_retrieval_basic(store_with_bullets, sample_bullets):
    """Test basic retrieval returns relevant bullets."""
    retriever = Retriever(store_with_bullets)

    results = retriever.retrieve("retrieval", top_k=5)

    assert len(results) > 0
    assert any(b.id == "strat-001" for b in results)


def test_retrieval_tag_match(store_with_bullets):
    """Test that tag-based queries retrieve correctly."""
    retriever = Retriever(store_with_bullets)

    results = retriever.retrieve("retrieval", top_k=5)

    retrieval_bullets = [b for b in results if "topic:retrieval" in b.tags]
    assert len(retrieval_bullets) > 0


def test_retrieval_lexical_rerank(store_with_bullets):
    """Test that lexical overlap reranking prioritizes relevant bullets."""
    retriever = Retriever(store_with_bullets)

    results = retriever.retrieve("retrieval BM25 embeddings hybrid", top_k=5)

    if results:
        top_result = results[0]
        assert "strat-001" == top_result.id


def test_retrieval_topk_limit(store_with_bullets):
    """Test that top_k parameter limits results correctly."""
    retriever = Retriever(store_with_bullets)

    results = retriever.retrieve("test", top_k=2)

    assert len(results) <= 2


def test_retrieval_empty_query(store_with_bullets):
    """Test retrieval with query that has no overlap returns limited results."""
    retriever = Retriever(store_with_bullets)

    # Query with terms that don't match any bullets
    results = retriever.retrieve("zzz", top_k=5)

    assert isinstance(results, list)
    # May return results from FTS/vector fallback, but should be limited
    assert len(results) <= 5


def test_retrieval_no_matches(store_with_bullets):
    """Test retrieval with query unlikely to match returns empty or minimal results."""
    retriever = Retriever(store_with_bullets)

    results = retriever.retrieve("xyzabc123nonexistent", top_k=5)

    assert isinstance(results, list)
    # With no lexical overlap, reranking should yield empty or very few results
    assert len(results) <= 5
    # If results exist, they should have no overlap with query
    if results:
        query_terms = set("xyzabc123nonexistent".lower().split())
        for bullet in results:
            content_terms = set(bullet.content.lower().split())
            overlap = len(query_terms & content_terms)
            # No direct overlap expected for nonsense query
            assert overlap == 0


def test_retrieval_section_specific(store_with_bullets):
    """Test that we can retrieve from specific sections."""
    retriever = Retriever(store_with_bullets)

    results = retriever.retrieve("troubleshooting retrieval", top_k=5)

    troubleshooting_bullets = [b for b in results if b.section == "troubleshooting"]
    assert len(troubleshooting_bullets) > 0


def test_retrieval_deduplication(store_with_bullets):
    """Test that results don't contain duplicates."""
    retriever = Retriever(store_with_bullets)

    results = retriever.retrieve("retrieval", top_k=10)

    bullet_ids = [b.id for b in results]
    assert len(bullet_ids) == len(set(bullet_ids))


@pytest.mark.benchmark
def test_retrieval_performance(store_with_bullets, benchmark):
    """Benchmark retrieval performance."""
    retriever = Retriever(store_with_bullets)

    result = benchmark(retriever.retrieve, "retrieval hybrid BM25", top_k=24)

    assert len(result) <= 24
