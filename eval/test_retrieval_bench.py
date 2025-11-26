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
from test_helpers import HybridStore, LightweightStore

from ace.core.retrieve import Retriever
from ace.core.schema import Bullet


@pytest.fixture
def sample_bullets() -> list[Bullet]:
    """Create a sample set of bullets for testing retrieval."""
    return [
        Bullet(
            id="strat-001",
            section="strategies_and_hard_rules",
            content="Use hybrid retrieval with BM25 and embeddings for best results",
            tags=["topic:retrieval", "stack:python"],
        ),
        Bullet(
            id="strat-002",
            section="strategies_and_hard_rules",
            content="Always validate JSON output from LLM reflections",
            tags=["topic:parsing", "robustness"],
        ),
        Bullet(
            id="tmpl-001",
            section="code_snippets_and_templates",
            content="Unit test template for merge operations",
            tags=["topic:testing", "merge"],
        ),
        Bullet(
            id="trbl-001",
            section="troubleshooting_and_pitfalls",
            content="If retrieval returns no results, check embedding model initialization",
            tags=["topic:retrieval", "debugging"],
        ),
        Bullet(
            id="code-001",
            section="code_snippets_and_templates",
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

    troubleshooting_bullets = [b for b in results if b.section == "troubleshooting_and_pitfalls"]
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


# ============================================================================
# GOLDEN TEST CASES - ACE-82
# These are hand-curated, precise test cases that verify exact retrieval
# behavior for critical scenarios. They should never regress.
# ============================================================================


@pytest.fixture
def golden_bullets() -> list[Bullet]:
    """
    Golden bullet set for precise retrieval testing.

    Covers realistic ACE scenarios: Python stack, DB ops,
    troubleshooting, curation policies, and MCP integration.
    """
    return [
        # Retrieval & DB domain
        Bullet(
            id="strat-golden-001",
            section="strategies_and_hard_rules",
            content=(
                "Use pgvector for production; FAISS for local dev. "
                "Hybrid retrieval with BM25+embedding always."
            ),
            tags=["topic:retrieval", "db:pgvector", "db:faiss", "stack:python"],
            helpful=5,
            harmful=0,
        ),
        Bullet(
            id="trbl-golden-001",
            section="troubleshooting_and_pitfalls",
            content=(
                "If pgvector extension missing: CREATE EXTENSION IF NOT EXISTS vector; "
                "then restart connection."
            ),
            tags=["topic:retrieval", "db:pgvector", "error:extension"],
            helpful=3,
            harmful=0,
        ),
        # Curation & delta policies
        Bullet(
            id="strat-golden-002",
            section="strategies_and_hard_rules",
            content=(
                "Never rewrite playbook wholesale. "
                "Only emit ADD/PATCH/DEPRECATE ops. Run refine weekly."
            ),
            tags=["topic:curation", "policy", "delta"],
            helpful=8,
            harmful=0,
        ),
        Bullet(
            id="code-golden-001",
            section="code_snippets_and_templates",
            content='delta = {"ops": [{"op":"ADD","new_bullet":{...}}]}; curator.commit(delta)',
            tags=["topic:curation", "example:delta", "stack:python"],
            helpful=4,
            harmful=0,
        ),
        # MCP & tooling
        Bullet(
            id="strat-golden-003",
            section="strategies_and_hard_rules",
            content=(
                "Expose MCP tools: ace.retrieve, ace.reflect, ace.curate, "
                "ace.commit, ace.refine. Use stdio transport."
            ),
            tags=["topic:mcp", "tool:server", "stack:python"],
            helpful=6,
            harmful=0,
        ),
        Bullet(
            id="trbl-golden-002",
            section="troubleshooting_and_pitfalls",
            content=(
                "MCP JSON parse error: strip markdown fencing (```json) "
                "before json.loads(); retry once."
            ),
            tags=["topic:mcp", "topic:parsing", "error:json"],
            helpful=7,
            harmful=0,
        ),
        # Testing & CI
        Bullet(
            id="tmpl-golden-001",
            section="code_snippets_and_templates",
            content=(
                "Integration test pattern: setup fixture → act (call API/tool) → "
                "assert expected delta/state → teardown."
            ),
            tags=["topic:testing", "pattern:integration"],
            helpful=3,
            harmful=0,
        ),
        Bullet(
            id="strat-golden-004",
            section="strategies_and_hard_rules",
            content=(
                "CI stages: lint→typecheck→unit→integration→refine-dry-run. "
                "Block merge on failures."
            ),
            tags=["topic:ci", "policy", "testing"],
            helpful=5,
            harmful=0,
        ),
        # Python stack specifics
        Bullet(
            id="code-golden-002",
            section="code_snippets_and_templates",
            content=(
                "with store.db.begin(): store.bullet_store.create_bullet(bullet); "
                "ensures atomic commit."
            ),
            tags=["stack:python", "db:sqlite", "pattern:transaction"],
            helpful=2,
            harmful=0,
        ),
        # Reflection & LLM integration
        Bullet(
            id="strat-golden-005",
            section="strategies_and_hard_rules",
            content=(
                "Reflector must emit strict JSON matching schema. "
                "No chain-of-thought. Retry on parse error."
            ),
            tags=["topic:reflection", "topic:parsing", "llm"],
            helpful=6,
            harmful=0,
        ),
    ]


@pytest.fixture
def golden_store(tmp_path, golden_bullets):
    """
    Fixture that provides a HybridStore populated with golden bullets.

    Uses real embeddings and FAISS to test the full hybrid retrieval pipeline.
    Ensures proper cleanup via yield pattern to prevent resource leaks.
    """
    db_path = str(tmp_path / "golden.db")
    store = HybridStore(db_path)

    for bullet in golden_bullets:
        store.save_bullet(bullet)

    yield store

    # Cleanup: close DB and FAISS handles
    store.close()


@pytest.mark.retrieval_regression
def test_golden_retrieval_pgvector_query(golden_store):
    """
    GOLDEN TEST 1: Query for pgvector troubleshooting should return exact bullet.

    Scenario: Developer hits missing pgvector extension error.
    Expected: trbl-golden-001 must be in top-3 results.
    """
    retriever = Retriever(golden_store)
    results = retriever.retrieve("pgvector extension missing error", top_k=3)

    result_ids = [b.id for b in results]
    assert "trbl-golden-001" in result_ids, \
        f"Expected trbl-golden-001 in top-3 for pgvector error query, got {result_ids}"


@pytest.mark.retrieval_regression
def test_golden_retrieval_delta_curation_policy(golden_store):
    """
    GOLDEN TEST 2: Query about delta policies retrieves relevant curation bullets.

    Scenario: Agent needs to understand curation delta rules.
    Expected: strat-golden-002 (delta policy) or code-golden-001 (delta example)
              must be in top-3 results.

    Note: Retriever ranks by lexical overlap only; it does not use helpful scores.
    """
    retriever = Retriever(golden_store)
    results = retriever.retrieve("curation delta policy add patch deprecate", top_k=3)

    assert len(results) > 0, "Should retrieve at least one result for delta policy query"
    result_ids = [b.id for b in results]
    assert "strat-golden-002" in result_ids or "code-golden-001" in result_ids, \
        f"Expected delta-related bullets in top-3, got {result_ids}"


@pytest.mark.retrieval_regression
def test_golden_retrieval_mcp_json_error(golden_store):
    """
    GOLDEN TEST 3: MCP JSON parse error query should retrieve specific troubleshooting.

    Scenario: MCP tool returns JSON with markdown fencing, causing parse failure.
    Expected: trbl-golden-002 must be in top-2 results.
    """
    retriever = Retriever(golden_store)
    results = retriever.retrieve("MCP JSON parse error markdown fencing", top_k=2)

    result_ids = [b.id for b in results]
    assert "trbl-golden-002" in result_ids, \
        f"Expected trbl-golden-002 in top-2 for MCP JSON error, got {result_ids}"


@pytest.mark.retrieval_regression
def test_golden_retrieval_multi_tag_filtering(golden_store):
    """
    GOLDEN TEST 4: Multi-tag query should retrieve bullets with overlapping tags.

    Scenario: Query spans multiple domains (retrieval + database).
    Expected: Results should include bullets tagged with both topic:retrieval AND db:*.
    """
    retriever = Retriever(golden_store)
    results = retriever.retrieve("retrieval database pgvector faiss", top_k=5)

    # Should retrieve strat-golden-001 and trbl-golden-001 (both have retrieval+db tags)
    result_ids = [b.id for b in results]
    assert "strat-golden-001" in result_ids or "trbl-golden-001" in result_ids, \
        f"Expected retrieval+db tagged bullets in results, got {result_ids}"

    # Check tag overlap
    retrieval_and_db_bullets = [
        b for b in results
        if any("topic:retrieval" in tag for tag in b.tags)
        and any(tag.startswith("db:") for tag in b.tags)
    ]
    assert len(retrieval_and_db_bullets) > 0, \
        "Should retrieve at least one bullet with both topic:retrieval and db:* tags"


@pytest.mark.retrieval_regression
def test_golden_retrieval_lexical_rerank_precision(golden_store):
    """
    GOLDEN TEST 5: Lexical overlap reranking should prioritize high-overlap bullets.

    Scenario: Query with specific terms that match bullet content precisely.
    Expected: Bullets with more lexical overlap rank higher.

    Note: Current retriever uses lexical overlap only (not helpful/harmful scores).
    This test validates actual reranking behavior.
    """
    retriever = Retriever(golden_store)
    # Query designed to match strat-golden-002 very well (many matching terms)
    results = retriever.retrieve("playbook wholesale ADD PATCH DEPRECATE ops refine", top_k=5)

    assert len(results) >= 1, "Should retrieve at least one result"

    # The bullet with highest lexical overlap should rank near the top
    # strat-golden-002 content: "Never rewrite playbook wholesale.
    # Only emit ADD/PATCH/DEPRECATE ops. Run refine weekly."
    result_ids = [b.id for b in results[:3]]
    assert "strat-golden-002" in result_ids, \
        f"Expected strat-golden-002 in top-3 due to lexical overlap, got {result_ids}"
