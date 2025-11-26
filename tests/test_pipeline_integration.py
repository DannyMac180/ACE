# tests/test_pipeline_integration.py
"""
Full pipeline integration tests for ACE.
Tests the complete flow: retrieve → reflect → curate → commit
"""
import os
import tempfile

import pytest

from ace.core.manager import PlaybookManager
from ace.core.retrieve import Retriever
from ace.core.schema import Bullet, Delta, DeltaOp
from ace.core.storage.store_adapter import Store
from ace.curator import curate
from ace.reflector import BulletTag, CandidateBullet, Reflection


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for FAISS indices."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up temp directory and its contents
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def store_with_bullets(temp_db, temp_index_dir):
    """Create a store with some seed bullets."""
    index_path = os.path.join(temp_index_dir, "test_index.idx")

    # Patch the EmbeddingStore to use temp index path
    from ace.core.storage import embedding_store
    original_init = embedding_store.EmbeddingStore.__init__

    def patched_init(self, db_conn, index_path=index_path):
        original_init(self, db_conn, index_path)

    embedding_store.EmbeddingStore.__init__ = patched_init

    try:
        store = Store(temp_db)

        # Add seed bullets
        bullets = [
            Bullet(
                id="strat-001",
                section="strategies_and_hard_rules",
                content="Prefer hybrid retrieval: BM25 + embedding; rerank by lexical overlap",
                tags=["topic:retrieval", "stack:python"],
            ),
            Bullet(
                id="strat-002",
                section="strategies_and_hard_rules",
                content="Never rewrite the whole playbook. Only ADD/PATCH/DEPRECATE bullets",
                tags=["topic:curation", "policy"],
            ),
            Bullet(
                id="trbl-001",
                section="troubleshooting_and_pitfalls",
                content="Check FAISS index dimension mismatch if insertions fail",
                tags=["topic:vector", "tool:faiss"],
            ),
            Bullet(
                id="tmpl-001",
                section="code_snippets_and_templates",
                content="Unit test template: apply Delta ops and assert version increment",
                tags=["topic:testing"],
            ),
        ]

        for bullet in bullets:
            store.save_bullet(bullet)

        yield store

        # Properly close the store
        store.close()
    finally:
        # Restore original init
        embedding_store.EmbeddingStore.__init__ = original_init


def _find_bullet_by_id(bullets: list[Bullet], bullet_id: str) -> Bullet | None:
    """Helper to find a bullet by ID in a list of bullets."""
    for bullet in bullets:
        if bullet.id == bullet_id:
            return bullet
    return None


def test_full_pipeline_with_helpful_feedback(store_with_bullets):
    """Test complete pipeline: retrieve → reflect → curate → commit with helpful feedback."""

    # Step 1: Retrieve relevant bullets for a query
    retriever = Retriever(store_with_bullets)
    query = "retrieval application"
    retrieved_bullets = retriever.retrieve(query, top_k=2)

    assert len(retrieved_bullets) > 0
    # Verify retrieval found the relevant bullet
    assert any(b.id == "strat-001" for b in retrieved_bullets)

    # Step 2: Create a reflection marking a bullet as helpful
    reflection = Reflection(
        error_identification=None,
        root_cause_analysis=None,
        correct_approach="Use hybrid retrieval approach",
        key_insight="Hybrid retrieval improves accuracy",
        bullet_tags=[
            BulletTag(id="strat-001", tag="helpful")
        ],
        candidate_bullets=[],
    )

    # Step 3: Curate the reflection into a delta
    delta = curate(reflection)

    assert len(delta.ops) == 1
    assert delta.ops[0].op == "INCR_HELPFUL"
    assert delta.ops[0].target_id == "strat-001"

    # Step 4: Commit the delta using PlaybookManager which persists to Store
    initial_version = store_with_bullets.get_version()

    # Apply delta operations through PlaybookManager
    manager = PlaybookManager()
    playbook = store_with_bullets.load_playbook()
    manager.playbook = playbook

    for op in delta.ops:
        manager.apply_delta(op)

    # Save back to store
    store_with_bullets.set_version(manager.playbook.version)
    for bullet in manager.playbook.bullets:
        store_with_bullets.save_bullet(bullet)

    # Verify playbook was updated
    assert manager.playbook.version == initial_version + 1

    # Verify the helpful counter was incremented via public API
    updated_bullet = _find_bullet_by_id(manager.playbook.bullets, "strat-001")
    assert updated_bullet is not None
    assert updated_bullet.helpful == 1


def test_full_pipeline_with_new_bullet(store_with_bullets):
    """Test complete pipeline: retrieve → reflect → curate → commit with new bullet."""

    # Step 1: Retrieve bullets for context and verify retrieval works
    retriever = Retriever(store_with_bullets)
    query = "debug FAISS issues"
    assert len(retriever.retrieve(query, top_k=2)) > 0

    # Step 2: Create a reflection with a new candidate bullet
    reflection = Reflection(
        error_identification="FAISS dimension mismatch error",
        root_cause_analysis="Embedding model changed from 384 to 768 dims",
        correct_approach="Rebuild index with correct dimensions",
        key_insight="Always validate embedding dimensions match index",
        bullet_tags=[
            BulletTag(id="trbl-001", tag="helpful")
        ],
        candidate_bullets=[
            CandidateBullet(
                section="troubleshooting_and_pitfalls",
                content="Validate embedding dims match FAISS index dims before insertion",
                tags=["topic:vector", "tool:faiss", "error:dimension"],
            )
        ],
    )

    # Step 3: Curate the reflection into a delta
    delta = curate(reflection)

    assert len(delta.ops) == 2  # One INCR_HELPFUL, one ADD

    # Step 4: Commit using PlaybookManager
    manager = PlaybookManager()
    playbook = store_with_bullets.load_playbook()
    manager.playbook = playbook
    initial_count = len(manager.playbook.bullets)
    initial_version = manager.playbook.version

    for op in delta.ops:
        manager.apply_delta(op)

    # Save back to store
    store_with_bullets.set_version(manager.playbook.version)
    for bullet in manager.playbook.bullets:
        store_with_bullets.save_bullet(bullet)

    # Verify playbook was updated
    assert manager.playbook.version == initial_version + 2
    assert len(manager.playbook.bullets) == initial_count + 1

    # Verify the new bullet was added
    new_bullets = [b for b in manager.playbook.bullets if "Validate embedding dims" in b.content]
    assert len(new_bullets) == 1
    assert new_bullets[0].section == "troubleshooting_and_pitfalls"
    assert "topic:vector" in new_bullets[0].tags


def test_full_pipeline_with_harmful_feedback(store_with_bullets):
    """Test complete pipeline with harmful feedback leading to potential deprecation."""

    # Step 1: Verify retrieval works (don't need the results)
    retriever = Retriever(store_with_bullets)
    assert len(retriever.retrieve("testing strategies", top_k=2)) > 0

    # Step 2: Create reflection marking a bullet as harmful
    reflection = Reflection(
        error_identification="Test template was misleading",
        root_cause_analysis="Template didn't account for async tests",
        correct_approach="Use async-aware test patterns",
        key_insight="Templates must cover async scenarios",
        bullet_tags=[
            BulletTag(id="tmpl-001", tag="harmful")
        ],
        candidate_bullets=[
            CandidateBullet(
                section="code_snippets_and_templates",
                content="For async code, use pytest-asyncio and async def test_* patterns",
                tags=["topic:testing", "lang:python", "pattern:async"],
            )
        ],
    )

    # Step 3: Curate
    delta = curate(reflection)

    assert len(delta.ops) == 2  # INCR_HARMFUL + ADD

    # Step 4: Commit using PlaybookManager
    manager = PlaybookManager()
    playbook = store_with_bullets.load_playbook()
    manager.playbook = playbook

    for op in delta.ops:
        manager.apply_delta(op)

    # Save back to store
    store_with_bullets.set_version(manager.playbook.version)
    for bullet in manager.playbook.bullets:
        store_with_bullets.save_bullet(bullet)

    # Verify harmful counter was incremented
    harmful_bullet = _find_bullet_by_id(manager.playbook.bullets, "tmpl-001")
    assert harmful_bullet is not None
    assert harmful_bullet.harmful == 1

    # Verify new bullet was added as replacement
    new_bullets = [b for b in manager.playbook.bullets if "pytest-asyncio" in b.content]
    assert len(new_bullets) == 1


def test_full_pipeline_multiple_iterations(store_with_bullets):
    """Test multiple iterations of the pipeline showing evolution."""

    manager = PlaybookManager()
    manager.playbook = store_with_bullets.load_playbook()
    initial_version = manager.playbook.version

    # Iteration 1: Mark helpful and add new bullet
    reflection1 = Reflection(
        bullet_tags=[BulletTag(id="strat-001", tag="helpful")],
        candidate_bullets=[
            CandidateBullet(
                section="strategies_and_hard_rules",
                content="Use reranking to improve retrieval precision",
                tags=["topic:retrieval", "technique:rerank"],
            )
        ],
    )

    delta1 = curate(reflection1)
    for op in delta1.ops:
        manager.apply_delta(op)

    # Iteration 2: Mark another as helpful
    reflection2 = Reflection(
        bullet_tags=[BulletTag(id="strat-002", tag="helpful")],
        candidate_bullets=[],
    )

    delta2 = curate(reflection2)
    for op in delta2.ops:
        manager.apply_delta(op)

    # Save back to store
    store_with_bullets.set_version(manager.playbook.version)
    for bullet in manager.playbook.bullets:
        store_with_bullets.save_bullet(bullet)

    # Verify multiple iterations updated the playbook
    assert manager.playbook.version == initial_version + 3  # 2 ops from iter1, 1 from iter2
    assert len(manager.playbook.bullets) == 5  # 4 original + 1 new

    # Verify counters using public API
    bullet1 = _find_bullet_by_id(manager.playbook.bullets, "strat-001")
    bullet2 = _find_bullet_by_id(manager.playbook.bullets, "strat-002")
    assert bullet1 is not None and bullet1.helpful == 1
    assert bullet2 is not None and bullet2.helpful == 1


def test_pipeline_error_handling_missing_bullet(temp_db, temp_index_dir):
    """Test pipeline handles errors when referencing non-existent bullets."""

    index_path = os.path.join(temp_index_dir, "test_error.idx")

    from ace.core.storage import embedding_store
    original_init = embedding_store.EmbeddingStore.__init__

    def patched_init(self, db_conn, index_path=index_path):
        original_init(self, db_conn, index_path)

    embedding_store.EmbeddingStore.__init__ = patched_init

    try:
        store = Store(temp_db)
        store.save_bullet(
            Bullet(id="test-001", section="strategies_and_hard_rules", content="test", tags=[])
        )

        playbook = store.load_playbook()

        # Try to increment helpful on non-existent bullet
        reflection = Reflection(
            bullet_tags=[BulletTag(id="nonexistent-999", tag="helpful")],
            candidate_bullets=[],
        )

        delta = curate(reflection)

        # Apply through PlaybookManager - non-existent bullet will raise ValueError
        manager = PlaybookManager()
        manager.playbook = playbook

        # The manager should raise an error for non-existent bullet
        with pytest.raises(ValueError, match="Bullet not found"):
            for op in delta.ops:
                manager.apply_delta(op)

        store.close()
    finally:
        embedding_store.EmbeddingStore.__init__ = original_init


def test_pipeline_with_patch_operation(store_with_bullets):
    """Test pipeline with PATCH operation to update bullet content."""

    manager = PlaybookManager()
    manager.playbook = store_with_bullets.load_playbook()

    # Manually create a PATCH delta (curator doesn't generate these yet in current impl)
    delta = Delta(ops=[
        DeltaOp(
            op="PATCH",
            target_id="strat-001",
            patch="Prefer hybrid retrieval: BM25 + vector embedding; use reranking for precision",
        )
    ])

    original_bullet = _find_bullet_by_id(manager.playbook.bullets, "strat-001")
    assert original_bullet is not None
    original_content = original_bullet.content

    for op in delta.ops:
        manager.apply_delta(op)

    # Save back to store
    store_with_bullets.set_version(manager.playbook.version)
    for bullet in manager.playbook.bullets:
        store_with_bullets.save_bullet(bullet)

    # Verify content was patched
    patched_bullet = _find_bullet_by_id(manager.playbook.bullets, "strat-001")
    assert patched_bullet is not None
    assert patched_bullet.content != original_content
    assert "reranking for precision" in patched_bullet.content
