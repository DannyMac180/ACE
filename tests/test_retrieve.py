import os
import tempfile

import pytest

from ace.core.retrieve import Retriever
from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as handle:
        db_path = handle.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for FAISS indices."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def store_with_bullets(temp_db, temp_index_dir):
    """Create a store with bullets that exercise lexical reranking."""
    index_path = os.path.join(temp_index_dir, "test_index.idx")

    from ace.core.storage import embedding_store

    original_init = embedding_store.EmbeddingStore.__init__

    def patched_init(self, db_conn, index_path=index_path):
        original_init(self, db_conn, index_path)

    embedding_store.EmbeddingStore.__init__ = patched_init

    try:
        store = Store(temp_db)
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
        ]

        for bullet in bullets:
            store.save_bullet(bullet)

        yield store
        store.close()
    finally:
        embedding_store.EmbeddingStore.__init__ = original_init


def test_retrieve_normalizes_punctuation_for_lexical_reranking(store_with_bullets):
    retriever = Retriever(store_with_bullets)

    bullets = retriever.retrieve("retrieval application", top_k=2)

    assert [bullet.id for bullet in bullets][:1] == ["strat-001"]


def test_retrieve_normalizes_tag_tokens_for_overlap(store_with_bullets):
    retriever = Retriever(store_with_bullets)

    bullets = retriever.retrieve("python retrieval", top_k=1)

    assert [bullet.id for bullet in bullets] == ["strat-001"]
