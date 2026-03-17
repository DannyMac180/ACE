import pickle
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from ace.core.schema import Bullet
from ace.core.storage.bullet_store import BulletStore
from ace.core.storage.db import DatabaseConnection, init_schema
from ace.core.storage.embedding_store import EmbeddingStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mock_embedding(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    vector = np.zeros(384, dtype=np.float32)
    vector[sum(text.encode("utf-8")) % 384] = 1.0
    return vector


def test_mock_embeddings_preserve_near_duplicate_similarity(monkeypatch):
    monkeypatch.setenv("ACE_EMBEDDINGS", "mock")

    from ace.core.storage.embedding_store import generate_embedding

    existing = "Always validate input before processing"
    near_duplicate = "Validate all inputs before any processing"
    unrelated = "Check database connection timeout settings"

    existing_vec = generate_embedding(existing)
    near_duplicate_vec = generate_embedding(near_duplicate)
    unrelated_vec = generate_embedding(unrelated)

    near_duplicate_similarity = float(existing_vec @ near_duplicate_vec)
    unrelated_similarity = float(existing_vec @ unrelated_vec)

    assert near_duplicate_similarity > 0.90
    assert unrelated_similarity < near_duplicate_similarity


def test_mock_embeddings_do_not_import_sentence_transformers():
    script = textwrap.dedent(
        """
        import builtins
        import os

        os.environ["ACE_EMBEDDINGS"] = "mock"

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sentence_transformers":
                raise RuntimeError(
                    "sentence_transformers import should stay lazy for mock embeddings"
                )
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        from ace.core.storage.embedding_store import generate_embedding

        vector = generate_embedding("mock retrieval query")
        print(vector.shape[0])
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "384"


def test_sqlite_embedding_store_persists_index_updates_across_reopen(monkeypatch, tmp_path):
    monkeypatch.setattr("ace.core.storage.embedding_store.generate_embedding", _mock_embedding)

    db_url = f"sqlite:///{tmp_path / 'ace.db'}"
    index_path = tmp_path / "embeddings.faiss"

    db = DatabaseConnection(db_url)
    db.connect()
    init_schema(db)
    bullet_store = BulletStore(db)
    bullet_store.create_bullet(
        Bullet(
            id="b1",
            section="strategies_and_hard_rules",
            content="first retrieval bullet",
            added_at=datetime.now(UTC),
        )
    )
    bullet_store.create_bullet(
        Bullet(
            id="b2",
            section="strategies_and_hard_rules",
            content="second retrieval bullet",
            added_at=datetime.now(UTC),
        )
    )

    store = EmbeddingStore(db, index_path=str(index_path))
    store.add_embedding("b1", "alpha query")
    store.add_embedding("b2", "beta query")

    assert index_path.exists()
    assert index_path.with_suffix(index_path.suffix + ".mapping").exists()

    reopened_db = DatabaseConnection(db_url)
    reopened_db.connect()
    init_schema(reopened_db)
    reopened = EmbeddingStore(reopened_db, index_path=str(index_path))
    assert reopened.search("alpha query", top_k=2) == ["b1", "b2"]

    reopened.remove_embedding("b1")

    final_db = DatabaseConnection(db_url)
    final_db.connect()
    init_schema(final_db)
    final_store = EmbeddingStore(final_db, index_path=str(index_path))
    assert final_store.search("alpha query", top_k=2) == ["b2"]


def test_sqlite_embedding_store_removes_embedding_without_rebuild(monkeypatch, tmp_path):
    monkeypatch.setattr("ace.core.storage.embedding_store.generate_embedding", _mock_embedding)

    db_url = f"sqlite:///{tmp_path / 'ace.db'}"
    index_path = tmp_path / "embeddings.faiss"

    db = DatabaseConnection(db_url)
    db.connect()
    init_schema(db)
    bullet_store = BulletStore(db)
    for bullet_id, content in (
        ("b1", "first retrieval bullet"),
        ("b2", "second retrieval bullet"),
    ):
        bullet_store.create_bullet(
            Bullet(
                id=bullet_id,
                section="strategies_and_hard_rules",
                content=content,
                added_at=datetime.now(UTC),
            )
        )

    store = EmbeddingStore(db, index_path=str(index_path))
    store.add_embedding("b1", "alpha query")
    store.add_embedding("b2", "beta query")

    original_rebuild_index = store.rebuild_index

    def fail_rebuild() -> None:
        raise AssertionError("remove_embedding should not rebuild the full SQLite index")

    monkeypatch.setattr(store, "rebuild_index", fail_rebuild)
    store.remove_embedding("b1")

    assert store.search("alpha query", top_k=2) == ["b2"]

    monkeypatch.setattr(store, "rebuild_index", original_rebuild_index)
    reopened_db = DatabaseConnection(db_url)
    reopened_db.connect()
    init_schema(reopened_db)
    reopened = EmbeddingStore(reopened_db, index_path=str(index_path))
    assert reopened.search("alpha query", top_k=2) == ["b2"]


def test_sqlite_embedding_store_reopens_legacy_mapping_files(monkeypatch, tmp_path):
    monkeypatch.setattr("ace.core.storage.embedding_store.generate_embedding", _mock_embedding)

    db_url = f"sqlite:///{tmp_path / 'ace.db'}"
    index_path = tmp_path / "embeddings.faiss"

    db = DatabaseConnection(db_url)
    db.connect()
    init_schema(db)
    bullet_store = BulletStore(db)
    bullet_store.create_bullet(
        Bullet(
            id="b1",
            section="strategies_and_hard_rules",
            content="persistent retrieval bullet",
            added_at=datetime.now(UTC),
        )
    )

    store = EmbeddingStore(db, index_path=str(index_path))
    store.add_embedding("b1", "recoverable query")

    with open(index_path.with_suffix(index_path.suffix + ".mapping"), "wb") as handle:
        pickle.dump((store.id_to_idx, store.idx_to_id), handle)

    reopened_db = DatabaseConnection(db_url)
    reopened_db.connect()
    init_schema(reopened_db)
    reopened = EmbeddingStore(reopened_db, index_path=str(index_path))

    assert type(reopened.index).__name__ == "IndexIDMap2"
    assert reopened.search("recoverable query", top_k=1) == ["b1"]

    bullet_store = BulletStore(reopened_db)
    bullet_store.create_bullet(
        Bullet(
            id="b2",
            section="strategies_and_hard_rules",
            content="second retrieval bullet",
            added_at=datetime.now(UTC),
        )
    )
    reopened.add_embedding("b2", "follow up query")

    raw_index = faiss.read_index(str(index_path))
    assert type(raw_index).__name__ == "IndexIDMap2"
    assert reopened.search("follow up query", top_k=2) == ["b2", "b1"]


def test_sqlite_embedding_store_rebuilds_index_from_db_when_files_are_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("ace.core.storage.embedding_store.generate_embedding", _mock_embedding)

    db_url = f"sqlite:///{tmp_path / 'ace.db'}"
    index_path = tmp_path / "embeddings.faiss"

    db = DatabaseConnection(db_url)
    db.connect()
    init_schema(db)
    bullet_store = BulletStore(db)
    bullet_store.create_bullet(
        Bullet(
            id="b1",
            section="strategies_and_hard_rules",
            content="persistent retrieval bullet",
            added_at=datetime.now(UTC),
        )
    )
    store = EmbeddingStore(db, index_path=str(index_path))
    store.add_embedding("b1", "recoverable query")

    index_path.unlink()
    index_path.with_suffix(index_path.suffix + ".mapping").unlink()

    reopened_db = DatabaseConnection(db_url)
    reopened_db.connect()
    init_schema(reopened_db)
    reopened = EmbeddingStore(reopened_db, index_path=str(index_path))

    assert reopened.search("recoverable query", top_k=1) == ["b1"]
    assert index_path.exists()
    assert index_path.with_suffix(index_path.suffix + ".mapping").exists()
