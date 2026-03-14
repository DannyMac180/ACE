import os
import tempfile

import numpy as np

from ace.core.merge import Delta, apply_delta
from ace.core.schema import Bullet, Playbook
from ace.core.storage.store_adapter import Store


def _mock_embedding(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    vector = np.zeros(384, dtype=np.float32)
    vector[sum(text.encode("utf-8")) % 384] = 1.0
    return vector


def test_store_save_and_get(monkeypatch):
    monkeypatch.setattr("ace.core.storage.embedding_store.generate_embedding", _mock_embedding)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        db_path = f.name
    try:
        store = Store(db_path)
        bullet = Bullet(
            id="test-001",
            section="strategies_and_hard_rules",
            content="test",
            tags=["test"],
        )
        store.save_bullet(bullet)
        bullets = store.get_bullets()
        assert len(bullets) == 1
        assert bullets[0].id == "test-001"
    finally:
        os.unlink(db_path)


def test_store_preserves_database_urls(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDatabaseConnection:
        def __init__(self, db_url: str):
            captured["db_url"] = db_url

        def connect(self) -> None:
            captured["connected"] = True

    class FakeBulletStore:
        def __init__(self, _db: object):
            return None

    class FakeEmbeddingStore:
        def __init__(self, _db: object):
            return None

    def fake_init_schema(_db: object) -> None:
        captured["schema_initialized"] = True

    monkeypatch.setattr("ace.core.storage.store_adapter.DatabaseConnection", FakeDatabaseConnection)
    monkeypatch.setattr("ace.core.storage.store_adapter.BulletStore", FakeBulletStore)
    monkeypatch.setattr("ace.core.storage.store_adapter.EmbeddingStore", FakeEmbeddingStore)
    monkeypatch.setattr("ace.core.storage.store_adapter.init_schema", fake_init_schema)

    Store("sqlite:///ace.db")
    assert captured["db_url"] == "sqlite:///ace.db"
    assert captured["connected"] is True
    assert captured["schema_initialized"] is True

    Store("postgres://user:pass@localhost:5432/ace")
    assert captured["db_url"] == "postgres://user:pass@localhost:5432/ace"


def test_store_normalizes_legacy_sqlite_paths(monkeypatch):
    captured: dict[str, str] = {}

    class FakeDatabaseConnection:
        def __init__(self, db_url: str):
            captured["db_url"] = db_url

        def connect(self) -> None:
            return None

    class FakeBulletStore:
        def __init__(self, _db: object):
            return None

    class FakeEmbeddingStore:
        def __init__(self, _db: object):
            return None

    monkeypatch.setattr("ace.core.storage.store_adapter.DatabaseConnection", FakeDatabaseConnection)
    monkeypatch.setattr("ace.core.storage.store_adapter.BulletStore", FakeBulletStore)
    monkeypatch.setattr("ace.core.storage.store_adapter.EmbeddingStore", FakeEmbeddingStore)
    monkeypatch.setattr("ace.core.storage.store_adapter.init_schema", lambda _db: None)

    Store("ace.db")
    assert captured["db_url"] == "sqlite:///ace.db"


def test_store_allows_database_connection_to_resolve_default_url(monkeypatch):
    captured: dict[str, str | None] = {}

    class FakeDatabaseConnection:
        def __init__(self, db_url: str | None):
            captured["db_url"] = db_url

        def connect(self) -> None:
            return None

    class FakeBulletStore:
        def __init__(self, _db: object):
            return None

    class FakeEmbeddingStore:
        def __init__(self, _db: object):
            return None

    monkeypatch.setattr("ace.core.storage.store_adapter.DatabaseConnection", FakeDatabaseConnection)
    monkeypatch.setattr("ace.core.storage.store_adapter.BulletStore", FakeBulletStore)
    monkeypatch.setattr("ace.core.storage.store_adapter.EmbeddingStore", FakeEmbeddingStore)
    monkeypatch.setattr("ace.core.storage.store_adapter.init_schema", lambda _db: None)

    Store()
    assert captured["db_url"] is None


def test_store_tracks_playbook_snapshots_and_rolls_back_exact_state(monkeypatch):
    monkeypatch.setattr("ace.core.storage.embedding_store.generate_embedding", _mock_embedding)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        db_path = f.name
    try:
        store = Store(db_path)
        store.save_bullet(
            Bullet(
                id="test-001",
                section="strategies_and_hard_rules",
                content="first",
                tags=["topic:test"],
            )
        )

        initial = store.load_playbook()
        first_version = apply_delta(
            initial,
            Delta.from_dict(
                {
                    "ops": [
                        {
                            "op": "PATCH",
                            "target_id": "test-001",
                            "patch": "second",
                        }
                    ]
                }
            ),
            store,
        )
        second_version = apply_delta(
            first_version,
            Delta.from_dict(
                {
                    "ops": [
                        {
                            "op": "ADD",
                            "new_bullet": {
                                "id": "test-002",
                                "section": "troubleshooting_and_pitfalls",
                                "content": "added later",
                                "tags": ["topic:test"],
                            },
                        }
                    ]
                }
            ),
            store,
        )

        history = store.list_playbook_versions()

        assert [entry["version"] for entry in history][:3] == [2, 1, 0]
        assert store.get_playbook_version(1) is not None
        assert second_version.version == 2

        rolled_back = store.rollback_to_version(1)

        assert rolled_back.version == 1
        assert store.get_version() == 1
        assert [bullet.id for bullet in store.get_all_bullets()] == ["test-001"]
        assert store.get_bullet("test-001").content == "second"
        assert store.get_bullet("test-002") is None
        embedding_rows = store.db.fetchall(
            "SELECT vector FROM embeddings WHERE bullet_id = ?",
            ("test-001",),
        )
        assert embedding_rows
        expected = _mock_embedding("second")
        actual = np.frombuffer(embedding_rows[0][0], dtype=np.float32)
        assert np.array_equal(actual, expected)
    finally:
        os.unlink(db_path)


def test_load_playbook_data_replaces_removed_bullets(monkeypatch):
    monkeypatch.setattr("ace.core.storage.embedding_store.generate_embedding", _mock_embedding)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        db_path = f.name
    try:
        store = Store(db_path)
        store.save_bullet(
            Bullet(
                id="test-001",
                section="strategies_and_hard_rules",
                content="keep me",
            )
        )
        store.save_bullet(
            Bullet(
                id="test-002",
                section="troubleshooting_and_pitfalls",
                content="remove me",
            )
        )
        store.set_version(1)

        replacement = Playbook(
            version=7,
            bullets=[
                Bullet(
                    id="test-001",
                    section="strategies_and_hard_rules",
                    content="updated",
                    tags=["topic:import"],
                )
            ],
        )

        store.load_playbook_data(replacement)

        assert store.get_version() == 7
        assert store.get_bullet("test-001").content == "updated"
        assert store.get_bullet("test-002") is None
        assert store.get_playbook_version(7).model_dump() == replacement.model_dump()
    finally:
        os.unlink(db_path)
