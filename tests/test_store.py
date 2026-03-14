import os
import tempfile

from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store


def test_store_save_and_get():
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
