from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from ace.core.schema import Bullet
from ace.core.storage.bullet_store import BulletStore
from ace.core.storage.db import DatabaseConnection
from ace.core.storage.embedding_store import EmbeddingStore
from ace.core.storage.minhash_store import MinHashStore


def test_database_connection_rewrites_postgres_placeholders(monkeypatch) -> None:
    executed: list[tuple[str, tuple[object, ...]]] = []

    class FakeCursor:
        def execute(self, query: str, params: tuple[object, ...]) -> None:
            executed.append((query, params))

        def fetchall(self) -> list[tuple[str]]:
            return [("row",)]

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_connect(**_kwargs: object) -> FakeConnection:
        return FakeConnection()

    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=fake_connect))

    db = DatabaseConnection("postgres://user:pass@localhost:5432/ace")
    db.connect()

    db.execute("INSERT INTO demo (id, value) VALUES (?, ?)", ("id-1", 3))
    rows = db.fetchall("SELECT value FROM demo WHERE id = ?", ("id-1",))

    assert executed == [
        ("INSERT INTO demo (id, value) VALUES (%s, %s)", ("id-1", 3)),
        ("SELECT value FROM demo WHERE id = %s", ("id-1",)),
    ]
    assert rows == [("row",)]


def test_bullet_store_postgres_preserves_native_types() -> None:
    created: list[tuple[str, tuple[object, ...]]] = []
    now = datetime(2026, 3, 14, 10, 30, 0)

    class FakeDB:
        is_sqlite = False

        def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
            created.append((query, params))

        def fetchall(
            self, query: str, params: tuple[object, ...] = ()
        ) -> list[tuple[object, ...]]:
            if query.startswith("SELECT * FROM bullets WHERE id"):
                return [
                    (
                        "strat-001",
                        "strategies_and_hard_rules",
                        "Use pgvector for semantic retrieval",
                        ["topic:retrieval", "db:postgresql"],
                        2,
                        0,
                        now,
                        now,
                    )
                ]
            if "to_tsvector" in query:
                return [("strat-001",)]
            return []

    bullet_store = BulletStore(FakeDB())
    bullet = Bullet(
        id="strat-001",
        section="strategies_and_hard_rules",
        content="Use pgvector for semantic retrieval",
        tags=["topic:retrieval", "db:postgresql"],
        helpful=2,
        added_at=now,
    )

    bullet_store.create_bullet(bullet)
    loaded = bullet_store.get_bullet("strat-001")
    fts_ids = bullet_store.search_fts("pgvector retrieval", limit=4)

    assert created[0][1][3] == ["topic:retrieval", "db:postgresql"]
    assert created[0][1][6] is None
    assert created[0][1][7] == now
    assert loaded is not None
    assert loaded.tags == ["topic:retrieval", "db:postgresql"]
    assert loaded.added_at == now
    assert fts_ids == ["strat-001"]


def test_embedding_store_postgres_uses_pgvector_queries(monkeypatch) -> None:
    executed: list[tuple[str, tuple[object, ...]]] = []

    class FakeDB:
        is_sqlite = False

        def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
            executed.append((query, params))

        def fetchall(
            self, query: str, params: tuple[object, ...] = ()
        ) -> list[tuple[object, ...]]:
            executed.append((query, params))
            return [("strat-002",), ("strat-001",)]

    monkeypatch.setattr(
        "ace.core.storage.embedding_store.generate_embedding",
        lambda _text: np.array([0.25, 0.75], dtype=np.float32),
    )

    store = EmbeddingStore(FakeDB())
    store.add_embedding("strat-001", "retrieval bullet")
    results = store.search("retrieval query", top_k=2)
    store.remove_embedding("strat-001")
    store.save_index()
    store.rebuild_index()

    assert store.index is None
    assert "ON CONFLICT" in executed[0][0]
    assert executed[0][1] == ("strat-001", "[0.250000000,0.750000000]")
    assert "<=>" in executed[1][0]
    assert executed[1][1] == ("[0.250000000,0.750000000]", 2)
    assert executed[2] == ("DELETE FROM embeddings WHERE bullet_id = ?", ("strat-001",))
    assert results == ["strat-002", "strat-001"]


def test_minhash_store_postgres_uses_upsert() -> None:
    executed: list[tuple[str, tuple[object, ...]]] = []

    class FakeDB:
        is_sqlite = False

        def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
            executed.append((query, params))

    store = MinHashStore(FakeDB())
    store.add_signature("strat-001", "semantic dedup rule")

    assert "ON CONFLICT" in executed[0][0]
    assert executed[0][1][0] == "strat-001"
