"""
Test helpers for ACE evaluation harness.

Provides lightweight stubs and mocks that avoid heavy dependencies
like FAISS and SentenceTransformer for fast, hermetic unit tests.
"""

from typing import Any

from ace.core.schema import Bullet
from ace.core.storage.bullet_store import BulletStore
from ace.core.storage.db import DatabaseConnection


class MockEmbeddingStore:
    """Lightweight embedding store that uses simple lexical matching instead of FAISS."""

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.embeddings: dict[str, str] = {}

    def add_embedding(self, bullet_id: str, content: str) -> None:
        """Store content for simple lexical search."""
        self.embeddings[bullet_id] = content.lower()

    def search(self, query: str, top_k: int = 24) -> list[str]:
        """Simple lexical search without vectors."""
        query_lower = query.lower()
        scored = []
        for bullet_id, content in self.embeddings.items():
            score = sum(1 for term in query_lower.split() if term in content)
            if score > 0:
                scored.append((score, bullet_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [bullet_id for _, bullet_id in scored[:top_k]]

    def save_index(self) -> None:
        """No-op for mock."""
        pass


class LightweightStore:
    """Lightweight store for testing that avoids FAISS and embedding models."""

    def __init__(self, db_path: str, index_path: str | None = None):
        """Initialize with SQLite only, no FAISS."""
        from ace.core.storage.db import init_schema

        db_url = f"sqlite://{db_path}"
        self.db = DatabaseConnection(db_url)
        self.db.connect()
        init_schema(self.db)

        self.bullet_store = BulletStore(self.db)
        self.embedding_store = MockEmbeddingStore(self.db)

    def save_bullet(self, bullet: Bullet) -> None:
        """Save or update a bullet."""
        existing = self.bullet_store.get_bullet(bullet.id)
        if existing:
            self.bullet_store.update_bullet(bullet)
        else:
            self.bullet_store.create_bullet(bullet)
        self.embedding_store.add_embedding(bullet.id, bullet.content)

    def get_bullet(self, bullet_id: str) -> Bullet | None:
        """Retrieve a single bullet by ID."""
        return self.bullet_store.get_bullet(bullet_id)

    def close(self) -> None:
        """Close database connection."""
        self.embedding_store.save_index()
        self.db.close()
