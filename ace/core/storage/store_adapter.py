"""
Unified Store adapter that wraps the storage layer components.

This provides a backward-compatible interface for code that used the old Store class,
while delegating to the proper storage/ module implementations.
"""

from ace.core.schema import Bullet, Playbook

from .bullet_store import BulletStore
from .db import DatabaseConnection, init_schema
from .embedding_store import EmbeddingStore


class Store:
    """Unified storage interface for ACE playbook."""

    def __init__(self, db_path: str = "ace.db"):
        """
        Initialize the store with SQLite and FAISS backing.

        Args:
            db_path: Path to SQLite database file (default: ace.db)
        """
        db_url = f"sqlite://{db_path}"
        self.db = DatabaseConnection(db_url)
        self.db.connect()
        init_schema(self.db)

        self.bullet_store = BulletStore(self.db)
        self.embedding_store = EmbeddingStore(self.db)

    def save_bullet(self, bullet: Bullet) -> None:
        """Save or update a bullet in the store."""
        existing = self.bullet_store.get_bullet(bullet.id)
        if existing:
            self.bullet_store.update_bullet(bullet)
        else:
            self.bullet_store.create_bullet(bullet)
        # Update embeddings
        self.embedding_store.add_embedding(bullet.id, bullet.content)

    def get_bullets(self) -> list[Bullet]:
        """Retrieve all bullets from the store."""
        return self.bullet_store.list_bullets(limit=10000)

    def get_bullet(self, bullet_id: str) -> Bullet | None:
        """Retrieve a single bullet by ID."""
        return self.bullet_store.get_bullet(bullet_id)

    def get_all_bullets(self) -> list[Bullet]:
        """Retrieve all bullets (alias for get_bullets)."""
        return self.get_bullets()

    def get_version(self) -> int:
        """Get current playbook version."""
        rows = self.db.fetchall("SELECT version FROM playbook_version LIMIT 1")
        return rows[0][0] if rows else 0

    def set_version(self, version: int) -> None:
        """Set playbook version."""
        # Initialize version table if needed
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS playbook_version (
                version INTEGER PRIMARY KEY
            )
        """)
        self.db.execute("DELETE FROM playbook_version")
        self.db.execute("INSERT INTO playbook_version (version) VALUES (?)", (version,))

    def load_playbook(self) -> Playbook:
        """Load the current playbook from the database."""
        bullets = self.get_all_bullets()
        version = self.get_version()
        return Playbook(version=version, bullets=bullets)

    def close(self) -> None:
        """Close database connections and save indices."""
        self.embedding_store.save_index()
        self.db.close()
