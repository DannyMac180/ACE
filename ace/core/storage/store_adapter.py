"""
Unified Store adapter that wraps the storage layer components.

This provides a backward-compatible interface for code that used the old Store class,
while delegating to the proper storage/ module implementations.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

from ace.core.schema import Bullet, Playbook

from .bullet_store import BulletStore
from .db import DatabaseConnection, init_schema
from .embedding_store import EmbeddingStore


class Store:
    """Unified storage interface for ACE playbook."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize the store with a database URL or legacy SQLite path.

        Args:
            db_path: Database URL or filesystem path to a SQLite database.
        """
        db_url = self._normalize_db_url(db_path)
        self.db = DatabaseConnection(db_url)
        self.db.connect()
        init_schema(self.db)

        self.bullet_store = BulletStore(self.db)
        self.embedding_store = EmbeddingStore(self.db)
        if hasattr(self.db, "fetchall"):
            self._ensure_snapshot_exists(self.get_version())

    @staticmethod
    def _normalize_db_url(db_path: str | None) -> str | None:
        """Accept full database URLs while preserving legacy path inputs."""
        if db_path is None:
            return None

        parsed = urlparse(db_path)
        if parsed.scheme in {"sqlite", "postgres", "postgresql"}:
            return db_path

        path = Path(db_path).expanduser()
        return f"sqlite:///{path.as_posix()}"

    def save_bullet(self, bullet: Bullet) -> None:
        """Save or update a bullet in the store."""
        existing = self.bullet_store.get_bullet(bullet.id)
        if existing:
            self.bullet_store.update_bullet(bullet)
        else:
            self.bullet_store.create_bullet(bullet)
        # Update embeddings
        self.embedding_store.add_embedding(bullet.id, bullet.content)

    def delete_bullet(self, bullet_id: str) -> None:
        """Delete a bullet from the store."""
        self.embedding_store.remove_embedding(bullet_id)
        self.db.execute("DELETE FROM minhash_sigs WHERE bullet_id = ?", (bullet_id,))
        self.bullet_store.delete_bullet(bullet_id)

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

    def _write_version(self, version: int) -> None:
        """Persist the active playbook version number."""
        self.db.execute("DELETE FROM playbook_version")
        self.db.execute("INSERT INTO playbook_version (version) VALUES (?)", (version,))

    def _snapshot_exists(self, version: int) -> bool:
        rows = self.db.fetchall(
            "SELECT 1 FROM playbook_snapshots WHERE version = ? LIMIT 1",
            (version,),
        )
        return bool(rows)

    def _ensure_snapshot_exists(self, version: int) -> None:
        if self._snapshot_exists(version):
            return
        self.save_snapshot(Playbook(version=version, bullets=self.get_all_bullets()))

    def save_snapshot(self, playbook: Playbook) -> None:
        """Persist an immutable snapshot for a playbook version."""
        snapshot = json.dumps(playbook.model_dump(), default=str)
        existing = self._snapshot_exists(playbook.version)
        if existing:
            self.db.execute(
                "UPDATE playbook_snapshots SET snapshot = ? WHERE version = ?",
                (snapshot, playbook.version),
            )
        else:
            self.db.execute(
                "INSERT INTO playbook_snapshots (version, snapshot) VALUES (?, ?)",
                (playbook.version, snapshot),
            )

    def set_version(
        self,
        version: int,
        snapshot_playbook: Playbook | None = None,
        *,
        record_snapshot: bool = True,
    ) -> None:
        """Set playbook version."""
        self._write_version(version)
        if not record_snapshot:
            return

        playbook = snapshot_playbook or Playbook(version=version, bullets=self.get_all_bullets())
        self.save_snapshot(playbook)

    def load_playbook(self) -> Playbook:
        """Load the current playbook from the database."""
        bullets = self.get_all_bullets()
        version = self.get_version()
        return Playbook(version=version, bullets=bullets)

    def get_playbook_version(self, version: int) -> Playbook | None:
        """Load a historical playbook snapshot by version."""
        rows = self.db.fetchall(
            "SELECT snapshot FROM playbook_snapshots WHERE version = ?",
            (version,),
        )
        if not rows:
            return None
        raw_snapshot = rows[0][0]
        if isinstance(raw_snapshot, str):
            payload = json.loads(raw_snapshot)
        else:
            payload = raw_snapshot
        return Playbook.model_validate(payload)

    def list_playbook_versions(self) -> list[dict[str, str | int]]:
        """Return available playbook versions ordered newest-first."""
        rows = self.db.fetchall(
            "SELECT version, created_at FROM playbook_snapshots ORDER BY version DESC"
        )
        return [{"version": row[0], "created_at": row[1]} for row in rows]

    def replace_playbook(self, playbook: Playbook, *, record_snapshot: bool = True) -> None:
        """Replace the current playbook contents with the provided playbook."""
        existing_ids = {bullet.id for bullet in self.get_all_bullets()}
        target_ids = {bullet.id for bullet in playbook.bullets}

        for bullet_id in existing_ids - target_ids:
            self.delete_bullet(bullet_id)

        for bullet in playbook.bullets:
            self.save_bullet(bullet)

        self.set_version(
            playbook.version,
            snapshot_playbook=playbook,
            record_snapshot=record_snapshot,
        )

    def load_playbook_data(self, playbook: Playbook) -> None:
        """Import playbook data into the store.

        Args:
            playbook: Playbook object to import (replaces current data)
        """
        self.replace_playbook(playbook)

    def rollback_to_version(self, version: int) -> Playbook:
        """Restore the current playbook to a previously snapshotted version."""
        playbook = self.get_playbook_version(version)
        if playbook is None:
            raise ValueError(f"Playbook version {version} not found")

        self.replace_playbook(playbook, record_snapshot=False)
        return playbook

    def close(self) -> None:
        """Close database connections and save indices."""
        self.embedding_store.save_index()
        self.db.close()
