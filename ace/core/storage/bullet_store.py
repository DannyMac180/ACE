import json
from datetime import datetime
from typing import Any

from ace.core.schema import Bullet

from .db import DatabaseConnection


class BulletStore:
    def __init__(self, db_conn: DatabaseConnection):
        self.db = db_conn

    @staticmethod
    def _deserialize_tags(raw_tags: Any) -> list[str]:
        if raw_tags is None:
            return []
        if isinstance(raw_tags, str):
            parsed = json.loads(raw_tags)
            if isinstance(parsed, list):
                return [str(tag) for tag in parsed]
            return []
        if isinstance(raw_tags, (list, tuple)):
            return [str(tag) for tag in raw_tags]
        return []

    @staticmethod
    def _deserialize_datetime(raw_value: Any) -> datetime | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, datetime):
            return raw_value
        if isinstance(raw_value, str):
            return datetime.fromisoformat(raw_value)
        return None

    def create_bullet(self, bullet: Bullet) -> None:
        tags_value: Any = json.dumps(bullet.tags) if self.db.is_sqlite else bullet.tags
        last_used_value: Any = (
            bullet.last_used.isoformat()
            if self.db.is_sqlite and bullet.last_used
            else bullet.last_used
        )
        added_at_value: Any = bullet.added_at.isoformat() if self.db.is_sqlite else bullet.added_at
        self.db.execute(
            """INSERT INTO bullets (id, section, content, tags, helpful,
               harmful, last_used, added_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bullet.id,
                bullet.section,
                bullet.content,
                tags_value,
                bullet.helpful,
                bullet.harmful,
                last_used_value,
                added_at_value,
            ),
        )

    def get_bullet(self, bullet_id: str) -> Bullet | None:
        rows = self.db.fetchall("SELECT * FROM bullets WHERE id = ?", (bullet_id,))
        if not rows:
            return None
        row = rows[0]
        tags = self._deserialize_tags(row[3])
        last_used = self._deserialize_datetime(row[6])
        added_at = self._deserialize_datetime(row[7])
        assert added_at is not None
        return Bullet(
            id=row[0],
            section=row[1],
            content=row[2],
            tags=tags,
            helpful=row[4],
            harmful=row[5],
            last_used=last_used,
            added_at=added_at,
        )

    def update_bullet(self, bullet: Bullet) -> None:
        tags_value: Any = json.dumps(bullet.tags) if self.db.is_sqlite else bullet.tags
        last_used_value: Any = (
            bullet.last_used.isoformat()
            if self.db.is_sqlite and bullet.last_used
            else bullet.last_used
        )
        self.db.execute(
            """UPDATE bullets SET section=?, content=?, tags=?, helpful=?,
               harmful=?, last_used=? WHERE id=?""",
            (
                bullet.section,
                bullet.content,
                tags_value,
                bullet.helpful,
                bullet.harmful,
                last_used_value,
                bullet.id,
            ),
        )

    def delete_bullet(self, bullet_id: str) -> None:
        self.db.execute("DELETE FROM bullets WHERE id = ?", (bullet_id,))

    def list_bullets(self, limit: int = 100, offset: int = 0) -> list[Bullet]:
        rows = self.db.fetchall("SELECT * FROM bullets LIMIT ? OFFSET ?", (limit, offset))
        bullets = []
        for row in rows:
            tags = self._deserialize_tags(row[3])
            last_used = self._deserialize_datetime(row[6])
            added_at = self._deserialize_datetime(row[7])
            assert added_at is not None
            bullets.append(
                Bullet(
                    id=row[0],
                    section=row[1],
                    content=row[2],
                    tags=tags,
                    helpful=row[4],
                    harmful=row[5],
                    last_used=last_used,
                    added_at=added_at,
                )
            )
        return bullets

    def search_fts(self, query: str, limit: int = 24) -> list[str]:
        if self.db.is_sqlite:
            rows = self.db.fetchall(
                "SELECT id FROM bullets_fts WHERE content MATCH ? LIMIT ?", (query, limit)
            )
        else:
            rows = self.db.fetchall(
                """SELECT id FROM bullets
                   WHERE to_tsvector('english', content)
                   @@ websearch_to_tsquery('english', ?)
                   LIMIT ?""",
                (query, limit),
            )
        return [row[0] for row in rows]
