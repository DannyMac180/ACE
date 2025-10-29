import json
from datetime import datetime

from core.schema import Bullet

from .db import DatabaseConnection


class BulletStore:
    def __init__(self, db_conn: DatabaseConnection):
        self.db = db_conn

    def create_bullet(self, bullet: Bullet) -> None:
        tags_json = json.dumps(bullet.tags)
        last_used_str = bullet.last_used.isoformat() if bullet.last_used else None
        added_at_str = bullet.added_at.isoformat()
        self.db.execute(
            'INSERT INTO bullets (id, section, content, tags, helpful, harmful, last_used, added_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (bullet.id, bullet.section, bullet.content, tags_json, bullet.helpful, bullet.harmful, last_used_str, added_at_str)
        )

    def get_bullet(self, bullet_id: str) -> Bullet | None:
        rows = self.db.fetchall('SELECT * FROM bullets WHERE id = ?', (bullet_id,))
        if not rows:
            return None
        row = rows[0]
        tags = json.loads(row[3]) if row[3] else []
        last_used = datetime.fromisoformat(row[6]) if row[6] else None
        added_at = datetime.fromisoformat(row[7])
        return Bullet(
            id=row[0], section=row[1], content=row[2], tags=tags,
            helpful=row[4], harmful=row[5], last_used=last_used, added_at=added_at
        )

    def update_bullet(self, bullet: Bullet) -> None:
        tags_json = json.dumps(bullet.tags)
        last_used_str = bullet.last_used.isoformat() if bullet.last_used else None
        self.db.execute(
            'UPDATE bullets SET section=?, content=?, tags=?, helpful=?, harmful=?, last_used=? WHERE id=?',
            (bullet.section, bullet.content, tags_json, bullet.helpful, bullet.harmful, last_used_str, bullet.id)
        )

    def delete_bullet(self, bullet_id: str) -> None:
        self.db.execute('DELETE FROM bullets WHERE id = ?', (bullet_id,))

    def list_bullets(self, limit: int = 100, offset: int = 0) -> list[Bullet]:
        rows = self.db.fetchall('SELECT * FROM bullets LIMIT ? OFFSET ?', (limit, offset))
        bullets = []
        for row in rows:
            tags = json.loads(row[3]) if row[3] else []
            last_used = datetime.fromisoformat(row[6]) if row[6] else None
            added_at = datetime.fromisoformat(row[7])
            bullets.append(Bullet(
                id=row[0], section=row[1], content=row[2], tags=tags,
                helpful=row[4], harmful=row[5], last_used=last_used, added_at=added_at
            ))
        return bullets

    def search_fts(self, query: str, limit: int = 24) -> list[str]:
        # Return bullet IDs matching FTS query
        rows = self.db.fetchall('SELECT id FROM bullets_fts WHERE content MATCH ? LIMIT ?', (query, limit))
        return [row[0] for row in rows]
