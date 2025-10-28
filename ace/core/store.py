# ace/core/store.py
import sqlite3
import json
from typing import List, Optional
from .schema import Bullet, Playbook

class Store:
    def __init__(self, db_path: str = "ace.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bullets (
                    id TEXT PRIMARY KEY,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT,
                    helpful INTEGER DEFAULT 0,
                    harmful INTEGER DEFAULT 0,
                    last_used TEXT,
                    added_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS playbook_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            # Insert initial version if not exists
            conn.execute("INSERT OR IGNORE INTO playbook_version (version) VALUES (0)")

    def save_bullet(self, bullet: Bullet):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bullets (id, section, content, tags, helpful, harmful, last_used, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bullet.id, bullet.section, bullet.content,
                json.dumps(bullet.tags), bullet.helpful, bullet.harmful,
                bullet.last_used.isoformat() if bullet.last_used else None,
                bullet.added_at.isoformat()
            ))

    def get_bullets(self) -> List[Bullet]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM bullets").fetchall()
            return [Bullet(
                id=row[0], section=row[1], content=row[2],
                tags=json.loads(row[3]) if row[3] else [],
                helpful=row[4], harmful=row[5],
                last_used=row[6], added_at=row[7]
            ) for row in rows]

    def get_version(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT version FROM playbook_version").fetchone()[0]

    def set_version(self, version: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE playbook_version SET version = ?", (version,))
