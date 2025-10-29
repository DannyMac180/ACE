import os
import sqlite3
from typing import Any
from urllib.parse import urlparse


class DatabaseConnection:
    def __init__(self, db_url: str | None = None):
        resolved_url = db_url or os.getenv("ACE_DB_URL") or "sqlite:///ace.db"
        self.db_url: str = resolved_url
        self.conn: Any | None = None
        self.is_sqlite = self.db_url.startswith("sqlite://")

    def connect(self):
        if self.is_sqlite:
            db_path = self.db_url.replace("sqlite://", "")
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
        else:
            # Assuming postgres URL
            import psycopg2

            parsed = urlparse(self.db_url)
            self.conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path.lstrip("/") if parsed.path else "",
            )

    def close(self):
        if self.conn:
            self.conn.close()

    def execute(self, query: str, params: tuple = ()) -> Any:
        if not self.conn:
            self.connect()
        assert self.conn is not None
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor

    def fetchall(self, query: str, params: tuple = ()) -> list[Any]:
        if not self.conn:
            self.connect()
        assert self.conn is not None
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return list(cursor.fetchall())


def init_schema(db_conn: DatabaseConnection):
    if db_conn.is_sqlite:
        # Create bullets table
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS bullets (
                id TEXT PRIMARY KEY,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,  -- JSON string
                helpful INTEGER DEFAULT 0,
                harmful INTEGER DEFAULT 0,
                last_used TEXT,
                added_at TEXT NOT NULL
            )
        """)
        # FTS5 virtual table for full-text search
        db_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS bullets_fts USING fts5(
                id UNINDEXED,
                content,
                content='bullets',
                content_rowid='rowid'
            )
        """)
        # Triggers to keep FTS updated
        db_conn.execute("""
            CREATE TRIGGER IF NOT EXISTS bullets_fts_insert AFTER INSERT ON bullets
            BEGIN
                INSERT INTO bullets_fts(rowid, id, content) VALUES (new.rowid, new.id, new.content);
            END
        """)
        db_conn.execute("""
            CREATE TRIGGER IF NOT EXISTS bullets_fts_delete AFTER DELETE ON bullets
            BEGIN
                DELETE FROM bullets_fts WHERE rowid = old.rowid;
            END
        """)
        db_conn.execute("""
            CREATE TRIGGER IF NOT EXISTS bullets_fts_update AFTER UPDATE ON bullets
            BEGIN
                UPDATE bullets_fts SET content = new.content WHERE rowid = new.rowid;
            END
        """)
        # Table for embeddings
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bullet_id TEXT PRIMARY KEY,
                vector BLOB,
                FOREIGN KEY (bullet_id) REFERENCES bullets(id)
            )
        """)
        # Table for minhash signatures
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS minhash_sigs (
                bullet_id TEXT PRIMARY KEY,
                signature BLOB,
                FOREIGN KEY (bullet_id) REFERENCES bullets(id)
            )
        """)
    else:
        # Postgres schema
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS bullets (
                id TEXT PRIMARY KEY,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT[],
                helpful INTEGER DEFAULT 0,
                harmful INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                added_at TIMESTAMP NOT NULL
            )
        """)
        # pgvector extension for embeddings
        db_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bullet_id TEXT PRIMARY KEY,
                vector VECTOR(384),  -- assuming 384 dim
                FOREIGN KEY (bullet_id) REFERENCES bullets(id)
            )
        """)
        # For minhash, use bytea
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS minhash_sigs (
                bullet_id TEXT PRIMARY KEY,
                signature BYTEA,
                FOREIGN KEY (bullet_id) REFERENCES bullets(id)
            )
        """)
        # GIN index for tags
        db_conn.execute("CREATE INDEX IF NOT EXISTS idx_bullets_tags ON bullets USING GIN (tags)")
        # Full-text search index
        db_conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_bullets_content ON bullets
               USING GIN (to_tsvector('english', content))"""
        )
