import os

from ace.core.storage.db import DatabaseConnection


def test_sqlite_relative_url_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = DatabaseConnection("sqlite:///ace.db")
    db.connect()
    try:
        assert (tmp_path / "ace.db").exists()
    finally:
        db.close()


def test_sqlite_absolute_url_preserves_path(tmp_path):
    db_path = tmp_path / "nested" / "ace.db"
    url = f"sqlite:////{db_path.as_posix().lstrip('/')}"
    db = DatabaseConnection(url)
    db.connect()
    try:
        assert db_path.exists()
    finally:
        db.close()
