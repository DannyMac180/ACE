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
    url = f"sqlite:///{db_path.as_posix()}"
    db = DatabaseConnection(url)
    db.connect()
    try:
        assert db_path.exists()
    finally:
        db.close()


def test_sqlite_connections_allow_cross_thread_usage(monkeypatch):
    captured: dict[str, object] = {}

    class FakeConnection:
        def execute(self, _query: str) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_connect(path: str, **kwargs: object) -> FakeConnection:
        captured["path"] = path
        captured["kwargs"] = kwargs
        return FakeConnection()

    monkeypatch.setattr("ace.core.storage.db.sqlite3.connect", fake_connect)

    db = DatabaseConnection("sqlite:///ace.db")
    db.connect()

    assert captured["path"] == "ace.db"
    assert captured["kwargs"] == {"check_same_thread": False}
