import pytest
import tempfile
import os
from ace.core.store import Store
from ace.core.schema import Bullet

def test_store_save_and_get():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        db_path = f.name
    try:
        store = Store(db_path)
        bullet = Bullet(id="test-001", section="strategies", content="test", tags=["test"])
        store.save_bullet(bullet)
        bullets = store.get_bullets()
        assert len(bullets) == 1
        assert bullets[0].id == "test-001"
    finally:
        os.unlink(db_path)
