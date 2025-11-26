import os
import tempfile

from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store


def test_store_save_and_get():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        db_path = f.name
    try:
        store = Store(db_path)
        bullet = Bullet(
            id="test-001",
            section="strategies_and_hard_rules",
            content="test",
            tags=["test"],
        )
        store.save_bullet(bullet)
        bullets = store.get_bullets()
        assert len(bullets) == 1
        assert bullets[0].id == "test-001"
    finally:
        os.unlink(db_path)
