import importlib.util
from pathlib import Path


def _load_seed_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "seed.py"
    spec = importlib.util.spec_from_file_location("ace_seed_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed_module = _load_seed_module()


class FakeStore:
    def __init__(self) -> None:
        self.saved_ids: list[str] = []
        self.closed = False

    def save_bullet(self, bullet) -> None:
        self.saved_ids.append(bullet.id)

    def get_version(self) -> int:
        return 0

    def get_bullets(self) -> list[str]:
        return self.saved_ids

    def close(self) -> None:
        self.closed = True


def test_seed_initial_playbook_closes_store(monkeypatch):
    fake_store = FakeStore()
    monkeypatch.setattr(seed_module, "Store", lambda *_args, **_kwargs: fake_store)

    seed_module.seed_initial_playbook()

    assert fake_store.closed is True
    assert len(fake_store.saved_ids) == 11
