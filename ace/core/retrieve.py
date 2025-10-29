# ace/core/retrieve.py

from .schema import Bullet
from .store import Store


class Retriever:
    def __init__(self, store: Store):
        self.store = store

    def retrieve(self, query: str, top_k: int = 24) -> list[Bullet]:
        # Minimal implementation: simple keyword match
        all_bullets = self.store.get_bullets()
        # Filter by query in content or tags
        matching = [b for b in all_bullets if query.lower() in b.content.lower() or any(query.lower() in tag.lower() for tag in b.tags)]
        return matching[:top_k]
