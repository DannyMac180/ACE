# ace/core/retrieve.py

from .schema import Bullet
from .storage.store_adapter import Store


class Retriever:
    def __init__(self, store: Store):
        self.store = store

    def retrieve(self, query: str, top_k: int = 24) -> list[Bullet]:
        # Hybrid retrieval: FTS (BM25-like) + vector search
        # Fetch 2x top_k from each method for better coverage
        fts_ids = self.store.bullet_store.search_fts(query, limit=top_k * 2)
        vector_ids = self.store.embedding_store.search(query, top_k=top_k * 2)

        # Combine into unique set of candidate IDs
        candidate_ids = list(dict.fromkeys(fts_ids + vector_ids))

        # Retrieve full Bullet objects
        bullets = []
        for bullet_id in candidate_ids:
            bullet = self.store.get_bullet(bullet_id)
            if bullet:
                bullets.append(bullet)

        # Rerank by lexical overlap with query terms
        query_terms = set(query.lower().split())
        scored_bullets = []
        for bullet in bullets:
            # Score by overlap in content and tags
            content_terms = set(bullet.content.lower().split())
            tag_terms = set(" ".join(bullet.tags).lower().split())
            all_terms = content_terms | tag_terms
            overlap = len(query_terms & all_terms)
            scored_bullets.append((overlap, bullet))

        # Sort by score descending and return top_k
        scored_bullets.sort(key=lambda x: x[0], reverse=True)
        return [bullet for _, bullet in scored_bullets[:top_k]]
