
from core.schema import Bullet
from core.storage.bullet_store import BulletStore
from core.storage.db import DatabaseConnection
from core.storage.embedding_store import EmbeddingStore


class HybridRetriever:
    def __init__(self, db_conn: DatabaseConnection, bullet_store: BulletStore, embedding_store: EmbeddingStore):
        self.db = db_conn
        self.bullet_store = bullet_store
        self.embedding_store = embedding_store

    def retrieve(self, query: str, top_k: int = 24) -> list[Bullet]:
        # Lexical search
        lexical_ids = set(self.bullet_store.search_fts(query, top_k * 2))  # More candidates

        # Vector search
        vector_ids = set(self.embedding_store.search(query, top_k * 2))

        # Union
        candidate_ids = lexical_ids.union(vector_ids)

        # Simple reranking: score as 1 if in both, 0.5 if in one
        scores = {}
        for cid in candidate_ids:
            score = 0
            if cid in lexical_ids:
                score += 0.5
            if cid in vector_ids:
                score += 0.5
            scores[cid] = score

        # Sort by score desc, take top_k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        # Fetch bullets
        bullets = []
        for bid in sorted_ids:
            bullet = self.bullet_store.get_bullet(bid)
            if bullet:
                bullets.append(bullet)

        return bullets
