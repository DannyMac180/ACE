import os
import pickle
from typing import Any, Optional

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .db import DatabaseConnection

# Load embedding model (all-MiniLM-L6-v2: 384d, Apache 2.0 license)
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def generate_embedding(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return np.array(embedding, dtype=np.float32)


class EmbeddingStore:
    def __init__(self, db_conn: DatabaseConnection, index_path: str = "faiss_index.idx"):
        self.db = db_conn
        self.index_path = index_path
        self.index: Optional[Any] = None
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: dict[int, str] = {}
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # Load mappings
            if os.path.exists(self.index_path + ".mapping"):
                with open(self.index_path + ".mapping", "rb") as f:
                    self.id_to_idx, self.idx_to_id = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(384)  # Cosine similarity
            self.id_to_idx = {}
            self.idx_to_id = {}

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".mapping", "wb") as f:
            pickle.dump((self.id_to_idx, self.idx_to_id), f)

    def add_embedding(self, bullet_id: str, text: str):
        if bullet_id in self.id_to_idx:
            return  # Already exists
        vector = generate_embedding(text)
        assert self.index is not None
        idx = self.index.ntotal
        self.index.add(vector.reshape(1, -1))
        self.id_to_idx[bullet_id] = idx
        self.idx_to_id[idx] = bullet_id
        # Persist to DB
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings (bullet_id, vector) VALUES (?, ?)",
            (bullet_id, vector.tobytes()),
        )

    def search(self, query: str, top_k: int = 24) -> list[str]:
        assert self.index is not None
        vector = generate_embedding(query)
        distances, indices = self.index.search(vector.reshape(1, -1), top_k)
        return [self.idx_to_id[idx] for idx in indices[0] if idx != -1]

    def remove_embedding(self, bullet_id: str):
        if bullet_id not in self.id_to_idx:
            return
        idx = self.id_to_idx[bullet_id]
        # FAISS doesn't support removal easily, so rebuild index
        # For simplicity, mark as removed or rebuild
        # TODO: Implement proper removal
        del self.id_to_idx[bullet_id]
        del self.idx_to_id[idx]
        self.db.execute("DELETE FROM embeddings WHERE bullet_id = ?", (bullet_id,))
        # Rebuild index
        self.rebuild_index()

    def rebuild_index(self):
        self.index = faiss.IndexFlatIP(384)
        assert self.index is not None
        rows = self.db.fetchall("SELECT bullet_id, vector FROM embeddings")
        for bullet_id, vector_bytes in rows:
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            idx = self.index.ntotal
            self.index.add(vector.reshape(1, -1))
            self.id_to_idx[bullet_id] = idx
            self.idx_to_id[idx] = bullet_id
        self.save_index()
