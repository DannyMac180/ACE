import hashlib
import os
import pickle
from typing import TYPE_CHECKING, Any

import faiss  # type: ignore
import numpy as np

from .db import DatabaseConnection

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Load embedding model (all-MiniLM-L6-v2: 384d, Apache 2.0 license)
_model: Any | None = None
_model_name: str | None = None


def _get_model_name() -> str:
    return os.getenv("ACE_EMBEDDINGS", "sentence-transformers/all-MiniLM-L6-v2")


def _uses_mock_embeddings(model_name: str) -> bool:
    return model_name.lower() in {"mock", "deterministic", "test"}


def _generate_mock_embedding(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(384).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def _get_model() -> "SentenceTransformer":
    global _model, _model_name
    model_name = _get_model_name()
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        _model_name = model_name
    return _model


def generate_embedding(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    model_name = _get_model_name()
    if _uses_mock_embeddings(model_name):
        return _generate_mock_embedding(text)
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return np.array(embedding, dtype=np.float32)


class EmbeddingStore:
    def __init__(self, db_conn: DatabaseConnection, index_path: str = "faiss_index.idx"):
        self.db = db_conn
        self.index_path = index_path
        self.index: Any | None = None
        self.id_to_idx: dict[str, int] = {}
        self.idx_to_id: dict[int, str] = {}
        self.load_index()

    def _sqlite_embedding_count(self) -> int:
        rows = self.db.fetchall("SELECT COUNT(*) FROM embeddings")
        return int(rows[0][0]) if rows else 0

    def load_index(self):
        if not self.db.is_sqlite:
            self.index = None
            self.id_to_idx = {}
            self.idx_to_id = {}
            return

        index_exists = os.path.exists(self.index_path)
        mapping_exists = os.path.exists(self.index_path + ".mapping")

        if index_exists:
            self.index = faiss.read_index(self.index_path)
            # Load mappings
            if mapping_exists:
                with open(self.index_path + ".mapping", "rb") as f:
                    self.id_to_idx, self.idx_to_id = pickle.load(f)
            else:
                self.id_to_idx = {}
                self.idx_to_id = {}
        else:
            self.index = faiss.IndexFlatIP(384)  # Cosine similarity
            self.id_to_idx = {}
            self.idx_to_id = {}

        expected_rows = self._sqlite_embedding_count()
        observed_rows = len(self.id_to_idx)
        observed_index_size = self.index.ntotal if self.index is not None else 0
        if expected_rows and (
            not index_exists
            or not mapping_exists
            or observed_rows != expected_rows
            or observed_index_size != expected_rows
        ):
            self.rebuild_index()

    def save_index(self):
        if not self.db.is_sqlite:
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".mapping", "wb") as f:
            pickle.dump((self.id_to_idx, self.idx_to_id), f)

    @staticmethod
    def _to_pgvector_literal(vector: np.ndarray[tuple[int], np.dtype[np.float32]]) -> str:
        return "[" + ",".join(f"{float(value):.9f}" for value in vector) + "]"

    def add_embedding(self, bullet_id: str, text: str):
        vector = generate_embedding(text)
        if not self.db.is_sqlite:
            self.db.execute(
                """INSERT INTO embeddings (bullet_id, vector)
                   VALUES (?, ?::vector)
                   ON CONFLICT (bullet_id) DO UPDATE SET vector = EXCLUDED.vector""",
                (bullet_id, self._to_pgvector_literal(vector)),
            )
            return
        assert self.index is not None
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings (bullet_id, vector) VALUES (?, ?)",
            (bullet_id, vector.tobytes()),
        )
        if bullet_id in self.id_to_idx:
            self.rebuild_index()
            return
        idx = self.index.ntotal
        self.index.add(vector.reshape(1, -1))
        self.id_to_idx[bullet_id] = idx
        self.idx_to_id[idx] = bullet_id
        self.save_index()

    def search(self, query: str, top_k: int = 24) -> list[str]:
        vector = generate_embedding(query)
        if not self.db.is_sqlite:
            rows = self.db.fetchall(
                """SELECT bullet_id FROM embeddings
                   ORDER BY vector <=> ?::vector
                   LIMIT ?""",
                (self._to_pgvector_literal(vector), top_k),
            )
            return [row[0] for row in rows]
        assert self.index is not None
        distances, indices = self.index.search(vector.reshape(1, -1), top_k)
        return [self.idx_to_id[idx] for idx in indices[0] if idx != -1]

    def remove_embedding(self, bullet_id: str):
        if not self.db.is_sqlite:
            self.db.execute("DELETE FROM embeddings WHERE bullet_id = ?", (bullet_id,))
            return
        self.db.execute("DELETE FROM embeddings WHERE bullet_id = ?", (bullet_id,))
        if bullet_id not in self.id_to_idx:
            return
        idx = self.id_to_idx[bullet_id]
        # FAISS doesn't support removal easily, so rebuild index
        # For simplicity, mark as removed or rebuild
        # TODO: Implement proper removal
        del self.id_to_idx[bullet_id]
        del self.idx_to_id[idx]
        # Rebuild index
        self.rebuild_index()

    def rebuild_index(self):
        if not self.db.is_sqlite:
            return
        self.index = faiss.IndexFlatIP(384)
        assert self.index is not None
        self.id_to_idx = {}
        self.idx_to_id = {}
        rows = self.db.fetchall("SELECT bullet_id, vector FROM embeddings")
        for bullet_id, vector_bytes in rows:
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            idx = self.index.ntotal
            self.index.add(vector.reshape(1, -1))
            self.id_to_idx[bullet_id] = idx
            self.idx_to_id[idx] = bullet_id
        self.save_index()
