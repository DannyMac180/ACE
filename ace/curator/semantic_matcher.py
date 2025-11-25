# ace/curator/semantic_matcher.py

import numpy as np
from datasketch import MinHash  # type: ignore

from ace.core.schema import Bullet
from ace.core.storage.embedding_store import generate_embedding


class SemanticMatcher:
    """
    Checks semantic similarity between candidate bullets and existing playbook bullets.

    Uses both cosine similarity (embedding) and MinHash Jaccard (lexical) to detect duplicates.
    A match is found if cosine > threshold OR jaccard > 0.85.
    """

    def __init__(self, threshold: float = 0.90, jaccard_threshold: float = 0.85):
        self.threshold = threshold
        self.jaccard_threshold = jaccard_threshold

    def find_duplicate(
        self, candidate_content: str, existing_bullets: list[Bullet]
    ) -> Bullet | None:
        """
        Find a semantically similar bullet in existing_bullets.

        Args:
            candidate_content: The content of the candidate bullet to check
            existing_bullets: List of existing Bullet objects to compare against

        Returns:
            The first matching Bullet if a near-duplicate is found, None otherwise
        """
        if not existing_bullets:
            return None

        candidate_embedding = generate_embedding(candidate_content)
        candidate_minhash = self._generate_minhash(candidate_content)

        for existing_bullet in existing_bullets:
            if self._is_duplicate(
                candidate_embedding, candidate_minhash, existing_bullet
            ):
                return existing_bullet

        return None

    def _is_duplicate(
        self,
        candidate_embedding: np.ndarray,
        candidate_minhash: MinHash,
        existing_bullet: Bullet,
    ) -> bool:
        """Check if candidate is a near-duplicate of existing bullet."""
        existing_embedding = generate_embedding(existing_bullet.content)
        cosine_sim = self._cosine_similarity(candidate_embedding, existing_embedding)

        existing_minhash = self._generate_minhash(existing_bullet.content)
        jaccard_sim = candidate_minhash.jaccard(existing_minhash)

        return (cosine_sim > self.threshold) or (jaccard_sim > self.jaccard_threshold)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two embedding vectors."""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        return float(np.dot(vec1_norm, vec2_norm))

    def _generate_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """Generate MinHash signature for text."""
        m = MinHash(num_perm=num_perm)
        for word in text.split():
            m.update(word.encode("utf8"))
        return m
