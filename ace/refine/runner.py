# ace/refine/runner.py

import numpy as np
from datasketch import MinHash  # type: ignore

from ace.core.schema import Delta, Playbook, RefineOp, RefineResult
from ace.core.storage.embedding_store import generate_embedding
from ace.curator import curate
from ace.reflector.schema import Reflection


class RefineRunner:
    """
    Orchestrates the refinement pipeline: Curator -> Dedup -> Consolidate -> Archive.

    This is the main coordination class for processing reflections and generating
    refined playbook updates.
    """

    def __init__(
        self,
        playbook: Playbook,
        threshold: float = 0.90,
        archive_ratio: float = 0.75,
        curator_threshold: float | None = None,
    ):
        """
        Initialize the RefineRunner.

        Args:
            playbook: The current Playbook to refine against
            threshold: Similarity threshold for deduplication stage (default: 0.90)
            archive_ratio: Harmful ratio threshold for archival (default: 0.75)
            curator_threshold: Similarity threshold for curator's semantic dedup.
                             If None, defaults to 0.90. Set to None or a high value
                             to enable semantic dedup; the curator only checks for
                             duplicates when this is >= 0.5 (otherwise all candidates
                             become ADD ops without dedup checking).
        """
        self.playbook = playbook
        self.threshold = threshold
        self.archive_ratio = archive_ratio
        self.curator_threshold = curator_threshold if curator_threshold is not None else 0.90

    def run(self, reflection: Reflection) -> RefineResult:
        """
        Execute the full refinement pipeline.

        Pipeline stages:
        1. Curator: Convert reflection to delta operations
        2. Deduplication: Find near-duplicate bullets
        3. Consolidation: Merge duplicates and transfer counters (stubbed)
        4. Archival: Remove low-utility bullets (stubbed)

        Args:
            reflection: The Reflection object to process

        Returns:
            RefineResult: Summary of merge and archive operations
        """
        # Stage 1: Curator - convert reflection to delta
        # Only enable semantic dedup when curator_threshold is reasonable (>= 0.5)
        # This prevents low thresholds from treating all candidates as duplicates
        if self.curator_threshold >= 0.5:
            delta = curate(
                reflection,
                existing_bullets=self.playbook.bullets,
                threshold=self.curator_threshold,
            )
        else:
            # Skip semantic dedup - all candidates become ADD ops
            delta = curate(reflection)

        # Stage 2: Deduplication - find near-duplicates
        # Skip deduplication when threshold is very low (caller wants to disable dedup)
        if self.threshold >= 0.5:
            merge_ops = self.deduplicate(delta)
        else:
            merge_ops = []

        # Stage 3: Consolidation (stubbed)
        self._consolidate(merge_ops)

        # Stage 4: Archive (stubbed)
        archive_ops = self._archive()

        # Combine results
        result = RefineResult(
            merged=len(merge_ops), archived=len(archive_ops), ops=merge_ops + archive_ops
        )

        return result

    def deduplicate(self, delta: Delta) -> list[RefineOp]:
        """
        Find near-duplicate bullets using embedding cosine similarity and minhash.

        Checks candidate bullets (from ADD operations) against existing playbook bullets.
        Uses both cosine similarity (>threshold) OR minhash Jaccard (>0.85) to detect duplicates.

        Args:
            delta: The delta containing candidate bullets to check

        Returns:
            List of MERGE operations for near-duplicates
        """
        merge_ops: list[RefineOp] = []

        # Extract candidate bullets from ADD operations
        candidate_bullets: list[tuple[str, dict]] = []
        for op in delta.ops:
            if op.op == "ADD" and op.new_bullet:
                # Create a temporary bullet ID for tracking
                temp_id = f"candidate-{len(candidate_bullets)}"
                candidate_bullets.append((temp_id, op.new_bullet))

        # Compare each candidate against existing playbook bullets
        for candidate_id, candidate_data in candidate_bullets:
            candidate_content = candidate_data.get("content", "")
            candidate_embedding = generate_embedding(candidate_content)
            candidate_minhash = self._generate_minhash(candidate_content)

            for existing_bullet in self.playbook.bullets:
                # Calculate cosine similarity
                existing_embedding = generate_embedding(existing_bullet.content)
                cosine_sim = self._cosine_similarity(candidate_embedding, existing_embedding)

                # Calculate minhash Jaccard similarity
                existing_minhash = self._generate_minhash(existing_bullet.content)
                jaccard_sim = candidate_minhash.jaccard(existing_minhash)

                # Check if duplicate based on either metric
                is_duplicate = (cosine_sim > self.threshold) or (jaccard_sim > 0.85)

                if is_duplicate:
                    # Create MERGE operation: keep existing bullet, merge candidate into it
                    merge_op = RefineOp(
                        op="MERGE", target_ids=[candidate_id], survivor_id=existing_bullet.id
                    )
                    merge_ops.append(merge_op)
                    break  # Only merge with first match

        return merge_ops

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two embedding vectors."""
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        # Compute dot product
        return float(np.dot(vec1_norm, vec2_norm))

    def _generate_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """Generate MinHash signature for text."""
        m = MinHash(num_perm=num_perm)
        for word in text.split():
            m.update(word.encode("utf8"))
        return m

    def _consolidate(self, merge_ops: list[RefineOp]) -> None:
        """
        Merge duplicate bullets, keeping clearest content and transferring counters.

        For each MERGE operation:
        - Find the survivor bullet in the playbook
        - Transfer helpful/harmful counters from target bullets to survivor
        - Remove target bullets if they exist in the playbook

        Args:
            merge_ops: List of MERGE operations to execute
        """
        # Build a lookup map for existing bullets
        bullet_map = {bullet.id: bullet for bullet in self.playbook.bullets}

        for merge_op in merge_ops:
            if merge_op.op != "MERGE" or not merge_op.survivor_id:
                continue

            # Find survivor bullet
            survivor = bullet_map.get(merge_op.survivor_id)
            if not survivor:
                continue  # Skip if survivor doesn't exist

            # Transfer counters from each target bullet
            for target_id in merge_op.target_ids:
                target = bullet_map.get(target_id)
                if target:
                    # Transfer counters
                    survivor.helpful += target.helpful
                    survivor.harmful += target.harmful

                    # Remove target from playbook
                    self.playbook.bullets.remove(target)
                    # Remove from map to avoid double-processing
                    del bullet_map[target_id]
                # If target_id is a candidate (not in playbook), skip - it was never added

    def _archive(self) -> list[RefineOp]:
        """
        Archive low-utility bullets based on harmful/helpful ratio.

        Bullets are archived if their harmful ratio (harmful / (helpful + harmful))
        exceeds the archive_ratio threshold.

        Returns:
            List of ARCHIVE operations
        """
        archive_ops: list[RefineOp] = []
        bullets_to_remove = []

        for bullet in self.playbook.bullets:
            total_count = bullet.helpful + bullet.harmful

            # Only evaluate bullets with feedback
            if total_count > 0:
                harmful_ratio = bullet.harmful / total_count

                if harmful_ratio > self.archive_ratio:
                    # Create ARCHIVE operation
                    archive_op = RefineOp(op="ARCHIVE", target_ids=[bullet.id])
                    archive_ops.append(archive_op)
                    bullets_to_remove.append(bullet)

        # Remove archived bullets from playbook
        for bullet in bullets_to_remove:
            self.playbook.bullets.remove(bullet)

        return archive_ops


def refine(
    reflection: Reflection,
    playbook: Playbook,
    threshold: float = 0.90,
    archive_ratio: float = 0.75,
    curator_threshold: float | None = None,
) -> RefineResult:
    """
    Main entry point for the refinement pipeline.

    Orchestrates: Curator -> Deduplication -> Consolidation -> Archival

    Args:
        reflection: A Reflection object from the Reflector
        playbook: The current Playbook to refine against
        threshold: Similarity threshold for deduplication stage (default: 0.90)
        archive_ratio: Harmful ratio threshold for archival (default: 0.75)
        curator_threshold: Similarity threshold for curator's semantic dedup.
                          If None, defaults to 0.90. When < 0.5, semantic dedup
                          is disabled and all candidates become ADD ops.

    Returns:
        RefineResult: Summary of operations performed (merged count, archived count, ops)
    """
    runner = RefineRunner(
        playbook=playbook,
        threshold=threshold,
        archive_ratio=archive_ratio,
        curator_threshold=curator_threshold,
    )
    return runner.run(reflection)
