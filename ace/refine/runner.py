# ace/refine/runner.py
from typing import List
import numpy as np
from ace.core.schema import Delta, RefineResult, RefineOp, Playbook
from ace.reflector.schema import Reflection
from ace.curator import curate
from ace.core.storage.embedding_store import generate_embedding
from datasketch import MinHash


class RefineRunner:
    """
    Orchestrates the refinement pipeline: Curator -> Dedup -> Consolidate -> Archive.
    
    This is the main coordination class for processing reflections and generating
    refined playbook updates.
    """
    
    def __init__(self, playbook: Playbook, threshold: float = 0.90):
        """
        Initialize the RefineRunner.
        
        Args:
            playbook: The current Playbook to refine against
            threshold: Similarity threshold for deduplication (default: 0.90)
        """
        self.playbook = playbook
        self.threshold = threshold
    
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
        delta = curate(reflection)
        
        # Stage 2: Deduplication - find near-duplicates
        merge_ops = self._deduplicate(delta)
        
        # Stage 3: Consolidation (stubbed)
        self._consolidate(merge_ops)
        
        # Stage 4: Archive (stubbed)
        archive_ops = self._archive()
        
        # Combine results
        result = RefineResult(
            merged=len(merge_ops),
            archived=len(archive_ops),
            ops=merge_ops + archive_ops
        )
        
        return result
    
    def _deduplicate(self, delta: Delta) -> List[RefineOp]:
        """
        Find near-duplicate bullets using embedding cosine similarity and minhash.
        
        Checks candidate bullets (from ADD operations) against existing playbook bullets.
        Uses both cosine similarity (>threshold) OR minhash Jaccard (>0.85) to detect duplicates.
        
        Args:
            delta: The delta containing candidate bullets to check
            
        Returns:
            List of MERGE operations for near-duplicates
        """
        merge_ops: List[RefineOp] = []
        
        # Extract candidate bullets from ADD operations
        candidate_bullets = []
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
                        op="MERGE",
                        target_ids=[candidate_id],
                        survivor_id=existing_bullet.id
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
            m.update(word.encode('utf8'))
        return m
    
    def _consolidate(self, merge_ops: List[RefineOp]) -> None:
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
    
    def _archive(self) -> List[RefineOp]:
        """
        Archive low-utility bullets based on harmful/helpful ratio.
        
        Stub implementation - to be completed in future task.
        
        Returns:
            List of ARCHIVE operations
        """
        return []


def refine(reflection: Reflection, playbook: Playbook, threshold: float = 0.90) -> RefineResult:
    """
    Main entry point for the refinement pipeline.
    
    Orchestrates: Curator -> Deduplication -> Consolidation -> Archival
    
    Args:
        reflection: A Reflection object from the Reflector
        playbook: The current Playbook to refine against
        threshold: Similarity threshold for deduplication (default: 0.90)
        
    Returns:
        RefineResult: Summary of operations performed (merged count, archived count, ops)
    """
    runner = RefineRunner(playbook=playbook, threshold=threshold)
    return runner.run(reflection)
