# ace/pipeline.py
"""
Full ACE pipeline orchestration.

Paper loop: Query → Retrieve → Generator → Trajectory → Reflector → Insights →
Curator → Delta → Merge → Updated Playbook

This module wires all ACE components together into a single cohesive pipeline.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.retrieve import Retriever
from ace.core.schema import Bullet, Playbook
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.generator.generator import Generator
from ace.generator.schemas import Trajectory
from ace.reflector.reflector import Reflector
from ace.reflector.schema import Reflection

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full pipeline cycle."""

    playbook: Playbook
    trajectory: Trajectory
    reflection: Reflection
    delta_ops_applied: int
    retrieved_bullets: list[Bullet]


class Pipeline:
    """
    Full ACE pipeline: Query → Retrieve → Generator → Reflector → Curator → Merge → Playbook.

    This class orchestrates the complete adaptation cycle described in the ACE paper,
    wiring together all components into a cohesive pipeline.
    """

    def __init__(
        self,
        store: Store | None = None,
        db_path: str = "ace.db",
        generator_max_steps: int = 10,
        retrieval_top_k: int = 24,
        reflector_model: str = "gpt-4o-mini",
        reflector_temperature: float = 0.3,
        curator_threshold: float = 0.90,
    ):
        """
        Initialize the Pipeline.

        Args:
            store: Optional pre-configured Store instance
            db_path: Path to SQLite database (used if store not provided)
            generator_max_steps: Maximum steps for generator execution
            retrieval_top_k: Number of bullets to retrieve
            reflector_model: OpenAI model for reflection
            reflector_temperature: LLM temperature for reflection
            curator_threshold: Similarity threshold for duplicate detection
        """
        self.store = store or Store(db_path)
        self.retrieval_top_k = retrieval_top_k
        self.curator_threshold = curator_threshold

        self.retriever = Retriever(self.store)
        self.reflector = Reflector(
            model=reflector_model,
            temperature=reflector_temperature,
        )

        self.generator = Generator(
            max_steps=generator_max_steps,
            retriever=self.retriever,
            retrieval_top_k=retrieval_top_k,
        )

    def run_full_cycle(
        self,
        query: str,
        execute_fn: Callable[[str], str] | None = None,
        auto_commit: bool = True,
    ) -> PipelineResult:
        """
        Run the complete ACE adaptation cycle.

        This implements the paper's loop:
        1. Retrieve relevant playbook bullets for the query
        2. Generator executes the task (with bullet context)
        3. Reflector analyzes the trajectory to produce insights
        4. Curator converts insights into delta operations
        5. Merge applies the delta to update the playbook

        Args:
            query: The task or goal to accomplish
            execute_fn: Optional custom tool executor for the generator.
                       If None, uses a default simulator.
            auto_commit: If True, automatically apply delta to playbook.
                        If False, return result without persisting changes.

        Returns:
            PipelineResult containing the updated playbook, trajectory, reflection,
            and metrics about the cycle.
        """
        logger.info(f"Starting full cycle for query: {query}")

        # 1. Retrieve relevant bullets (for duplicate checking later)
        retrieved_bullets = self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        logger.info(f"Retrieved {len(retrieved_bullets)} bullets for context")

        # 2. Run generator to execute task and produce trajectory
        if execute_fn:
            self.generator.tool_executor = execute_fn

        trajectory = self.generator.run(query)
        logger.info(
            f"Generator completed: {trajectory.total_steps} steps, "
            f"status={trajectory.final_status}"
        )

        # 3. Reflector analyzes trajectory to produce insights
        reflection = self.reflector.reflect_on_trajectory(trajectory)
        logger.info(
            f"Reflection generated: {len(reflection.candidate_bullets)} candidate bullets, "
            f"{len(reflection.bullet_tags)} bullet tags"
        )

        # 4. Curator converts insights to delta operations
        existing_bullets = self.store.get_all_bullets()
        delta = curate(
            reflection,
            existing_bullets=existing_bullets,
            threshold=self.curator_threshold,
        )
        logger.info(f"Curated {len(delta.ops)} delta operations")

        # 5. Load current playbook
        playbook = self.store.load_playbook()

        # 6. Merge: apply delta to update playbook
        if auto_commit and delta.ops:
            merge_delta = MergeDelta.from_dict(delta.model_dump())
            playbook = apply_delta(playbook, merge_delta, self.store)
            logger.info(f"Applied delta, playbook now at version {playbook.version}")
        else:
            if not auto_commit:
                logger.info("Auto-commit disabled, skipping delta application")
            elif not delta.ops:
                logger.info("No delta operations to apply")

        return PipelineResult(
            playbook=playbook,
            trajectory=trajectory,
            reflection=reflection,
            delta_ops_applied=len(delta.ops) if auto_commit else 0,
            retrieved_bullets=retrieved_bullets,
        )

    def run_retrieve_only(self, query: str) -> list[Bullet]:
        """
        Run retrieval only (no execution or adaptation).

        Useful for inspecting what context would be provided to the generator.

        Args:
            query: Query to retrieve bullets for

        Returns:
            List of retrieved bullets
        """
        return self.retriever.retrieve(query, top_k=self.retrieval_top_k)

    def run_with_feedback(
        self,
        query: str,
        execute_fn: Callable[[str], str] | None = None,
        code_diff: str = "",
        test_output: str = "",
        logs: str = "",
        env_meta: dict[str, Any] | None = None,
        auto_commit: bool = True,
    ) -> PipelineResult:
        """
        Run the pipeline with explicit execution feedback.

        This variant allows passing explicit environment feedback (code_diff, test_output,
        logs) instead of relying on the generator to capture it. Useful when the execution
        happens externally (e.g., CI system, IDE, or MCP client).

        Args:
            query: The task or goal
            execute_fn: Optional custom tool executor
            code_diff: Code changes made during execution
            test_output: Test results or output
            logs: Execution logs
            env_meta: Additional environment metadata
            auto_commit: If True, automatically apply delta

        Returns:
            PipelineResult with all cycle data
        """
        logger.info(f"Starting cycle with explicit feedback for: {query}")

        # 1. Retrieve bullets for context
        retrieved_bullets = self.retriever.retrieve(query, top_k=self.retrieval_top_k)

        # 2. Run generator
        if execute_fn:
            self.generator.tool_executor = execute_fn
        trajectory = self.generator.run(query)

        # 3. Reflector with explicit feedback
        retrieved_bullet_dicts = [
            {"id": b.id, "content": b.content} for b in retrieved_bullets
        ]

        reflection = self.reflector.reflect(
            query=query,
            retrieved_bullet_ids=trajectory.used_bullet_ids,
            code_diff=code_diff,
            test_output=test_output,
            logs=logs,
            env_meta=env_meta or {"final_status": trajectory.final_status},
            retrieved_bullets=retrieved_bullet_dicts,
        )

        # 4. Curator
        existing_bullets = self.store.get_all_bullets()
        delta = curate(
            reflection,
            existing_bullets=existing_bullets,
            threshold=self.curator_threshold,
        )

        # 5. Load and update playbook
        playbook = self.store.load_playbook()
        if auto_commit and delta.ops:
            merge_delta = MergeDelta.from_dict(delta.model_dump())
            playbook = apply_delta(playbook, merge_delta, self.store)
            logger.info(f"Applied delta, playbook now at version {playbook.version}")

        return PipelineResult(
            playbook=playbook,
            trajectory=trajectory,
            reflection=reflection,
            delta_ops_applied=len(delta.ops) if auto_commit else 0,
            retrieved_bullets=retrieved_bullets,
        )


def run_full_cycle(
    query: str,
    execute_fn: Callable[[str], str] | None = None,
    db_path: str = "ace.db",
    auto_commit: bool = True,
) -> PipelineResult:
    """
    Convenience function to run a full ACE cycle.

    Args:
        query: The task or goal to accomplish
        execute_fn: Optional custom tool executor
        db_path: Path to SQLite database
        auto_commit: If True, automatically apply delta

    Returns:
        PipelineResult with updated playbook and cycle data
    """
    pipeline = Pipeline(db_path=db_path)
    return pipeline.run_full_cycle(query, execute_fn, auto_commit)
