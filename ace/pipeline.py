# ace/pipeline.py
"""
Full ACE pipeline orchestration.

Paper loop: Query → Retrieve → Generator → Trajectory → Reflector → Insights →
Curator → Delta → Merge → Updated Playbook

This module wires all ACE components together into a single cohesive pipeline.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.retrieve import Retriever
from ace.core.schema import Bullet, Playbook
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.generator.generator import Generator
from ace.generator.schemas import Trajectory, TrajectoryDoc
from ace.llm import LLMClient
from ace.reflector.reflector import Reflector
from ace.reflector.schema import Reflection

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Execution timings and LLM usage for a pipeline run."""

    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    reflect_ms: float = 0.0
    curate_ms: float = 0.0
    merge_ms: float = 0.0
    total_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class PipelineResult:
    """Result of a full pipeline cycle."""

    playbook: Playbook
    trajectory: Trajectory
    reflection: Reflection
    delta_ops_applied: int
    retrieved_bullets: list[Bullet]
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)


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
        llm_client: LLMClient | None = None,
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
            llm_client: LLM client for reflection. If None, creates from config.
            reflector_temperature: LLM temperature for reflection
            curator_threshold: Similarity threshold for duplicate detection
        """
        self.store = store or Store(db_path)
        self.retrieval_top_k = retrieval_top_k
        self.curator_threshold = curator_threshold

        self.retriever = Retriever(self.store)
        self.reflector = Reflector(
            llm_client=llm_client,
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
        overall_start = perf_counter()

        # 1. Retrieve relevant bullets (for duplicate checking later)
        retrieve_start = perf_counter()
        retrieved_bullets = self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        retrieve_ms = (perf_counter() - retrieve_start) * 1000
        logger.info(f"Retrieved {len(retrieved_bullets)} bullets for context")

        # 2. Run generator to execute task and produce trajectory
        # Stash and restore executor to avoid persisting custom executor across calls
        original_executor = self.generator.tool_executor
        generate_start = perf_counter()
        try:
            if execute_fn:
                self.generator.tool_executor = execute_fn

            trajectory = self.generator.run(query)
        finally:
            self.generator.tool_executor = original_executor
        generate_ms = (perf_counter() - generate_start) * 1000

        logger.info(
            f"Generator completed: {trajectory.total_steps} steps, "
            f"status={trajectory.final_status}"
        )

        # 3. Convert trajectory to TrajectoryDoc and reflect
        trajectory_doc = trajectory.to_trajectory_doc()
        reflect_start = perf_counter()
        reflection = self.reflector.reflect(trajectory_doc)
        reflect_ms = (perf_counter() - reflect_start) * 1000
        logger.info(
            f"Reflection generated: {len(reflection.candidate_bullets)} candidate bullets, "
            f"{len(reflection.bullet_tags)} bullet tags"
        )

        # 4. Curator converts insights to delta operations
        existing_bullets = self.store.get_all_bullets()
        curate_start = perf_counter()
        delta = curate(
            reflection,
            existing_bullets=existing_bullets,
            threshold=self.curator_threshold,
        )
        curate_ms = (perf_counter() - curate_start) * 1000
        logger.info(f"Curated {len(delta.ops)} delta operations")

        # 5. Load current playbook
        merge_start = perf_counter()
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
        merge_ms = (perf_counter() - merge_start) * 1000

        usage_metrics = getattr(self.reflector, "last_usage_metrics", None)
        metrics = PipelineMetrics(
            retrieve_ms=round(retrieve_ms, 2),
            generate_ms=round(generate_ms, 2),
            reflect_ms=round(reflect_ms, 2),
            curate_ms=round(curate_ms, 2),
            merge_ms=round(merge_ms, 2),
            total_ms=round((perf_counter() - overall_start) * 1000, 2),
            llm_calls=getattr(usage_metrics, "llm_calls", 0),
            prompt_tokens=getattr(usage_metrics, "prompt_tokens", 0),
            completion_tokens=getattr(usage_metrics, "completion_tokens", 0),
            total_tokens=getattr(usage_metrics, "total_tokens", 0),
        )

        return PipelineResult(
            playbook=playbook,
            trajectory=trajectory,
            reflection=reflection,
            delta_ops_applied=len(delta.ops) if auto_commit else 0,
            retrieved_bullets=retrieved_bullets,
            metrics=metrics,
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
        code_diff: str = "",
        test_output: str = "",
        logs: str = "",
        env_meta: dict[str, Any] | None = None,
        auto_commit: bool = True,
    ) -> PipelineResult:
        """
        Run the pipeline with explicit execution feedback (no generator execution).

        This variant skips the generator entirely and uses explicit environment feedback
        (code_diff, test_output, logs) for reflection. Use this when execution happens
        externally (e.g., CI system, IDE, or MCP client) and you want to curate insights
        from that real execution rather than running a simulated one.

        Args:
            query: The task or goal that was executed externally
            code_diff: Code changes made during execution
            test_output: Test results or output
            logs: Execution logs
            env_meta: Additional environment metadata
            auto_commit: If True, automatically apply delta

        Returns:
            PipelineResult with reflection and delta data (trajectory will be empty)
        """
        logger.info(f"Starting cycle with explicit feedback for: {query}")
        overall_start = perf_counter()

        # 1. Retrieve bullets for context and duplicate checking
        retrieve_start = perf_counter()
        retrieved_bullets = self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        retrieve_ms = (perf_counter() - retrieve_start) * 1000
        logger.info(f"Retrieved {len(retrieved_bullets)} bullets for context")

        # 2. Build retrieved bullet info for reflector
        retrieved_bullet_ids = [b.id for b in retrieved_bullets]
        retrieved_bullet_dicts = [
            {"id": b.id, "content": b.content} for b in retrieved_bullets
        ]

        # 3. Build TrajectoryDoc from explicit feedback (no generator execution)
        trajectory_doc = TrajectoryDoc(
            query=query,
            retrieved_bullet_ids=retrieved_bullet_ids,
            code_diff=code_diff,
            test_output=test_output,
            logs=logs,
            env_meta=env_meta or {},
        )
        reflect_start = perf_counter()
        reflection = self.reflector.reflect(
            trajectory_doc,
            retrieved_bullets=retrieved_bullet_dicts,
        )
        reflect_ms = (perf_counter() - reflect_start) * 1000
        logger.info(
            f"Reflection generated: {len(reflection.candidate_bullets)} candidate bullets, "
            f"{len(reflection.bullet_tags)} bullet tags"
        )

        # 4. Curator converts insights to delta operations
        existing_bullets = self.store.get_all_bullets()
        curate_start = perf_counter()
        delta = curate(
            reflection,
            existing_bullets=existing_bullets,
            threshold=self.curator_threshold,
        )
        curate_ms = (perf_counter() - curate_start) * 1000
        logger.info(f"Curated {len(delta.ops)} delta operations")

        # 5. Load and update playbook
        merge_start = perf_counter()
        playbook = self.store.load_playbook()
        if auto_commit and delta.ops:
            merge_delta = MergeDelta.from_dict(delta.model_dump())
            playbook = apply_delta(playbook, merge_delta, self.store)
            logger.info(f"Applied delta, playbook now at version {playbook.version}")
        merge_ms = (perf_counter() - merge_start) * 1000

        # Create an empty trajectory since execution happened externally
        empty_trajectory = Trajectory(
            steps=[],
            initial_goal=query,
            final_status="success",
            used_bullet_ids=retrieved_bullet_ids,
        )

        usage_metrics = getattr(self.reflector, "last_usage_metrics", None)
        metrics = PipelineMetrics(
            retrieve_ms=round(retrieve_ms, 2),
            generate_ms=0.0,
            reflect_ms=round(reflect_ms, 2),
            curate_ms=round(curate_ms, 2),
            merge_ms=round(merge_ms, 2),
            total_ms=round((perf_counter() - overall_start) * 1000, 2),
            llm_calls=getattr(usage_metrics, "llm_calls", 0),
            prompt_tokens=getattr(usage_metrics, "prompt_tokens", 0),
            completion_tokens=getattr(usage_metrics, "completion_tokens", 0),
            total_tokens=getattr(usage_metrics, "total_tokens", 0),
        )

        return PipelineResult(
            playbook=playbook,
            trajectory=empty_trajectory,
            reflection=reflection,
            delta_ops_applied=len(delta.ops) if auto_commit else 0,
            retrieved_bullets=retrieved_bullets,
            metrics=metrics,
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
