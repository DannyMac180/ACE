# ace_mcp_server/server.py
import json
from typing import Any

from fastmcp import FastMCP

from ace.core.merge import Delta as MergeDelta
from ace.core.merge import apply_delta
from ace.core.retrieve import Retriever
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.generator.schemas import Trajectory, TrajectoryDoc
from ace.pipeline import Pipeline
from ace.refine.runner import refine
from ace.reflector.reflector import Reflector
from ace.reflector.schema import Reflection

app = FastMCP("ACE MCP Server")

store = Store()
retriever = Retriever(store)
reflector = Reflector()
pipeline = Pipeline(store=store)


@app.tool()
def ace_retrieve(query: str, top_k: int = 24) -> list[dict[str, Any]]:
    """
    Retrieve relevant playbook bullets for a query.

    Args:
        query: Search query string
        top_k: Maximum number of bullets to return (default: 24)

    Returns:
        List of bullet dictionaries
    """
    bullets = retriever.retrieve(query, top_k)
    return [b.model_dump() for b in bullets]


@app.tool()
def ace_record_trajectory(doc: dict[str, Any]) -> str:
    """
    Record a trajectory document.

    Args:
        doc: Dictionary containing trajectory data

    Returns:
        Trajectory ID string
    """
    trajectory = Trajectory(**doc)
    return trajectory.initial_goal


@app.tool()
def ace_reflect(doc: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a reflection from task execution data.

    Args:
        doc: Dictionary with query, retrieved_bullet_ids, code_diff, test_output, logs, env_meta

    Returns:
        Reflection dictionary
    """
    trajectory_doc = TrajectoryDoc(
        query=doc.get("query", ""),
        retrieved_bullet_ids=doc.get("retrieved_bullet_ids", []),
        code_diff=doc.get("code_diff", ""),
        test_output=doc.get("test_output", ""),
        logs=doc.get("logs", ""),
        env_meta=doc.get("env_meta") or {},
    )
    reflection = reflector.reflect(trajectory_doc)
    return {
        "error_identification": reflection.error_identification,
        "root_cause_analysis": reflection.root_cause_analysis,
        "correct_approach": reflection.correct_approach,
        "key_insight": reflection.key_insight,
        "bullet_tags": [{"id": bt.id, "tag": bt.tag} for bt in reflection.bullet_tags],
        "candidate_bullets": [
            {"section": cb.section, "content": cb.content, "tags": cb.tags}
            for cb in reflection.candidate_bullets
        ],
    }


@app.tool()
def ace_curate(reflection_data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a reflection into delta operations.

    Args:
        reflection_data: Dictionary containing reflection data

    Returns:
        Delta dictionary with ops list
    """
    reflection = Reflection(**reflection_data)
    existing_bullets = store.get_all_bullets()
    delta = curate(reflection, existing_bullets=existing_bullets)
    return delta.model_dump()


@app.tool()
def ace_commit(delta: dict[str, Any]) -> dict[str, int]:
    """
    Apply a delta to the playbook and return the new version.

    Args:
        delta: Dictionary containing delta operations

    Returns:
        Dict with new version number
    """
    playbook = store.load_playbook()
    merge_delta = MergeDelta.from_dict(delta)
    new_playbook = apply_delta(playbook, merge_delta, store)
    return {"version": new_playbook.version}


@app.tool()
def ace_refine(threshold: float = 0.90) -> dict[str, int]:
    """
    Run refinement pipeline: dedup near-duplicates, consolidate, archive low-utility bullets.

    Args:
        threshold: Cosine similarity threshold for deduplication (default: 0.90)

    Returns:
        Dict with 'merged' and 'archived' counts
    """
    playbook = store.load_playbook()
    empty_reflection = Reflection()
    result = refine(empty_reflection, playbook, threshold=threshold)

    for bullet in playbook.bullets:
        store.save_bullet(bullet)

    return {"merged": result.merged, "archived": result.archived}


@app.tool()
def ace_stats() -> dict[str, Any]:
    """
    Get playbook statistics.

    Returns:
        Dict with num_bullets, helpful_ratio, and other stats
    """
    playbook = store.load_playbook()
    helpful_sum = sum(b.helpful for b in playbook.bullets)
    harmful_sum = sum(b.harmful for b in playbook.bullets)
    return {
        "version": playbook.version,
        "num_bullets": len(playbook.bullets),
        "helpful_ratio": helpful_sum / max(helpful_sum + harmful_sum, 1),
    }


@app.tool()
def ace_pipeline(
    query: str,
    code_diff: str = "",
    test_output: str = "",
    logs: str = "",
    auto_commit: bool = True,
) -> dict[str, Any]:
    """
    Run the full ACE pipeline: retrieve → generate → reflect → curate → merge.

    This is the main entry point for running a complete adaptation cycle.

    Args:
        query: The task or goal to accomplish
        code_diff: Optional code changes from execution
        test_output: Optional test results
        logs: Optional execution logs
        auto_commit: Whether to automatically apply changes (default: True)

    Returns:
        Dict with playbook_version, trajectory info, and operation counts
    """
    if code_diff or test_output or logs:
        result = pipeline.run_with_feedback(
            query=query,
            code_diff=code_diff,
            test_output=test_output,
            logs=logs,
            auto_commit=auto_commit,
        )
    else:
        result = pipeline.run_full_cycle(
            query=query,
            auto_commit=auto_commit,
        )

    return {
        "playbook_version": result.playbook.version,
        "trajectory_steps": result.trajectory.total_steps,
        "trajectory_status": result.trajectory.final_status,
        "candidate_bullets": len(result.reflection.candidate_bullets),
        "bullet_tags": len(result.reflection.bullet_tags),
        "delta_ops_applied": result.delta_ops_applied,
        "retrieved_bullets": len(result.retrieved_bullets),
    }


@app.resource("ace://playbook.json")
def get_playbook() -> str:
    """Get the full playbook as JSON."""
    playbook = store.load_playbook()
    return json.dumps(playbook.model_dump(), default=str)
