"""
ACE MCP Server - in-library Model Context Protocol surface for ACE.

Exposes the documented ACE contract tools plus the older legacy tool names.
"""

import json
import uuid
from typing import Any

from fastmcp import FastMCP

from ace.core.config import load_config
from ace.core.logging_utils import configure_logging
from ace.core.merge import Delta, apply_delta
from ace.core.retrieve import Retriever
from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store
from ace.curator.curator import curate
from ace.generator.schemas import Trajectory, TrajectoryDoc
from ace.refine.runner import refine
from ace.reflector.reflector import Reflector
from ace.reflector.schema import BulletTag, CandidateBullet, Reflection

mcp = FastMCP("ACE Playbook Server")

_trajectory_store: dict[str, TrajectoryDoc] = {}


def _serialize_bullet(bullet: Bullet) -> dict[str, Any]:
    return bullet.model_dump(mode="json")


def _serialize_reflection(reflection: Reflection) -> dict[str, Any]:
    return {
        "error_identification": reflection.error_identification,
        "root_cause_analysis": reflection.root_cause_analysis,
        "correct_approach": reflection.correct_approach,
        "key_insight": reflection.key_insight,
        "bullet_tags": [{"id": tag.id, "tag": tag.tag} for tag in reflection.bullet_tags],
        "candidate_bullets": [
            {
                "section": candidate.section,
                "content": candidate.content,
                "tags": candidate.tags,
            }
            for candidate in reflection.candidate_bullets
        ],
    }


def _coerce_trajectory_doc(doc: dict[str, Any]) -> TrajectoryDoc:
    if "query" in doc:
        payload = {
            "query": doc["query"],
            "retrieved_bullet_ids": doc.get("retrieved_bullet_ids", []),
            "code_diff": doc.get("code_diff", ""),
            "test_output": doc.get("test_output", ""),
            "logs": doc.get("logs", ""),
            "env_meta": doc.get("env_meta") or {},
            "tools_used": doc.get("tools_used", []),
            "trajectory_id": doc.get("trajectory_id", f"traj-{uuid.uuid4().hex[:12]}"),
        }
        return TrajectoryDoc(
            **payload,
        )

    trajectory = Trajectory(**doc)
    return trajectory.to_trajectory_doc()


def _load_tracked_trajectory(trajectory_id: str) -> TrajectoryDoc:
    doc = _trajectory_store.get(trajectory_id)
    if doc is None:
        raise ValueError(f"Trajectory not found: {trajectory_id}")
    return doc


def _coerce_reflection(reflection_data: dict[str, Any]) -> Reflection:
    return Reflection(
        error_identification=reflection_data.get("error_identification"),
        root_cause_analysis=reflection_data.get("root_cause_analysis"),
        correct_approach=reflection_data.get("correct_approach"),
        key_insight=reflection_data.get("key_insight"),
        bullet_tags=[BulletTag(**tag) for tag in reflection_data.get("bullet_tags", [])],
        candidate_bullets=[
            CandidateBullet(**candidate)
            for candidate in reflection_data.get("candidate_bullets", [])
        ],
        iteration=reflection_data.get("iteration", 0),
        parent_id=reflection_data.get("parent_id"),
    )


def _ace_retrieve_impl(query: str, top_k: int = 24) -> list[dict[str, Any]]:
    store = Store()
    try:
        retriever = Retriever(store)
        return [_serialize_bullet(bullet) for bullet in retriever.retrieve(query, top_k)]
    finally:
        store.close()


def _ace_record_trajectory_impl(doc: dict[str, Any]) -> str:
    trajectory_doc = _coerce_trajectory_doc(doc)
    _trajectory_store[trajectory_doc.trajectory_id] = trajectory_doc
    return trajectory_doc.trajectory_id


def _ace_reflect_impl(doc: dict[str, Any]) -> dict[str, Any]:
    if "trajectory_id" in doc and len(doc) == 1:
        trajectory_doc = _load_tracked_trajectory(str(doc["trajectory_id"]))
    else:
        trajectory_doc = _coerce_trajectory_doc(doc)

    reflection = Reflector().reflect(trajectory_doc)
    return _serialize_reflection(reflection)


def _ace_curate_impl(reflection_data: dict[str, Any]) -> dict[str, Any]:
    store = Store()
    try:
        reflection = _coerce_reflection(reflection_data)
        delta = curate(reflection, existing_bullets=store.get_all_bullets())
        return delta.model_dump(mode="json")
    finally:
        store.close()


def _ace_commit_impl(delta: dict[str, Any]) -> dict[str, int]:
    store = Store()
    try:
        playbook = store.load_playbook()
        updated_playbook = apply_delta(playbook, Delta.from_dict(delta), store)
        return {"version": updated_playbook.version}
    finally:
        store.close()


def _ace_refine_impl(threshold: float = 0.90) -> dict[str, int]:
    store = Store()
    try:
        playbook = store.load_playbook()
        result = refine(Reflection(), playbook, threshold=threshold)
        store.replace_playbook(playbook)
        return {"merged": result.merged, "archived": result.archived}
    finally:
        store.close()


def _ace_stats_impl() -> dict[str, Any]:
    store = Store()
    try:
        playbook = store.load_playbook()
        helpful_sum = sum(bullet.helpful for bullet in playbook.bullets)
        harmful_sum = sum(bullet.harmful for bullet in playbook.bullets)
        return {
            "version": playbook.version,
            "num_bullets": len(playbook.bullets),
            "helpful_ratio": helpful_sum / max(helpful_sum + harmful_sum, 1),
        }
    finally:
        store.close()


@mcp.tool()
async def status() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@mcp.tool()
async def record_trajectory(
    query: str,
    retrieved_bullet_ids: list[str] | None = None,
    code_diff: str = "",
    test_output: str = "",
    logs: str = "",
    env_meta: dict[str, Any] | None = None,
    tools_used: list[str] | None = None,
) -> dict[str, str]:
    """Legacy trajectory recording endpoint returning a wrapped identifier."""
    trajectory_id = _ace_record_trajectory_impl(
        {
            "query": query,
            "retrieved_bullet_ids": retrieved_bullet_ids or [],
            "code_diff": code_diff,
            "test_output": test_output,
            "logs": logs,
            "env_meta": env_meta or {},
            "tools_used": tools_used or [],
        }
    )
    return {"trajectory_id": trajectory_id}


@mcp.tool()
async def reflect(trajectory_id: str) -> dict[str, Any]:
    """Legacy reflector endpoint that resolves a stored trajectory first."""
    try:
        return _ace_reflect_impl({"trajectory_id": trajectory_id})
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
async def commit(delta: dict[str, Any]) -> dict[str, Any]:
    """Legacy commit endpoint retaining the historical success wrapper."""
    try:
        result = _ace_commit_impl(delta)
        return {"success": True, "message": "Delta applied successfully", **result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool(name="ace.retrieve")
async def ace_retrieve_contract(query: str, top_k: int = 24) -> list[dict[str, Any]]:
    """Stable ACE retrieval contract."""
    return _ace_retrieve_impl(query, top_k)


@mcp.tool()
async def ace_retrieve(query: str, top_k: int = 24) -> list[dict[str, Any]]:
    """Legacy underscore alias for retrieval."""
    return _ace_retrieve_impl(query, top_k)


@mcp.tool(name="ace.record_trajectory")
async def ace_record_trajectory_contract(doc: dict[str, Any]) -> str:
    """Stable ACE trajectory recording contract."""
    return _ace_record_trajectory_impl(doc)


@mcp.tool()
async def ace_record_trajectory(doc: dict[str, Any]) -> str:
    """Legacy underscore alias for trajectory recording."""
    return _ace_record_trajectory_impl(doc)


@mcp.tool(name="ace.reflect")
async def ace_reflect_contract(doc: dict[str, Any]) -> dict[str, Any]:
    """Stable ACE reflection contract."""
    return _ace_reflect_impl(doc)


@mcp.tool()
async def ace_reflect(doc: dict[str, Any]) -> dict[str, Any]:
    """Legacy underscore alias for reflection."""
    return _ace_reflect_impl(doc)


@mcp.tool(name="ace.curate")
async def ace_curate_contract(reflection: dict[str, Any]) -> dict[str, Any]:
    """Stable ACE curation contract."""
    return _ace_curate_impl(reflection)


@mcp.tool()
async def ace_curate(reflection: dict[str, Any]) -> dict[str, Any]:
    """Legacy underscore alias for curation."""
    return _ace_curate_impl(reflection)


@mcp.tool(name="ace.commit")
async def ace_commit_contract(delta: dict[str, Any]) -> dict[str, int]:
    """Stable ACE deterministic commit contract."""
    return _ace_commit_impl(delta)


@mcp.tool()
async def ace_commit(delta: dict[str, Any]) -> dict[str, int]:
    """Legacy underscore alias for commit."""
    return _ace_commit_impl(delta)


@mcp.tool(name="ace.refine")
async def ace_refine_contract(threshold: float = 0.90) -> dict[str, int]:
    """Stable ACE refinement contract."""
    return _ace_refine_impl(threshold)


@mcp.tool()
async def ace_refine(threshold: float = 0.90) -> dict[str, int]:
    """Legacy underscore alias for refinement."""
    return _ace_refine_impl(threshold)


@mcp.tool(name="ace.stats")
async def ace_stats_contract() -> dict[str, Any]:
    """Stable ACE stats contract."""
    return _ace_stats_impl()


@mcp.tool()
async def ace_stats() -> dict[str, Any]:
    """Legacy underscore alias for stats."""
    return _ace_stats_impl()


@mcp.resource("ace://playbook.json")
async def get_playbook_json() -> str:
    """
    Resource handler for ace://playbook.json.

    Returns the full playbook as JSON containing version and all bullets.
    """
    store = Store()
    try:
        playbook = store.load_playbook()
        return json.dumps(playbook.model_dump(mode="json"), indent=2)
    finally:
        store.close()


def main() -> None:
    """
    Entry point for running the MCP server with Uvicorn.

    Usage:
        python -m ace.mcp.server
    """
    import uvicorn

    config = load_config()
    configure_logging(config.logging.level, config.logging.format)
    uvicorn.run("ace.mcp.server:mcp", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
