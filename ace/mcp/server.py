"""
ACE MCP Server - Model Context Protocol implementation for ACE.

Exposes endpoints for playbook retrieval, reflection, curation, and refinement.
Built with FastMCP (FastAPI-based MCP implementation).
"""

from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP

from ace.core.merge import Delta, apply_delta
from ace.core.storage.store_adapter import Store
from ace.generator.schemas import TrajectoryDoc
from ace.reflector.reflector import Reflector

mcp = FastMCP("ACE Playbook Server")

_trajectory_store: dict[str, TrajectoryDoc] = {}


@mcp.tool()
async def status() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Status response indicating server is operational
    """
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
) -> dict:
    """
    Record a trajectory from task execution.

    Stores the trajectory and returns a trajectory_id that can be used
    with ace.reflect to generate insights.

    Args:
        query: The task or query that was executed
        retrieved_bullet_ids: IDs of playbook bullets retrieved and used
        code_diff: Code changes made during execution
        test_output: Test results or output
        logs: Execution logs, errors, stack traces
        env_meta: Environment metadata (final_status, tool versions, etc.)
        tools_used: List of tools or actions invoked during execution

    Returns:
        dict: {"trajectory_id": str} for use with ace.reflect
    """
    try:
        doc = TrajectoryDoc(
            query=query,
            retrieved_bullet_ids=retrieved_bullet_ids or [],
            code_diff=code_diff,
            test_output=test_output,
            logs=logs,
            env_meta=env_meta or {},
            tools_used=tools_used or [],
        )
        _trajectory_store[doc.trajectory_id] = doc
        return {"trajectory_id": doc.trajectory_id}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def reflect(trajectory_id: str) -> dict:
    """
    Generate a Reflection from a recorded trajectory.

    Takes a trajectory_id from ace.record_trajectory and produces
    structured insights (error analysis, candidate bullets, bullet tags).

    Args:
        trajectory_id: ID returned from ace.record_trajectory

    Returns:
        dict: Reflection object with error_identification, root_cause_analysis,
              correct_approach, key_insight, bullet_tags, candidate_bullets
    """
    try:
        doc = _trajectory_store.get(trajectory_id)
        if doc is None:
            return {"error": f"Trajectory not found: {trajectory_id}"}

        reflector = Reflector()
        reflection = reflector.reflect(doc)

        return {
            "error_identification": reflection.error_identification,
            "root_cause_analysis": reflection.root_cause_analysis,
            "correct_approach": reflection.correct_approach,
            "key_insight": reflection.key_insight,
            "bullet_tags": [
                {"id": bt.id, "tag": bt.tag} for bt in reflection.bullet_tags
            ],
            "candidate_bullets": [
                {"section": cb.section, "content": cb.content, "tags": cb.tags}
                for cb in reflection.candidate_bullets
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def commit(delta: dict) -> dict:
    """
    Commit a playbook delta.

    Applies the given delta operations (ADD/PATCH/DEPRECATE/INCR_*) to the playbook.

    Args:
        delta: Dictionary containing delta operations in the format:
               {"ops": [{"op": "ADD|PATCH|DEPRECATE|INCR_HELPFUL|INCR_HARMFUL", ...}]}

    Returns:
        dict: Success response with new version or error details
    """
    try:
        store = Store()
        playbook = store.load_playbook()
        delta_obj = Delta.from_dict(delta)
        updated_playbook = apply_delta(playbook, delta_obj, store)
        return {
            "success": True,
            "message": "Delta applied successfully",
            "version": updated_playbook.version,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.resource("playbook://json")
async def get_playbook_json() -> str:
    """
    Resource handler for ace://playbook.json

    Returns the full playbook as JSON containing version and all bullets.

    Returns:
        str: JSON-serialized playbook with version and bullets
    """
    import json

    store = Store()
    version = store.get_version()
    bullets = store.get_bullets()

    playbook_dict = {"version": version, "bullets": [asdict(b) for b in bullets]}

    return json.dumps(playbook_dict, indent=2, default=str)


def main() -> None:
    """
    Entry point for running the MCP server with Uvicorn.

    Usage:
        python -m ace.mcp.server
    """
    import uvicorn

    uvicorn.run("ace.mcp.server:mcp", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
