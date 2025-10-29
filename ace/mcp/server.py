"""
ACE MCP Server - Model Context Protocol implementation for ACE.

Exposes endpoints for playbook retrieval, reflection, curation, and refinement.
Built with FastMCP (FastAPI-based MCP implementation).
"""
from dataclasses import asdict

from fastmcp import FastMCP

from ace.core.merge import Delta, apply_delta
from ace.core.store import Store

mcp = FastMCP("ACE Playbook Server")


@mcp.tool()
async def status() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Status response indicating server is operational
    """
    return {"status": "ok"}


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
        return {"success": True, "message": "Delta applied successfully", "version": updated_playbook.version}
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

    playbook_dict = {
        "version": version,
        "bullets": [asdict(b) for b in bullets]
    }

    return json.dumps(playbook_dict, indent=2, default=str)


def main() -> None:
    """
    Entry point for running the MCP server with Uvicorn.

    Usage:
        python -m ace.mcp.server
    """
    import uvicorn
    uvicorn.run(
        "ace.mcp.server:mcp",
        host="127.0.0.1",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
