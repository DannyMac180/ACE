# ace_mcp_server/server.py
from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP

from ace.core.reflect import Reflector
from ace.core.retrieve import Retriever
from ace.core.schema import Playbook
from ace.core.storage.store_adapter import Store
from ace.refine import refine
from ace.reflector.schema import Reflection

app = FastMCP("ACE MCP Server")

store = Store()
retriever = Retriever(store)
reflector = Reflector()


@app.tool()
def ace_retrieve(query: str, top_k: int = 24) -> list[dict[str, Any]]:
    bullets = retriever.retrieve(query, top_k)
    return [asdict(b) for b in bullets]


@app.tool()
def ace_record_trajectory(doc: dict[str, Any]) -> str:
    # Stub: record trajectory and return id
    return "traj-0001"


@app.tool()
def ace_reflect(doc: dict[str, Any]) -> dict[str, Any]:
    return reflector.reflect(doc)


@app.tool()
def ace_curate(reflection_data: dict[str, Any]) -> dict[str, int]:
    """
    Process a reflection and curate the playbook (add bullets, update counters).

    Args:
        reflection_data: Dictionary containing reflection data

    Returns:
        Dict with 'merged' and 'archived' counts
    """
    # Load current playbook state
    bullets = store.get_bullets()
    version = store.get_version()
    playbook = Playbook(version=version, bullets=bullets)

    # Deserialize reflection data into Reflection object
    reflection = Reflection(**reflection_data)

    # Run refine with dedup/archive disabled (threshold=0.0, archive_ratio=1.0)
    result = refine(reflection, playbook, threshold=0.0, archive_ratio=1.0)

    # Persist updated bullets back to store
    for bullet in playbook.bullets:
        store.save_bullet(bullet)

    return {"merged": result.merged, "archived": result.archived}


@app.tool()
def ace_commit(delta: dict[str, Any]) -> dict[str, int]:
    # Stub: apply delta and return new version
    return {"version": 1}


@app.tool()
def ace_refine(threshold: float = 0.90) -> dict[str, int]:
    """
    Run refinement pipeline: dedup near-duplicates, consolidate, archive low-utility bullets.

    Args:
        threshold: Cosine similarity threshold for deduplication (default: 0.90)

    Returns:
        Dict with 'merged' and 'archived' counts
    """
    # Get current playbook
    bullets = store.get_bullets()
    version = store.get_version()
    playbook = Playbook(version=version, bullets=bullets)

    # Run refine with empty reflection (dedup/archive only)
    empty_reflection = Reflection()
    result = refine(empty_reflection, playbook, threshold=threshold)

    # Persist updated playbook bullets to store
    for bullet in playbook.bullets:
        store.save_bullet(bullet)

    return {"merged": result.merged, "archived": result.archived}


@app.tool()
def ace_stats() -> dict[str, Any]:
    bullets = store.get_bullets()
    helpful_sum = sum(b.helpful for b in bullets)
    return {
        "num_bullets": len(bullets),
        "helpful_ratio": helpful_sum / len(bullets) if bullets else 0,
    }


@app.resource("ace://playbook.json")
def get_playbook() -> str:
    bullets = store.get_bullets()
    playbook = Playbook(version=store.get_version(), bullets=bullets)
    import json

    return json.dumps(playbook.__dict__)
