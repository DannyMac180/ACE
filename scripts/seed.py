#!/usr/bin/env python3
"""Seed initial playbook with high-leverage bullets."""

from ace.core.schema import Bullet
from ace.core.store import Store


def seed_initial_playbook() -> None:
    """Create and persist initial seed bullets to bootstrap the playbook."""
    store = Store()

    seed_bullets = [
        Bullet(
            id="strat-00001",
            section="strategies",
            content="Prefer hybrid retrieval: BM25 + embedding; rerank by lexical overlap with query terms; default top_k=24.",
            tags=["topic:retrieval", "stack:python"]
        ),
        Bullet(
            id="strat-00002",
            section="strategies",
            content="Never rewrite the whole playbook. Only ADD/PATCH/DEPRECATE bullets; run refine for dedup.",
            tags=["topic:curation", "policy"]
        ),
        Bullet(
            id="strat-00003",
            section="strategies",
            content="Consider bullets near-duplicate if cosine>0.90 OR minhash Jaccard>0.85; keep clearer text; transfer counters.",
            tags=["topic:refine", "retrieval"]
        ),
        Bullet(
            id="trbl-00001",
            section="troubleshooting",
            content="Reflector/Curator must emit valid JSON without markdown fencing; reject and retry on parse errors.",
            tags=["topic:parsing", "robustness"]
        ),
        Bullet(
            id="tmpl-00001",
            section="templates",
            content="Unit test template for merge: apply Delta ops and assert version increment + idempotency.",
            tags=["topic:testing"]
        ),
        Bullet(
            id="code-00001",
            section="code_snippets",
            content="Expose MCP tools 'ace.retrieve|reflect|curate|commit|refine|stats'; resource 'ace://playbook.json'.",
            tags=["topic:mcp"]
        ),
        Bullet(
            id="seed-retrieval-hygiene",
            section="strategies",
            content="Ensure retrieval queries are specific and context-aware to prevent irrelevant or conflicting information from polluting the context.",
            tags=["topic:retrieval", "discipline:hygiene"]
        ),
        Bullet(
            id="strat-00004",
            section="strategies",
            content="Always write a failing test before implementing new functionality or fixing a bug.",
            tags=["topic:testing", "discipline:hygiene"]
        ),
        Bullet(
            id="code-00002",
            section="code_snippets",
            content="The MCP tool shape is defined by the following JSON schema: { 'type': 'object', 'properties': { 'tool_name': {'type': 'string', 'description': 'Name of the tool'}, 'tool_description': {'type': 'string', 'description': 'Description of the tool'}, 'tool_parameters': {'type': 'object', 'description': 'JSON schema for tool parameters'} } }",
            tags=["topic:mcp", "type:tool_shape"]
        ),
        Bullet(
            id="strat-00005",
            section="strategies",
            content="Bullets are considered near-duplicates if their embedding cosine similarity exceeds 0.90 or their minhash Jaccard index exceeds 0.85. The bullet with the clearer, more concise text should be retained.",
            tags=["topic:refine", "policy:deduplication"]
        ),
        Bullet(
            id="strat-00006",
            section="strategies",
            content="When generating structured output, especially tool calls or data schemas, always ensure the output is strictly valid JSON, enclosed in triple backticks if necessary, and avoid extraneous text.",
            tags=["topic:generation", "policy:json_strictness"]
        ),
    ]

    for bullet in seed_bullets:
        store.save_bullet(bullet)

    print(f"✓ Seeded {len(seed_bullets)} initial bullets")
    print(f"  Version: {store.get_version()}")
    print(f"  Total bullets in store: {len(store.get_bullets())}")


if __name__ == "__main__":
    seed_initial_playbook()
