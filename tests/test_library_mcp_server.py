import json

import pytest

from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store

fastmcp = pytest.importorskip(
    "fastmcp",
    reason="Library MCP server tests require the optional fastmcp dependency tree.",
)
Client = fastmcp.Client


def parse_resource_content(content):
    """Parse resource content from fastmcp Client."""
    if isinstance(content, list) and len(content) > 0:
        item = content[0]
        if hasattr(item, "text"):
            return item.text
    if hasattr(content, "text"):
        return content.text
    return str(content)


@pytest.mark.asyncio
async def test_library_mcp_server_exposes_documented_playbook_resource(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.setenv("ACE_DB_URL", f"sqlite:///{tmp_path / 'ace.db'}")
    monkeypatch.chdir(tmp_path)

    store = Store()
    store.save_bullet(
        Bullet(
            id="strat-001",
            section="strategies_and_hard_rules",
            content="Prefer hybrid retrieval for recall-sensitive queries",
            tags=["topic:retrieval"],
        )
    )
    playbook = store.load_playbook()
    store.close()

    from ace.mcp.server import mcp

    async with Client(mcp) as client:
        resources = await client.list_resources()
        resource_uris = [str(resource.uri) for resource in resources]

        assert "ace://playbook.json" in resource_uris

        content = await client.read_resource("ace://playbook.json")
        payload = json.loads(parse_resource_content(content))

    assert payload == playbook.model_dump(mode="json")
