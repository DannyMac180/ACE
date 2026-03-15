import json
import sys

import pytest

from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store
from ace.reflector.schema import BulletTag, CandidateBullet, Reflection

fastmcp = pytest.importorskip(
    "fastmcp",
    reason="Library MCP server tests require the optional fastmcp dependency tree.",
)
Client = fastmcp.Client

EXPECTED_LIBRARY_TOOLS = [
    "ace_retrieve",
    "ace_record_trajectory",
    "ace_reflect",
    "ace_curate",
    "ace_commit",
    "ace_refine",
    "ace_stats",
    "ace.retrieve",
    "ace.record_trajectory",
    "ace.reflect",
    "ace.curate",
    "ace.commit",
    "ace.refine",
    "ace.stats",
]


def parse_tool_result(result):
    """Parse a tool result from fastmcp Client."""
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            item = content[0]
            if hasattr(item, "text"):
                return item.text
        return str(content)
    if hasattr(result, "text"):
        return result.text
    return str(result)


def parse_resource_content(content):
    """Parse resource content from fastmcp Client."""
    if isinstance(content, list) and len(content) > 0:
        item = content[0]
        if hasattr(item, "text"):
            return item.text
    if hasattr(content, "text"):
        return content.text
    return str(content)


@pytest.fixture
def library_mcp_server(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("ACE_DB_URL", f"sqlite:///{tmp_path / 'ace.db'}")
    monkeypatch.setenv("ACE_EMBEDDINGS", "mock")
    monkeypatch.chdir(tmp_path)

    store = Store()
    store.save_bullet(
        Bullet(
            id="strat-001",
            section="strategies_and_hard_rules",
            content="Prefer hybrid retrieval for recall-sensitive queries",
            tags=["topic:retrieval"],
            helpful=3,
        )
    )
    store.save_bullet(
        Bullet(
            id="trbl-001",
            section="troubleshooting_and_pitfalls",
            content="Reject markdown-fenced JSON from reflector outputs",
            tags=["topic:parsing"],
            harmful=1,
        )
    )
    store.close()

    for mod_name in list(sys.modules):
        if mod_name.startswith("ace.mcp.server"):
            del sys.modules[mod_name]

    import ace.mcp.server as server_module

    class FakeReflector:
        def reflect(self, doc):
            return Reflection(
                error_identification=f"Observed issue for {doc.query}",
                root_cause_analysis="The execution context needs a reusable tactic.",
                correct_approach="Capture the fix as a concise bullet.",
                key_insight="Short, tagged bullets improve future retrieval.",
                bullet_tags=[BulletTag(id="strat-001", tag="helpful")],
                candidate_bullets=[
                    CandidateBullet(
                        section="strategies_and_hard_rules",
                        content="Tag reusable fixes with topic metadata for retrieval",
                        tags=["topic:retrieval", "repo:ace"],
                    )
                ],
            )

    monkeypatch.setattr(server_module, "Reflector", FakeReflector)
    server_module._trajectory_store.clear()

    yield server_module.mcp


@pytest.mark.asyncio
async def test_library_mcp_server_exposes_documented_tool_contracts(library_mcp_server):
    async with Client(library_mcp_server) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

    for expected_tool in EXPECTED_LIBRARY_TOOLS:
        assert expected_tool in tool_names


@pytest.mark.asyncio
async def test_library_mcp_server_supports_dotted_retrieve_contract(library_mcp_server):
    async with Client(library_mcp_server) as client:
        result = await client.call_tool("ace.retrieve", {"query": "retrieval", "top_k": 5})

    parsed = json.loads(parse_tool_result(result))
    assert isinstance(parsed, list)
    assert parsed[0]["id"] == "strat-001"


@pytest.mark.asyncio
async def test_library_mcp_server_records_and_reflects_trajectory(library_mcp_server):
    doc = {
        "query": "Add regression coverage for MCP contracts",
        "retrieved_bullet_ids": ["strat-001"],
        "test_output": "pytest -q",
        "logs": "all green",
    }

    async with Client(library_mcp_server) as client:
        record_result = await client.call_tool("ace.record_trajectory", {"doc": doc})
        trajectory_id = parse_tool_result(record_result)
        reflect_result = await client.call_tool(
            "ace.reflect",
            {"doc": {"trajectory_id": trajectory_id}},
        )

    reflection = json.loads(parse_tool_result(reflect_result))
    assert trajectory_id.startswith("traj-")
    assert reflection["bullet_tags"] == [{"id": "strat-001", "tag": "helpful"}]
    assert reflection["candidate_bullets"][0]["tags"] == ["topic:retrieval", "repo:ace"]


@pytest.mark.asyncio
async def test_library_mcp_server_supports_curate_commit_refine_and_stats(library_mcp_server):
    reflection = {
        "bullet_tags": [{"id": "strat-001", "tag": "helpful"}],
        "candidate_bullets": [
            {
                "section": "strategies_and_hard_rules",
                "content": "Persist stable MCP contracts in the library server",
                "tags": ["topic:mcp", "repo:ace"],
            }
        ],
    }

    async with Client(library_mcp_server) as client:
        curate_result = await client.call_tool("ace.curate", {"reflection": reflection})
        delta = json.loads(parse_tool_result(curate_result))

        commit_result = await client.call_tool("ace.commit", {"delta": delta})
        commit_payload = json.loads(parse_tool_result(commit_result))

        refine_result = await client.call_tool("ace.refine", {"threshold": 0.9})
        refine_payload = json.loads(parse_tool_result(refine_result))

        stats_result = await client.call_tool("ace.stats", {})
        stats_payload = json.loads(parse_tool_result(stats_result))

    assert delta["ops"]
    assert commit_payload["version"] >= 1
    assert set(refine_payload) == {"merged", "archived"}
    assert stats_payload["num_bullets"] >= 2
    assert "helpful_ratio" in stats_payload


@pytest.mark.asyncio
async def test_library_mcp_server_exposes_documented_playbook_resource(
    library_mcp_server, tmp_path
):
    store = Store(str(tmp_path / "ace.db"))
    playbook = store.load_playbook()
    store.close()

    async with Client(library_mcp_server) as client:
        resources = await client.list_resources()
        resource_uris = [str(resource.uri) for resource in resources]
        assert "ace://playbook.json" in resource_uris

        content = await client.read_resource("ace://playbook.json")
        payload = json.loads(parse_resource_content(content))

    assert payload == playbook.model_dump(mode="json")
