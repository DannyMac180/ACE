# tests/test_mcp_server.py
"""
Tests for the ACE MCP Server.

Tests that:
1. All expected tools are exposed (ace_retrieve, ace_reflect, ace_curate, ace_commit,
   ace_refine, ace_stats, ace_record_trajectory)
2. Each tool returns schema-correct responses
3. The ace://playbook.json resource returns the same structure as `ace playbook dump`
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest
from fastmcp import Client

from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store


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
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for FAISS indices."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def store_with_bullets(temp_db, temp_index_dir):
    """Create a store with seed bullets for testing."""
    index_path = os.path.join(temp_index_dir, "test_index.idx")

    from ace.core.storage import embedding_store

    original_init = embedding_store.EmbeddingStore.__init__

    def patched_init(self, db_conn, index_path=index_path):
        original_init(self, db_conn, index_path)

    embedding_store.EmbeddingStore.__init__ = patched_init  # type: ignore[method-assign]

    try:
        store = Store(temp_db)

        bullets = [
            Bullet(
                id="strat-001",
                section="strategies_and_hard_rules",
                content="Prefer hybrid retrieval: BM25 + embedding for better recall",
                tags=["topic:retrieval", "stack:python"],
                helpful=3,
                harmful=0,
            ),
            Bullet(
                id="strat-002",
                section="strategies_and_hard_rules",
                content="Never rewrite the whole playbook. Only ADD/PATCH/DEPRECATE bullets",
                tags=["topic:curation", "policy"],
                helpful=5,
                harmful=1,
            ),
            Bullet(
                id="trbl-001",
                section="troubleshooting_and_pitfalls",
                content="Check FAISS index dimension mismatch if insertions fail",
                tags=["topic:vector", "tool:faiss"],
                helpful=2,
                harmful=0,
            ),
        ]

        for bullet in bullets:
            store.save_bullet(bullet)

        yield store
        store.close()
    finally:
        embedding_store.EmbeddingStore.__init__ = original_init  # type: ignore[method-assign]


@pytest.fixture
def mcp_server(store_with_bullets):
    """Create an MCP server instance with mocked dependencies."""
    # Remove cached module to force reimport with clean state
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("ace_mcp_server"):
            del sys.modules[mod_name]

    # Now import after conftest has set env
    # Patch the module-level instances
    import ace_mcp_server.server as server_module
    from ace_mcp_server.server import app

    # Create mock retriever that uses the test store
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [store_with_bullets.get_bullet("strat-001")]

    # Patch module-level instances
    server_module.store = store_with_bullets
    server_module.retriever = mock_retriever

    yield app


EXPECTED_TOOLS = [
    "ace_retrieve",
    "ace_reflect",
    "ace_curate",
    "ace_commit",
    "ace_refine",
    "ace_stats",
    "ace_record_trajectory",
    "ace_pipeline",
]


class TestMCPServerToolListing:
    """Tests for MCP server tool discovery."""

    @pytest.mark.asyncio
    async def test_lists_all_expected_tools(self, mcp_server):
        """MCP server should expose all expected tools."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            for expected_tool in EXPECTED_TOOLS:
                assert (
                    expected_tool in tool_names
                ), f"Missing tool: {expected_tool}"

    @pytest.mark.asyncio
    async def test_tool_count(self, mcp_server):
        """MCP server should have the expected number of tools."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            assert len(tools) >= len(
                EXPECTED_TOOLS
            ), f"Expected at least {len(EXPECTED_TOOLS)} tools, got {len(tools)}"


class TestAceRetrieveTool:
    """Tests for ace_retrieve tool."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_list(self, mcp_server):
        """ace_retrieve should return a list of bullet dicts."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "ace_retrieve", {"query": "retrieval", "top_k": 5}
            )

            assert result is not None
            data = parse_tool_result(result)
            parsed = json.loads(data) if isinstance(data, str) else data
            assert isinstance(parsed, list)

    @pytest.mark.asyncio
    async def test_retrieve_bullets_have_required_fields(self, mcp_server):
        """Retrieved bullets should have required schema fields."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "ace_retrieve", {"query": "retrieval", "top_k": 5}
            )

            data = parse_tool_result(result)
            bullets = json.loads(data) if isinstance(data, str) else data

            if bullets:
                bullet = bullets[0]
                required_fields = ["id", "section", "content", "tags"]
                for field in required_fields:
                    assert field in bullet, f"Missing field: {field}"


class TestAceCommitTool:
    """Tests for ace_commit tool."""

    @pytest.mark.asyncio
    async def test_commit_returns_version(self, mcp_server):
        """ace_commit should return new version number."""
        async with Client(mcp_server) as client:
            delta = {"ops": [{"op": "INCR_HELPFUL", "target_id": "strat-001"}]}
            result = await client.call_tool("ace_commit", {"delta": delta})

            assert result is not None
            data = parse_tool_result(result)
            commit_result = json.loads(data) if isinstance(data, str) else data

            assert "version" in commit_result, "Result should have 'version' field"
            assert isinstance(
                commit_result["version"], int
            ), "'version' should be an integer"


class TestAceRefineTool:
    """Tests for ace_refine tool."""

    @pytest.mark.asyncio
    async def test_refine_returns_counts(self, mcp_server):
        """ace_refine should return merged and archived counts."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("ace_refine", {"threshold": 0.90})

            assert result is not None
            data = parse_tool_result(result)
            refine_result = json.loads(data) if isinstance(data, str) else data

            assert "merged" in refine_result, "Result should have 'merged' field"
            assert "archived" in refine_result, "Result should have 'archived' field"
            assert isinstance(refine_result["merged"], int)
            assert isinstance(refine_result["archived"], int)


class TestAceStatsTool:
    """Tests for ace_stats tool."""

    @pytest.mark.asyncio
    async def test_stats_returns_playbook_stats(self, mcp_server):
        """ace_stats should return playbook statistics."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("ace_stats", {})

            assert result is not None
            data = parse_tool_result(result)
            stats = json.loads(data) if isinstance(data, str) else data

            expected_fields = ["version", "num_bullets", "helpful_ratio"]
            for field in expected_fields:
                assert field in stats, f"Missing stats field: {field}"

    @pytest.mark.asyncio
    async def test_stats_num_bullets_is_correct(self, mcp_server):
        """ace_stats num_bullets should match actual bullet count."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("ace_stats", {})

            data = parse_tool_result(result)
            stats = json.loads(data) if isinstance(data, str) else data

            assert stats["num_bullets"] == 3, "Expected 3 seed bullets"


class TestAceRecordTrajectoryTool:
    """Tests for ace_record_trajectory tool."""

    @pytest.mark.asyncio
    async def test_record_trajectory_returns_id(self, mcp_server):
        """ace_record_trajectory should return a trajectory identifier."""
        async with Client(mcp_server) as client:
            doc = {
                "initial_goal": "Test goal",
                "steps": [],
                "final_status": "success",
            }
            result = await client.call_tool("ace_record_trajectory", {"doc": doc})

            assert result is not None
            data = parse_tool_result(result)
            assert data is not None and len(data) > 0


class TestPlaybookResource:
    """Tests for the ace://playbook.json resource."""

    @pytest.mark.asyncio
    async def test_playbook_resource_exists(self, mcp_server):
        """ace://playbook.json resource should be listed."""
        async with Client(mcp_server) as client:
            resources = await client.list_resources()
            resource_uris = [str(r.uri) for r in resources]

            assert any(
                "playbook.json" in uri for uri in resource_uris
            ), f"Expected ace://playbook.json in {resource_uris}"

    @pytest.mark.asyncio
    async def test_playbook_resource_returns_valid_json(self, mcp_server):
        """ace://playbook.json should return valid JSON."""
        async with Client(mcp_server) as client:
            resources = await client.list_resources()
            playbook_uri = None
            for r in resources:
                if "playbook.json" in str(r.uri):
                    playbook_uri = str(r.uri)
                    break

            if playbook_uri:
                content = await client.read_resource(playbook_uri)
                text = parse_resource_content(content)
                playbook = json.loads(text)

                assert "version" in playbook, "Playbook should have 'version' field"
                assert "bullets" in playbook, "Playbook should have 'bullets' field"
                assert isinstance(playbook["bullets"], list)

    @pytest.mark.asyncio
    async def test_playbook_resource_matches_store(self, mcp_server, store_with_bullets):
        """ace://playbook.json should match the playbook from store."""
        async with Client(mcp_server) as client:
            resources = await client.list_resources()
            playbook_uri = None
            for r in resources:
                if "playbook.json" in str(r.uri):
                    playbook_uri = str(r.uri)
                    break

            if playbook_uri:
                content = await client.read_resource(playbook_uri)
                text = parse_resource_content(content)
                resource_playbook = json.loads(text)

                store_playbook = store_with_bullets.load_playbook()
                store_playbook_dict = store_playbook.model_dump()

                assert resource_playbook["version"] == store_playbook_dict["version"]
                assert len(resource_playbook["bullets"]) == len(
                    store_playbook_dict["bullets"]
                )


class TestPlaybookResourceMatchesDump:
    """Test that ace://playbook.json matches what `ace playbook dump` would output."""

    @pytest.mark.asyncio
    async def test_resource_structure_matches_cli_dump(
        self, mcp_server, store_with_bullets
    ):
        """ace://playbook.json should have the same structure as CLI dump."""
        async with Client(mcp_server) as client:
            resources = await client.list_resources()
            playbook_uri = None
            for r in resources:
                if "playbook.json" in str(r.uri):
                    playbook_uri = str(r.uri)
                    break

            if playbook_uri:
                content = await client.read_resource(playbook_uri)
                text = parse_resource_content(content)
                resource_playbook = json.loads(text)

                playbook = store_with_bullets.load_playbook()
                cli_dump = playbook.model_dump()

                assert set(resource_playbook.keys()) == set(
                    cli_dump.keys()
                ), "Keys should match"

                if resource_playbook["bullets"] and cli_dump["bullets"]:
                    resource_bullet_keys = set(resource_playbook["bullets"][0].keys())
                    cli_bullet_keys = set(cli_dump["bullets"][0].keys())
                    assert (
                        resource_bullet_keys == cli_bullet_keys
                    ), "Bullet fields should match"


class TestMCPServerRoundTrip:
    """End-to-end round-trip tests for MCP tools."""

    @pytest.mark.asyncio
    async def test_retrieve_then_tag_roundtrip(self, mcp_server):
        """Retrieve a bullet, then mark it as helpful via commit."""
        async with Client(mcp_server) as client:
            initial_stats = await client.call_tool("ace_stats", {})
            stats_data = parse_tool_result(initial_stats)
            initial_version = json.loads(stats_data)["version"]

            delta = {"ops": [{"op": "INCR_HELPFUL", "target_id": "strat-001"}]}
            commit_result = await client.call_tool("ace_commit", {"delta": delta})
            commit_data = parse_tool_result(commit_result)
            new_version = json.loads(commit_data)["version"]

            assert (
                new_version > initial_version
            ), "Version should increment after commit"
