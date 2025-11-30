"""Tests for online serving module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ace.serve.runner import OnlineServer, create_app
from ace.serve.schema import (
    AdaptationMode,
    FeedbackRequest,
    FeedbackResponse,
    OnlineStats,
    RetrieveResponse,
    WarmupSource,
)


class TestAdaptationMode:
    """Test AdaptationMode enum."""

    def test_offline_mode(self):
        assert AdaptationMode.OFFLINE == "offline"
        assert AdaptationMode.OFFLINE.value == "offline"

    def test_online_mode(self):
        assert AdaptationMode.ONLINE == "online"
        assert AdaptationMode.ONLINE.value == "online"


class TestFeedbackRequest:
    """Test FeedbackRequest schema."""

    def test_minimal_request(self):
        req = FeedbackRequest(query="test query")
        assert req.query == "test query"
        assert req.retrieved_bullet_ids == []
        assert req.code_diff == ""
        assert req.test_output == ""
        assert req.logs == ""
        assert req.env_meta is None
        assert req.execution_success is None
        assert req.error_message is None

    def test_full_request(self):
        req = FeedbackRequest(
            query="test query",
            retrieved_bullet_ids=["b1", "b2"],
            code_diff="diff content",
            test_output="PASSED",
            logs="some logs",
            env_meta={"key": "value"},
            execution_success=True,
            error_message=None,
        )
        assert req.query == "test query"
        assert len(req.retrieved_bullet_ids) == 2
        assert req.execution_success is True


class TestFeedbackResponse:
    """Test FeedbackResponse schema."""

    def test_success_response(self):
        resp = FeedbackResponse(
            success=True,
            ops_applied=3,
            playbook_version=5,
            adaptation_ms=42.5,
            message="Applied 3 operations",
        )
        assert resp.success is True
        assert resp.ops_applied == 3
        assert resp.playbook_version == 5
        assert resp.adaptation_ms == 42.5

    def test_failure_response(self):
        resp = FeedbackResponse(
            success=False,
            message="Error occurred",
        )
        assert resp.success is False
        assert resp.ops_applied == 0


class TestWarmupSource:
    """Test WarmupSource enum."""

    def test_none_source(self):
        assert WarmupSource.NONE == "none"
        assert WarmupSource.NONE.value == "none"

    def test_file_source(self):
        assert WarmupSource.FILE == "file"
        assert WarmupSource.FILE.value == "file"

    def test_database_source(self):
        assert WarmupSource.DATABASE == "database"
        assert WarmupSource.DATABASE.value == "database"


class TestOnlineStats:
    """Test OnlineStats schema."""

    def test_default_stats(self):
        stats = OnlineStats(session_id="test-123")
        assert stats.session_id == "test-123"
        assert stats.requests_processed == 0
        assert stats.total_ops_applied == 0
        assert stats.helpful_feedback_count == 0
        assert stats.harmful_feedback_count == 0
        assert stats.avg_adaptation_ms == 0.0
        assert stats.warmup_source == WarmupSource.NONE
        assert stats.warmup_bullets_loaded == 0
        assert stats.warmup_playbook_version == 0

    def test_stats_with_warmup(self):
        stats = OnlineStats(
            session_id="test-456",
            warmup_source=WarmupSource.FILE,
            warmup_bullets_loaded=10,
            warmup_playbook_version=5,
        )
        assert stats.warmup_source == WarmupSource.FILE
        assert stats.warmup_bullets_loaded == 10
        assert stats.warmup_playbook_version == 5


class TestOnlineServer:
    """Test OnlineServer class."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        playbook = MagicMock()
        playbook.version = 1
        playbook.bullets = []
        store.load_playbook.return_value = playbook
        return store

    @pytest.fixture
    def mock_reflector(self):
        reflector = MagicMock()
        reflector.reflect.return_value = MagicMock(
            error_identification=None,
            root_cause_analysis=None,
            correct_approach=None,
            key_insight="test insight",
            bullet_tags=[],
            candidate_bullets=[],
        )
        return reflector

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        return retriever

    def test_init_with_defaults(self, mock_store, mock_reflector, mock_retriever):
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )
        assert server.mode == AdaptationMode.ONLINE
        assert server.auto_adapt is True
        assert server.session_id is not None
        assert len(server.session_id) == 8

    def test_init_no_auto_adapt(self, mock_store, mock_reflector, mock_retriever):
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            auto_adapt=False,
        )
        assert server.auto_adapt is False

    def test_init_cold_start(self, mock_store, mock_reflector, mock_retriever):
        """Test that empty database results in cold start."""
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )
        assert server.stats.warmup_source == WarmupSource.NONE
        assert server.stats.warmup_bullets_loaded == 0
        assert server.stats.warmup_playbook_version == 0

    def test_init_database_warmup(self, mock_reflector, mock_retriever):
        """Test that existing database bullets are detected as warmup."""
        store = MagicMock()
        playbook = MagicMock()
        playbook.version = 3
        mock_bullet = MagicMock()
        playbook.bullets = [mock_bullet, mock_bullet, mock_bullet]
        store.load_playbook.return_value = playbook

        server = OnlineServer(
            store=store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )
        assert server.stats.warmup_source == WarmupSource.DATABASE
        assert server.stats.warmup_bullets_loaded == 3
        assert server.stats.warmup_playbook_version == 3

    def test_init_file_warmup(self, mock_store, mock_reflector, mock_retriever):
        """Test warm-start from a playbook JSON file."""
        playbook_data = {
            "version": 5,
            "bullets": [
                {
                    "id": "strat-001",
                    "section": "strategies_and_hard_rules",
                    "content": "Test bullet",
                    "tags": ["topic:test"],
                    "helpful": 0,
                    "harmful": 0,
                },
                {
                    "id": "strat-002",
                    "section": "strategies_and_hard_rules",
                    "content": "Second bullet",
                    "tags": ["topic:test"],
                    "helpful": 1,
                    "harmful": 0,
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(playbook_data, f)
            temp_path = f.name

        try:
            server = OnlineServer(
                store=mock_store,
                reflector=mock_reflector,
                retriever=mock_retriever,
                warmup_path=temp_path,
            )
            assert server.stats.warmup_source == WarmupSource.FILE
            assert server.stats.warmup_bullets_loaded == 2
            assert server.stats.warmup_playbook_version == 5
            mock_store.load_playbook_data.assert_called_once()
        finally:
            Path(temp_path).unlink()

    def test_init_file_warmup_not_found(self, mock_store, mock_reflector, mock_retriever):
        """Test that missing warmup file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            OnlineServer(
                store=mock_store,
                reflector=mock_reflector,
                retriever=mock_retriever,
                warmup_path="/nonexistent/path/playbook.json",
            )
        assert "Warmup playbook not found" in str(exc_info.value)

    def test_init_file_warmup_invalid_json(self, mock_store, mock_reflector, mock_retriever):
        """Test that invalid JSON in warmup file raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                OnlineServer(
                    store=mock_store,
                    reflector=mock_reflector,
                    retriever=mock_retriever,
                    warmup_path=temp_path,
                )
        finally:
            Path(temp_path).unlink()

    def test_retrieve(self, mock_store, mock_reflector, mock_retriever):
        mock_bullet = MagicMock()
        mock_bullet.model_dump.return_value = {"id": "b1", "content": "test"}
        mock_retriever.retrieve.return_value = [mock_bullet]

        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )
        result = server.retrieve("test query", top_k=10)

        assert isinstance(result, RetrieveResponse)
        assert len(result.bullets) == 1
        assert result.retrieval_ms >= 0
        mock_retriever.retrieve.assert_called_once_with("test query", top_k=10)

    def test_process_feedback_no_ops(self, mock_store, mock_reflector, mock_retriever):
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )

        with patch("ace.serve.runner.curate") as mock_curate:
            mock_delta = MagicMock()
            mock_delta.ops = []
            mock_curate.return_value = mock_delta

            request = FeedbackRequest(query="test query")
            result = server.process_feedback(request)

            assert result.success is True
            assert result.ops_applied == 0
            assert result.message == "No adaptation needed"

    def test_process_feedback_with_ops(self, mock_store, mock_reflector, mock_retriever):
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )

        with patch("ace.serve.runner.curate") as mock_curate:
            with patch("ace.serve.runner.apply_delta") as mock_apply:
                mock_op = MagicMock()
                mock_op.op = "ADD"
                mock_delta = MagicMock()
                mock_delta.ops = [mock_op]
                mock_delta.model_dump.return_value = {"ops": [{"op": "ADD"}]}
                mock_curate.return_value = mock_delta

                new_playbook = MagicMock()
                new_playbook.version = 2
                mock_apply.return_value = new_playbook

                request = FeedbackRequest(
                    query="test query",
                    test_output="PASSED",
                    execution_success=True,
                )
                result = server.process_feedback(request)

                assert result.success is True
                assert result.ops_applied == 1
                assert result.playbook_version == 2

    def test_get_stats(self, mock_store, mock_reflector, mock_retriever):
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )
        stats = server.get_stats()
        assert isinstance(stats, OnlineStats)
        assert stats.session_id == server.session_id

    def test_get_playbook_version(self, mock_store, mock_reflector, mock_retriever):
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )
        version = server.get_playbook_version()
        assert version == 1


class TestCreateApp:
    """Test FastAPI app creation."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        playbook = MagicMock()
        playbook.version = 1
        playbook.bullets = []
        store.load_playbook.return_value = playbook
        return store

    def test_health_endpoint(self, mock_store):
        with patch("ace.serve.runner.Reflector"):
            with patch("ace.serve.runner.Retriever"):
                app = create_app(store=mock_store)
                with TestClient(app) as client:
                    response = client.get("/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ok"
                    assert data["mode"] == "online"

    def test_stats_endpoint(self, mock_store):
        with patch("ace.serve.runner.Reflector"):
            with patch("ace.serve.runner.Retriever"):
                app = create_app(store=mock_store)
                with TestClient(app) as client:
                    response = client.get("/stats")
                    assert response.status_code == 200
                    data = response.json()
                    assert "session_id" in data
                    assert "requests_processed" in data
                    assert "warmup_source" in data
                    assert "warmup_bullets_loaded" in data
                    assert "warmup_playbook_version" in data

    def test_stats_endpoint_with_warmup(self, mock_store):
        playbook_data = {
            "version": 10,
            "bullets": [
                {
                    "id": "strat-001",
                    "section": "strategies_and_hard_rules",
                    "content": "Test bullet",
                    "tags": [],
                    "helpful": 0,
                    "harmful": 0,
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(playbook_data, f)
            temp_path = f.name

        try:
            with patch("ace.serve.runner.Reflector"):
                with patch("ace.serve.runner.Retriever"):
                    app = create_app(store=mock_store, warmup_path=temp_path)
                    with TestClient(app) as client:
                        response = client.get("/stats")
                        assert response.status_code == 200
                        data = response.json()
                        assert data["warmup_source"] == "file"
                        assert data["warmup_bullets_loaded"] == 1
                        assert data["warmup_playbook_version"] == 10
        finally:
            Path(temp_path).unlink()

    def test_playbook_version_endpoint(self, mock_store):
        with patch("ace.serve.runner.Reflector"):
            with patch("ace.serve.runner.Retriever"):
                app = create_app(store=mock_store)
                with TestClient(app) as client:
                    response = client.get("/playbook/version")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["version"] == 1

    def test_retrieve_endpoint(self, mock_store):
        with patch("ace.serve.runner.Reflector"):
            with patch("ace.serve.runner.Retriever") as mock_ret_cls:
                mock_retriever = MagicMock()
                mock_retriever.retrieve.return_value = []
                mock_ret_cls.return_value = mock_retriever

                app = create_app(store=mock_store)
                with TestClient(app) as client:
                    response = client.post("/retrieve", json={"query": "test", "top_k": 5})
                    assert response.status_code == 200
                    data = response.json()
                    assert "bullets" in data
                    assert "retrieval_ms" in data

    def test_feedback_endpoint(self, mock_store):
        with patch("ace.serve.runner.Reflector") as mock_ref_cls:
            with patch("ace.serve.runner.Retriever"):
                with patch("ace.serve.runner.curate") as mock_curate:
                    mock_reflector = MagicMock()
                    mock_reflector.reflect.return_value = MagicMock()
                    mock_ref_cls.return_value = mock_reflector

                    mock_delta = MagicMock()
                    mock_delta.ops = []
                    mock_curate.return_value = mock_delta

                    app = create_app(store=mock_store)
                    with TestClient(app) as client:
                        response = client.post("/feedback", json={"query": "test query"})
                        assert response.status_code == 200
                        data = response.json()
                        assert data["success"] is True


class TestAutoRefine:
    """Test auto-refine policy in OnlineServer."""

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        playbook = MagicMock()
        playbook.version = 1
        playbook.bullets = []
        store.load_playbook.return_value = playbook
        return store

    @pytest.fixture
    def mock_reflector(self):
        reflector = MagicMock()
        reflector.reflect.return_value = MagicMock(
            error_identification=None,
            root_cause_analysis=None,
            correct_approach=None,
            key_insight="test insight",
            bullet_tags=[],
            candidate_bullets=[],
        )
        return reflector

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        return retriever

    def test_init_with_auto_refine_every(self, mock_store, mock_reflector, mock_retriever):
        """Test that auto_refine_every is properly initialized."""
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            auto_refine_every=100,
        )
        assert server.auto_refine_every == 100
        assert server._delta_count_since_refine == 0

    def test_init_with_max_bullets(self, mock_store, mock_reflector, mock_retriever):
        """Test that max_bullets is properly initialized."""
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            max_bullets=500,
        )
        assert server.max_bullets == 500

    def test_delta_count_increments_on_feedback(self, mock_store, mock_reflector, mock_retriever):
        """Test that delta count increments after processing feedback with ops."""
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            auto_refine_every=10,
        )

        with patch("ace.serve.runner.curate") as mock_curate:
            with patch("ace.serve.runner.apply_delta") as mock_apply:
                mock_op = MagicMock()
                mock_op.op = "ADD"
                mock_delta = MagicMock()
                mock_delta.ops = [mock_op]
                mock_delta.model_dump.return_value = {"ops": [{"op": "ADD"}]}
                mock_curate.return_value = mock_delta

                new_playbook = MagicMock()
                new_playbook.version = 2
                mock_apply.return_value = new_playbook

                request = FeedbackRequest(query="test query")
                server.process_feedback(request)

                assert server._delta_count_since_refine == 1

    def test_auto_refine_triggers_on_delta_count(self, mock_store, mock_reflector, mock_retriever):
        """Test that auto-refine triggers when delta count reaches threshold."""
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            auto_refine_every=2,
        )

        with patch("ace.serve.runner.curate") as mock_curate:
            with patch("ace.serve.runner.apply_delta") as mock_apply:
                with patch("ace.serve.runner.run_refine") as mock_refine:
                    mock_op = MagicMock()
                    mock_op.op = "ADD"
                    mock_delta = MagicMock()
                    mock_delta.ops = [mock_op]
                    mock_delta.model_dump.return_value = {"ops": [{"op": "ADD"}]}
                    mock_curate.return_value = mock_delta

                    new_playbook = MagicMock()
                    new_playbook.version = 2
                    mock_apply.return_value = new_playbook

                    mock_refine_result = MagicMock()
                    mock_refine_result.merged = 1
                    mock_refine_result.archived = 0
                    mock_refine.return_value = mock_refine_result

                    request = FeedbackRequest(query="test")
                    server.process_feedback(request)
                    mock_refine.assert_not_called()

                    server.process_feedback(request)
                    mock_refine.assert_called_once()

                    assert server._delta_count_since_refine == 0
                    assert server.stats.auto_refine_runs == 1
                    assert server.stats.auto_refine_merged == 1

    def test_auto_refine_triggers_on_max_bullets(self, mock_reflector, mock_retriever):
        """Test that auto-refine triggers when bullet count exceeds max_bullets."""
        store = MagicMock()
        playbook = MagicMock()
        playbook.version = 1
        mock_bullets = [MagicMock() for _ in range(6)]
        playbook.bullets = mock_bullets
        store.load_playbook.return_value = playbook

        server = OnlineServer(
            store=store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            max_bullets=5,
        )

        with patch("ace.serve.runner.curate") as mock_curate:
            with patch("ace.serve.runner.apply_delta") as mock_apply:
                with patch("ace.serve.runner.run_refine") as mock_refine:
                    mock_op = MagicMock()
                    mock_op.op = "ADD"
                    mock_delta = MagicMock()
                    mock_delta.ops = [mock_op]
                    mock_delta.model_dump.return_value = {"ops": [{"op": "ADD"}]}
                    mock_curate.return_value = mock_delta

                    new_playbook = MagicMock()
                    new_playbook.version = 2
                    mock_apply.return_value = new_playbook

                    mock_refine_result = MagicMock()
                    mock_refine_result.merged = 2
                    mock_refine_result.archived = 1
                    mock_refine.return_value = mock_refine_result

                    request = FeedbackRequest(query="test")
                    server.process_feedback(request)

                    mock_refine.assert_called_once()
                    assert server.stats.auto_refine_runs == 1
                    assert server.stats.auto_refine_merged == 2
                    assert server.stats.auto_refine_archived == 1

    def test_no_auto_refine_when_disabled(self, mock_store, mock_reflector, mock_retriever):
        """Test that auto-refine does not trigger when disabled (both params = 0)."""
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            auto_refine_every=0,
            max_bullets=0,
        )

        with patch("ace.serve.runner.curate") as mock_curate:
            with patch("ace.serve.runner.apply_delta") as mock_apply:
                with patch("ace.serve.runner.run_refine") as mock_refine:
                    mock_op = MagicMock()
                    mock_op.op = "ADD"
                    mock_delta = MagicMock()
                    mock_delta.ops = [mock_op]
                    mock_delta.model_dump.return_value = {"ops": [{"op": "ADD"}]}
                    mock_curate.return_value = mock_delta

                    new_playbook = MagicMock()
                    new_playbook.version = 2
                    mock_apply.return_value = new_playbook

                    request = FeedbackRequest(query="test")
                    for _ in range(10):
                        server.process_feedback(request)

                    mock_refine.assert_not_called()
                    assert server.stats.auto_refine_runs == 0

    def test_stats_include_auto_refine_fields(self, mock_store, mock_reflector, mock_retriever):
        """Test that stats include auto-refine tracking fields."""
        server = OnlineServer(
            store=mock_store,
            reflector=mock_reflector,
            retriever=mock_retriever,
        )
        stats = server.get_stats()
        assert hasattr(stats, "auto_refine_runs")
        assert hasattr(stats, "auto_refine_merged")
        assert hasattr(stats, "auto_refine_archived")
        assert stats.auto_refine_runs == 0
        assert stats.auto_refine_merged == 0
        assert stats.auto_refine_archived == 0

    def test_auto_refine_deletes_removed_bullets_from_store(self, mock_reflector, mock_retriever):
        """Test that auto-refine properly deletes archived bullets from the store."""
        store = MagicMock()
        
        bullet1 = MagicMock()
        bullet1.id = "bullet-1"
        bullet2 = MagicMock()
        bullet2.id = "bullet-2"
        bullet3 = MagicMock()
        bullet3.id = "bullet-3"
        
        playbook = MagicMock()
        playbook.version = 1
        playbook.bullets = [bullet1, bullet2, bullet3]
        store.load_playbook.return_value = playbook

        server = OnlineServer(
            store=store,
            reflector=mock_reflector,
            retriever=mock_retriever,
            max_bullets=2,
        )

        with patch("ace.serve.runner.curate") as mock_curate:
            with patch("ace.serve.runner.apply_delta") as mock_apply:
                with patch("ace.serve.runner.run_refine") as mock_refine:
                    mock_op = MagicMock()
                    mock_op.op = "ADD"
                    mock_delta = MagicMock()
                    mock_delta.ops = [mock_op]
                    mock_delta.model_dump.return_value = {"ops": [{"op": "ADD"}]}
                    mock_curate.return_value = mock_delta

                    new_playbook = MagicMock()
                    new_playbook.version = 2
                    mock_apply.return_value = new_playbook

                    def simulate_refine(reflection, pb, threshold):
                        pb.bullets = [bullet1, bullet2]
                        result = MagicMock()
                        result.merged = 0
                        result.archived = 1
                        return result

                    mock_refine.side_effect = simulate_refine

                    request = FeedbackRequest(query="test")
                    server.process_feedback(request)

                    store.delete_bullet.assert_called_once_with("bullet-3")
                    
                    assert store.save_bullet.call_count == 2
                    saved_ids = [call[0][0].id for call in store.save_bullet.call_args_list]
                    assert "bullet-1" in saved_ids
                    assert "bullet-2" in saved_ids
