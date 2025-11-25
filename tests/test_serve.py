"""Tests for online serving module."""

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
