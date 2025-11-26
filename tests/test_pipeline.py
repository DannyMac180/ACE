# tests/test_pipeline.py
"""
Unit tests for the ACE Pipeline orchestration module.
Tests the full cycle: Query → Retrieve → Generator → Reflector → Curator → Merge
"""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ace.core.schema import Bullet, Playbook
from ace.core.storage.store_adapter import Store
from ace.generator.schemas import Trajectory
from ace.pipeline import Pipeline, PipelineResult, run_full_cycle
from ace.reflector.schema import BulletTag, CandidateBullet, Reflection


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
    """Create a store with seed bullets."""
    index_path = os.path.join(temp_index_dir, "test_index.idx")

    from ace.core.storage import embedding_store
    original_init = embedding_store.EmbeddingStore.__init__

    def patched_init(self, db_conn, index_path=index_path):
        original_init(self, db_conn, index_path)

    embedding_store.EmbeddingStore.__init__ = patched_init

    try:
        store = Store(temp_db)

        bullets = [
            Bullet(
                id="strat-001",
                section="strategies_and_hard_rules",
                content="Prefer hybrid retrieval: BM25 + embedding for better recall",
                tags=["topic:retrieval", "stack:python"],
            ),
            Bullet(
                id="strat-002",
                section="strategies_and_hard_rules",
                content="Never rewrite the whole playbook. Only ADD/PATCH/DEPRECATE bullets",
                tags=["topic:curation", "policy"],
            ),
            Bullet(
                id="trbl-001",
                section="troubleshooting_and_pitfalls",
                content="Check FAISS index dimension mismatch if insertions fail",
                tags=["topic:vector", "tool:faiss"],
            ),
        ]

        for bullet in bullets:
            store.save_bullet(bullet)

        yield store
        store.close()
    finally:
        embedding_store.EmbeddingStore.__init__ = original_init


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    @patch("ace.pipeline.Reflector")
    def test_pipeline_creates_store_from_db_path(self, mock_reflector_class, temp_db):
        """Pipeline should create a Store from db_path if none provided."""
        pipeline = Pipeline(db_path=temp_db)
        assert pipeline.store is not None
        assert pipeline.retriever is not None
        assert pipeline.reflector is not None
        assert pipeline.generator is not None

    @patch("ace.pipeline.Reflector")
    def test_pipeline_uses_provided_store(
        self, mock_reflector_class, store_with_bullets
    ):
        """Pipeline should use provided Store instance."""
        pipeline = Pipeline(store=store_with_bullets)
        assert pipeline.store is store_with_bullets


class TestPipelineRetrieveOnly:
    """Tests for the retrieve-only functionality."""

    @patch("ace.pipeline.Reflector")
    def test_retrieve_only_returns_bullets(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_retrieve_only should return relevant bullets."""
        pipeline = Pipeline(store=store_with_bullets)
        bullets = pipeline.run_retrieve_only("retrieval strategies")

        assert isinstance(bullets, list)
        assert all(isinstance(b, Bullet) for b in bullets)

    @patch("ace.pipeline.Reflector")
    def test_retrieve_only_respects_top_k(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_retrieve_only should respect retrieval_top_k."""
        pipeline = Pipeline(store=store_with_bullets, retrieval_top_k=1)
        bullets = pipeline.run_retrieve_only("retrieval")

        assert len(bullets) <= 1


class TestPipelineFullCycle:
    """Tests for the full pipeline cycle."""

    @patch("ace.pipeline.Reflector")
    def test_full_cycle_returns_pipeline_result(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_full_cycle should return a complete PipelineResult."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_trajectory.return_value = Reflection(
            error_identification=None,
            root_cause_analysis=None,
            correct_approach="Use hybrid retrieval",
            key_insight="Hybrid is better",
            bullet_tags=[],
            candidate_bullets=[],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        result = pipeline.run_full_cycle("test query", auto_commit=False)

        assert isinstance(result, PipelineResult)
        assert isinstance(result.playbook, Playbook)
        assert isinstance(result.trajectory, Trajectory)
        assert isinstance(result.reflection, Reflection)
        assert isinstance(result.delta_ops_applied, int)
        assert isinstance(result.retrieved_bullets, list)

    @patch("ace.pipeline.Reflector")
    def test_full_cycle_with_custom_executor(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_full_cycle should use custom tool executor if provided."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_trajectory.return_value = Reflection(
            bullet_tags=[],
            candidate_bullets=[],
        )
        mock_reflector_class.return_value = mock_reflector

        custom_executor = MagicMock(return_value="Custom execution result")

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        result = pipeline.run_full_cycle(
            "test query",
            execute_fn=custom_executor,
            auto_commit=False,
        )

        assert custom_executor.called
        assert result.trajectory.total_steps > 0

    @patch("ace.pipeline.Reflector")
    def test_full_cycle_applies_delta_when_auto_commit(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_full_cycle should apply delta when auto_commit=True."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_trajectory.return_value = Reflection(
            bullet_tags=[BulletTag(id="strat-001", tag="helpful")],
            candidate_bullets=[],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        initial_version = store_with_bullets.get_version()
        result = pipeline.run_full_cycle("test query", auto_commit=True)

        assert result.playbook.version > initial_version
        assert result.delta_ops_applied > 0

    @patch("ace.pipeline.Reflector")
    def test_full_cycle_dry_run(self, mock_reflector_class, store_with_bullets):
        """run_full_cycle with auto_commit=False should not modify playbook."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_trajectory.return_value = Reflection(
            bullet_tags=[BulletTag(id="strat-001", tag="helpful")],
            candidate_bullets=[
                CandidateBullet(
                    section="strategies_and_hard_rules",
                    content="New strategy bullet",
                    tags=["topic:test"],
                )
            ],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        initial_version = store_with_bullets.get_version()
        initial_bullet_count = len(store_with_bullets.get_all_bullets())

        result = pipeline.run_full_cycle("test query", auto_commit=False)

        assert result.playbook.version == initial_version
        assert result.delta_ops_applied == 0
        assert len(store_with_bullets.get_all_bullets()) == initial_bullet_count

    @patch("ace.pipeline.Reflector")
    def test_executor_restored_after_run(self, mock_reflector_class, store_with_bullets):
        """Custom execute_fn should not persist after run_full_cycle completes."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_trajectory.return_value = Reflection(
            bullet_tags=[],
            candidate_bullets=[],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        original_executor = pipeline.generator.tool_executor
        custom_executor = MagicMock(return_value="custom result")

        pipeline.run_full_cycle("test", execute_fn=custom_executor, auto_commit=False)

        assert pipeline.generator.tool_executor is original_executor
        assert pipeline.generator.tool_executor is not custom_executor

    @patch("ace.pipeline.Reflector")
    def test_executor_restored_even_on_error(
        self, mock_reflector_class, store_with_bullets
    ):
        """Executor should be restored even if generator raises an exception."""
        mock_reflector_class.return_value = MagicMock()

        pipeline = Pipeline(store=store_with_bullets)
        original_executor = pipeline.generator.tool_executor

        def failing_executor(action: str) -> str:
            raise RuntimeError("Simulated failure")

        pipeline.generator.retriever = None

        try:
            pipeline.run_full_cycle("test", execute_fn=failing_executor, auto_commit=False)
        except RuntimeError:
            pass

        assert pipeline.generator.tool_executor is original_executor


class TestPipelineWithFeedback:
    """Tests for pipeline with explicit feedback."""

    @patch("ace.pipeline.Reflector")
    def test_with_feedback_passes_explicit_context(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_with_feedback should pass explicit feedback to reflector."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = Reflection(
            bullet_tags=[],
            candidate_bullets=[],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        pipeline.run_with_feedback(
            query="test query",
            code_diff="--- a/file.py\n+++ b/file.py",
            test_output="PASSED 5 tests",
            logs="INFO: Process completed",
            auto_commit=False,
        )

        mock_reflector.reflect.assert_called_once()
        call_kwargs = mock_reflector.reflect.call_args.kwargs
        assert call_kwargs["code_diff"] == "--- a/file.py\n+++ b/file.py"
        assert call_kwargs["test_output"] == "PASSED 5 tests"
        assert call_kwargs["logs"] == "INFO: Process completed"

    @patch("ace.pipeline.Reflector")
    def test_with_feedback_skips_generator(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_with_feedback should NOT run the generator."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = Reflection(
            bullet_tags=[],
            candidate_bullets=[],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector
        pipeline.generator.run = MagicMock()

        result = pipeline.run_with_feedback(
            query="test query",
            code_diff="some diff",
            auto_commit=False,
        )

        pipeline.generator.run.assert_not_called()
        assert result.trajectory.steps == []
        assert result.trajectory.initial_goal == "test query"

    @patch("ace.pipeline.Reflector")
    def test_with_feedback_uses_retrieved_bullet_ids(
        self, mock_reflector_class, store_with_bullets
    ):
        """run_with_feedback should use retrieved bullet IDs for reflection."""
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = Reflection(
            bullet_tags=[],
            candidate_bullets=[],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        pipeline.run_with_feedback(
            query="retrieval",
            code_diff="diff",
            auto_commit=False,
        )

        call_kwargs = mock_reflector.reflect.call_args.kwargs
        assert "retrieved_bullet_ids" in call_kwargs
        assert len(call_kwargs["retrieved_bullet_ids"]) > 0


class TestConvenienceFunction:
    """Tests for the run_full_cycle convenience function."""

    @patch("ace.pipeline.Pipeline")
    def test_run_full_cycle_creates_pipeline(self, mock_pipeline_class, temp_db):
        """run_full_cycle function should create and run a pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.run_full_cycle.return_value = PipelineResult(
            playbook=Playbook(version=1, bullets=[]),
            trajectory=Trajectory(
                steps=[],
                initial_goal="test",
                final_status="success",
            ),
            reflection=Reflection(bullet_tags=[], candidate_bullets=[]),
            delta_ops_applied=0,
            retrieved_bullets=[],
        )
        mock_pipeline_class.return_value = mock_pipeline

        result = run_full_cycle("test query", db_path=temp_db)

        mock_pipeline_class.assert_called_once_with(db_path=temp_db)
        mock_pipeline.run_full_cycle.assert_called_once_with(
            "test query", None, True
        )
        assert isinstance(result, PipelineResult)


class TestPipelineWithNewBullets:
    """Tests for pipeline adding new bullets."""

    @patch("ace.pipeline.Reflector")
    def test_adds_candidate_bullets(self, mock_reflector_class, store_with_bullets):
        """Pipeline should add candidate bullets from reflection."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_trajectory.return_value = Reflection(
            error_identification="Test error",
            root_cause_analysis="Root cause",
            correct_approach="Fix approach",
            key_insight="Key insight",
            bullet_tags=[],
            candidate_bullets=[
                CandidateBullet(
                    section="troubleshooting_and_pitfalls",
                    content="New troubleshooting tip from pipeline test",
                    tags=["topic:test", "source:pipeline"],
                )
            ],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        initial_count = len(store_with_bullets.get_all_bullets())
        result = pipeline.run_full_cycle("test query", auto_commit=True)

        final_count = len(store_with_bullets.get_all_bullets())
        assert final_count == initial_count + 1
        assert result.delta_ops_applied == 1

        new_bullets = [
            b for b in store_with_bullets.get_all_bullets()
            if "New troubleshooting tip from pipeline test" in b.content
        ]
        assert len(new_bullets) == 1


class TestPipelineIntegration:
    """Integration tests (requires mocking LLM calls)."""

    @patch("ace.pipeline.Reflector")
    def test_full_integration_flow(self, mock_reflector_class, store_with_bullets):
        """Test complete flow with mocked reflector."""
        mock_reflector = MagicMock()
        mock_reflector.reflect_on_trajectory.return_value = Reflection(
            error_identification=None,
            root_cause_analysis=None,
            correct_approach="Implemented solution",
            key_insight="Learned something new",
            bullet_tags=[
                BulletTag(id="strat-001", tag="helpful"),
            ],
            candidate_bullets=[
                CandidateBullet(
                    section="strategies_and_hard_rules",
                    content="Integration test bullet",
                    tags=["topic:integration"],
                )
            ],
        )
        mock_reflector_class.return_value = mock_reflector

        pipeline = Pipeline(store=store_with_bullets)
        pipeline.reflector = mock_reflector

        pipeline.generator.retriever = None

        def safe_executor(action: str) -> str:
            return "done"

        result = pipeline.run_full_cycle(
            query="retrieval",
            execute_fn=safe_executor,
            auto_commit=True,
        )

        assert result.trajectory.final_status == "success"
        assert result.delta_ops_applied == 2

        strat_001 = store_with_bullets.get_bullet("strat-001")
        assert strat_001 is not None
        assert strat_001.helpful == 1
