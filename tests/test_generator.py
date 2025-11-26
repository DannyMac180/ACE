from unittest.mock import MagicMock

import pytest

from ace.core.schema import Bullet
from ace.generator.generator import Generator
from ace.generator.schemas import Trajectory


class TestTrajectorySchema:
    """Tests for the updated Trajectory schema."""

    def test_trajectory_includes_bullet_tracking_fields(self):
        trajectory = Trajectory(
            initial_goal="test goal",
            final_status="success",
            used_bullet_ids=["strat-001", "strat-002"],
            bullet_feedback={"strat-001": "helpful", "strat-002": "harmful"},
        )
        assert trajectory.used_bullet_ids == ["strat-001", "strat-002"]
        assert trajectory.bullet_feedback == {"strat-001": "helpful", "strat-002": "harmful"}

    def test_trajectory_default_values(self):
        trajectory = Trajectory(initial_goal="test", final_status="success")
        assert trajectory.used_bullet_ids == []
        assert trajectory.bullet_feedback == {}


class TestGeneratorWithoutRetriever:
    """Tests for Generator without retriever (existing behavior)."""

    def test_generator_runs_without_retriever(self):
        gen = Generator(max_steps=3)
        traj = gen.run("test goal")
        assert traj.final_status == "success"
        assert traj.total_steps > 0
        assert traj.used_bullet_ids == []

    def test_generator_custom_tool_executor(self):
        def custom_executor(action: str) -> str:
            return f"Custom result for: {action}"

        gen = Generator(max_steps=2, tool_executor=custom_executor)
        traj = gen.run("test goal")
        assert "Custom result" in traj.steps[0].observation


class TestGeneratorWithRetriever:
    """Tests for Generator with retriever integration."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever with some test bullets."""
        retriever = MagicMock()
        retriever.store = MagicMock()

        test_bullets = [
            Bullet(
                id="strat-001",
                section="strategies_and_hard_rules",
                content="Use hybrid retrieval for best results",
                tags=["topic:retrieval"],
            ),
            Bullet(
                id="tmpl-001",
                section="code_snippets_and_templates",
                content="Always validate input before processing",
                tags=["topic:validation"],
            ),
        ]

        retriever.retrieve.return_value = test_bullets
        retriever.store.get_bullet.side_effect = lambda bid: next(
            (b for b in test_bullets if b.id == bid), None
        )
        return retriever

    def test_generator_with_retriever_tracks_bullets(self, mock_retriever):
        gen = Generator(max_steps=2, retriever=mock_retriever, retrieval_top_k=5)
        traj = gen.run("implement retrieval")

        assert "strat-001" in traj.used_bullet_ids
        assert "tmpl-001" in traj.used_bullet_ids
        mock_retriever.retrieve.assert_called()

    def test_generator_retrieves_before_each_step(self, mock_retriever):
        gen = Generator(max_steps=3, retriever=mock_retriever)
        gen.run("test goal")

        assert mock_retriever.retrieve.call_count >= 1

    def test_generator_builds_query_with_context(self, mock_retriever):
        gen = Generator(max_steps=2, retriever=mock_retriever)
        gen.run("my test goal")

        first_call_query = mock_retriever.retrieve.call_args_list[0][0][0]
        assert "my test goal" in first_call_query

    def test_bullet_context_appears_in_thought(self, mock_retriever):
        gen = Generator(max_steps=2, retriever=mock_retriever)
        traj = gen.run("test goal")

        thought_with_context = traj.steps[0].thought
        assert "Relevant playbook guidance" in thought_with_context
        assert "hybrid retrieval" in thought_with_context

    def test_generator_respects_retrieval_top_k(self, mock_retriever):
        gen = Generator(max_steps=2, retriever=mock_retriever, retrieval_top_k=10)
        gen.run("test")

        _, kwargs = mock_retriever.retrieve.call_args
        assert kwargs.get("top_k") == 10


class TestGeneratorBulletRetrieval:
    """Tests for bullet retrieval helper methods."""

    def test_retrieve_bullets_without_retriever(self):
        gen = Generator()
        result = gen._retrieve_bullets("query", set())
        assert result == []

    def test_format_bullets_without_retriever(self):
        gen = Generator()
        result = gen._format_bullets_for_reasoning(["id1", "id2"])
        assert result == ""

    def test_format_bullets_empty_list(self):
        mock_retriever = MagicMock()
        gen = Generator(retriever=mock_retriever)
        result = gen._format_bullets_for_reasoning([])
        assert result == ""

    def test_format_bullets_includes_section(self):
        mock_retriever = MagicMock()
        mock_retriever.store.get_bullet.return_value = Bullet(
            id="strat-001",
            section="strategies_and_hard_rules",
            content="Test content",
            tags=[],
        )
        gen = Generator(retriever=mock_retriever)
        result = gen._format_bullets_for_reasoning(["strat-001"])

        assert "[strategies_and_hard_rules]" in result
        assert "Test content" in result
