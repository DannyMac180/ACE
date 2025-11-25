# tests/test_reflector.py
from unittest.mock import MagicMock, patch

import pytest

from ace.generator.schemas import Step, Trajectory
from ace.reflector import (
    BulletTag,
    CandidateBullet,
    QualityParseError,
    RefinementQuality,
    Reflection,
    ReflectionParseError,
    Reflector,
    parse_quality,
    parse_reflection,
)


def test_parse_reflection_valid():
    """Test parsing valid reflection JSON."""
    json_str = """
    {
      "error_identification": "Test failed due to missing import",
      "root_cause_analysis": "Module was not installed",
      "correct_approach": "Add dependency to requirements",
      "key_insight": "Always check imports before testing",
      "bullet_tags": [
        {"id": "strat-00091", "tag": "helpful"}
      ],
      "candidate_bullets": [
        {
          "section": "troubleshooting",
          "content": "Check imports before running tests",
          "tags": ["topic:testing", "tool:pytest"]
        }
      ]
    }
    """

    reflection = parse_reflection(json_str)

    assert reflection.error_identification == "Test failed due to missing import"
    assert reflection.root_cause_analysis == "Module was not installed"
    assert reflection.correct_approach == "Add dependency to requirements"
    assert reflection.key_insight == "Always check imports before testing"
    assert len(reflection.bullet_tags) == 1
    assert reflection.bullet_tags[0].id == "strat-00091"
    assert reflection.bullet_tags[0].tag == "helpful"
    assert len(reflection.candidate_bullets) == 1
    assert reflection.candidate_bullets[0].section == "troubleshooting"
    assert reflection.candidate_bullets[0].content == "Check imports before running tests"
    assert "topic:testing" in reflection.candidate_bullets[0].tags


def test_parse_reflection_with_markdown_fencing():
    """Test parsing reflection with markdown code fencing."""
    json_str = """```json
    {
      "error_identification": "Test failed",
      "root_cause_analysis": null,
      "correct_approach": null,
      "key_insight": null,
      "bullet_tags": [],
      "candidate_bullets": []
    }
    ```"""

    reflection = parse_reflection(json_str)
    assert reflection.error_identification == "Test failed"
    assert reflection.root_cause_analysis is None


def test_parse_reflection_minimal():
    """Test parsing minimal reflection with no optional fields."""
    json_str = '{"bullet_tags": [], "candidate_bullets": []}'

    reflection = parse_reflection(json_str)
    assert reflection.error_identification is None
    assert reflection.root_cause_analysis is None
    assert reflection.correct_approach is None
    assert reflection.key_insight is None
    assert len(reflection.bullet_tags) == 0
    assert len(reflection.candidate_bullets) == 0


def test_parse_reflection_invalid_json():
    """Test that invalid JSON raises ReflectionParseError."""
    json_str = "{ invalid json }"

    with pytest.raises(ReflectionParseError, match="Invalid JSON"):
        parse_reflection(json_str)


def test_parse_reflection_invalid_bullet_tag():
    """Test that invalid bullet tag raises error."""
    json_str = """
    {
      "bullet_tags": [
        {"id": "strat-00091", "tag": "invalid"}
      ],
      "candidate_bullets": []
    }
    """

    with pytest.raises(ReflectionParseError, match="Invalid tag value"):
        parse_reflection(json_str)


def test_parse_reflection_invalid_section():
    """Test that invalid section raises error."""
    json_str = """
    {
      "bullet_tags": [],
      "candidate_bullets": [
        {
          "section": "invalid_section",
          "content": "Some content",
          "tags": []
        }
      ]
    }
    """

    with pytest.raises(ReflectionParseError, match="Invalid section"):
        parse_reflection(json_str)


def test_parse_reflection_missing_required_fields():
    """Test that missing required fields raises error."""
    json_str = """
    {
      "bullet_tags": [],
      "candidate_bullets": [
        {
          "section": "strategies"
        }
      ]
    }
    """

    with pytest.raises(ReflectionParseError, match="must have"):
        parse_reflection(json_str)


def test_bullet_tag_creation():
    """Test BulletTag dataclass creation."""
    bt = BulletTag(id="strat-00091", tag="helpful")
    assert bt.id == "strat-00091"
    assert bt.tag == "helpful"


def test_candidate_bullet_creation():
    """Test CandidateBullet dataclass creation."""
    cb = CandidateBullet(
        section="strategies",
        content="Use hybrid retrieval",
        tags=["topic:retrieval"],
    )
    assert cb.section == "strategies"
    assert cb.content == "Use hybrid retrieval"
    assert cb.tags == ["topic:retrieval"]


def test_reflection_creation():
    """Test Reflection dataclass creation."""
    reflection = Reflection(
        error_identification="Error",
        root_cause_analysis="Root cause",
        correct_approach="Correct way",
        key_insight="Insight",
        bullet_tags=[BulletTag(id="test-1", tag="helpful")],
        candidate_bullets=[
            CandidateBullet(
                section="strategies",
                content="Test content",
                tags=["tag1"],
            )
        ],
    )
    assert reflection.error_identification == "Error"
    assert len(reflection.bullet_tags) == 1
    assert len(reflection.candidate_bullets) == 1


# --- Tests for reflect_on_trajectory ---


class TestExtractTrajectoryContext:
    """Tests for _extract_trajectory_context helper."""

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_extract_code_diff(self):
        """Test extraction of code diffs from trajectory."""
        trajectory = Trajectory(
            initial_goal="Fix bug",
            final_status="success",
            steps=[
                Step(
                    action="edit file",
                    observation="--- a/file.py\n+++ b/file.py\n@@ -1,2 +1,3 @@",
                    thought="Need to fix import",
                ),
                Step(
                    action="run tests",
                    observation="All tests passed",
                    thought="Verify fix",
                ),
            ],
        )

        reflector = Reflector()
        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "---" in code_diff
        assert "+++" in code_diff
        assert code_diff != ""

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_extract_test_output(self):
        """Test extraction of test output from trajectory."""
        trajectory = Trajectory(
            initial_goal="Run tests",
            final_status="failure",
            steps=[
                Step(
                    action="pytest",
                    observation="FAILED test_example.py::test_add - AssertionError",
                    thought="Test failed",
                ),
            ],
        )

        reflector = Reflector()
        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "FAILED" in test_output
        assert "AssertionError" in test_output

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_extract_logs(self):
        """Test extraction of logs from trajectory."""
        trajectory = Trajectory(
            initial_goal="Debug issue",
            final_status="failure",
            steps=[
                Step(
                    action="run script",
                    observation="ERROR: Connection refused\nstderr: timeout",
                    thought="Connection issue",
                ),
            ],
        )

        reflector = Reflector()
        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "ERROR:" in logs
        assert "stderr:" in logs

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_extract_empty_trajectory(self):
        """Test extraction from trajectory with no matching patterns."""
        trajectory = Trajectory(
            initial_goal="Simple task",
            final_status="success",
            steps=[
                Step(
                    action="ls",
                    observation="file1.txt file2.txt",
                    thought="List files",
                ),
            ],
        )

        reflector = Reflector()
        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert code_diff == ""
        assert test_output == ""
        assert logs == ""

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_extract_multiple_steps(self):
        """Test extraction combines multiple relevant steps."""
        trajectory = Trajectory(
            initial_goal="Complex task",
            final_status="partial",
            steps=[
                Step(
                    action="edit",
                    observation="modified file.py",
                    thought="Change code",
                ),
                Step(
                    action="test",
                    observation="pytest: 2 passed, 1 failed",
                    thought="Run tests",
                ),
                Step(
                    action="edit again",
                    observation="created file new.py",
                    thought="Add new file",
                ),
            ],
        )

        reflector = Reflector()
        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "modified" in code_diff
        assert "created file" in code_diff
        assert "pytest" in test_output


class TestReflectOnTrajectory:
    """Tests for reflect_on_trajectory method."""

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_passes_used_bullet_ids(self):
        """Test that used_bullet_ids from trajectory are passed to reflect."""
        reflector = Reflector()
        reflector.reflect = MagicMock(
            return_value=Reflection(
                key_insight="Test insight",
                bullet_tags=[],
                candidate_bullets=[],
            )
        )

        trajectory = Trajectory(
            initial_goal="Test task",
            final_status="success",
            used_bullet_ids=["strat-001", "tmpl-002"],
            steps=[],
        )

        reflector.reflect_on_trajectory(trajectory)

        reflector.reflect.assert_called_once()
        call_args = reflector.reflect.call_args
        assert call_args.kwargs["retrieved_bullet_ids"] == ["strat-001", "tmpl-002"]

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_passes_initial_goal_as_query(self):
        """Test that initial_goal is passed as query."""
        reflector = Reflector()
        reflector.reflect = MagicMock(return_value=Reflection())

        trajectory = Trajectory(
            initial_goal="Implement feature X",
            final_status="success",
            steps=[],
        )

        reflector.reflect_on_trajectory(trajectory)

        call_args = reflector.reflect.call_args
        assert call_args.kwargs["query"] == "Implement feature X"

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_passes_env_meta(self):
        """Test that trajectory metadata is passed as env_meta."""
        reflector = Reflector()
        reflector.reflect = MagicMock(return_value=Reflection())

        trajectory = Trajectory(
            initial_goal="Task",
            final_status="failure",
            steps=[Step(action="a", observation="o", thought="t")],
            bullet_feedback={"strat-001": "helpful"},
        )

        reflector.reflect_on_trajectory(trajectory)

        call_args = reflector.reflect.call_args
        env_meta = call_args.kwargs["env_meta"]
        assert env_meta["final_status"] == "failure"
        assert env_meta["total_steps"] == 1
        assert env_meta["bullet_feedback"] == {"strat-001": "helpful"}


# --- Tests for RefinementQuality and parse_quality ---


class TestRefinementQuality:
    """Tests for RefinementQuality dataclass."""

    def test_creation_and_overall_score(self):
        """Test RefinementQuality computes overall_score correctly."""
        quality = RefinementQuality(
            specificity=0.8,
            actionability=0.6,
            redundancy=0.2,  # low redundancy is good
        )
        # (0.8 + 0.6 + (1 - 0.2)) / 3 = (0.8 + 0.6 + 0.8) / 3 = 0.733...
        assert abs(quality.overall_score - 0.7333) < 0.01

    def test_with_feedback(self):
        """Test RefinementQuality with feedback."""
        quality = RefinementQuality(
            specificity=0.5,
            actionability=0.5,
            redundancy=0.5,
            feedback="Needs more specific insights",
        )
        assert quality.feedback == "Needs more specific insights"
        assert quality.overall_score == 0.5


class TestParseQuality:
    """Tests for parse_quality function."""

    def test_valid_quality_json(self):
        """Test parsing valid quality JSON."""
        json_str = """
        {
          "specificity": 0.85,
          "actionability": 0.7,
          "redundancy": 0.15,
          "feedback": "Good insights"
        }
        """
        quality = parse_quality(json_str)
        assert quality.specificity == 0.85
        assert quality.actionability == 0.7
        assert quality.redundancy == 0.15
        assert quality.feedback == "Good insights"

    def test_quality_with_markdown_fencing(self):
        """Test parsing quality with markdown code fencing."""
        json_str = """```json
        {
          "specificity": 0.9,
          "actionability": 0.8,
          "redundancy": 0.1
        }
        ```"""
        quality = parse_quality(json_str)
        assert quality.specificity == 0.9
        assert quality.feedback == ""

    def test_quality_missing_field(self):
        """Test that missing field raises error."""
        json_str = """{"specificity": 0.8, "actionability": 0.7}"""
        with pytest.raises(QualityParseError, match="Missing required field"):
            parse_quality(json_str)

    def test_quality_out_of_range(self):
        """Test that out-of-range value raises error."""
        json_str = """{"specificity": 1.5, "actionability": 0.7, "redundancy": 0.3}"""
        with pytest.raises(QualityParseError, match="must be between"):
            parse_quality(json_str)

    def test_quality_invalid_json(self):
        """Test that invalid JSON raises error."""
        with pytest.raises(QualityParseError, match="Invalid JSON"):
            parse_quality("{ not valid }")


# --- Tests for iterative refinement ---


class TestIterativeRefinement:
    """Tests for iterative refinement in Reflector."""

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_refinement_rounds_default(self):
        """Test default refinement_rounds is 1 (no refinement)."""
        reflector = Reflector()
        reflector.refinement_rounds = 1
        reflector.quality_threshold = 0.7
        reflector.max_retries = 3
        reflector.model = "gpt-4o-mini"
        reflector.temperature = 0.3
        reflector.client = MagicMock()

        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
        {
          "error_identification": "Test error",
          "bullet_tags": [],
          "candidate_bullets": []
        }
        """
        reflector.client.chat.completions.create.return_value = mock_response

        reflection = reflector.reflect(
            query="test query",
            retrieved_bullet_ids=[],
        )

        assert reflection.error_identification == "Test error"
        # Only 1 call since refinement_rounds=1
        assert reflector.client.chat.completions.create.call_count == 1

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_refinement_stops_when_quality_threshold_met(self):
        """Test refinement stops early when quality threshold is met."""
        reflector = Reflector()
        reflector.refinement_rounds = 3
        reflector.quality_threshold = 0.7
        reflector.max_retries = 3
        reflector.model = "gpt-4o-mini"
        reflector.temperature = 0.3
        reflector.client = MagicMock()

        # First call: initial reflection
        initial_response = MagicMock()
        initial_response.choices[0].message.content = """
        {"error_identification": "Initial", "bullet_tags": [], "candidate_bullets": []}
        """

        # Second call: quality eval (high quality, should stop)
        quality_response = MagicMock()
        quality_response.choices[0].message.content = """
        {"specificity": 0.9, "actionability": 0.8, "redundancy": 0.1, "feedback": ""}
        """

        reflector.client.chat.completions.create.side_effect = [
            initial_response,
            quality_response,
        ]

        reflection = reflector.reflect(
            query="test query",
            retrieved_bullet_ids=[],
        )

        assert reflection.error_identification == "Initial"
        # 1 for initial + 1 for quality eval = 2 calls
        assert reflector.client.chat.completions.create.call_count == 2

    @patch.object(Reflector, "__init__", lambda self, **kwargs: None)
    def test_refinement_iterates_until_max_rounds(self):
        """Test refinement continues until max rounds when quality is low."""
        reflector = Reflector()
        reflector.refinement_rounds = 2
        reflector.quality_threshold = 0.9  # High threshold
        reflector.max_retries = 3
        reflector.model = "gpt-4o-mini"
        reflector.temperature = 0.3
        reflector.client = MagicMock()

        # Initial reflection
        initial_response = MagicMock()
        initial_response.choices[0].message.content = """
        {"error_identification": "V1", "bullet_tags": [], "candidate_bullets": []}
        """

        # Quality eval: low quality
        quality_response = MagicMock()
        quality_response.choices[0].message.content = (
            '{"specificity": 0.4, "actionability": 0.5, '
            '"redundancy": 0.6, "feedback": "Be more specific"}'
        )

        # Refined reflection
        refined_response = MagicMock()
        refined_response.choices[0].message.content = """
        {"error_identification": "V2 - refined", "bullet_tags": [], "candidate_bullets": []}
        """

        reflector.client.chat.completions.create.side_effect = [
            initial_response,
            quality_response,
            refined_response,
        ]

        reflection = reflector.reflect(
            query="test query",
            retrieved_bullet_ids=[],
        )

        # Should get the refined version
        assert reflection.error_identification == "V2 - refined"
        # 1 initial + 1 quality + 1 refinement = 3 calls
        assert reflector.client.chat.completions.create.call_count == 3

    def test_reflector_init_with_refinement_params(self):
        """Test Reflector can be initialized with refinement parameters."""
        with patch("ace.reflector.reflector.OpenAI"):
            reflector = Reflector(refinement_rounds=3, quality_threshold=0.8)
            assert reflector.refinement_rounds == 3
            assert reflector.quality_threshold == 0.8

    def test_refinement_rounds_minimum_is_one(self):
        """Test that refinement_rounds is at least 1."""
        with patch("ace.reflector.reflector.OpenAI"):
            reflector = Reflector(refinement_rounds=0)
            assert reflector.refinement_rounds == 1

            reflector2 = Reflector(refinement_rounds=-5)
            assert reflector2.refinement_rounds == 1
