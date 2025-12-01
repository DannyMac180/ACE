# tests/test_reflector.py
from unittest.mock import MagicMock, patch

import pytest

from ace.generator.schemas import Step, Trajectory, TrajectoryDoc
from ace.llm import CompletionResponse, LLMClient, MockLLMClient
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
          "section": "troubleshooting_and_pitfalls",
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
    assert reflection.candidate_bullets[0].section == "troubleshooting_and_pitfalls"
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
          "section": "strategies_and_hard_rules"
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
        section="strategies_and_hard_rules",
        content="Use hybrid retrieval",
        tags=["topic:retrieval"],
    )
    assert cb.section == "strategies_and_hard_rules"
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
                section="strategies_and_hard_rules",
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

    def test_extract_code_diff(self):
        """Test extraction of code diffs from trajectory."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

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

        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "---" in code_diff
        assert "+++" in code_diff
        assert code_diff != ""

    def test_extract_test_output(self):
        """Test extraction of test output from trajectory."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

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

        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "FAILED" in test_output
        assert "AssertionError" in test_output

    def test_extract_logs(self):
        """Test extraction of logs from trajectory."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

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

        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "ERROR:" in logs
        assert "stderr:" in logs

    def test_extract_empty_trajectory(self):
        """Test extraction from trajectory with no matching patterns."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

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

        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert code_diff == ""
        assert test_output == ""
        assert logs == ""

    def test_extract_multiple_steps(self):
        """Test extraction combines multiple relevant steps."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

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

        code_diff, test_output, logs = reflector._extract_trajectory_context(trajectory)

        assert "modified" in code_diff
        assert "created file" in code_diff
        assert "pytest" in test_output


class TestReflectOnTrajectory:
    """Tests for reflect_on_trajectory method."""

    def test_passes_used_bullet_ids(self):
        """Test that used_bullet_ids from trajectory are passed to reflect."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)
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
        doc_arg = call_args.args[0]
        assert isinstance(doc_arg, TrajectoryDoc)
        assert doc_arg.retrieved_bullet_ids == ["strat-001", "tmpl-002"]

    def test_passes_initial_goal_as_query(self):
        """Test that initial_goal is passed as query."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)
        reflector.reflect = MagicMock(return_value=Reflection())

        trajectory = Trajectory(
            initial_goal="Implement feature X",
            final_status="success",
            steps=[],
        )

        reflector.reflect_on_trajectory(trajectory)

        call_args = reflector.reflect.call_args
        doc_arg = call_args.args[0]
        assert isinstance(doc_arg, TrajectoryDoc)
        assert doc_arg.query == "Implement feature X"

    def test_passes_env_meta(self):
        """Test that trajectory metadata is passed as env_meta."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)
        reflector.reflect = MagicMock(return_value=Reflection())

        trajectory = Trajectory(
            initial_goal="Task",
            final_status="failure",
            steps=[Step(action="a", observation="o", thought="t")],
            bullet_feedback={"strat-001": "helpful"},
        )

        reflector.reflect_on_trajectory(trajectory)

        call_args = reflector.reflect.call_args
        doc_arg = call_args.args[0]
        assert isinstance(doc_arg, TrajectoryDoc)
        env_meta = doc_arg.env_meta
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


# --- Tests for LLM client injection ---


class TestLLMClientInjection:
    """Tests for LLM client injection in Reflector."""

    def test_reflector_uses_injected_client(self):
        """Test that Reflector uses the injected LLM client."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = CompletionResponse(
            text='{"error_identification": "Test", "bullet_tags": [], "candidate_bullets": []}'
        )

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test", retrieved_bullet_ids=[])
        reflection = reflector.reflect(doc)

        mock_client.complete.assert_called_once()
        assert reflection.error_identification == "Test"

    def test_reflector_uses_mock_client_for_testing(self):
        """Test that Reflector works with MockLLMClient."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.return_value = CompletionResponse(
            text='{"key_insight": "Mock insight", "bullet_tags": [], "candidate_bullets": []}'
        )

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect(doc)

        assert reflection.key_insight == "Mock insight"

    @patch("ace.reflector.reflector.create_llm_client")
    def test_reflector_creates_client_from_factory_when_not_provided(
        self, mock_factory
    ):
        """Test that Reflector creates client from factory when none provided."""
        mock_client = MagicMock(spec=LLMClient)
        mock_factory.return_value = mock_client

        reflector = Reflector()

        mock_factory.assert_called_once()
        assert reflector.client is mock_client


# --- Tests for iterative refinement ---


class TestIterativeRefinement:
    """Tests for iterative refinement in Reflector."""

    def test_refinement_rounds_default(self):
        """Test default refinement_rounds is 1 (no refinement)."""
        mock_client = MagicMock(spec=LLMClient)
        response_json = (
            '{"error_identification": "Test error", '
            '"bullet_tags": [], "candidate_bullets": []}'
        )
        mock_client.complete.return_value = CompletionResponse(text=response_json)

        reflector = Reflector(llm_client=mock_client)

        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect(doc)

        assert reflection.error_identification == "Test error"
        # Only 1 call since refinement_rounds=1
        assert mock_client.complete.call_count == 1

    def test_refinement_stops_when_quality_threshold_met(self):
        """Test refinement stops early when quality threshold is met."""
        mock_client = MagicMock(spec=LLMClient)

        initial_json = (
            '{"error_identification": "Initial", '
            '"bullet_tags": [], "candidate_bullets": []}'
        )
        quality_json = (
            '{"specificity": 0.9, "actionability": 0.8, '
            '"redundancy": 0.1, "feedback": ""}'
        )
        # First call: initial reflection
        # Second call: quality eval (high quality, should stop)
        mock_client.complete.side_effect = [
            CompletionResponse(text=initial_json),
            CompletionResponse(text=quality_json),
        ]

        reflector = Reflector(
            llm_client=mock_client,
            refinement_rounds=3,
            quality_threshold=0.7,
        )

        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect(doc)

        assert reflection.error_identification == "Initial"
        # 1 for initial + 1 for quality eval = 2 calls
        assert mock_client.complete.call_count == 2

    def test_refinement_iterates_until_max_rounds(self):
        """Test refinement continues until max rounds when quality is low."""
        mock_client = MagicMock(spec=LLMClient)

        initial_json = (
            '{"error_identification": "V1", '
            '"bullet_tags": [], "candidate_bullets": []}'
        )
        quality_json = (
            '{"specificity": 0.4, "actionability": 0.5, '
            '"redundancy": 0.6, "feedback": "Be more specific"}'
        )
        refined_json = (
            '{"error_identification": "V2 - refined", '
            '"bullet_tags": [], "candidate_bullets": []}'
        )
        mock_client.complete.side_effect = [
            CompletionResponse(text=initial_json),  # Initial reflection
            CompletionResponse(text=quality_json),  # Quality eval: low quality
            CompletionResponse(text=refined_json),  # Refined reflection
        ]

        reflector = Reflector(
            llm_client=mock_client,
            refinement_rounds=2,
            quality_threshold=0.9,  # High threshold
        )

        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect(doc)

        # Should get the refined version
        assert reflection.error_identification == "V2 - refined"
        # 1 initial + 1 quality + 1 refinement = 3 calls
        assert mock_client.complete.call_count == 3

    def test_reflector_init_with_refinement_params(self):
        """Test Reflector can be initialized with refinement parameters."""
        mock_client = MockLLMClient()
        reflector = Reflector(
            llm_client=mock_client,
            refinement_rounds=3,
            quality_threshold=0.8,
        )
        assert reflector.refinement_rounds == 3
        assert reflector.quality_threshold == 0.8

    def test_refinement_rounds_minimum_is_one(self):
        """Test that refinement_rounds is at least 1."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client, refinement_rounds=0)
        assert reflector.refinement_rounds == 1

        reflector2 = Reflector(llm_client=mock_client, refinement_rounds=-5)
        assert reflector2.refinement_rounds == 1


# --- Tests for multi-pass reflection ---


class TestMultiPassReflection:
    """Tests for reflect_multi method."""

    def test_reflect_multi_single_pass_returns_single_reflection(self):
        """Test that num_passes=1 returns single reflection with correct fields."""
        mock_client = MagicMock(spec=LLMClient)
        response_json = (
            '{"error_identification": "Test error", '
            '"bullet_tags": [{"id": "strat-001", "tag": "helpful"}], '
            '"candidate_bullets": [{"section": "strategies_and_hard_rules", '
            '"content": "Test bullet", "tags": ["topic:test"]}]}'
        )
        mock_client.complete.return_value = CompletionResponse(text=response_json)

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect_multi(doc, num_passes=1)

        assert reflection.error_identification == "Test error"
        assert reflection.iteration == 0
        assert reflection.parent_id is not None
        assert mock_client.complete.call_count == 1

    def test_reflect_multi_multiple_passes(self):
        """Test that multiple passes are executed."""
        mock_client = MagicMock(spec=LLMClient)
        response_json = (
            '{"error_identification": "Test error", '
            '"bullet_tags": [], "candidate_bullets": []}'
        )
        mock_client.complete.return_value = CompletionResponse(text=response_json)

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect_multi(doc, num_passes=3)

        assert mock_client.complete.call_count == 3
        assert reflection.iteration == 3  # Number of passes
        assert reflection.parent_id is not None

    def test_reflect_multi_deduplicates_candidate_bullets(self):
        """Test that similar candidate bullets are deduplicated."""
        mock_client = MagicMock(spec=LLMClient)

        response1 = (
            '{"error_identification": "Error", "bullet_tags": [], '
            '"candidate_bullets": [{"section": "strategies_and_hard_rules", '
            '"content": "Use hybrid retrieval for better results", "tags": ["topic:retrieval"]}]}'
        )
        response2 = (
            '{"error_identification": "Error", "bullet_tags": [], '
            '"candidate_bullets": [{"section": "strategies_and_hard_rules", '
            '"content": "Use hybrid retrieval for improved results", "tags": ["topic:search"]}]}'
        )

        mock_client.complete.side_effect = [
            CompletionResponse(text=response1),
            CompletionResponse(text=response2),
        ]

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect_multi(doc, num_passes=2, similarity_threshold=0.5)

        # Should deduplicate to 1 bullet and merge tags
        assert len(reflection.candidate_bullets) == 1
        assert "topic:retrieval" in reflection.candidate_bullets[0].tags
        assert "topic:search" in reflection.candidate_bullets[0].tags

    def test_reflect_multi_keeps_distinct_bullets(self):
        """Test that distinct bullets are kept separate."""
        mock_client = MagicMock(spec=LLMClient)

        response1 = (
            '{"error_identification": "Error", "bullet_tags": [], '
            '"candidate_bullets": [{"section": "strategies_and_hard_rules", '
            '"content": "Use caching for performance", "tags": ["topic:perf"]}]}'
        )
        response2 = (
            '{"error_identification": "Error", "bullet_tags": [], '
            '"candidate_bullets": [{"section": "troubleshooting_and_pitfalls", '
            '"content": "Check database connections on startup", "tags": ["topic:db"]}]}'
        )

        mock_client.complete.side_effect = [
            CompletionResponse(text=response1),
            CompletionResponse(text=response2),
        ]

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect_multi(doc, num_passes=2)

        # Different sections, should keep both
        assert len(reflection.candidate_bullets) == 2

    def test_reflect_multi_aggregates_bullet_tags_majority_vote(self):
        """Test that bullet_tags are aggregated using majority vote."""
        mock_client = MagicMock(spec=LLMClient)

        # 2 helpful, 1 harmful -> helpful wins
        response1 = (
            '{"error_identification": "Error", '
            '"bullet_tags": [{"id": "strat-001", "tag": "helpful"}], '
            '"candidate_bullets": []}'
        )
        response2 = (
            '{"error_identification": "Error", '
            '"bullet_tags": [{"id": "strat-001", "tag": "helpful"}], '
            '"candidate_bullets": []}'
        )
        response3 = (
            '{"error_identification": "Error", '
            '"bullet_tags": [{"id": "strat-001", "tag": "harmful"}], '
            '"candidate_bullets": []}'
        )

        mock_client.complete.side_effect = [
            CompletionResponse(text=response1),
            CompletionResponse(text=response2),
            CompletionResponse(text=response3),
        ]

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect_multi(doc, num_passes=3)

        assert len(reflection.bullet_tags) == 1
        assert reflection.bullet_tags[0].id == "strat-001"
        assert reflection.bullet_tags[0].tag == "helpful"

    def test_reflect_multi_tie_goes_to_helpful(self):
        """Test that ties in bullet_tags voting go to helpful."""
        mock_client = MagicMock(spec=LLMClient)

        response1 = (
            '{"error_identification": "Error", '
            '"bullet_tags": [{"id": "strat-001", "tag": "helpful"}], '
            '"candidate_bullets": []}'
        )
        response2 = (
            '{"error_identification": "Error", '
            '"bullet_tags": [{"id": "strat-001", "tag": "harmful"}], '
            '"candidate_bullets": []}'
        )

        mock_client.complete.side_effect = [
            CompletionResponse(text=response1),
            CompletionResponse(text=response2),
        ]

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test query", retrieved_bullet_ids=[])
        reflection = reflector.reflect_multi(doc, num_passes=2)

        assert len(reflection.bullet_tags) == 1
        assert reflection.bullet_tags[0].tag == "helpful"

    def test_reflect_multi_num_passes_zero_treated_as_one(self):
        """Test that num_passes < 1 is treated as 1."""
        mock_client = MagicMock(spec=LLMClient)
        response_json = '{"bullet_tags": [], "candidate_bullets": []}'
        mock_client.complete.return_value = CompletionResponse(text=response_json)

        reflector = Reflector(llm_client=mock_client)
        doc = TrajectoryDoc(query="test", retrieved_bullet_ids=[])
        reflection = reflector.reflect_multi(doc, num_passes=0)

        assert mock_client.complete.call_count == 1
        assert reflection.iteration == 0


class TestTextSimilarity:
    """Tests for _text_similarity helper method."""

    def test_identical_texts(self):
        """Test identical texts have similarity 1.0."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

        similarity = reflector._text_similarity(
            "hello world foo bar",
            "hello world foo bar"
        )
        assert similarity == 1.0

    def test_completely_different_texts(self):
        """Test completely different texts have similarity 0.0."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

        similarity = reflector._text_similarity(
            "hello world",
            "goodbye universe"
        )
        assert similarity == 0.0

    def test_partial_overlap(self):
        """Test partial overlap gives expected similarity."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

        # "hello world" vs "hello there" -> intersection={hello}, union={hello,world,there}
        similarity = reflector._text_similarity("hello world", "hello there")
        assert abs(similarity - 1 / 3) < 0.01

    def test_empty_texts(self):
        """Test empty texts."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

        assert reflector._text_similarity("", "") == 1.0
        assert reflector._text_similarity("hello", "") == 0.0
        assert reflector._text_similarity("", "world") == 0.0

    def test_case_insensitive(self):
        """Test similarity is case-insensitive."""
        mock_client = MockLLMClient()
        reflector = Reflector(llm_client=mock_client)

        similarity = reflector._text_similarity("Hello World", "hello world")
        assert similarity == 1.0


class TestReflectionSchema:
    """Tests for Reflection schema with iteration and parent_id."""

    def test_reflection_has_iteration_field(self):
        """Test Reflection has iteration field with default 0."""
        reflection = Reflection()
        assert reflection.iteration == 0

    def test_reflection_has_parent_id_field(self):
        """Test Reflection has parent_id field with default None."""
        reflection = Reflection()
        assert reflection.parent_id is None

    def test_reflection_iteration_and_parent_set(self):
        """Test Reflection can have iteration and parent_id set."""
        reflection = Reflection(
            iteration=2,
            parent_id="parent-abc-123",
        )
        assert reflection.iteration == 2
        assert reflection.parent_id == "parent-abc-123"
