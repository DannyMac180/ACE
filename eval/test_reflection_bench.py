"""
Reflection benchmark tests for ACE reflection system.

Tests reflection quality and correctness to ensure:
- Valid JSON schema output
- Appropriate bullet tagging (helpful/harmful)
- Quality candidate bullets generation
- Retry logic on parse errors
- Key insights are actionable and reusable

Uses mocked LLM responses for deterministic, fast testing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from ace.reflector.parser import ReflectionParseError
from ace.reflector.reflector import Reflector
from ace.reflector.schema import Reflection


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client that returns controlled responses."""
    with patch("ace.reflector.reflector.OpenAI") as mock:
        yield mock.return_value


@pytest.fixture
def valid_reflection_json() -> str:
    """Valid reflection JSON for successful parsing."""
    return json.dumps({
        "error_identification": "TypeError: expected string, got None",
        "root_cause_analysis": "Function argument validation missing",
        "correct_approach": "Add type checking before processing",
        "key_insight": "Always validate input types in public APIs",
        "bullet_tags": [
            {"id": "strat-001", "tag": "helpful"},
            {"id": "code-002", "tag": "harmful"}
        ],
        "candidate_bullets": [
            {
                "section": "troubleshooting",
                "content": "TypeError on None: add explicit None checks before operations",
                "tags": ["topic:validation", "error:TypeError", "stack:python"]
            }
        ]
    })


@pytest.fixture
def malformed_json_then_valid(valid_reflection_json) -> list[str]:
    """Sequence of responses: first malformed, then valid."""
    return [
        "This is not valid JSON at all!",  # Invalid JSON
        valid_reflection_json  # Valid on retry
    ]


def test_reflection_valid_json_parsing(mock_openai_client, valid_reflection_json):
    """Test successful reflection with valid JSON response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=valid_reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector(model="gpt-4o-mini")
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="fix type error",
        retrieved_bullet_ids=["strat-001", "code-002"],
        test_output="TypeError: expected string, got None"
    )

    assert isinstance(reflection, Reflection)
    assert reflection.error_identification == "TypeError: expected string, got None"
    assert reflection.root_cause_analysis == "Function argument validation missing"
    assert len(reflection.bullet_tags) == 2
    assert len(reflection.candidate_bullets) == 1


def test_reflection_retry_on_parse_error(mock_openai_client, malformed_json_then_valid):
    """Test that reflector retries on JSON parse errors."""
    responses = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=resp))])
        for resp in malformed_json_then_valid
    ]
    mock_openai_client.chat.completions.create.side_effect = responses

    reflector = Reflector(model="gpt-4o-mini", max_retries=3)
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="fix parse error",
        retrieved_bullet_ids=["strat-001"],
        logs="Failed to parse JSON"
    )

    assert isinstance(reflection, Reflection)
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_reflection_max_retries_exceeded(mock_openai_client):
    """Test that reflector raises error after max retries."""
    bad_json = "this is not json at all"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=bad_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector(model="gpt-4o-mini", max_retries=2)
    reflector.client = mock_openai_client

    with pytest.raises(ReflectionParseError) as exc_info:
        reflector.reflect(
            query="test retry limit",
            retrieved_bullet_ids=[]
        )

    assert "Failed to parse reflection after 2 attempts" in str(exc_info.value)
    assert mock_openai_client.chat.completions.create.call_count == 2


def test_reflection_empty_response(mock_openai_client):
    """Test that reflector handles empty LLM response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=None))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector(model="gpt-4o-mini", max_retries=1)
    reflector.client = mock_openai_client

    with pytest.raises(ReflectionParseError) as exc_info:
        reflector.reflect(
            query="test empty response",
            retrieved_bullet_ids=[]
        )

    assert "Empty response from LLM" in str(exc_info.value)


def test_reflection_bullet_tagging_quality(mock_openai_client):
    """Test that bullet tags are correctly identified as helpful/harmful."""
    reflection_json = json.dumps({
        "error_identification": None,
        "root_cause_analysis": None,
        "correct_approach": None,
        "key_insight": "Bullet strat-100 was very helpful",
        "bullet_tags": [
            {"id": "strat-100", "tag": "helpful"},
            {"id": "trbl-050", "tag": "harmful"}
        ],
        "candidate_bullets": []
    })

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="tag bullets",
        retrieved_bullet_ids=["strat-100", "trbl-050"]
    )

    assert len(reflection.bullet_tags) == 2
    helpful_tags = [t for t in reflection.bullet_tags if t.tag == "helpful"]
    harmful_tags = [t for t in reflection.bullet_tags if t.tag == "harmful"]

    assert len(helpful_tags) == 1
    assert len(harmful_tags) == 1
    assert helpful_tags[0].id == "strat-100"
    assert harmful_tags[0].id == "trbl-050"


def test_reflection_candidate_bullet_quality(mock_openai_client):
    """Test that candidate bullets have proper structure and tags."""
    reflection_json = json.dumps({
        "error_identification": "Database connection timeout",
        "root_cause_analysis": "Connection pool exhausted under load",
        "correct_approach": "Increase pool size and add connection retry logic",
        "key_insight": "Monitor connection pool metrics in production",
        "bullet_tags": [],
        "candidate_bullets": [
            {
                "section": "troubleshooting",
                "content": "Connection timeout: increase pool_size in pgvector config",
                "tags": ["db:pgvector", "error:timeout", "topic:retrieval"]
            },
            {
                "section": "strategies",
                "content": "Add exponential backoff retry for database connections",
                "tags": ["db:pgvector", "pattern:retry", "robustness"]
            }
        ]
    })

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="fix database timeout",
        retrieved_bullet_ids=[],
        logs="psycopg2.OperationalError: connection timeout"
    )

    assert len(reflection.candidate_bullets) == 2

    # Check first bullet
    bullet1 = reflection.candidate_bullets[0]
    assert bullet1.section == "troubleshooting"
    assert "timeout" in bullet1.content.lower()
    assert "db:pgvector" in bullet1.tags
    assert any("error:" in tag for tag in bullet1.tags)

    # Check second bullet
    bullet2 = reflection.candidate_bullets[1]
    assert bullet2.section == "strategies"
    assert "retry" in bullet2.content.lower()
    assert len(bullet2.tags) >= 2


def test_reflection_minimal_valid_output(mock_openai_client):
    """Test that reflection with minimal fields is still valid."""
    minimal_json = json.dumps({
        "error_identification": None,
        "root_cause_analysis": None,
        "correct_approach": None,
        "key_insight": None,
        "bullet_tags": [],
        "candidate_bullets": []
    })

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=minimal_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="minimal test",
        retrieved_bullet_ids=[]
    )

    assert isinstance(reflection, Reflection)
    assert reflection.error_identification is None
    assert len(reflection.bullet_tags) == 0
    assert len(reflection.candidate_bullets) == 0


def test_reflection_with_all_fields_populated(mock_openai_client, valid_reflection_json):
    """Test reflection with all possible fields populated."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=valid_reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="comprehensive test",
        retrieved_bullet_ids=["strat-001", "code-002"],
        code_diff="+ def validate_input(x):\n+     if x is None:\n+         raise ValueError",
        test_output="All tests passed",
        logs="INFO: validation added",
        env_meta={"repo": "ace", "branch": "feat/validation"}
    )

    assert reflection.error_identification is not None
    assert reflection.root_cause_analysis is not None
    assert reflection.correct_approach is not None
    assert reflection.key_insight is not None
    assert len(reflection.bullet_tags) > 0
    assert len(reflection.candidate_bullets) > 0


@pytest.mark.benchmark
def test_reflection_performance(benchmark, mock_openai_client, valid_reflection_json):
    """Benchmark reflection parsing and validation performance."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=valid_reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    result = benchmark(
        reflector.reflect,
        query="benchmark test",
        retrieved_bullet_ids=["test-001"]
    )

    assert isinstance(result, Reflection)


# ============================================================================
# QUALITY CHECKS - These validate reflection output quality
# ============================================================================


def test_quality_candidate_bullets_are_concise(mock_openai_client):
    """Quality check: candidate bullets should be short and actionable."""
    reflection_json = json.dumps({
        "error_identification": None,
        "root_cause_analysis": None,
        "correct_approach": None,
        "key_insight": None,
        "bullet_tags": [],
        "candidate_bullets": [
            {
                "section": "strategies",
                "content": "Use hybrid retrieval: BM25 + embeddings; default top_k=24",
                "tags": ["topic:retrieval"]
            }
        ]
    })

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="test conciseness",
        retrieved_bullet_ids=[]
    )

    for bullet in reflection.candidate_bullets:
        # Bullets should be < 200 chars for reusability
        assert len(bullet.content) < 200, \
            f"Bullet too long ({len(bullet.content)} chars): {bullet.content}"
        # Should not be prose paragraphs (no multiple sentences with
        # "However", "Additionally", etc.)
        prose_markers = ["however", "additionally", "furthermore", "therefore"]
        content_lower = bullet.content.lower()
        assert not any(marker in content_lower for marker in prose_markers), \
            f"Bullet contains prose narrative: {bullet.content}"


def test_quality_candidate_bullets_have_tags(mock_openai_client):
    """Quality check: candidate bullets must have relevant tags."""
    reflection_json = json.dumps({
        "error_identification": None,
        "root_cause_analysis": None,
        "correct_approach": None,
        "key_insight": None,
        "bullet_tags": [],
        "candidate_bullets": [
            {
                "section": "strategies",
                "content": "Always validate JSON output from LLM reflections",
                "tags": ["topic:parsing", "robustness", "llm"]
            }
        ]
    })

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="test tags",
        retrieved_bullet_ids=[]
    )

    for bullet in reflection.candidate_bullets:
        # Every bullet should have at least one tag
        assert len(bullet.tags) > 0, f"Bullet missing tags: {bullet.content}"
        # Tags should follow namespace:value format where appropriate
        for tag in bullet.tags:
            # At least some tags should be namespaced (topic:, db:, error:, stack:, etc.)
            if ":" in tag:
                parts = tag.split(":")
                assert len(parts) == 2, f"Invalid tag format: {tag}"
                assert parts[0] and parts[1], f"Empty namespace or value in tag: {tag}"


def test_quality_insights_are_actionable(mock_openai_client):
    """Quality check: key insights should be actionable, not generic."""
    good_insight_json = json.dumps({
        "error_identification": None,
        "root_cause_analysis": None,
        "correct_approach": None,
        "key_insight": "Use BM25 lexical search before vector similarity for better recall",
        "bullet_tags": [],
        "candidate_bullets": []
    })

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=good_insight_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="test insight quality",
        retrieved_bullet_ids=[]
    )

    if reflection.key_insight:
        insight_lower = reflection.key_insight.lower()
        # Should not be generic advice
        generic_phrases = [
            "it's important to",
            "make sure to",
            "don't forget",
            "be careful",
            "always remember"
        ]
        assert not any(phrase in insight_lower for phrase in generic_phrases), \
            f"Insight too generic: {reflection.key_insight}"
        # Should be specific and technical
        assert len(reflection.key_insight.split()) >= 5, \
            f"Insight too short to be actionable: {reflection.key_insight}"


def test_quality_no_duplicate_candidate_bullets(mock_openai_client):
    """Quality check: candidate bullets should not have near-duplicates."""
    reflection_json = json.dumps({
        "error_identification": None,
        "root_cause_analysis": None,
        "correct_approach": None,
        "key_insight": None,
        "bullet_tags": [],
        "candidate_bullets": [
            {
                "section": "strategies",
                "content": "Use hybrid retrieval with BM25 and embeddings",
                "tags": ["topic:retrieval"]
            },
            {
                "section": "troubleshooting",
                "content": "Connection timeout: check network and increase timeout setting",
                "tags": ["error:timeout"]
            }
        ]
    })

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=reflection_json))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    reflector = Reflector()
    reflector.client = mock_openai_client

    reflection = reflector.reflect(
        query="test dedup",
        retrieved_bullet_ids=[]
    )

    contents = [b.content for b in reflection.candidate_bullets]
    # Simple check: no exact duplicates
    assert len(contents) == len(set(contents)), \
        f"Candidate bullets contain duplicates: {contents}"
