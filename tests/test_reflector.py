# tests/test_reflector.py
import pytest
from ace.reflector import (
    Reflection,
    BulletTag,
    CandidateBullet,
    parse_reflection,
    ReflectionParseError,
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
