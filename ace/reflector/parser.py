# ace/reflector/parser.py
import json

from ace.core.metrics import get_tracker

from .schema import BulletTag, CandidateBullet, RefinementQuality, Reflection


class ReflectionParseError(Exception):
    """Raised when reflection JSON parsing fails."""

    pass


def parse_reflection(json_str: str) -> Reflection:
    """Parse JSON string into Reflection object with strict validation.

    Args:
        json_str: Raw JSON string from LLM (may contain markdown fencing)

    Returns:
        Reflection: Validated reflection object

    Raises:
        ReflectionParseError: If JSON is invalid or schema doesn't match
    """
    tracker = get_tracker()

    # Strip markdown fencing if present
    cleaned = json_str.strip()
    if cleaned.startswith("```"):
        # Remove markdown code blocks
        lines = cleaned.split("\n")
        # Find first line after opening fence
        start_idx = 1
        if lines[0].startswith("```json"):
            start_idx = 1
        # Find closing fence
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        cleaned = "\n".join(lines[start_idx:end_idx])

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        tracker.record_attempt(
            success=False,
            error_type="JSONDecodeError",
            error_message=str(e),
            schema_type="reflection",
        )
        raise ReflectionParseError(f"Invalid JSON: {e}") from None

    if not isinstance(data, dict):
        tracker.record_attempt(
            success=False,
            error_type="InvalidTopLevelType",
            error_message="JSON must be an object",
            schema_type="reflection",
        )
        raise ReflectionParseError("JSON must be an object")

    # Parse bullet_tags
    bullet_tags = []
    try:
        if "bullet_tags" in data:
            if not isinstance(data["bullet_tags"], list):
                raise ReflectionParseError("bullet_tags must be a list")
            for bt in data["bullet_tags"]:
                if not isinstance(bt, dict):
                    raise ReflectionParseError("Each bullet_tag must be an object")
                if "id" not in bt or "tag" not in bt:
                    raise ReflectionParseError("bullet_tag must have 'id' and 'tag' fields")
                if bt["tag"] not in ["helpful", "harmful"]:
                    raise ReflectionParseError(f"Invalid tag value: {bt['tag']}")
                bullet_tags.append(BulletTag(id=bt["id"], tag=bt["tag"]))

        # Parse candidate_bullets
        candidate_bullets = []
        if "candidate_bullets" in data:
            if not isinstance(data["candidate_bullets"], list):
                raise ReflectionParseError("candidate_bullets must be a list")
            for cb in data["candidate_bullets"]:
                if not isinstance(cb, dict):
                    raise ReflectionParseError("Each candidate_bullet must be an object")
                if "section" not in cb or "content" not in cb:
                    raise ReflectionParseError(
                        "candidate_bullet must have 'section' and 'content' fields"
                    )

                section = cb["section"]
                valid_sections = [
                    "strategies_and_hard_rules",
                    "code_snippets_and_templates",
                    "troubleshooting_and_pitfalls",
                    "domain_facts_and_references",
                ]
                if section not in valid_sections:
                    raise ReflectionParseError(f"Invalid section: {section}")

                tags = cb.get("tags", [])
                if not isinstance(tags, list):
                    raise ReflectionParseError("tags must be a list")

                candidate_bullets.append(
                    CandidateBullet(
                        section=section,
                        content=cb["content"],
                        tags=tags,
                    )
                )

        # Build Reflection
        reflection = Reflection(
            error_identification=data.get("error_identification"),
            root_cause_analysis=data.get("root_cause_analysis"),
            correct_approach=data.get("correct_approach"),
            key_insight=data.get("key_insight"),
            bullet_tags=bullet_tags,
            candidate_bullets=candidate_bullets,
        )

        tracker.record_attempt(success=True, schema_type="reflection")
        return reflection

    except ReflectionParseError as e:
        tracker.record_attempt(
            success=False,
            error_type="SchemaValidationError",
            error_message=str(e),
            schema_type="reflection",
        )
        raise


class QualityParseError(Exception):
    """Raised when quality evaluation JSON parsing fails."""

    pass


def parse_quality(json_str: str) -> RefinementQuality:
    """Parse JSON string into RefinementQuality object.

    Args:
        json_str: Raw JSON string from LLM (may contain markdown fencing)

    Returns:
        RefinementQuality: Validated quality assessment

    Raises:
        QualityParseError: If JSON is invalid or schema doesn't match
    """
    cleaned = json_str.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        start_idx = 1
        if lines[0].startswith("```json"):
            start_idx = 1
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        cleaned = "\n".join(lines[start_idx:end_idx])

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise QualityParseError(f"Invalid JSON: {e}") from None

    if not isinstance(data, dict):
        raise QualityParseError("JSON must be an object")

    required_fields = ["specificity", "actionability", "redundancy"]
    for field in required_fields:
        if field not in data:
            raise QualityParseError(f"Missing required field: {field}")
        if not isinstance(data[field], (int, float)):
            raise QualityParseError(f"Field {field} must be a number")
        if not 0.0 <= data[field] <= 1.0:
            raise QualityParseError(f"Field {field} must be between 0.0 and 1.0")

    return RefinementQuality(
        specificity=float(data["specificity"]),
        actionability=float(data["actionability"]),
        redundancy=float(data["redundancy"]),
        feedback=data.get("feedback", ""),
    )
