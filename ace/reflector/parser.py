# ace/reflector/parser.py
import json
from .schema import Reflection, BulletTag, CandidateBullet


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
        raise ReflectionParseError(f"Invalid JSON: {e}")
    
    if not isinstance(data, dict):
        raise ReflectionParseError("JSON must be an object")
    
    # Parse bullet_tags
    bullet_tags = []
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
                raise ReflectionParseError("candidate_bullet must have 'section' and 'content' fields")
            
            section = cb["section"]
            valid_sections = ["strategies", "templates", "troubleshooting", "code_snippets", "facts"]
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
    return Reflection(
        error_identification=data.get("error_identification"),
        root_cause_analysis=data.get("root_cause_analysis"),
        correct_approach=data.get("correct_approach"),
        key_insight=data.get("key_insight"),
        bullet_tags=bullet_tags,
        candidate_bullets=candidate_bullets,
    )
