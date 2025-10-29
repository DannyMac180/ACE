# ace/reflector/schema.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BulletTag:
    id: str
    tag: Literal["helpful", "harmful"]

@dataclass
class CandidateBullet:
    section: Literal["strategies", "templates", "troubleshooting", "code_snippets", "facts"]
    content: str
    tags: list[str] = field(default_factory=list)

@dataclass
class Reflection:
    error_identification: str | None = None
    root_cause_analysis: str | None = None
    correct_approach: str | None = None
    key_insight: str | None = None
    bullet_tags: list[BulletTag] = field(default_factory=list)
    candidate_bullets: list[CandidateBullet] = field(default_factory=list)
