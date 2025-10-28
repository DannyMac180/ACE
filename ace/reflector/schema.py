# ace/reflector/schema.py
from dataclasses import dataclass, field
from typing import Literal, Optional, List

@dataclass
class BulletTag:
    id: str
    tag: Literal["helpful", "harmful"]

@dataclass
class CandidateBullet:
    section: Literal["strategies", "templates", "troubleshooting", "code_snippets", "facts"]
    content: str
    tags: List[str] = field(default_factory=list)

@dataclass
class Reflection:
    error_identification: Optional[str] = None
    root_cause_analysis: Optional[str] = None
    correct_approach: Optional[str] = None
    key_insight: Optional[str] = None
    bullet_tags: List[BulletTag] = field(default_factory=list)
    candidate_bullets: List[CandidateBullet] = field(default_factory=list)
