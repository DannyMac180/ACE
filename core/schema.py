# ACE/core/schema.py
from dataclasses import dataclass, field
from typing import Literal, Optional, List
from datetime import datetime

Section = Literal["strategies", "code_snippets", "troubleshooting", "facts", "templates"]

@dataclass
class Bullet:
    id: str                      # e.g., "strat-00091"
    section: Section
    content: str                 # text or code snippet
    tags: List[str] = field(default_factory=list)   # "domain:finance", "tool:appworld.phone_api"
    helpful: int = 0
    harmful: int = 0
    last_used: Optional[datetime] = None
    added_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Playbook:
    version: int
    bullets: List[Bullet]
