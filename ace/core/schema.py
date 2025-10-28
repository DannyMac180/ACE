# ACE/core/schema.py
from dataclasses import dataclass, field
from typing import Literal, Optional, List
from datetime import datetime

Section = Literal["strategies", "templates", "troubleshooting", "code_snippets", "facts"]

@dataclass
class Bullet:
    id: str                      # unique, stable (e.g., "strat-00091")
    section: Section
    content: str                 # short, reusable, domain-rich
    tags: List[str] = field(default_factory=list)   # e.g., ["repo:ace","topic:retrieval","db:pg"]
    helpful: int = 0
    harmful: int = 0
    last_used: Optional[datetime] = None
    added_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Playbook:
    version: int
    bullets: List[Bullet]

OpType = Literal["ADD", "PATCH", "DEPRECATE", "INCR_HELPFUL", "INCR_HARMFUL"]

@dataclass
class DeltaOp:
    op: OpType
    target_id: Optional[str] = None
    new_bullet: Optional[dict] = None  # Will contain section, content, tags
    patch: Optional[str] = None

@dataclass
class Delta:
    ops: List[DeltaOp] = field(default_factory=list)

RefineOpType = Literal["MERGE", "ARCHIVE"]

@dataclass
class RefineOp:
    op: RefineOpType
    target_ids: List[str] = field(default_factory=list)  # IDs being merged/archived
    survivor_id: Optional[str] = None  # For MERGE ops, the ID kept

@dataclass
class RefineResult:
    merged: int = 0
    archived: int = 0
    ops: List[RefineOp] = field(default_factory=list)
