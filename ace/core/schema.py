from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Section = Literal[
    "strategies_and_hard_rules",
    "code_snippets_and_templates",
    "troubleshooting_and_pitfalls",
    "domain_facts_and_references",
]


class Bullet(BaseModel):
    id: str
    section: Section
    content: str
    tags: list[str] = Field(default_factory=list)
    helpful: int = 0
    harmful: int = 0
    last_used: datetime | None = None
    added_at: datetime = Field(default_factory=datetime.utcnow)


class Reflection(BaseModel):
    summary: str
    critique: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DeltaBullet(BaseModel):
    section: Section
    content: str
    tags: list[str] = Field(default_factory=list)
    id: str | None = None  # Optional: for idempotent replay


class DeltaOp(BaseModel):
    op: str
    target_id: str | None = None
    new_bullet: dict | None = None
    patch: str | None = None


class Delta(BaseModel):
    ops: list[DeltaOp] = Field(default_factory=list)


class RefineOp(BaseModel):
    op: str
    survivor_id: str | None = None
    target_ids: list[str] = Field(default_factory=list)


class RefineResult(BaseModel):
    merged: int
    archived: int
    ops: list[RefineOp] = Field(default_factory=list)


class Playbook(BaseModel):
    version: int
    bullets: list[Bullet] = Field(default_factory=list)
