from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Section = Literal["strategies", "templates", "troubleshooting", "code_snippets", "facts"]


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


class Delta(BaseModel):
    op: str
    target_id: str | None = None
    new_bullet: DeltaBullet | None = None
    patch: str | None = None


class Playbook(BaseModel):
    version: int
    bullets: list[Bullet] = Field(default_factory=list)
