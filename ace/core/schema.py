from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

Section = Literal[
    "strategies_and_hard_rules",
    "code_snippets_and_templates",
    "troubleshooting_and_pitfalls",
    "domain_facts_and_references",
]

# Backward compatibility: map old section names to new ones
SECTION_MIGRATION_MAP: dict[str, Section] = {
    # Old names -> New names
    "strategies": "strategies_and_hard_rules",
    "templates": "code_snippets_and_templates",
    "code_snippets": "code_snippets_and_templates",
    "troubleshooting": "troubleshooting_and_pitfalls",
    "facts": "domain_facts_and_references",
    # New names map to themselves
    "strategies_and_hard_rules": "strategies_and_hard_rules",
    "code_snippets_and_templates": "code_snippets_and_templates",
    "troubleshooting_and_pitfalls": "troubleshooting_and_pitfalls",
    "domain_facts_and_references": "domain_facts_and_references",
}


def normalize_section(value: str) -> Section:
    """Normalize section value, mapping old names to new names for backward compatibility."""
    if value in SECTION_MIGRATION_MAP:
        return SECTION_MIGRATION_MAP[value]
    raise ValueError(
        f"Invalid section: {value}. Must be one of: {list(SECTION_MIGRATION_MAP.keys())}"
    )


# Type that accepts both old and new section names, normalizing to new
NormalizedSection = Annotated[Section, BeforeValidator(normalize_section)]


class Bullet(BaseModel):
    id: str
    section: NormalizedSection
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
    section: NormalizedSection
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
