# ace/reflector/schema.py
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BulletTag:
    id: str
    tag: Literal["helpful", "harmful"]


@dataclass
class CandidateBullet:
    section: Literal[
        "strategies_and_hard_rules",
        "code_snippets_and_templates",
        "troubleshooting_and_pitfalls",
        "domain_facts_and_references",
    ]
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


@dataclass
class RefinementQuality:
    """Quality assessment for a Reflection, used in iterative refinement."""
    specificity: float  # 0-1: How specific vs generic are the insights
    actionability: float  # 0-1: How actionable are the candidate bullets
    redundancy: float  # 0-1: How much overlap with existing bullets (lower=better)
    overall_score: float = field(init=False)  # Computed average
    feedback: str = ""  # Suggestions for improvement

    def __post_init__(self):
        self.overall_score = (self.specificity + self.actionability + (1 - self.redundancy)) / 3
