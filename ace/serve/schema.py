"""Schemas for online serving mode."""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class AdaptationMode(str, Enum):
    """Adaptation mode for ACE.

    - OFFLINE: Batch processing with ground-truth labels available
    - ONLINE: Test-time sequential adaptation with execution feedback only
    """

    OFFLINE = "offline"
    ONLINE = "online"


class FeedbackRequest(BaseModel):
    """Request for online adaptation based on execution feedback.

    Unlike offline mode which has ground-truth labels, online mode
    relies solely on execution feedback (test output, logs, errors).
    """

    query: str
    retrieved_bullet_ids: list[str] = Field(default_factory=list)
    code_diff: str = ""
    test_output: str = ""
    logs: str = ""
    env_meta: dict | None = None
    execution_success: bool | None = None
    error_message: str | None = None


class FeedbackResponse(BaseModel):
    """Response from online adaptation."""

    success: bool
    ops_applied: int = 0
    playbook_version: int = 0
    adaptation_ms: float = 0.0
    message: str = ""


class WarmupSource(str, Enum):
    """Source of warmup data.

    - NONE: Cold start, no preloaded playbook
    - FILE: Preloaded from a JSON file
    - DATABASE: Started with existing database playbook
    """

    NONE = "none"
    FILE = "file"
    DATABASE = "database"


class OnlineStats(BaseModel):
    """Statistics for online serving session."""

    session_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    requests_processed: int = 0
    total_ops_applied: int = 0
    helpful_feedback_count: int = 0
    harmful_feedback_count: int = 0
    avg_adaptation_ms: float = 0.0
    warmup_source: WarmupSource = WarmupSource.NONE
    warmup_bullets_loaded: int = 0
    warmup_playbook_version: int = 0


class RetrieveRequest(BaseModel):
    """Request for retrieving bullets."""

    query: str
    top_k: int = 24


class RetrieveResponse(BaseModel):
    """Response with retrieved bullets."""

    bullets: list[dict]
    retrieval_ms: float = 0.0
