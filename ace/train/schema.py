"""Schemas for multi-epoch training."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class TrainSample(BaseModel):
    """A training sample supporting both labeled and unlabeled modes.

    This is the primary input format for ACE training data:
    - Labeled mode: ground_truth is present for metric computation
    - Unlabeled mode: feedback-only signals (code_diff, test_output, etc.)

    Attributes:
        query: The task query or prompt
        input: Optional structured input data for the task
        ground_truth: Expected output for labeled samples (enables metrics)
        feedback: Execution feedback matching Reflection inputs
    """

    query: str
    input: dict | None = None
    ground_truth: str | dict | None = None
    feedback: dict | None = Field(
        default=None,
        description="Feedback signals: code_diff, test_output, logs, env_meta, success",
    )

    @property
    def is_labeled(self) -> bool:
        """Check if this sample has ground truth for evaluation."""
        return self.ground_truth is not None

    def get_feedback_field(self, field: str, default: str = "") -> str:
        """Extract a feedback field safely."""
        if self.feedback is None:
            return default
        return self.feedback.get(field, default)


class TrainingSample(BaseModel):
    """A single training sample from offline adaptation data.

    Represents a task execution record with inputs/outputs that can be used
    to train the playbook through reflect→curate→commit.

    Note: For new implementations, prefer TrainSample which supports
    labeled/unlabeled modes with ground_truth.
    """

    id: str
    query: str
    retrieved_bullet_ids: list[str] = Field(default_factory=list)
    code_diff: str = ""
    test_output: str = ""
    logs: str = ""
    env_meta: dict | None = None
    success: bool | None = None

    def to_train_sample(self, sample_id: str | None = None) -> "TrainSample":
        """Convert to TrainSample format for unified processing."""
        return TrainSample(
            query=self.query,
            input={"id": sample_id or self.id},
            ground_truth=None,
            feedback={
                "code_diff": self.code_diff,
                "test_output": self.test_output,
                "logs": self.logs,
                "env_meta": self.env_meta,
                "success": self.success,
                "retrieved_bullet_ids": self.retrieved_bullet_ids,
            },
        )


class SampleEpochRecord(BaseModel):
    """Record of a sample being processed in a specific epoch."""

    sample_id: str
    epoch: int
    processed_at: datetime = Field(default_factory=_utcnow)
    ops_applied: int = 0
    playbook_version_before: int = 0
    playbook_version_after: int = 0


class EpochMetadata(BaseModel):
    """Metadata for a training epoch."""

    epoch: int
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    samples_processed: int = 0
    total_ops_applied: int = 0
    playbook_version_start: int = 0
    playbook_version_end: int = 0


class TrainingState(BaseModel):
    """Persistent state for multi-epoch training.

    Tracks progress across epochs and which samples have been processed.
    """

    current_epoch: int = 0
    total_epochs: int = 1
    epochs: list[EpochMetadata] = Field(default_factory=list)
    sample_records: list[SampleEpochRecord] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None

    def get_processed_samples_for_epoch(self, epoch: int) -> set[str]:
        """Get sample IDs processed in a given epoch."""
        return {r.sample_id for r in self.sample_records if r.epoch == epoch}

    def record_sample(
        self,
        sample_id: str,
        epoch: int,
        ops_applied: int,
        version_before: int,
        version_after: int,
    ) -> None:
        """Record that a sample was processed."""
        self.sample_records.append(
            SampleEpochRecord(
                sample_id=sample_id,
                epoch=epoch,
                ops_applied=ops_applied,
                playbook_version_before=version_before,
                playbook_version_after=version_after,
            )
        )

    def start_epoch(self, epoch: int, playbook_version: int) -> None:
        """Start a new epoch."""
        self.current_epoch = epoch
        self.epochs.append(
            EpochMetadata(
                epoch=epoch,
                playbook_version_start=playbook_version,
            )
        )

    def complete_epoch(
        self, epoch: int, samples_processed: int, ops_applied: int, playbook_version: int
    ) -> None:
        """Complete an epoch."""
        for meta in self.epochs:
            if meta.epoch == epoch:
                meta.completed_at = _utcnow()
                meta.samples_processed = samples_processed
                meta.total_ops_applied = ops_applied
                meta.playbook_version_end = playbook_version
                break

    def is_epoch_in_progress(self, epoch: int) -> bool:
        """Check if an epoch was started but not completed."""
        for meta in self.epochs:
            if meta.epoch == epoch:
                return meta.completed_at is None
        return False

    def is_epoch_completed(self, epoch: int) -> bool:
        """Check if an epoch was completed."""
        for meta in self.epochs:
            if meta.epoch == epoch:
                return meta.completed_at is not None
        return False


class TrainingResult(BaseModel):
    """Result of a training run."""

    epochs_completed: int
    total_samples_processed: int
    total_ops_applied: int
    playbook_version_start: int
    playbook_version_end: int
    duration_seconds: float
    state: TrainingState
