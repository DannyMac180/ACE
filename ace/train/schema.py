"""Schemas for multi-epoch training."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class TrainingSample(BaseModel):
    """A single training sample from offline adaptation data.

    Represents a task execution record with inputs/outputs that can be used
    to train the playbook through reflect→curate→commit.
    """

    id: str
    query: str
    retrieved_bullet_ids: list[str] = Field(default_factory=list)
    code_diff: str = ""
    test_output: str = ""
    logs: str = ""
    env_meta: dict | None = None
    success: bool | None = None


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
