from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """Represents a single step in a task execution trajectory."""

    action: str = Field(..., description="Action taken in this step")
    observation: str = Field(..., description="Result or observation from the action")
    thought: str = Field(..., description="Reasoning or thought process for this step")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this step occurred"
    )


class Trajectory(BaseModel):
    """Represents the complete execution trajectory of a task."""

    steps: list[Step] = Field(default_factory=list, description="Sequence of steps taken")
    initial_goal: str = Field(..., description="The original goal or task")
    final_status: Literal["success", "failure", "partial"] = Field(
        ..., description="Final outcome status"
    )
    total_steps: int = Field(default=0, description="Total number of steps executed")
    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="When execution started"
    )
    completed_at: datetime | None = Field(default=None, description="When execution completed")

    def model_post_init(self, __context):
        """Auto-calculate total_steps from steps list."""
        if self.total_steps == 0:
            self.total_steps = len(self.steps)
