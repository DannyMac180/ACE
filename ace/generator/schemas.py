import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TrajectoryDoc(BaseModel):
    """
    Document capturing complete task execution context for reflection.

    This is the public schema that external agents (MCP clients, CI systems, etc.)
    should use when recording trajectories. The Reflector accepts this directly.
    """

    trajectory_id: str = Field(
        default_factory=lambda: f"traj-{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this trajectory",
    )
    query: str = Field(..., description="The task or query that was executed")
    retrieved_bullet_ids: list[str] = Field(
        default_factory=list,
        description="IDs of playbook bullets retrieved and used during execution",
    )
    code_diff: str = Field(default="", description="Code changes made during execution")
    test_output: str = Field(default="", description="Test results or output")
    logs: str = Field(default="", description="Execution logs, errors, stack traces")
    env_meta: dict = Field(
        default_factory=dict,
        description="Environment metadata (e.g., final_status, tool versions, runtime info)",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="List of tools or actions invoked during execution",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this trajectory was recorded",
    )


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
    used_bullet_ids: list[str] = Field(
        default_factory=list, description="Bullet IDs retrieved and used during generation"
    )
    bullet_feedback: dict[str, str] = Field(
        default_factory=dict,
        description="Feedback on bullets used: bullet_id -> 'helpful' | 'harmful'",
    )

    def model_post_init(self, __context):
        """Auto-calculate total_steps from steps list."""
        if self.total_steps == 0:
            self.total_steps = len(self.steps)

    def to_trajectory_doc(self) -> TrajectoryDoc:
        """
        Convert this Trajectory to a TrajectoryDoc for reflection.

        Extracts code_diff, test_output, and logs from step observations
        based on content patterns.

        Returns:
            TrajectoryDoc ready for reflection
        """
        code_diffs: list[str] = []
        test_outputs: list[str] = []
        logs: list[str] = []
        tools_used: list[str] = []

        for step in self.steps:
            obs = step.observation.lower()
            tools_used.append(step.action)

            if any(
                pattern in obs
                for pattern in ["diff", "+++", "---", "@@", "modified", "created file"]
            ):
                code_diffs.append(f"Step: {step.action}\n{step.observation}")

            if any(
                pattern in obs
                for pattern in [
                    "passed",
                    "failed",
                    "error",
                    "pytest",
                    "test_",
                    "assert",
                    "traceback",
                ]
            ):
                test_outputs.append(f"Step: {step.action}\n{step.observation}")

            if any(
                pattern in obs
                for pattern in ["exception", "error:", "warning:", "log:", "stderr"]
            ):
                logs.append(f"Step: {step.action}\n{step.observation}")

        return TrajectoryDoc(
            query=self.initial_goal,
            retrieved_bullet_ids=self.used_bullet_ids,
            code_diff="\n\n".join(code_diffs) if code_diffs else "",
            test_output="\n\n".join(test_outputs) if test_outputs else "",
            logs="\n\n".join(logs) if logs else "",
            env_meta={
                "final_status": self.final_status,
                "total_steps": self.total_steps,
                "bullet_feedback": self.bullet_feedback,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            },
            tools_used=tools_used,
            timestamp=self.completed_at or self.started_at,
        )
