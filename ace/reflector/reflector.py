# ace/reflector/reflector.py
from typing import TYPE_CHECKING

from ace.llm import LLMClient, Message, create_llm_client

from .parser import QualityParseError, ReflectionParseError, parse_quality, parse_reflection
from .prompts import format_quality_eval_prompt, format_refinement_prompt, format_reflector_prompt
from .schema import RefinementQuality, Reflection

if TYPE_CHECKING:
    from ace.generator.schemas import Trajectory


class Reflector:
    """Reflector component that generates Reflection objects from task outcomes."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_retries: int = 3,
        temperature: float = 0.3,
        refinement_rounds: int = 1,
        quality_threshold: float = 0.7,
    ):
        """Initialize Reflector.

        Args:
            llm_client: LLM client for generating reflections. If None, creates one
                        from config using the factory.
            max_retries: Maximum retry attempts on parse errors
            temperature: LLM temperature (lower = more deterministic)
            refinement_rounds: Maximum refinement iterations (1 = no refinement)
            quality_threshold: Quality score threshold (0-1) to stop early
        """
        self.max_retries = max_retries
        self.temperature = temperature
        self.refinement_rounds = max(1, refinement_rounds)
        self.quality_threshold = quality_threshold
        self.client = llm_client if llm_client is not None else create_llm_client()

    def reflect(
        self,
        query: str,
        retrieved_bullet_ids: list[str],
        code_diff: str = "",
        test_output: str = "",
        logs: str = "",
        env_meta: dict | None = None,
        retrieved_bullets: list[dict[str, str]] | None = None,
    ) -> Reflection:
        """Generate a Reflection from task execution data with optional iterative refinement.

        If refinement_rounds > 1, the reflection will be evaluated for quality
        and refined until the quality threshold is met or max rounds reached.

        Args:
            query: The task or query that was executed
            retrieved_bullet_ids: IDs of bullets retrieved for this task
            code_diff: Code changes made during execution
            test_output: Test results or output
            logs: Execution logs
            env_meta: Additional environment metadata
            retrieved_bullets: List of dicts with 'id' and 'content' for redundancy check

        Returns:
            Reflection: Parsed reflection object

        Raises:
            ReflectionParseError: If parsing fails after max_retries
        """
        reflection = self._generate_initial_reflection(
            query=query,
            retrieved_bullet_ids=retrieved_bullet_ids,
            code_diff=code_diff,
            test_output=test_output,
            logs=logs,
            env_meta=env_meta,
        )

        if self.refinement_rounds <= 1:
            return reflection

        bullets_for_eval = retrieved_bullets or []

        for _round_num in range(1, self.refinement_rounds):
            quality = self._evaluate_quality(query, bullets_for_eval, reflection)

            if quality.overall_score >= self.quality_threshold:
                break

            if not quality.feedback:
                break

            reflection = self._refine_reflection(query, reflection, quality.feedback)

        return reflection

    def _generate_initial_reflection(
        self,
        query: str,
        retrieved_bullet_ids: list[str],
        code_diff: str = "",
        test_output: str = "",
        logs: str = "",
        env_meta: dict | None = None,
    ) -> Reflection:
        """Generate the initial reflection (with parse retries)."""
        system_prompt, user_prompt = format_reflector_prompt(
            query=query,
            retrieved_bullet_ids=retrieved_bullet_ids,
            code_diff=code_diff,
            test_output=test_output,
            logs=logs,
            env_meta=env_meta,
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.complete(
                    messages=[
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=user_prompt),
                    ],
                    temperature=self.temperature,
                )

                json_str = response.text
                if not json_str:
                    raise ReflectionParseError("Empty response from LLM")

                return parse_reflection(json_str)

            except ReflectionParseError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    user_prompt += (
                        f"\n\nPrevious attempt failed: {e}. "
                        "Please output ONLY valid JSON without markdown fencing."
                    )
                    continue
                else:
                    raise ReflectionParseError(
                        f"Failed to parse reflection after {self.max_retries} "
                        f"attempts. Last error: {e}"
                    ) from None

        raise ReflectionParseError(f"Unexpected error: {last_error}")

    def _evaluate_quality(
        self,
        query: str,
        retrieved_bullets: list[dict[str, str]],
        reflection: Reflection,
    ) -> RefinementQuality:
        """Evaluate quality of a reflection.

        Args:
            query: The original task query
            retrieved_bullets: List of dicts with 'id' and 'content' for redundancy check
            reflection: The reflection to evaluate
        """
        reflection_json = self._reflection_to_json(reflection)
        system_prompt, user_prompt = format_quality_eval_prompt(
            query=query,
            retrieved_bullets=retrieved_bullets,
            reflection_json=reflection_json,
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.complete(
                    messages=[
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=user_prompt),
                    ],
                    temperature=self.temperature,
                )

                json_str = response.text
                if not json_str:
                    raise QualityParseError("Empty response from LLM")

                return parse_quality(json_str)

            except QualityParseError as e:
                if attempt < self.max_retries - 1:
                    user_prompt += (
                        f"\n\nPrevious attempt failed: {e}. "
                        "Please output ONLY valid JSON without markdown fencing."
                    )
                    continue
                else:
                    # Return high score with empty feedback to skip refinement
                    return RefinementQuality(
                        specificity=1.0,
                        actionability=1.0,
                        redundancy=0.0,
                        feedback="",
                    )

        return RefinementQuality(
            specificity=1.0, actionability=1.0, redundancy=0.0, feedback=""
        )

    def _refine_reflection(
        self,
        query: str,
        reflection: Reflection,
        feedback: str,
    ) -> Reflection:
        """Refine a reflection based on quality feedback."""
        reflection_json = self._reflection_to_json(reflection)
        system_prompt, user_prompt = format_refinement_prompt(
            query=query,
            reflection_json=reflection_json,
            feedback=feedback,
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.complete(
                    messages=[
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=user_prompt),
                    ],
                    temperature=self.temperature,
                )

                json_str = response.text
                if not json_str:
                    raise ReflectionParseError("Empty response from LLM")

                return parse_reflection(json_str)

            except ReflectionParseError as e:
                if attempt < self.max_retries - 1:
                    user_prompt += (
                        f"\n\nPrevious attempt failed: {e}. "
                        "Please output ONLY valid JSON without markdown fencing."
                    )
                    continue
                else:
                    return reflection

        return reflection

    def _reflection_to_json(self, reflection: Reflection) -> str:
        """Convert Reflection to JSON string."""
        import json

        data = {
            "error_identification": reflection.error_identification,
            "root_cause_analysis": reflection.root_cause_analysis,
            "correct_approach": reflection.correct_approach,
            "key_insight": reflection.key_insight,
            "bullet_tags": [
                {"id": bt.id, "tag": bt.tag} for bt in reflection.bullet_tags
            ],
            "candidate_bullets": [
                {"section": cb.section, "content": cb.content, "tags": cb.tags}
                for cb in reflection.candidate_bullets
            ],
        }
        return json.dumps(data, indent=2)

    def reflect_on_trajectory(self, trajectory: "Trajectory") -> Reflection:
        """Generate a Reflection from a complete Trajectory.

        This is a trajectory-aware helper that extracts code_diff, test_output, and logs
        from trajectory steps and automatically passes used_bullet_ids.

        Args:
            trajectory: The complete execution trajectory

        Returns:
            Reflection: Parsed reflection object
        """
        code_diff, test_output, logs = self._extract_trajectory_context(trajectory)

        return self.reflect(
            query=trajectory.initial_goal,
            retrieved_bullet_ids=trajectory.used_bullet_ids,
            code_diff=code_diff,
            test_output=test_output,
            logs=logs,
            env_meta={
                "final_status": trajectory.final_status,
                "total_steps": trajectory.total_steps,
                "bullet_feedback": trajectory.bullet_feedback,
            },
        )

    def _extract_trajectory_context(
        self, trajectory: "Trajectory"
    ) -> tuple[str, str, str]:
        """Extract code_diff, test_output, and logs from trajectory steps.

        Scans step observations and actions for patterns indicating:
        - Code changes (diffs, file modifications)
        - Test results (pass/fail patterns, pytest output)
        - Logs (error messages, stack traces)

        Args:
            trajectory: The trajectory to extract context from

        Returns:
            Tuple of (code_diff, test_output, logs)
        """
        code_diffs: list[str] = []
        test_outputs: list[str] = []
        logs: list[str] = []

        for step in trajectory.steps:
            obs = step.observation.lower()

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

        return (
            "\n\n".join(code_diffs) if code_diffs else "",
            "\n\n".join(test_outputs) if test_outputs else "",
            "\n\n".join(logs) if logs else "",
        )
