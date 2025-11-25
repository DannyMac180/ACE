import logging
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ace.generator.schemas import Step, Trajectory

if TYPE_CHECKING:
    from ace.core.retrieve import Retriever

logger = logging.getLogger(__name__)


class Generator:
    """
    Generator component that executes tasks and captures complete trajectories.

    Implements a basic ReAct-style loop: Reason → Act → Observe.
    """

    def __init__(
        self,
        max_steps: int = 10,
        tool_executor: Callable[[str], str] | None = None,
        retriever: "Retriever | None" = None,
        retrieval_top_k: int = 8,
    ):
        """
        Initialize the Generator.

        Args:
            max_steps: Maximum number of steps to execute before stopping
            tool_executor: Optional callable to execute actions/tools
            retriever: Optional Retriever to fetch relevant bullets before reasoning
            retrieval_top_k: Number of bullets to retrieve per step
        """
        self.max_steps = max_steps
        self.tool_executor = tool_executor or self._default_tool_executor
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k

    def _default_tool_executor(self, action: str) -> str:
        """
        Default tool executor that simulates simple actions.

        Args:
            action: The action to execute

        Returns:
            Simulated observation result
        """
        action_lower = action.lower()

        if "search" in action_lower:
            return f"Found 3 relevant results for query in '{action}'"
        elif "read" in action_lower:
            return f"Successfully read content from '{action}'"
        elif "write" in action_lower:
            return f"Successfully wrote content in '{action}'"
        elif "analyze" in action_lower:
            return f"Analysis complete for '{action}'"
        else:
            return f"Executed action: {action}"

    def _retrieve_bullets(self, query: str, used_bullet_ids: set[str]) -> list[str]:
        """
        Retrieve relevant bullets for the current step.

        Args:
            query: Query string for retrieval
            used_bullet_ids: Set to track bullet IDs (mutated in place)

        Returns:
            List of bullet IDs retrieved
        """
        if not self.retriever:
            return []

        bullets = self.retriever.retrieve(query, top_k=self.retrieval_top_k)
        step_bullet_ids = []
        for bullet in bullets:
            used_bullet_ids.add(bullet.id)
            step_bullet_ids.append(bullet.id)
            logger.debug(f"Retrieved bullet {bullet.id}: {bullet.content[:50]}...")

        return step_bullet_ids

    def _format_bullets_for_reasoning(self, bullet_ids: list[str]) -> str:
        """
        Format retrieved bullets as context for reasoning.

        Args:
            bullet_ids: IDs of bullets to format

        Returns:
            Formatted bullet context string
        """
        if not self.retriever or not bullet_ids:
            return ""

        bullet_lines = []
        for bullet_id in bullet_ids:
            bullet = self.retriever.store.get_bullet(bullet_id)
            if bullet:
                bullet_lines.append(f"- [{bullet.section}] {bullet.content}")

        if bullet_lines:
            return "Relevant playbook guidance:\n" + "\n".join(bullet_lines)
        return ""

    def _reason(
        self, goal: str, steps_so_far: list[Step], bullet_context: str = ""
    ) -> str:
        """
        Generate reasoning/thought for the next action.

        Args:
            goal: The original goal
            steps_so_far: Steps executed so far
            bullet_context: Formatted bullet context from retrieval

        Returns:
            Reasoning string
        """
        base_thought = ""
        if not steps_so_far:
            base_thought = (
                f"Starting to work on goal: {goal}. Need to break this down into steps."
            )
        elif len(steps_so_far) >= self.max_steps - 1:
            base_thought = "Approaching step limit, need to wrap up and finalize results."
        else:
            last_obs = steps_so_far[-1].observation if steps_so_far else ""
            base_thought = (
                f"Based on previous observation '{last_obs}', "
                f"determining next action for goal: {goal}"
            )

        if bullet_context:
            return f"{base_thought}\n\n{bullet_context}"
        return base_thought

    def _decide_action(self, thought: str, step_num: int, goal: str) -> str:
        """
        Decide the next action based on reasoning.

        Args:
            thought: Current reasoning
            step_num: Current step number
            goal: Original goal

        Returns:
            Action to execute
        """
        if step_num == 0:
            return f"Search for information about: {goal}"
        elif step_num == 1:
            return f"Analyze search results for: {goal}"
        elif step_num == 2:
            return f"Read detailed documentation about: {goal}"
        elif step_num == 3:
            return f"Write summary of findings for: {goal}"
        else:
            return f"Finalize work on: {goal}"

    def _should_continue(self, steps: list[Step], goal: str) -> bool:
        """
        Determine if the loop should continue.

        Args:
            steps: Steps executed so far
            goal: Original goal

        Returns:
            True if should continue, False otherwise
        """
        if len(steps) >= self.max_steps:
            logger.info(f"Reached max steps ({self.max_steps})")
            return False

        if len(steps) >= 4:
            return False

        return True

    def run(self, goal: str) -> Trajectory:
        """
        Execute a task with full trajectory capture.

        Implements a basic ReAct loop:
        1. Retrieve relevant playbook bullets for context
        2. Reason about the current state and goal
        3. Decide on an action
        4. Execute the action
        5. Observe the result
        6. Log the complete step
        7. Repeat until goal is achieved or max steps reached

        Args:
            goal: The task/goal to accomplish

        Returns:
            Complete Trajectory object with all steps and metadata
        """
        logger.info(f"Starting Generator run for goal: {goal}")

        started_at = datetime.utcnow()
        steps: list[Step] = []
        used_bullet_ids: set[str] = set()
        bullet_feedback: dict[str, str] = {}

        try:
            while self._should_continue(steps, goal):
                step_num = len(steps)

                # Build retrieval query from goal and recent context
                if steps:
                    last_obs = steps[-1].observation
                    retrieval_query = f"{goal} {last_obs}"
                else:
                    retrieval_query = goal

                # Retrieve relevant bullets before reasoning
                step_bullet_ids = self._retrieve_bullets(retrieval_query, used_bullet_ids)
                bullet_context = self._format_bullets_for_reasoning(step_bullet_ids)
                if step_bullet_ids:
                    logger.debug(f"Step {step_num} - Retrieved {len(step_bullet_ids)} bullets")

                # Reason (with bullet context)
                thought = self._reason(goal, steps, bullet_context)
                logger.debug(f"Step {step_num} - Thought: {thought}")

                # Act
                action = self._decide_action(thought, step_num, goal)
                logger.debug(f"Step {step_num} - Action: {action}")

                # Observe
                observation = self.tool_executor(action)
                logger.debug(f"Step {step_num} - Observation: {observation}")

                # Record step
                step = Step(
                    action=action,
                    observation=observation,
                    thought=thought,
                    timestamp=datetime.utcnow(),
                )
                steps.append(step)
                logger.info(f"Completed step {step_num + 1}/{self.max_steps}")

            final_status: Literal["success", "failure", "partial"] = (
                "success" if len(steps) > 0 else "failure"
            )

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            final_status = "failure"

        completed_at = datetime.utcnow()

        trajectory = Trajectory(
            steps=steps,
            initial_goal=goal,
            final_status=final_status,
            total_steps=len(steps),
            started_at=started_at,
            completed_at=completed_at,
            used_bullet_ids=list(used_bullet_ids),
            bullet_feedback=bullet_feedback,
        )

        logger.info(
            f"Generator run completed: {len(steps)} steps, "
            f"status={final_status}, "
            f"bullets_used={len(used_bullet_ids)}, "
            f"duration={completed_at - started_at}"
        )

        return trajectory
