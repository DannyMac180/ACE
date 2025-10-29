import logging
from collections.abc import Callable
from datetime import datetime

from ace.generator.schemas import Step, Trajectory

logger = logging.getLogger(__name__)


class Generator:
    """
    Generator component that executes tasks and captures complete trajectories.

    Implements a basic ReAct-style loop: Reason → Act → Observe.
    """

    def __init__(self, max_steps: int = 10, tool_executor: Callable[[str], str] | None = None):
        """
        Initialize the Generator.

        Args:
            max_steps: Maximum number of steps to execute before stopping
            tool_executor: Optional callable to execute actions/tools
        """
        self.max_steps = max_steps
        self.tool_executor = tool_executor or self._default_tool_executor

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

    def _reason(self, goal: str, steps_so_far: list[Step]) -> str:
        """
        Generate reasoning/thought for the next action.

        Args:
            goal: The original goal
            steps_so_far: Steps executed so far

        Returns:
            Reasoning string
        """
        if not steps_so_far:
            return f"Starting to work on goal: {goal}. Need to break this down into steps."

        step_num = len(steps_so_far)
        if step_num >= self.max_steps - 1:
            return "Approaching step limit, need to wrap up and finalize results."

        last_obs = steps_so_far[-1].observation if steps_so_far else ""
        return (
            f"Based on previous observation '{last_obs}', determining next action for goal: {goal}"
        )

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
        1. Reason about the current state and goal
        2. Decide on an action
        3. Execute the action
        4. Observe the result
        5. Log the complete step
        6. Repeat until goal is achieved or max steps reached

        Args:
            goal: The task/goal to accomplish

        Returns:
            Complete Trajectory object with all steps and metadata
        """
        logger.info(f"Starting Generator run for goal: {goal}")

        started_at = datetime.utcnow()
        steps: list[Step] = []

        try:
            while self._should_continue(steps, goal):
                step_num = len(steps)

                # Reason
                thought = self._reason(goal, steps)
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

            final_status = "success" if len(steps) > 0 else "failure"

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
        )

        logger.info(
            f"Generator run completed: {len(steps)} steps, "
            f"status={final_status}, "
            f"duration={completed_at - started_at}"
        )

        return trajectory
