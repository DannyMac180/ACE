# ace/reflector/reflector.py
import os

from openai import OpenAI

from .parser import ReflectionParseError, parse_reflection
from .prompts import format_reflector_prompt
from .schema import Reflection


class Reflector:
    """Reflector component that generates Reflection objects from task outcomes."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        temperature: float = 0.3,
    ):
        """Initialize Reflector.

        Args:
            model: OpenAI model to use
            max_retries: Maximum retry attempts on parse errors
            temperature: LLM temperature (lower = more deterministic)
        """
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def reflect(
        self,
        query: str,
        retrieved_bullet_ids: list[str],
        code_diff: str = "",
        test_output: str = "",
        logs: str = "",
        env_meta: dict | None = None,
    ) -> Reflection:
        """Generate a Reflection from task execution data.

        Args:
            query: The task or query that was executed
            retrieved_bullet_ids: IDs of bullets retrieved for this task
            code_diff: Code changes made during execution
            test_output: Test results or output
            logs: Execution logs
            env_meta: Additional environment metadata

        Returns:
            Reflection: Parsed reflection object

        Raises:
            ReflectionParseError: If parsing fails after max_retries
        """
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
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )

                json_str = response.choices[0].message.content
                if not json_str:
                    raise ReflectionParseError("Empty response from LLM")

                reflection = parse_reflection(json_str)
                return reflection

            except ReflectionParseError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Add guidance for retry
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

        # Should never reach here, but just in case
        raise ReflectionParseError(f"Unexpected error: {last_error}")
