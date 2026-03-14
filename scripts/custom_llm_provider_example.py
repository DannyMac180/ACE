"""Executable example for integrating a custom LLM provider with ACE."""

from __future__ import annotations

import json

from ace.generator.schemas import TrajectoryDoc
from ace.llm import CompletionResponse, LLMClient, Message
from ace.reflector import Reflector


class StaticReflectionClient(LLMClient):
    """Minimal custom client that returns a fixed Reflection payload."""

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        del messages, kwargs
        return CompletionResponse(
            text=json.dumps(
                {
                    "error_identification": "Pytest failed after a missing retry policy",
                    "root_cause_analysis": "The client retried inconsistently across commands",
                    "correct_approach": "Centralize retry handling in one wrapper",
                    "key_insight": (
                        "Custom providers only need to implement the "
                        "LLMClient interface."
                    ),
                    "bullet_tags": [{"id": "strat-00001", "tag": "helpful"}],
                    "candidate_bullets": [
                        {
                            "section": "strategies_and_hard_rules",
                            "content": (
                                "Inject custom LLMClient implementations "
                                "instead of forking ACE components."
                            ),
                            "tags": ["topic:llm", "topic:integration", "repo:ace"],
                        }
                    ],
                }
            )
        )


def main() -> None:
    doc = TrajectoryDoc(
        query="stabilize provider retries",
        retrieved_bullet_ids=["strat-00001"],
        code_diff="diff --git a/client.py b/client.py",
        test_output="FAILED tests/test_provider.py::test_retries",
        logs="TimeoutError: upstream provider request exceeded deadline",
        env_meta={"repo": "ace", "provider": "custom"},
    )

    reflector = Reflector(llm_client=StaticReflectionClient())
    reflection = reflector.reflect(doc)

    print("Custom provider reflection summary")
    print(f"Error: {reflection.error_identification}")
    print(f"Insight: {reflection.key_insight}")
    print(f"Bullet tags: {len(reflection.bullet_tags)}")
    print(f"Candidate bullets: {len(reflection.candidate_bullets)}")
    print(f"First candidate: {reflection.candidate_bullets[0].content}")


if __name__ == "__main__":
    main()
