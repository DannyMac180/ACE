#!/usr/bin/env python3
"""Generate a deterministic reflection for the ACE proof demo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from ace.generator.schemas import TrajectoryDoc
from ace.llm.client import LLMClient
from ace.llm.schemas import CompletionResponse, Message
from ace.reflector.reflector import Reflector


class StaticReflectionClient(LLMClient):
    """Small deterministic client used to keep the proof demo offline and repeatable."""

    def __init__(self, reflection_json: str) -> None:
        self.reflection_json = reflection_json

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        return CompletionResponse(text=self.reflection_json)


def build_reflection_payload(doc: TrajectoryDoc) -> dict[str, object]:
    helpful_ids = doc.retrieved_bullet_ids[:2] or ["strat-00001", "seed-retrieval-hygiene"]

    return {
        "error_identification": "The first retrieval attempt was too generic.",
        "root_cause_analysis": "The query omitted the concrete ACE action and tool surface.",
        "correct_approach": (
            "Include the exact task outcome and CLI surface in the retrieval query "
            "before reflecting on the run."
        ),
        "key_insight": (
            "Specific task wording helps ACE surface tactical bullets instead of "
            "broad hygiene advice."
        ),
        "bullet_tags": [{"id": bullet_id, "tag": "helpful"} for bullet_id in helpful_ids],
        "candidate_bullets": [
            {
                "section": "strategies_and_hard_rules",
                "content": (
                    "When retrieval returns a generic result, tighten the query with the "
                    "concrete task and tool names before reflecting."
                ),
                "tags": ["repo:ace", "topic:retrieval", "tool:cli"],
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc", help="Path to a trajectory doc JSON file.")
    args = parser.parse_args()

    with open(args.doc, encoding="utf-8") as handle:
        doc = TrajectoryDoc.model_validate_json(handle.read())

    reflection_payload = build_reflection_payload(doc)
    reflector = Reflector(
        llm_client=StaticReflectionClient(json.dumps(reflection_payload)),
        max_retries=1,
        refinement_rounds=1,
        quality_threshold=0.0,
    )
    reflection = reflector.reflect(doc)

    json.dump(asdict(reflection), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
