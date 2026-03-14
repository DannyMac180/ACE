"""Regression coverage for injected custom LLM provider examples."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ace.core.schema import Bullet
from ace.core.storage.store_adapter import Store
from ace.generator.schemas import TrajectoryDoc
from ace.llm import CompletionResponse, LLMClient, Message
from ace.pipeline import Pipeline
from ace.reflector import Reflector


class DeterministicCustomClient(LLMClient):
    """Custom client used to prove ACE accepts injected providers."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        del messages, kwargs
        self.calls += 1
        return CompletionResponse(
            text="""
            {
              "error_identification": "Custom provider handled reflection",
              "root_cause_analysis": "The injected client returned strict JSON",
              "correct_approach": "Pass llm_client explicitly",
              "key_insight": "Injection is the stable extension seam",
              "bullet_tags": [{"id": "strat-001", "tag": "helpful"}],
              "candidate_bullets": [
                {
                  "section": "strategies_and_hard_rules",
                  "content": "Inject provider clients instead of patching ACE internals.",
                  "tags": ["topic:llm", "repo:ace"]
                }
              ]
            }
            """
        )


def _make_store(db_path: str, index_path: str) -> Store:
    from ace.core.storage import embedding_store

    original_init = embedding_store.EmbeddingStore.__init__

    def patched_init(self, db_conn, saved_index_path=index_path):
        original_init(self, db_conn, saved_index_path)

    embedding_store.EmbeddingStore.__init__ = patched_init
    try:
        store = Store(db_path)
    finally:
        embedding_store.EmbeddingStore.__init__ = original_init
    return store


def test_custom_provider_example_script_runs():
    result = subprocess.run(
        [sys.executable, "scripts/custom_llm_provider_example.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "ACE_LLM_PROVIDER": "mock"},
    )

    assert "Custom provider reflection summary" in result.stdout
    assert "Candidate bullets: 1" in result.stdout
    assert "Inject custom LLMClient implementations" in result.stdout


def test_pipeline_accepts_injected_custom_llm_client():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = str(Path(temp_dir) / "ace.db")
        index_path = str(Path(temp_dir) / "embeddings.faiss")
        store = _make_store(db_path, index_path)

        try:
            store.save_bullet(
                Bullet(
                    id="strat-001",
                    section="strategies_and_hard_rules",
                    content="Prefer hybrid retrieval before giving up on sparse recall.",
                    tags=["topic:retrieval"],
                )
            )

            client = DeterministicCustomClient()
            pipeline = Pipeline(store=store, llm_client=client, retrieval_top_k=1)

            result = pipeline.run_with_feedback(
                query="debug custom provider integration",
                code_diff="diff --git a/llm.py b/llm.py",
                test_output="FAILED tests/test_provider.py::test_integration",
                logs="ValueError: provider payload malformed",
                auto_commit=False,
            )

            assert client.calls == 1
            assert result.reflection.error_identification == "Custom provider handled reflection"
            assert result.reflection.key_insight == "Injection is the stable extension seam"
            assert result.delta_ops_applied == 0
            assert result.retrieved_bullets[0].id == "strat-001"
        finally:
            store.close()


def test_reflector_accepts_injected_custom_llm_client():
    client = DeterministicCustomClient()
    reflector = Reflector(llm_client=client)
    doc = TrajectoryDoc(
        query="stabilize provider integration",
        retrieved_bullet_ids=["strat-001"],
        code_diff="diff --git a/provider.py b/provider.py",
        test_output="FAILED tests/test_provider.py::test_retry_policy",
        logs="TimeoutError: request exceeded deadline",
        env_meta={"provider": "custom"},
    )

    reflection = reflector.reflect(doc)

    assert client.calls == 1
    assert reflection.correct_approach == "Pass llm_client explicitly"
    assert reflection.candidate_bullets[0].tags == ["topic:llm", "repo:ace"]
