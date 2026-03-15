# tests/conftest.py
"""Pytest configuration and shared fixtures."""

import os

import pytest

# Set test environment before any ace imports
# Only set if not explicitly overridden by a specific test
# This ensures the mock LLM provider is used for most tests
# without breaking config tests that test specific providers
if "ACE_LLM_PROVIDER" not in os.environ:
    os.environ["ACE_LLM_PROVIDER"] = "mock"


@pytest.fixture(autouse=True)
def _default_mock_embeddings(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Use deterministic embeddings in tests unless a test opts into a real model."""
    module_name = getattr(request.module, "__name__", "")
    if module_name in {"test_config", "tests.test_config"}:
        return
    if "ACE_EMBEDDINGS" not in os.environ:
        monkeypatch.setenv("ACE_EMBEDDINGS", "mock")
