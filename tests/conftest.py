# tests/conftest.py
"""Pytest configuration and shared fixtures."""

import os

# Set test environment before any ace imports
# Only set if not explicitly overridden by a specific test
# This ensures the mock LLM provider is used for most tests
# without breaking config tests that test specific providers
if "ACE_LLM_PROVIDER" not in os.environ:
    os.environ["ACE_LLM_PROVIDER"] = "mock"
