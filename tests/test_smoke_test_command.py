"""Test for the smoke-test-model CLI command."""

import json
import subprocess
import sys


def test_smoke_test_model_with_mock_provider():
    """Test smoke test with mock provider (no API key required)."""
    # Set environment to use mock provider
    env = {
        "ACE_LLM_PROVIDER": "mock",
        "ACE_LLM_MODEL": "mock-model",
    }

    result = subprocess.run(
        [sys.executable, "-m", "ace.cli", "smoke-test-model"],
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, **env},
    )

    # Mock provider should work without API keys
    assert result.returncode == 0
    assert "Testing LLM provider: mock" in result.stdout
    assert "✓ LLM client created successfully" in result.stdout
    assert "✓ LLM request successful" in result.stdout
    assert "✓ Smoke test PASSED" in result.stdout


def test_smoke_test_model_json_output_with_mock():
    """Test smoke test JSON output with mock provider."""
    env = {
        "ACE_LLM_PROVIDER": "mock",
        "ACE_LLM_MODEL": "mock-model",
    }

    result = subprocess.run(
        [sys.executable, "-m", "ace.cli", "smoke-test-model", "--json"],
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, **env},
    )

    assert result.returncode == 0

    # Should contain JSON output
    assert '"status":' in result.stdout or '"status" :' in result.stdout

    # Parse JSON to verify structure
    try:
        # Extract JSON from output (may have other text before/after)
        lines = result.stdout.strip().split("\n")
        json_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                # Found start of JSON, join remaining lines
                json_line = "\n".join(lines[i:])
                break

        if json_line:
            output = json.loads(json_line)
            assert output["status"] == "success"
            assert output["provider"] == "mock"
    except (json.JSONDecodeError, KeyError):
        # If JSON parsing fails, at least check that success indicators are present
        assert '"status": "success"' in result.stdout or '"status":"success"' in result.stdout


def test_smoke_test_model_missing_api_key_for_openrouter():
    """Test smoke test fails gracefully when OpenRouter API key is missing."""
    env = {
        "ACE_LLM_PROVIDER": "openrouter",
        "ACE_LLM_MODEL": "openai/gpt-4o-mini",
    }

    # Make sure OPENROUTER_API_KEY is not set
    env_without_key = {**subprocess.os.environ, **env}
    env_without_key.pop("OPENROUTER_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "-m", "ace.cli", "smoke-test-model"],
        capture_output=True,
        text=True,
        env=env_without_key,
    )

    # Should fail with exit code 1
    assert result.returncode == 1
    assert "✗ Configuration error" in result.stdout
    assert "api key" in result.stdout.lower()


def test_smoke_test_model_help():
    """Test that smoke-test-model command appears in help."""
    result = subprocess.run(
        [sys.executable, "-m", "ace.cli", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "smoke-test-model" in result.stdout
    assert "Test LLM provider connectivity" in result.stdout
