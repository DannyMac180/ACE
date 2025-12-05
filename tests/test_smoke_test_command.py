"""Test for the smoke-test-model CLI command."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from ace.cli import cmd_smoke_test_model
from ace.llm.schemas import CompletionResponse, Message


@pytest.fixture
def mock_args():
    """Create mock CLI arguments."""
    args = argparse.Namespace()
    args.json = False
    return args


@pytest.fixture
def mock_args_json():
    """Create mock CLI arguments with JSON output."""
    args = argparse.Namespace()
    args.json = True
    return args


def test_smoke_test_model_success(mock_args, capsys):
    """Test successful smoke test."""
    with patch("ace.cli.load_config") as mock_load_config, patch(
        "ace.cli.create_llm_client"
    ) as mock_create_client:
        # Setup mocks
        mock_config = MagicMock()
        mock_config.llm.provider = "openrouter"
        mock_config.llm.model = "openai/gpt-4o-mini"
        mock_config.llm.temperature = 0.0
        mock_config.llm.max_tokens = 2000
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_client.complete.return_value = CompletionResponse(
            text="Hello from ACE! I can read your message."
        )
        mock_create_client.return_value = mock_client

        # Run command
        cmd_smoke_test_model(mock_args)

        # Verify
        captured = capsys.readouterr()
        assert "Testing LLM provider: openrouter" in captured.out
        assert "Model: openai/gpt-4o-mini" in captured.out
        assert "✓ LLM client created successfully" in captured.out
        assert "✓ LLM request successful" in captured.out
        assert "✓ Smoke test PASSED" in captured.out
        assert "Hello from ACE!" in captured.out

        # Verify client was called correctly
        mock_create_client.assert_called_once_with(mock_config.llm)
        assert mock_client.complete.call_count == 1
        call_args = mock_client.complete.call_args[0]
        assert len(call_args[0]) == 1
        assert isinstance(call_args[0][0], Message)
        assert call_args[0][0].role == "user"


def test_smoke_test_model_json_output(mock_args_json, capsys):
    """Test smoke test with JSON output."""
    with patch("ace.cli.load_config") as mock_load_config, patch(
        "ace.cli.create_llm_client"
    ) as mock_create_client:
        # Setup mocks
        mock_config = MagicMock()
        mock_config.llm.provider = "mock"
        mock_config.llm.model = "mock-model"
        mock_config.llm.temperature = 0.5
        mock_config.llm.max_tokens = 1000
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_client.complete.return_value = CompletionResponse(text="Test response")
        mock_create_client.return_value = mock_client

        # Run command
        cmd_smoke_test_model(mock_args_json)

        # Verify
        captured = capsys.readouterr()
        assert '"status": "success"' in captured.out
        assert '"provider": "mock"' in captured.out
        assert '"model": "mock-model"' in captured.out


def test_smoke_test_model_config_error(mock_args, capsys):
    """Test smoke test with configuration error."""
    with patch("ace.cli.load_config") as mock_load_config, patch(
        "ace.cli.create_llm_client"
    ) as mock_create_client, pytest.raises(SystemExit) as exc_info:
        # Setup mocks
        mock_config = MagicMock()
        mock_config.llm.provider = "openrouter"
        mock_config.llm.model = "test-model"
        mock_config.llm.temperature = 0.0
        mock_config.llm.max_tokens = 2000
        mock_load_config.return_value = mock_config

        mock_create_client.side_effect = ValueError("API key not found")

        # Run command (should exit with code 1)
        cmd_smoke_test_model(mock_args)

    # Verify
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "✗ Configuration error" in captured.out
    assert "API key not found" in captured.out


def test_smoke_test_model_request_error(mock_args, capsys):
    """Test smoke test with LLM request error."""
    with patch("ace.cli.load_config") as mock_load_config, patch(
        "ace.cli.create_llm_client"
    ) as mock_create_client, pytest.raises(SystemExit) as exc_info:
        # Setup mocks
        mock_config = MagicMock()
        mock_config.llm.provider = "openrouter"
        mock_config.llm.model = "openai/gpt-4o-mini"
        mock_config.llm.temperature = 0.0
        mock_config.llm.max_tokens = 2000
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("Network error")
        mock_create_client.return_value = mock_client

        # Run command (should exit with code 1)
        cmd_smoke_test_model(mock_args)

    # Verify
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "✗ LLM request failed" in captured.out
    assert "Network error" in captured.out
    assert "Troubleshooting tips:" in captured.out
    assert "OPENROUTER_API_KEY" in captured.out
