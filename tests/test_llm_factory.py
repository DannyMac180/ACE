# tests/test_llm_factory.py
from unittest.mock import patch

import pytest

from ace.core.config import LLMConfig
from ace.llm import MockLLMClient, OpenRouterClient, create_llm_client


class TestCreateLLMClient:
    """Tests for create_llm_client factory function."""

    def test_creates_mock_client(self):
        """Test factory creates MockLLMClient for 'mock' provider."""
        config = LLMConfig(
            provider="mock",
            model="test-model",
            temperature=0.5,
            max_tokens=100,
        )

        client = create_llm_client(config)

        assert isinstance(client, MockLLMClient)

    def test_creates_mock_client_case_insensitive(self):
        """Test factory handles provider name case-insensitively."""
        config = LLMConfig(
            provider="MOCK",
            model="test-model",
            temperature=0.5,
            max_tokens=100,
        )

        client = create_llm_client(config)

        assert isinstance(client, MockLLMClient)

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_creates_openrouter_client(self):
        """Test factory creates OpenRouterClient for 'openrouter' provider."""
        config = LLMConfig(
            provider="openrouter",
            model="openai/gpt-4",
            temperature=0.7,
            max_tokens=2000,
        )

        client = create_llm_client(config)

        assert isinstance(client, OpenRouterClient)
        assert client.model == "openai/gpt-4"
        assert client.default_temperature == 0.7
        assert client.default_max_tokens == 2000

    def test_raises_for_unsupported_provider(self):
        """Test factory raises ValueError for unsupported provider."""
        config = LLMConfig(
            provider="unsupported-provider",
            model="some-model",
            temperature=0.5,
            max_tokens=100,
        )

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client(config)

    @patch("ace.llm.factory.get_config")
    def test_uses_global_config_when_none_provided(self, mock_get_config):
        """Test factory uses global config when no config provided."""
        mock_llm_config = LLMConfig(
            provider="mock",
            model="test",
            temperature=0.5,
            max_tokens=100,
        )
        mock_get_config.return_value.llm = mock_llm_config

        client = create_llm_client()

        mock_get_config.assert_called_once()
        assert isinstance(client, MockLLMClient)
