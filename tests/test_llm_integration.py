import os
from unittest.mock import Mock, patch

import pytest
import requests

from ace.llm import CompletionResponse, Message, OpenRouterClient


class TestOpenRouterClientIntegration:
    """Integration tests for OpenRouterClient with mocked HTTP responses."""

    @pytest.fixture
    def mock_api_key(self):
        return "sk-or-v1-test-key-123"

    @pytest.fixture
    def openrouter_client(self, mock_api_key):
        return OpenRouterClient(api_key=mock_api_key, model="openai/gpt-5")

    def test_initialization_with_api_key(self, mock_api_key):
        client = OpenRouterClient(api_key=mock_api_key)
        assert client.api_key == mock_api_key
        assert client.model == "openai/gpt-5"
        assert client.reasoning_effort == "medium"

    def test_initialization_from_env_var(self, monkeypatch, mock_api_key):
        monkeypatch.setenv("OPENROUTER_API_KEY", mock_api_key)
        client = OpenRouterClient()
        assert client.api_key == mock_api_key

    def test_initialization_without_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenRouter API key must be provided"):
            OpenRouterClient()

    def test_initialization_with_custom_model(self, mock_api_key):
        client = OpenRouterClient(api_key=mock_api_key, model="anthropic/claude-3-opus")
        assert client.model == "anthropic/claude-3-opus"

    def test_initialization_with_custom_parameters(self, mock_api_key):
        client = OpenRouterClient(
            api_key=mock_api_key,
            model="openai/gpt-5",
            site_url="https://example.com",
            app_name="Test App",
            default_max_tokens=1000,
            default_temperature=0.5,
            reasoning_effort="high",
        )
        assert client.site_url == "https://example.com"
        assert client.app_name == "Test App"
        assert client.default_max_tokens == 1000
        assert client.default_temperature == 0.5
        assert client.reasoning_effort == "high"

    @patch("requests.post")
    def test_complete_successful_request(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a test response from the API"}}],
            "usage": {"total_tokens": 50},
        }
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Hello, world!")]
        response = openrouter_client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert response.text == "This is a test response from the API"

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://openrouter.ai/api/v1/chat/completions"

        payload = call_args[1]["json"]
        assert payload["model"] == "openai/gpt-5"
        assert payload["messages"] == [{"role": "user", "content": "Hello, world!"}]
        assert payload["reasoning"] == {"effort": "medium"}

    @patch("requests.post")
    def test_complete_with_custom_parameters(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"total_tokens": 30},
        }
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        openrouter_client.complete(messages, temperature=0.9, max_tokens=500, top_p=0.95)

        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 500
        assert payload["top_p"] == 0.95

    @patch("requests.post")
    def test_complete_with_site_url_and_app_name(self, mock_post, mock_api_key):
        client = OpenRouterClient(
            api_key=mock_api_key, site_url="https://myapp.com", app_name="My App"
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        client.complete(messages)

        headers = mock_post.call_args[1]["headers"]
        assert headers["HTTP-Referer"] == "https://myapp.com"
        assert headers["X-Title"] == "My App"

    @patch("requests.post")
    def test_complete_with_multiple_messages(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_post.return_value = mock_response

        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
        ]
        openrouter_client.complete(messages)

        payload = mock_post.call_args[1]["json"]
        assert len(payload["messages"]) == 4
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][3]["content"] == "Second question"

    @patch("requests.post")
    def test_complete_http_error(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]

        with pytest.raises((requests.exceptions.RequestException, KeyError)):
            openrouter_client.complete(messages)

    @patch("requests.post")
    def test_complete_no_choices_in_response(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]

        with pytest.raises(ValueError, match="No choices returned"):
            openrouter_client.complete(messages)

    @patch("requests.post")
    def test_complete_malformed_response(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "structure"}
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]

        with pytest.raises((KeyError, ValueError)):
            openrouter_client.complete(messages)

    @patch("requests.post")
    def test_complete_with_reasoning_effort_high(self, mock_post, mock_api_key):
        client = OpenRouterClient(api_key=mock_api_key, reasoning_effort="high")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        client.complete(messages)

        payload = mock_post.call_args[1]["json"]
        assert payload["reasoning"] == {"effort": "high"}

    @patch("requests.post")
    def test_complete_with_reasoning_effort_none(self, mock_post, mock_api_key):
        client = OpenRouterClient(api_key=mock_api_key, reasoning_effort=None)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        client.complete(messages)

        payload = mock_post.call_args[1]["json"]
        assert "reasoning" not in payload

    @patch("requests.post")
    def test_complete_with_timeout(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        openrouter_client.complete(messages, timeout=30)

        assert mock_post.call_args[1]["timeout"] == 30

    @patch("requests.post")
    def test_complete_with_default_timeout(self, mock_post, openrouter_client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="Test")]
        openrouter_client.complete(messages)

        assert mock_post.call_args[1]["timeout"] == 60


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping live API test",
)
class TestOpenRouterClientLiveAPI:
    """Live API integration tests - only run when OPENROUTER_API_KEY is set."""

    def test_live_api_call(self):
        client = OpenRouterClient(model="openai/gpt-5")
        messages = [
            Message(role="system", content="You are a helpful assistant. Be concise."),
            Message(role="user", content="Say 'Hello' in one word."),
        ]

        response = client.complete(messages, max_tokens=10)

        assert isinstance(response, CompletionResponse)
        assert len(response.text) > 0
        assert isinstance(response.text, str)
