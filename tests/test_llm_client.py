import pytest

from ace.llm import CompletionResponse, LLMClient, Message, MockLLMClient


class TestLLMClient:
    """Tests for the abstract LLMClient interface."""

    def test_llm_client_is_abstract(self):
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore

    def test_llm_client_subclass_must_implement_complete(self):
        class IncompleteLLMClient(LLMClient):
            pass

        with pytest.raises(TypeError):
            IncompleteLLMClient()  # type: ignore

    def test_llm_client_subclass_with_complete_works(self):
        class CompleteLLMClient(LLMClient):
            def complete(self, messages, **kwargs):
                return CompletionResponse(text="test")

        client = CompleteLLMClient()
        assert client is not None
        result = client.complete([])
        assert isinstance(result, CompletionResponse)


class TestMockLLMClient:
    """Tests for the MockLLMClient implementation."""

    def test_initialization(self):
        client = MockLLMClient()
        assert client.response_prefix == "Mock response:"

    def test_initialization_with_custom_prefix(self):
        client = MockLLMClient(response_prefix="Custom:")
        assert client.response_prefix == "Custom:"

    def test_complete_with_empty_messages(self):
        client = MockLLMClient()
        response = client.complete([])
        assert isinstance(response, CompletionResponse)
        assert "No messages provided" in response.text

    def test_complete_with_reflect_keyword(self):
        client = MockLLMClient()
        messages = [Message(role="user", content="Please reflect on this error")]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert "minor issue" in response.text
        assert "Root cause" in response.text
        assert "validation" in response.text

    def test_complete_with_error_keyword(self):
        client = MockLLMClient()
        messages = [Message(role="user", content="There was an error in the code")]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert "minor issue" in response.text

    def test_complete_with_curate_keyword(self):
        client = MockLLMClient()
        messages = [Message(role="user", content="Curate these findings")]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert "Proposing the following changes" in response.text
        assert '"op": "ADD"' in response.text
        assert "strategies" in response.text

    def test_complete_with_delta_keyword(self):
        client = MockLLMClient()
        messages = [Message(role="user", content="Generate a delta")]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert "Proposing the following changes" in response.text
        assert '"op":' in response.text

    def test_complete_with_analyze_keyword(self):
        client = MockLLMClient()
        messages = [Message(role="user", content="Analyze this system")]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert "Analysis complete" in response.text
        assert "functioning as expected" in response.text

    def test_complete_with_generic_message(self):
        client = MockLLMClient()
        messages = [Message(role="user", content="What is the weather today?")]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert "Mock response:" in response.text
        assert "What is the weather today?" in response.text

    def test_complete_with_multiple_messages(self):
        client = MockLLMClient()
        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="Tell me something"),
        ]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert "4 message(s)" in response.text

    def test_complete_respects_custom_prefix(self):
        client = MockLLMClient(response_prefix="TEST:")
        messages = [Message(role="user", content="Generic request")]
        response = client.complete(messages)

        assert "TEST:" in response.text
        assert "Mock response:" not in response.text

    def test_complete_with_long_content(self):
        client = MockLLMClient()
        long_content = "x" * 100
        messages = [Message(role="user", content=long_content)]
        response = client.complete(messages)

        assert isinstance(response, CompletionResponse)
        assert len(response.text) > 0

    def test_complete_kwargs_are_ignored(self):
        client = MockLLMClient()
        messages = [Message(role="user", content="Test")]
        response = client.complete(messages, temperature=0.5, max_tokens=100, extra_param="ignored")

        assert isinstance(response, CompletionResponse)

    def test_message_role_and_content(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_completion_response_structure(self):
        response = CompletionResponse(text="Test response")
        assert response.text == "Test response"
