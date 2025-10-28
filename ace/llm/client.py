import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from ace.llm.schemas import Message, CompletionResponse

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Provides a common interface for different LLM providers.
    """
    
    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """
        Generate a completion based on the input messages.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific parameters
            
        Returns:
            CompletionResponse with generated text
        """
        pass


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing and development.
    
    Returns simulated responses based on input message patterns.
    """
    
    def __init__(self, response_prefix: str = "Mock response:"):
        """
        Initialize the mock client.
        
        Args:
            response_prefix: Prefix to add to all mock responses
        """
        self.response_prefix = response_prefix
        logger.info("Initialized MockLLMClient")
    
    def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """
        Generate a mock completion based on input messages.
        
        Analyzes the last message to provide context-aware mock responses.
        
        Args:
            messages: List of conversation messages
            **kwargs: Ignored for mock client
            
        Returns:
            CompletionResponse with simulated text
        """
        if not messages:
            logger.warning("Empty messages list provided to MockLLMClient")
            return CompletionResponse(text=f"{self.response_prefix} No messages provided.")
        
        last_message = messages[-1]
        content_lower = last_message.content.lower()
        
        if "reflect" in content_lower or "error" in content_lower:
            response_text = (
                f"{self.response_prefix} Based on the execution, I identify the following:\n"
                "- The task encountered a minor issue in step 2\n"
                "- Root cause appears to be missing validation\n"
                "- Suggested improvement: Add input validation before processing"
            )
        elif "curate" in content_lower or "delta" in content_lower:
            response_text = (
                f"{self.response_prefix} Proposing the following changes:\n"
                '{"ops": [{"op": "ADD", "new_bullet": {"section": "strategies", '
                '"content": "Always validate inputs", "tags": ["topic:validation"]}}]}'
            )
        elif "analyze" in content_lower:
            response_text = (
                f"{self.response_prefix} Analysis complete. Key findings:\n"
                "1. System is functioning as expected\n"
                "2. Performance is within acceptable range\n"
                "3. No critical issues detected"
            )
        else:
            response_text = (
                f"{self.response_prefix} I understand your request about "
                f"'{last_message.content[:50]}...'. Here's a relevant response "
                f"based on {len(messages)} message(s) in the conversation."
            )
        
        logger.debug(f"MockLLMClient generated response of length {len(response_text)}")
        
        return CompletionResponse(text=response_text)


class OpenRouterClient(LLMClient):
    """
    OpenRouter LLM client for accessing multiple model providers.
    
    Provides access to various LLM providers through OpenRouter's unified API.
    """
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-5",
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        default_max_tokens: Optional[int] = None,
        default_temperature: float = 0.7,
        reasoning_effort: Optional[str] = "medium"
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (e.g., 'openai/gpt-5', 'anthropic/claude-3-opus')
            site_url: Optional site URL for rankings
            app_name: Optional app name for rankings
            default_max_tokens: Default maximum tokens to generate
            default_temperature: Default temperature for generation
            reasoning_effort: Reasoning effort level for reasoning models ('low', 'medium', 'high')
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided via api_key parameter "
                "or OPENROUTER_API_KEY environment variable"
            )
        
        self.model = model
        self.site_url = site_url
        self.app_name = app_name
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.reasoning_effort = reasoning_effort
        
        logger.info(f"Initialized OpenRouterClient with model: {model}")
    
    def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """
        Generate a completion using OpenRouter API.
        
        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            CompletionResponse with generated text
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        
        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": kwargs.get("temperature", self.default_temperature),
        }
        
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        
        if self.default_max_tokens or "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs.get("max_tokens", self.default_max_tokens)
        
        for optional_param in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if optional_param in kwargs:
                payload[optional_param] = kwargs[optional_param]
        
        logger.debug(f"Making OpenRouter API request to {self.model}")
        
        try:
            response = requests.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 60)
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError("No choices returned in OpenRouter response")
            
            content = data["choices"][0]["message"]["content"]
            
            logger.info(
                f"OpenRouter request successful. "
                f"Tokens: {data.get('usage', {}).get('total_tokens', 'unknown')}"
            )
            
            return CompletionResponse(text=content)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse OpenRouter response: {e}")
            raise
