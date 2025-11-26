"""Factory for creating LLM clients from configuration."""

from ace.core.config import LLMConfig, get_config
from ace.llm.client import LLMClient, MockLLMClient, OpenRouterClient


def create_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """Create an LLM client based on configuration.

    Args:
        config: LLMConfig to use. If None, loads from global config.

    Returns:
        LLMClient instance for the configured provider.

    Raises:
        ValueError: If provider is not supported.
    """
    if config is None:
        config = get_config().llm

    provider = config.provider.lower()

    if provider == "mock":
        return MockLLMClient()
    elif provider == "openrouter":
        return OpenRouterClient(
            model=config.model,
            default_temperature=config.temperature,
            default_max_tokens=config.max_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}. "
            f"Supported providers: mock, openrouter"
        )
