from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a message in a conversation with an LLM."""

    role: str = Field(
        ..., description="Role of the message sender (e.g., 'user', 'assistant', 'system')"
    )
    content: str = Field(..., description="Content of the message")


class TokenUsage(BaseModel):
    """Token usage metadata returned by an LLM provider."""

    prompt_tokens: int | None = Field(default=None, description="Prompt/input tokens used")
    completion_tokens: int | None = Field(
        default=None,
        description="Completion/output tokens used",
    )
    total_tokens: int | None = Field(default=None, description="Total tokens used")


class CompletionResponse(BaseModel):
    """Represents the response from an LLM completion request."""

    text: str = Field(..., description="The generated text response from the LLM")
    usage: TokenUsage | None = Field(
        default=None,
        description="Optional token usage metadata from the provider",
    )
