from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a message in a conversation with an LLM."""

    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")


class CompletionResponse(BaseModel):
    """Represents the response from an LLM completion request."""

    text: str = Field(..., description="The generated text response from the LLM")
