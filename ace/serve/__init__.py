"""Online serving module for test-time sequential adaptation.

Provides HTTP/MCP server that performs real-time adaptation using
execution feedback only (no ground-truth labels).
"""

from .runner import OnlineServer
from .schema import AdaptationMode, FeedbackRequest, FeedbackResponse, WarmupSource

__all__ = [
    "OnlineServer",
    "AdaptationMode",
    "FeedbackRequest",
    "FeedbackResponse",
    "WarmupSource",
]
