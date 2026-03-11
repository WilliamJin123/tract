"""LLM client infrastructure for Tract.

Provides OpenAI-compatible and Anthropic HTTP clients, pluggable
LLM/resolver protocols, and a built-in conflict resolver for
semantic merge operations.
"""

from tract.llm.client import OpenAIClient
from tract.llm.anthropic_client import (
    AnthropicClient,
    StreamEvent,
    TextDelta,
    ToolCallStart,
    ToolCallDelta,
    ThinkingDelta,
    UsageEvent,
    MessageDone,
)
from tract.llm.errors import (
    LLMAuthError,
    LLMClientError,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
    LLMToolUseError,
)
from tract.llm.protocols import LLMClient, Resolution, ResolverCallable
from tract.llm.resolver import OpenAIResolver

__all__ = [
    "OpenAIClient",
    "AnthropicClient",
    "OpenAIResolver",
    "LLMClient",
    "ResolverCallable",
    "Resolution",
    "LLMClientError",
    "LLMConfigError",
    "LLMRateLimitError",
    "LLMAuthError",
    "LLMResponseError",
    "LLMToolUseError",
    # Stream event types
    "StreamEvent",
    "TextDelta",
    "ToolCallStart",
    "ToolCallDelta",
    "ThinkingDelta",
    "UsageEvent",
    "MessageDone",
]
