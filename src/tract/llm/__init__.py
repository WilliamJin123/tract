"""LLM client infrastructure for Tract.

Provides an OpenAI-compatible HTTP client, pluggable LLM/resolver protocols,
and a built-in conflict resolver for semantic merge operations.
"""

from tract.llm.client import OpenAIClient
from tract.llm.errors import (
    LLMAuthError,
    LLMClientError,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
)
from tract.llm.protocols import LLMClient, Resolution, ResolverCallable

__all__ = [
    "OpenAIClient",
    "LLMClient",
    "ResolverCallable",
    "Resolution",
    "LLMClientError",
    "LLMConfigError",
    "LLMRateLimitError",
    "LLMAuthError",
    "LLMResponseError",
]
