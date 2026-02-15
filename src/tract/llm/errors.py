"""LLM-specific error hierarchy.

All LLM errors inherit from TraceError for consistent exception handling.
"""

from __future__ import annotations

from tract.exceptions import TraceError


class LLMClientError(TraceError):
    """Base for all LLM client errors."""


class LLMConfigError(LLMClientError):
    """Missing or invalid LLM configuration (e.g., no API key)."""


class LLMRateLimitError(LLMClientError):
    """Rate limited by the API (429).

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header),
            or None if not provided.
    """

    def __init__(self, message: str = "Rate limited", retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        if retry_after is not None:
            message = f"{message} (retry after {retry_after}s)"
        super().__init__(message)


class LLMAuthError(LLMClientError):
    """Authentication failed (401/403)."""


class LLMResponseError(LLMClientError):
    """Unexpected response format from LLM API."""
