"""LLM client and resolver protocols.

Defines pluggable interfaces for LLM interaction and conflict resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for pluggable LLM clients.

    Any object with chat() and close() methods matching this signature works.
    The built-in OpenAIClient implements this protocol.

    Custom clients can override ``extract_content()`` and ``extract_usage()``
    to support non-OpenAI response formats.  The defaults assume OpenAI-style
    responses (``choices[0].message.content`` and ``.usage``).
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send messages, return response dict."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...

    def extract_content(self, response: dict) -> str:
        """Extract assistant message content from an LLM response.

        Override this for non-OpenAI response formats.
        Default assumes ``response["choices"][0]["message"]["content"]``.
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Cannot extract content from response: {exc}. "
                f"Override extract_content() for custom formats."
            ) from exc

    def extract_usage(self, response: dict) -> dict | None:
        """Extract usage dict from an LLM response.

        Override this for non-OpenAI response formats.
        Default assumes ``response.get("usage")``.
        """
        return response.get("usage")


@dataclass
class Resolution:
    """Result of conflict resolution.

    Returned by resolvers to indicate how a conflict/issue should be handled.
    """

    action: Literal["resolved", "abort", "skip"]
    content: BaseModel | None = None
    content_text: str | None = None  # Raw text alternative to content model
    reasoning: str | None = None
    generation_config: dict | None = None


@runtime_checkable
class ResolverCallable(Protocol):
    """Protocol for conflict resolution callables.

    Can be a function, lambda, or class with __call__.
    The issue parameter is typed as object because the concrete types
    (ConflictInfo, RebaseWarning, CherryPickIssue) are defined in Plan 03-03.
    """

    def __call__(self, issue: object) -> Resolution:
        """Resolve an issue and return a Resolution."""
        ...
