"""LLM client and resolver protocols.

Defines pluggable interfaces for LLM interaction and conflict resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for pluggable LLM clients.

    Any object with chat() and close() methods matching this signature works.
    The built-in OpenAIClient implements this protocol.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Send messages, return response dict with 'choices' and 'usage'."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...


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
