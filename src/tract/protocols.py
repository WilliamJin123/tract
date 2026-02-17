"""Protocol definitions for Trace.

Defines pluggable interfaces (TokenCounter, ContextCompiler, TokenUsageExtractor)
and frozen dataclasses for structured output (Message, CompiledContext, TokenUsage).

No SQLAlchemy imports allowed in this module -- pure domain protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    """A single message in a compiled context."""

    role: str
    content: str
    name: str | None = None


@dataclass(frozen=True)
class CompiledContext:
    """Output of context compilation.

    Contains structured messages ready for LLM APIs,
    along with token count metadata.
    """

    messages: list[Message] = field(default_factory=list)
    token_count: int = 0
    commit_count: int = 0
    token_source: str = ""
    generation_configs: list[dict] = field(default_factory=list)
    commit_hashes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompileSnapshot:
    """Cached intermediate compilation state for incremental extension.

    Each position in ``messages`` corresponds to one effective commit.
    ``commit_hashes[i]`` is the commit that produced ``messages[i]``.
    ``message_token_counts[i]`` is the token count for ``messages[i]``
    (including per-message overhead, excluding the response primer).

    ``token_count`` equals ``sum(message_token_counts) + RESPONSE_PRIMER_TOKENS``
    when tiktoken-sourced, or the API-reported prompt_tokens when API-sourced.
    """

    head_hash: str
    messages: tuple[Message, ...]
    commit_count: int
    token_count: int
    token_source: str
    generation_configs: tuple[dict, ...] = ()
    commit_hashes: tuple[str, ...] = ()
    message_token_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by an LLM API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for pluggable token counting."""

    def count_text(self, text: str) -> int:
        """Count tokens in a plain text string."""
        ...

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens in a structured message list (including overhead)."""
        ...


@runtime_checkable
class ContextCompiler(Protocol):
    """Protocol for context compilation.

    Converts a commit chain into LLM-ready messages.
    """

    def compile(
        self,
        tract_id: str,
        head_hash: str,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
        include_edit_annotations: bool = False,
    ) -> CompiledContext:
        """Compile commits into structured messages for LLM consumption."""
        ...


@runtime_checkable
class TokenUsageExtractor(Protocol):
    """Protocol for extracting token usage from LLM API responses.

    Defined in Phase 1 for interface stability; provider-specific
    implementations added in Phase 3.
    """

    def extract(self, api_response: dict) -> TokenUsage | None:
        """Extract token usage from an API response dict.

        Returns None if the response format is not recognized.
        """
        ...
