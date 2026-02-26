"""Protocol definitions for Trace.

Defines pluggable interfaces (TokenCounter, ContextCompiler, TokenUsageExtractor)
and frozen dataclasses for structured output (Message, CompiledContext, TokenUsage).

No SQLAlchemy imports allowed in this module -- pure domain protocols.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

from tract.models.config import LLMConfig

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo


class ToolCallDict(TypedDict):
    """Canonical storage format for a single tool call."""

    id: str
    name: str
    arguments: dict
    type: str


class _ToolCallOpenAIFunction(TypedDict):
    """OpenAI function sub-object."""

    name: str
    arguments: str


class ToolCallOpenAIDict(TypedDict):
    """OpenAI wire format for a single tool call."""

    id: str
    type: str
    function: _ToolCallOpenAIFunction


@dataclass(frozen=True)
class ToolCall:
    """A tool/function invocation requested by the LLM.

    Provider-agnostic canonical representation. Arguments are always
    a parsed dict — OpenAI's JSON string is parsed at ingestion time.
    """

    id: str
    name: str
    arguments: dict
    type: str = "function"

    @classmethod
    def from_openai(cls, tc: dict) -> ToolCall:
        """Parse from OpenAI/compatible format.

        OpenAI sends arguments as a JSON string; this parses it to a dict.
        """
        raw_args = tc["function"]["arguments"]
        try:
            arguments = _json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (_json.JSONDecodeError, TypeError):
            arguments = {"_raw": raw_args}
        return cls(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=arguments,
            type=tc.get("type", "function"),
        )

    @classmethod
    def from_anthropic(cls, block: dict) -> ToolCall:
        """Parse from Anthropic tool_use content block."""
        return cls(
            id=block["id"],
            name=block["name"],
            arguments=block.get("input", {}),
        )

    @classmethod
    def from_dict(cls, d: ToolCallDict | dict[str, object]) -> ToolCall:
        """Reconstruct from a stored dict (e.g. metadata_json)."""
        return cls(
            id=d["id"],
            name=d["name"],
            arguments=d.get("arguments", {}),
            type=d.get("type", "function"),
        )

    def to_openai(self) -> ToolCallOpenAIDict:
        """Serialize to OpenAI wire format."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": _json.dumps(self.arguments),
            },
        }

    def to_dict(self) -> ToolCallDict:
        """Serialize for storage in metadata_json."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "type": self.type,
        }


@dataclass(frozen=True)
class Message:
    """A single message in a compiled context."""

    role: str
    content: str
    name: str | None = None
    token_count: int = 0
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    content_type: str | None = None


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
    generation_configs: list[LLMConfig | None] = field(default_factory=list)
    commit_hashes: list[str] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)

    def to_dicts(self) -> list[dict]:
        """Convert messages to a list of dicts with role/content keys.

        Returns a list suitable for most LLM APIs. Each dict has
        ``"role"`` and ``"content"`` keys, plus ``"name"`` when present.
        For tool-calling assistant messages, ``"tool_calls"`` is included.
        For tool result messages, ``"tool_call_id"`` is included.

        Returns:
            List of message dicts.
        """
        result: list[dict] = []
        for m in self.messages:
            d: dict = {"role": m.role, "content": m.content}
            if m.name is not None:
                d["name"] = m.name
            if m.tool_calls:
                d["tool_calls"] = [tc.to_openai() for tc in m.tool_calls]
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            result.append(d)
        return result

    def to_openai(self) -> list[dict]:
        """Convert messages to OpenAI chat completion format.

        OpenAI uses inline system messages, so this is identical
        to :meth:`to_dicts`.

        Returns:
            List of message dicts in OpenAI format.
        """
        return self.to_dicts()

    def to_anthropic(self) -> dict[str, object]:
        """Convert messages to Anthropic API format.

        Anthropic does not support ``role: "system"`` in the messages
        array.  System messages are extracted to a separate ``"system"``
        key and concatenated with ``"\\n\\n"``.

        Returns:
            Dict with ``"system"`` (str or None) and ``"messages"``
            (list of non-system message dicts).
        """
        system_parts: list[str] = []
        messages: list[dict] = []
        for m in self.messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                d: dict = {"role": m.role, "content": m.content}
                if m.name is not None:
                    d["name"] = m.name
                if m.tool_calls:
                    d["tool_calls"] = [tc.to_openai() for tc in m.tool_calls]
                if m.tool_call_id is not None:
                    d["tool_call_id"] = m.tool_call_id
                messages.append(d)
        return {
            "system": "\n\n".join(system_parts) if system_parts else None,
            "messages": messages,
        }

    def to_openai_params(self) -> dict[str, object]:
        """Full OpenAI API params dict with messages and tools.

        Returns a dict with ``"messages"`` and optionally ``"tools"``
        keys, suitable for passing to the OpenAI chat completions API.

        Returns:
            Dict with messages and tools (if any).
        """
        params: dict[str, object] = {"messages": self.to_dicts()}
        if self.tools:
            params["tools"] = list(self.tools)
        return params

    def to_anthropic_params(self) -> dict[str, object]:
        """Full Anthropic API params dict with system, messages, and tools.

        Returns a dict with ``"system"``, ``"messages"``, and optionally
        ``"tools"`` keys, suitable for passing to the Anthropic messages API.

        Returns:
            Dict with system, messages, and tools (if any).
        """
        result = self.to_anthropic()
        if self.tools:
            result["tools"] = list(self.tools)
        return result

    def __str__(self) -> str:
        return (
            f"CompiledContext(messages={self.commit_count},"
            f" tokens={self.token_count}, source={self.token_source})"
        )

    def pprint(
        self,
        *,
        max_chars: int | None = None,
        style: Literal["table", "chat", "compact"] = "table",
    ) -> None:
        """Pretty-print this compiled context using rich formatting.

        Args:
            max_chars: Max display characters before truncation. None (default)
                means no limit for ``"table"``/``"chat"``, and 500 for
                ``"compact"``. Pass an explicit int to override.
            style: ``"table"`` for a data table, ``"chat"`` for a chat
                transcript with panels per message, ``"compact"`` for a
                one-line-per-message summary.
        """
        from tract.formatting import pprint_compiled_context

        pprint_compiled_context(self, max_chars=max_chars, style=style)


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
    tool_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by an LLM API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ChatResponse:
    """Response from Tract.chat() or Tract.generate().

    Attributes:
        text: The assistant's response text.
        usage: Token usage from the API, or None if not reported.
        commit_info: CommitInfo for the assistant's commit.
        generation_config: The generation config captured from the request/response.
        prompt: The user message that triggered this response, or None when
            created via generate() (where the user committed separately).
    """

    text: str
    usage: TokenUsage | None
    commit_info: CommitInfo
    generation_config: LLMConfig
    prompt: str | None = None
    reasoning: str | None = None
    reasoning_commit: CommitInfo | None = None
    tool_calls: list[ToolCall] | None = None
    raw_response: dict | None = None

    def __str__(self) -> str:
        return self.text

    def pprint(self, *, max_chars: int | None = None) -> None:
        """Pretty-print this response using rich formatting.

        Args:
            max_chars: Max display characters before truncation.
                None (default) means no limit.
        """
        from tract.formatting import pprint_chat_response

        pprint_chat_response(self, max_chars=max_chars)


@dataclass(frozen=True)
class ToolTurn:
    """A paired tool-call assistant message and its tool result(s)."""

    call: CommitInfo
    results: list[CommitInfo]
    tool_names: list[str]

    @property
    def all_hashes(self) -> list[str]:
        """All commit hashes in this turn (call + results)."""
        return [self.call.commit_hash] + [r.commit_hash for r in self.results]

    @property
    def result_hashes(self) -> list[str]:
        """Just the result commit hashes."""
        return [r.commit_hash for r in self.results]

    @property
    def total_tokens(self) -> int:
        """Total tokens across call and all results."""
        return self.call.token_count + sum(r.token_count for r in self.results)


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
        include_reasoning: bool = False,
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
