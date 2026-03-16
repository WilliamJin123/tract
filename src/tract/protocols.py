"""Protocol definitions for Trace.

Defines pluggable interfaces (TokenCounter, ContextCompiler)
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
    priorities: list[str] = field(default_factory=list)
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

    def to_anthropic(self, *, cache_control: bool = False) -> dict[str, object]:
        """Convert messages to native Anthropic API format.

        Anthropic does not support ``role: "system"`` in the messages
        array.  System messages are extracted to a separate ``"system"``
        key.  Assistant messages with tool calls produce ``tool_use``
        content blocks.  Tool result messages produce ``tool_result``
        content blocks inside user messages.  Consecutive same-role
        messages are merged (Anthropic requires strict alternation).

        Args:
            cache_control: If True, add ``cache_control`` breakpoints for
                Anthropic's prompt caching (90% cost reduction on cached
                prefixes).  System messages get ephemeral cache control.
                The last PINNED/IMPORTANT message (or the midpoint message
                if no priorities exist) gets a cache_control marker,
                placing the stable/volatile boundary for maximum reuse.
                At most 2 breakpoints are added (system + one message),
                well under Anthropic's 4-breakpoint limit.

        Returns:
            Dict with ``"system"`` (str or None) and ``"messages"``
            (list of Anthropic-format message dicts).
        """
        system_parts: list[str] = []
        raw: list[dict] = []
        for m in self.messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "assistant":
                if m.tool_calls:
                    # Mixed content: text + tool_use blocks
                    blocks: list[dict] = []
                    if m.content:
                        blocks.append({"type": "text", "text": m.content})
                    for tc in m.tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                    raw.append({"role": "assistant", "content": blocks})
                else:
                    # Plain text assistant message
                    raw.append({"role": "assistant", "content": m.content or ""})
            elif m.tool_call_id is not None:
                # Tool result → user message with tool_result block
                raw.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content or "",
                    }],
                })
            else:
                raw.append({"role": m.role, "content": m.content})

        # Merge consecutive same-role messages for Anthropic alternation
        merged: list[dict] = []
        for msg in raw:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_content = merged[-1]["content"]
                new_content = msg["content"]
                prev_blocks = (
                    prev_content if isinstance(prev_content, list)
                    else ([{"type": "text", "text": prev_content}] if prev_content else [])
                )
                new_blocks = (
                    new_content if isinstance(new_content, list)
                    else ([{"type": "text", "text": new_content}] if new_content else [])
                )
                merged_blocks = [
                    b for b in (prev_blocks + new_blocks)
                    if not (
                        isinstance(b, dict)
                        and b.get("type") == "text"
                        and not b.get("text", "").strip()
                    )
                ]
                if not merged_blocks:
                    merged_blocks = [{"type": "text", "text": "(empty)"}]
                merged[-1]["content"] = merged_blocks
            else:
                merged.append(dict(msg))

        system_text = "\n\n".join(system_parts) if system_parts else None

        if cache_control:
            _apply_anthropic_cache_control(system_text, merged, self.priorities)
            # When cache_control is enabled the system value may have been
            # converted from a plain string to a block list.
            if system_text is not None:
                system_text = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

        return {
            "system": system_text,
            "messages": merged,
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

    def to_anthropic_params(self, *, cache_control: bool = False) -> dict[str, object]:
        """Full Anthropic API params dict with system, messages, and tools.

        Returns a dict with ``"system"``, ``"messages"``, and optionally
        ``"tools"`` keys, suitable for passing to the Anthropic messages API.
        Tool definitions are converted from OpenAI format to Anthropic format
        (``input_schema`` instead of ``parameters``).

        Args:
            cache_control: If True, add ``cache_control`` breakpoints for
                Anthropic's prompt caching.  Passed through to
                :meth:`to_anthropic`.

        Returns:
            Dict with system, messages, and tools (if any).
        """
        result = self.to_anthropic(cache_control=cache_control)
        if self.tools:
            anthropic_tools = []
            for tool in self.tools:
                func = tool.get("function", tool)
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get(
                        "parameters", func.get("input_schema", {})
                    ),
                })
            result["tools"] = anthropic_tools
        return result

    def to_text(
        self,
        *,
        include_roles: bool = True,
        separator: str = "\n\n",
    ) -> str:
        """Flatten compiled messages to plain text.

        Useful for embedding compiled context in agent prompts or
        other plain-text contexts where structured messages aren't needed.

        Args:
            include_roles: If True, prefix each message with ``[ROLE]:``.
                When a message has a ``name``, it appears as ``[ROLE (name)]:``.
            separator: String placed between messages.

        Returns:
            Plain text representation of all messages.
        """
        lines: list[str] = []
        for msg in self.messages:
            if include_roles:
                role = msg.role.upper()
                name = f" ({msg.name})" if msg.name else ""
                lines.append(f"[{role}{name}]:\n{msg.content}")
            else:
                lines.append(msg.content)
        return separator.join(lines)

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


# ---------------------------------------------------------------------------
# Anthropic prompt-caching helper
# ---------------------------------------------------------------------------

_CACHE_MARKER = {"type": "ephemeral"}

_STABLE_PRIORITIES = frozenset({"pinned", "important"})


def _apply_anthropic_cache_control(
    system_text: str | None,
    merged: list[dict],
    priorities: list[str],
) -> None:
    """Mutate *merged* in-place to add ``cache_control`` at the stable/volatile boundary.

    The system prompt is handled by the caller (converted to a block list).
    This function adds **one** ``cache_control`` marker to the message that
    sits at the boundary between stable (cacheable) and volatile content.

    Strategy:
    * If *priorities* contains PINNED or IMPORTANT entries, the boundary is
      the **last** message whose priority is PINNED or IMPORTANT.
    * Otherwise the boundary is the midpoint (``len(merged) // 2``), capped
      at index 0 minimum.
    * If there are no messages, nothing is modified.

    The marker is placed on the **last content block** of the chosen message
    (Anthropic requires ``cache_control`` on a content block, not the
    message envelope).  If the message content is a plain string it is
    converted to a single-element block list first.
    """
    if not merged:
        return

    # Determine boundary index.
    # priorities[i] corresponds to the i-th *commit*, not the i-th merged
    # message.  After system extraction + merging, the mapping isn't 1:1.
    # We use a conservative heuristic: scan priorities for the last
    # PINNED/IMPORTANT entry and use that index clamped to the merged range.
    boundary_idx: int | None = None
    if priorities:
        for i in range(len(priorities) - 1, -1, -1):
            if priorities[i] in _STABLE_PRIORITIES:
                boundary_idx = min(i, len(merged) - 1)
                break

    if boundary_idx is None:
        # Fallback: midpoint of merged messages (at least index 0).
        boundary_idx = max(len(merged) // 2 - 1, 0)

    _add_cache_control_to_message(merged, boundary_idx)


def _add_cache_control_to_message(messages: list[dict], idx: int) -> None:
    """Add ``cache_control`` to the last block of ``messages[idx]``."""
    msg = messages[idx]
    content = msg["content"]

    if isinstance(content, str):
        # Convert plain string to block list with cache_control.
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": _CACHE_MARKER}
        ]
    elif isinstance(content, list) and content:
        # Add cache_control to the last block.
        last_block = dict(content[-1])
        last_block["cache_control"] = _CACHE_MARKER
        content[-1] = last_block
    # else: empty list — nothing to annotate.


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
    priorities: tuple[str, ...] = ()
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
