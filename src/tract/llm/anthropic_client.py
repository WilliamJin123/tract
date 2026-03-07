"""Built-in Anthropic Messages API client using the official SDK.

Wraps the ``anthropic`` Python SDK.  Normalizes responses to OpenAI format
so the rest of tract (loop, compression, spawn, resolver) works unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Iterator

import anthropic

from tract.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 8192


class AnthropicClient:
    """Anthropic Messages API client using the official SDK.

    Implements the LLMClient protocol.  Accepts OpenAI-format messages
    and tool definitions, translates to Anthropic format for the API call,
    and normalizes the response back to OpenAI format so that all existing
    tract code (loop, compression, resolver, etc.) works unchanged.

    Usage::

        with AnthropicClient(api_key="sk-ant-...") as client:
            response = client.chat([{"role": "user", "content": "Hello"}])
            text = response["choices"][0]["message"]["content"]
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
        max_retries: int = 3,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        resolved_key = api_key or os.environ.get("TRACT_ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise LLMConfigError(
                "No API key provided. Pass api_key= or set TRACT_ANTHROPIC_API_KEY "
                "environment variable."
            )
        self._api_key = resolved_key
        self._base_url = base_url
        self._default_model = default_model
        self._default_max_tokens = default_max_tokens

        sdk_kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "max_retries": max_retries,
            "timeout": timeout,
        }
        if base_url:
            sdk_kwargs["base_url"] = base_url.rstrip("/")
        self._client = anthropic.Anthropic(**sdk_kwargs)

    # ------------------------------------------------------------------
    # LLMClient protocol: chat()
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send chat completion request, return OpenAI-format response dict.

        Accepts OpenAI-format messages (role/content strings, tool_calls,
        tool_call_id).  Translates to Anthropic format, calls the API,
        and normalizes the response back to OpenAI format.
        """
        create_kwargs = self._build_create_kwargs(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        try:
            response = self._client.messages.create(**create_kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(f"Authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMResponseError(f"API error ({e.status_code}): {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMResponseError(f"Connection error: {e}") from e

        return self._normalize_response(response)

    # ------------------------------------------------------------------
    # LLMClient protocol: extract_*
    # ------------------------------------------------------------------

    @staticmethod
    def extract_content(response: dict) -> str:
        """Extract text content from normalized OpenAI-format response."""
        try:
            return response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Cannot extract content from response: {exc}"
            ) from exc

    @staticmethod
    def extract_usage(response: dict) -> dict | None:
        """Extract usage dict from normalized response."""
        return response.get("usage")

    @staticmethod
    def extract_tool_calls(response: dict) -> list[dict]:
        """Extract tool calls from normalized OpenAI-format response."""
        try:
            msg = response["choices"][0]["message"]
            tcs = msg.get("tool_calls")
            if not tcs:
                return []
            result = []
            for tc in tcs:
                args = tc.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                result.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": args,
                })
            return result
        except (KeyError, IndexError, TypeError):
            return []

    @staticmethod
    def extract_reasoning(response: dict) -> tuple[str, str] | None:
        """Extract reasoning from the original Anthropic response.

        The normalized response stores the raw Anthropic content blocks
        in ``_anthropic_content`` for reasoning extraction.
        """
        raw_blocks = response.get("_anthropic_content")
        if raw_blocks and isinstance(raw_blocks, list):
            thinking_parts = []
            for block in raw_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    text = block.get("thinking", "")
                    if text:
                        thinking_parts.append(text)
            if thinking_parts:
                return ("\n\n".join(thinking_parts), "anthropic")
        return None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Stream a chat completion, yielding typed events.

        Yields :class:`StreamEvent` instances (TextDelta, ToolCallStart,
        ToolCallDelta, ThinkingDelta, UsageEvent, MessageDone).

        After the stream completes, ``MessageDone.response`` contains the
        full normalized OpenAI-format response dict (same as ``chat()``
        would return).
        """
        create_kwargs = self._build_create_kwargs(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        try:
            with self._client.messages.stream(**create_kwargs) as stream:
                for event in stream:
                    yielded = self._map_stream_event(event)
                    if yielded is not None:
                        yield yielded

                # SDK accumulates the full message; normalize it
                final_message = stream.get_final_message()
                normalized = self._normalize_response(final_message)
                yield UsageEvent(usage=normalized.get("usage", {}))
                yield MessageDone(response=normalized)

        except anthropic.AuthenticationError as e:
            raise LLMAuthError(f"Authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMResponseError(f"API error ({e.status_code}): {e}") from e

    @staticmethod
    def _map_stream_event(event: Any) -> StreamEvent | None:
        """Map an SDK stream event to a tract StreamEvent."""
        ev_type = event.type

        if ev_type == "content_block_start":
            block = event.content_block
            if hasattr(block, "type") and block.type == "tool_use":
                return ToolCallStart(
                    index=event.index, id=block.id, name=block.name,
                )

        elif ev_type == "content_block_delta":
            delta = event.delta
            delta_type = delta.type
            if delta_type == "text_delta":
                return TextDelta(text=delta.text)
            if delta_type == "input_json_delta":
                return ToolCallDelta(
                    index=event.index, partial_json=delta.partial_json,
                )
            if delta_type == "thinking_delta":
                return ThinkingDelta(text=delta.thinking)
            # signature_delta handled internally by SDK

        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying client."""
        self._client.close()

    def __enter__(self) -> AnthropicClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal: build API kwargs
    # ------------------------------------------------------------------

    def _build_create_kwargs(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs dict for messages.create() / messages.stream()."""
        system, anthropic_messages = self._translate_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "max_tokens": max_tokens or self._default_max_tokens,
            "messages": anthropic_messages,
        }
        if system is not None:
            create_kwargs["system"] = system
        if temperature is not None:
            create_kwargs["temperature"] = temperature

        # Translate tool definitions
        tools = kwargs.pop("tools", None)
        if tools:
            create_kwargs["tools"] = self._translate_tools(tools)

        # Translate stop → stop_sequences
        stop = kwargs.pop("stop", None)
        if stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            create_kwargs["stop_sequences"] = stop

        # Extended thinking
        thinking = kwargs.pop("thinking", None)
        if thinking is not None:
            create_kwargs["thinking"] = thinking

        # Tool choice translation
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice is not None:
            create_kwargs["tool_choice"] = self._translate_tool_choice(tool_choice)

        # Pass through remaining kwargs (cache_control, metadata, etc.)
        create_kwargs.update(kwargs)
        return create_kwargs

    # ------------------------------------------------------------------
    # Internal: message translation (OpenAI → Anthropic)
    # ------------------------------------------------------------------

    def _translate_messages(
        self, messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        """Translate OpenAI-format messages to Anthropic format.

        Returns (system_text, anthropic_messages).
        """
        system_parts: list[str] = []
        raw: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")

            if role == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)
                continue

            if role == "assistant":
                content_blocks = self._build_assistant_content(msg)
                raw.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                if msg.get("is_error"):
                    block["is_error"] = True
                raw.append({"role": "user", "content": [block]})

            else:
                # user or any other role
                raw.append({"role": "user", "content": msg.get("content", "")})

        # Merge consecutive same-role messages (Anthropic requires alternation)
        merged = self._merge_consecutive(raw)

        system = "\n\n".join(system_parts) if system_parts else None
        return system, merged

    @staticmethod
    def _build_assistant_content(msg: dict) -> list[dict] | str:
        """Build Anthropic content blocks for an assistant message.

        Handles text, tool_use, and thinking blocks.  Thinking blocks
        (with signatures) are preserved for tool use continuations as
        required by the Anthropic API.
        """
        text = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")
        thinking_blocks = msg.get("_thinking_blocks", [])

        if not tool_calls and not thinking_blocks:
            return text

        blocks: list[dict] = []

        # Thinking blocks must come first and in original order
        for tb in thinking_blocks:
            blocks.append(tb)

        if text:
            blocks.append({"type": "text", "text": text})

        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed = {"_raw": raw_args}
                else:
                    parsed = raw_args
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": func.get("name", ""),
                    "input": parsed,
                })

        return blocks

    @staticmethod
    def _merge_consecutive(messages: list[dict]) -> list[dict]:
        """Merge consecutive same-role messages for Anthropic.

        Anthropic requires strict user/assistant alternation.
        Consecutive messages of the same role are merged into a single
        message with a list of content blocks.
        """
        if not messages:
            return []

        merged: list[dict] = []
        for msg in messages:
            if merged and merged[-1]["role"] == msg["role"]:
                # Merge into previous message
                prev = merged[-1]
                prev_content = prev["content"]
                new_content = msg["content"]

                # Normalize both to lists of blocks
                prev_blocks = _to_content_blocks(prev_content)
                new_blocks = _to_content_blocks(new_content)

                prev["content"] = prev_blocks + new_blocks
            else:
                merged.append(dict(msg))

        return merged

    # ------------------------------------------------------------------
    # Internal: tool definition translation (OpenAI → Anthropic)
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_tools(tools: list[dict]) -> list[dict]:
        """Translate OpenAI-format tool definitions to Anthropic format.

        OpenAI: ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}``
        Anthropic: ``{"name": ..., "description": ..., "input_schema": ...}``
        """
        result = []
        for tool in tools:
            func = tool.get("function", tool)
            result.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", func.get("input_schema", {})),
            })
        return result

    @staticmethod
    def _translate_tool_choice(tool_choice: Any) -> dict:
        """Translate OpenAI tool_choice format to Anthropic format.

        OpenAI string shortcuts: "auto", "none", "required"
        OpenAI named: {"type": "function", "function": {"name": "..."}}
        Anthropic: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
        """
        if isinstance(tool_choice, str):
            mapping = {"required": "any", "auto": "auto", "none": "none"}
            return {"type": mapping.get(tool_choice, tool_choice)}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                return {
                    "type": "tool",
                    "name": tool_choice.get("function", {}).get("name", ""),
                }
            return tool_choice
        return {"type": "auto"}

    # ------------------------------------------------------------------
    # Internal: response normalization (Anthropic → OpenAI)
    # ------------------------------------------------------------------

    def _normalize_response(self, msg: Any) -> dict:
        """Convert Anthropic Messages API response to OpenAI format.

        Accepts either an ``anthropic.types.Message`` object (from SDK)
        or a raw dict (for testing).  Normalizes to the same OpenAI-format
        dict that ``OpenAIClient.chat()`` returns.
        """
        if isinstance(msg, dict):
            return self._normalize_dict_response(msg)

        # --- SDK Message object path ---
        content_blocks = msg.content

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        thinking_blocks: list[dict] = []

        for block in content_blocks:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
            elif block.type == "thinking":
                thinking_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })

        text = "\n".join(text_parts) if text_parts else None

        message: dict[str, Any] = {
            "role": "assistant",
            "content": text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if thinking_blocks:
            message["_thinking_blocks"] = thinking_blocks

        # Map stop_reason to finish_reason
        stop_reason = msg.stop_reason or "end_turn"
        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "pause_turn": "stop",
            "refusal": "stop",
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")

        # Map usage — preserve cache tokens
        usage_obj = msg.usage
        usage: dict[str, Any] = {
            "prompt_tokens": usage_obj.input_tokens,
            "completion_tokens": usage_obj.output_tokens,
            "total_tokens": usage_obj.input_tokens + usage_obj.output_tokens,
        }
        if usage_obj.cache_creation_input_tokens:
            usage["cache_creation_input_tokens"] = usage_obj.cache_creation_input_tokens
        if usage_obj.cache_read_input_tokens:
            usage["cache_read_input_tokens"] = usage_obj.cache_read_input_tokens

        # Serialize raw content blocks for reasoning extraction
        raw_blocks = []
        for b in content_blocks:
            raw_blocks.append(b.model_dump() if hasattr(b, "model_dump") else b)

        result: dict[str, Any] = {
            "id": msg.id,
            "object": "chat.completion",
            "model": msg.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
            "_anthropic_content": raw_blocks,
        }
        return result

    def _normalize_dict_response(self, data: dict) -> dict:
        """Normalize a raw dict response (used by tests)."""
        content_blocks = data.get("content", [])

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        thinking_blocks: list[dict] = []

        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif block_type == "thinking":
                thinking_blocks.append(block)

        text = "\n".join(text_parts) if text_parts else None

        message: dict[str, Any] = {
            "role": "assistant",
            "content": text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if thinking_blocks:
            message["_thinking_blocks"] = thinking_blocks

        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")

        raw_usage = data.get("usage", {})
        usage: dict[str, Any] = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": (
                raw_usage.get("input_tokens", 0)
                + raw_usage.get("output_tokens", 0)
            ),
        }
        if raw_usage.get("cache_creation_input_tokens"):
            usage["cache_creation_input_tokens"] = raw_usage["cache_creation_input_tokens"]
        if raw_usage.get("cache_read_input_tokens"):
            usage["cache_read_input_tokens"] = raw_usage["cache_read_input_tokens"]

        result: dict[str, Any] = {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": data.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
            "_anthropic_content": content_blocks,
        }
        return result


# ------------------------------------------------------------------
# Stream event types
# ------------------------------------------------------------------


class StreamEvent:
    """Base class for streaming events."""


class TextDelta(StreamEvent):
    """A chunk of text content."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class ToolCallStart(StreamEvent):
    """A tool call has started."""

    __slots__ = ("index", "id", "name")

    def __init__(self, index: int, id: str, name: str) -> None:
        self.index = index
        self.id = id
        self.name = name


class ToolCallDelta(StreamEvent):
    """Partial JSON for a tool call's arguments."""

    __slots__ = ("index", "partial_json")

    def __init__(self, index: int, partial_json: str) -> None:
        self.index = index
        self.partial_json = partial_json


class ThinkingDelta(StreamEvent):
    """A chunk of thinking/reasoning content."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class UsageEvent(StreamEvent):
    """Token usage information."""

    __slots__ = ("usage",)

    def __init__(self, usage: dict) -> None:
        self.usage = usage


class MessageDone(StreamEvent):
    """Stream complete.  ``response`` is the full normalized OpenAI-format dict."""

    __slots__ = ("response",)

    def __init__(self, response: dict) -> None:
        self.response = response


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _to_content_blocks(content: str | list) -> list[dict]:
    """Normalize message content to a list of Anthropic content blocks."""
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}]
    return []
