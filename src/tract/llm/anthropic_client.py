"""Built-in Anthropic Messages API client with tenacity retry.

Provides a sync HTTP client for the Anthropic Messages API.
Normalizes responses to OpenAI format so the rest of tract
(loop, compression, spawn, resolver) works without changes.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Iterator

import httpx
import tenacity

from tract.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
_AUTH_ERROR_STATUS_CODES = {401, 403}

_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_API_VERSION = "2023-06-01"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, LLMAuthError):
        return False
    if isinstance(exc, LLMRateLimitError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


class AnthropicClient:
    """Sync httpx client for the Anthropic Messages API.

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
        default_model: str = "claude-sonnet-4-20250514",
        timeout: float = 120.0,
        max_retries: int = 3,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_version: str = _DEFAULT_API_VERSION,
    ) -> None:
        self._api_key = api_key or os.environ.get("TRACT_ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise LLMConfigError(
                "No API key provided. Pass api_key= or set TRACT_ANTHROPIC_API_KEY "
                "environment variable."
            )
        self._base_url = (
            base_url
            or os.environ.get(
                "TRACT_ANTHROPIC_BASE_URL", "https://api.anthropic.com"
            )
        ).rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_max_tokens = default_max_tokens
        self._api_version = api_version
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": self._api_version,
            },
        )

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
        retryer = tenacity.Retrying(
            retry=tenacity.retry_if_exception(_is_retryable),
            wait=(
                tenacity.wait_exponential(multiplier=1, min=1, max=30)
                + tenacity.wait_random(0, 2)
            ),
            stop=tenacity.stop_after_attempt(self._max_retries),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        return retryer(
            self._do_chat,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _do_chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Execute a single Messages API request (no retry)."""
        system, anthropic_messages = self._translate_messages(messages)

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "max_tokens": max_tokens or self._default_max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature

        # Translate tool definitions from OpenAI to Anthropic format
        tools = kwargs.pop("tools", None)
        if tools:
            payload["tools"] = self._translate_tools(tools)

        # Translate stop → stop_sequences
        stop = kwargs.pop("stop", None)
        if stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            payload["stop_sequences"] = stop

        # Extended thinking
        thinking = kwargs.pop("thinking", None)
        if thinking is not None:
            payload["thinking"] = thinking

        # Pass through remaining kwargs
        payload.update(kwargs)

        response = self._client.post(
            f"{self._base_url}/v1/messages",
            json=payload,
        )

        if response.status_code in _AUTH_ERROR_STATUS_CODES:
            raise LLMAuthError(
                f"Authentication failed: HTTP {response.status_code} - "
                f"{response.text}"
            )

        if response.status_code == 429:
            retry_after_raw = response.headers.get("Retry-After")
            retry_after: float | None = None
            if retry_after_raw is not None:
                try:
                    retry_after = float(retry_after_raw)
                except (ValueError, TypeError):
                    pass
            raise LLMRateLimitError(
                f"Rate limited: HTTP 429 - {response.text}",
                retry_after=retry_after,
            )

        if response.status_code == 529:
            raise LLMRateLimitError(
                f"Overloaded: HTTP 529 - {response.text}"
            )

        response.raise_for_status()

        data = response.json()
        if "content" not in data:
            raise LLMResponseError(
                f"Unexpected response format: missing 'content' key. "
                f"Response: {data}"
            )
        return self._normalize_response(data)

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
        system, anthropic_messages = self._translate_messages(messages)

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "max_tokens": max_tokens or self._default_max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature

        tools = kwargs.pop("tools", None)
        if tools:
            payload["tools"] = self._translate_tools(tools)

        stop = kwargs.pop("stop", None)
        if stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            payload["stop_sequences"] = stop

        thinking = kwargs.pop("thinking", None)
        if thinking is not None:
            payload["thinking"] = thinking

        payload.update(kwargs)

        yield from self._do_stream(payload)

    def _do_stream(self, payload: dict) -> Iterator[StreamEvent]:
        """Execute streaming request and yield events."""
        with self._client.stream(
            "POST",
            f"{self._base_url}/v1/messages",
            json=payload,
        ) as response:
            if response.status_code in _AUTH_ERROR_STATUS_CODES:
                response.read()
                raise LLMAuthError(
                    f"Authentication failed: HTTP {response.status_code}"
                )
            if response.status_code == 429:
                response.read()
                raise LLMRateLimitError("Rate limited: HTTP 429")
            if response.status_code >= 400:
                response.read()
                response.raise_for_status()

            # Accumulation state
            content_blocks: list[dict] = []
            current_block_idx: int = -1
            text_parts: list[str] = []
            tool_inputs: dict[int, list[str]] = {}  # idx -> json fragments
            thinking_parts: dict[int, list[str]] = {}
            model_name: str = ""
            msg_id: str = ""
            usage: dict = {}
            stop_reason: str | None = None

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "message_start":
                    msg = event.get("message", {})
                    model_name = msg.get("model", "")
                    msg_id = msg.get("id", "")
                    usage = msg.get("usage", {})

                elif event_type == "content_block_start":
                    idx = event.get("index", 0)
                    block = event.get("content_block", {})
                    current_block_idx = idx
                    # Pad content_blocks if needed
                    while len(content_blocks) <= idx:
                        content_blocks.append({})
                    content_blocks[idx] = dict(block)

                    if block.get("type") == "tool_use":
                        tool_inputs[idx] = []
                        yield ToolCallStart(
                            index=idx,
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                        )

                elif event_type == "content_block_delta":
                    idx = event.get("index", current_block_idx)
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        text_parts.append(text)
                        yield TextDelta(text=text)

                    elif delta_type == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        if idx in tool_inputs:
                            tool_inputs[idx].append(partial)
                        yield ToolCallDelta(index=idx, partial_json=partial)

                    elif delta_type == "thinking_delta":
                        text = delta.get("thinking", "")
                        if idx not in thinking_parts:
                            thinking_parts[idx] = []
                        thinking_parts[idx].append(text)
                        yield ThinkingDelta(text=text)

                elif event_type == "content_block_stop":
                    idx = event.get("index", current_block_idx)
                    # Finalize tool input JSON
                    if idx in tool_inputs:
                        raw = "".join(tool_inputs[idx])
                        try:
                            parsed = json.loads(raw) if raw else {}
                        except json.JSONDecodeError:
                            parsed = {"_raw": raw}
                        if idx < len(content_blocks):
                            content_blocks[idx]["input"] = parsed
                    # Finalize thinking text
                    if idx in thinking_parts:
                        text = "".join(thinking_parts[idx])
                        if idx < len(content_blocks):
                            content_blocks[idx]["thinking"] = text

                elif event_type == "message_delta":
                    delta = event.get("delta", {})
                    stop_reason = delta.get("stop_reason", stop_reason)
                    delta_usage = event.get("usage", {})
                    if delta_usage:
                        usage.update(delta_usage)

                elif event_type == "message_stop":
                    pass  # handled after loop

            # Build the full Anthropic response for normalization
            full_text = "".join(text_parts)
            # Update text blocks with accumulated text
            for block in content_blocks:
                if block.get("type") == "text":
                    block["text"] = full_text

            anthropic_response = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "usage": usage,
            }

            normalized = self._normalize_response(anthropic_response)
            yield UsageEvent(usage=normalized.get("usage", {}))
            yield MessageDone(response=normalized)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self) -> AnthropicClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

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
                # OpenAI tool result → Anthropic tool_result content block
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
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
        """Build Anthropic content blocks for an assistant message."""
        text = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            return text

        blocks: list[dict] = []
        if text:
            blocks.append({"type": "text", "text": text})

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

    # ------------------------------------------------------------------
    # Internal: response normalization (Anthropic → OpenAI)
    # ------------------------------------------------------------------

    def _normalize_response(self, data: dict) -> dict:
        """Convert Anthropic Messages API response to OpenAI format.

        This allows all existing tract code that reads
        ``response["choices"][0]["message"]["content"]`` to work unchanged.
        """
        content_blocks = data.get("content", [])

        # Extract text content
        text_parts: list[str] = []
        tool_calls: list[dict] = []

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

        text = "\n".join(text_parts) if text_parts else None

        message: dict[str, Any] = {
            "role": "assistant",
            "content": text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Map stop_reason to finish_reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = finish_reason_map.get(stop_reason, "stop")

        # Map usage
        raw_usage = data.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": (
                raw_usage.get("input_tokens", 0)
                + raw_usage.get("output_tokens", 0)
            ),
        }

        result: dict[str, Any] = {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": data.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
            # Preserve raw Anthropic content for reasoning extraction
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
