"""Built-in OpenAI-compatible httpx client with tenacity retry.

Provides a sync HTTP client for OpenAI-compatible chat completion APIs.
Reads configuration from constructor arguments or environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import tenacity

from tract.llm.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
    LLMToolUseError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_AUTH_ERROR_STATUS_CODES = {401, 403}


def _is_retryable(exc: BaseException) -> bool:
    """Check if an exception is retryable.

    Retryable: 429, 500, 502, 503, 504, connection errors, tool_use_failed.
    Not retryable: 401, 403, other client errors.
    """
    if isinstance(exc, LLMAuthError):
        return False
    if isinstance(exc, (LLMRateLimitError, LLMToolUseError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


class OpenAIClient:
    """Sync httpx client for OpenAI-compatible chat completions.

    Implements the LLMClient protocol. Supports retry with exponential
    backoff for transient errors (429, 5xx). Fails immediately on
    authentication errors (401, 403).

    Usage::

        with OpenAIClient(api_key="sk-...") as client:
            response = client.chat([{"role": "user", "content": "Hello"}])
            text = OpenAIClient.extract_content(response)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            api_key: API key. Falls back to TRACT_OPENAI_API_KEY env var.
            base_url: API base URL. Falls back to TRACT_OPENAI_BASE_URL env var,
                then to https://api.openai.com/v1.
            default_model: Default model for chat requests.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for retryable errors.

        Raises:
            LLMConfigError: If no API key is provided or found in environment.
        """
        self._api_key = api_key or os.environ.get("TRACT_OPENAI_API_KEY", "")
        if not self._api_key:
            raise LLMConfigError(
                "No API key provided. Pass api_key= or set TRACT_OPENAI_API_KEY "
                "environment variable."
            )
        self._base_url = (
            base_url
            or os.environ.get("TRACT_OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send chat completion request with retry.

        Uses tenacity.Retrying programmatically (not as decorator) so that
        max_retries is configurable per-instance.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model to use. Falls back to default_model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional payload parameters forwarded to the API.

        Returns:
            Full response dict with 'choices', 'usage', 'model', etc.

        Raises:
            LLMAuthError: On 401/403 (no retry).
            LLMRateLimitError: On 429 after all retries exhausted.
            LLMResponseError: On unexpected response format.
            httpx.HTTPStatusError: On other non-retryable HTTP errors.
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
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Execute a single chat completion request (no retry)."""
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        response = self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
        )

        # Check for auth errors before raise_for_status
        if response.status_code in _AUTH_ERROR_STATUS_CODES:
            raise LLMAuthError(
                f"Authentication failed: HTTP {response.status_code} - "
                f"{response.text}"
            )

        # Check for rate limiting
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

        # Detect tool_use_failed (e.g. Groq returns 400 when the model's
        # tool-call JSON was truncated by max_tokens).  This is retryable.
        if response.status_code == 400:
            try:
                err_body = response.json()
            except Exception:
                err_body = {}
            err_info = err_body.get("error") if isinstance(err_body, dict) else None
            if isinstance(err_info, dict) and err_info.get("code") == "tool_use_failed":
                raise LLMToolUseError(
                    f"Tool call truncated (max_tokens too low?): "
                    f"{err_info.get('message', response.text)}"
                )

        response.raise_for_status()

        data = response.json()
        if "choices" not in data:
            raise LLMResponseError(
                f"Unexpected response format: missing 'choices' key. "
                f"Response: {data}"
            )
        return data

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self) -> OpenAIClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def extract_content(response: dict) -> str:
        """Extract the assistant's message content from a response dict.

        Args:
            response: Full API response dict.

        Returns:
            The content string from the first choice.

        Raises:
            LLMResponseError: If the response format is unexpected.
        """
        try:
            return response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Cannot extract content from response: {exc}. "
                f"Response: {response}"
            ) from exc

    @staticmethod
    def extract_usage(response: dict) -> dict | None:
        """Extract usage information from a response dict.

        Args:
            response: Full API response dict.

        Returns:
            Usage dict with prompt_tokens, completion_tokens, total_tokens,
            or None if not present.
        """
        return response.get("usage")

    @staticmethod
    def extract_tool_calls(response: dict) -> list[dict]:
        """Extract tool calls from an OpenAI response dict.

        Returns list of ``{"id": ..., "name": ..., "arguments": dict}``
        dicts, with arguments parsed from JSON strings.
        """
        try:
            msg = response["choices"][0]["message"]
            tcs = msg.get("tool_calls")
            if not tcs:
                return []
            result = []
            for tc in tcs:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                else:
                    args = raw_args
                result.append({
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "arguments": args,
                })
            return result
        except (KeyError, IndexError, TypeError):
            return []

    def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator:
        """Stream a chat completion, yielding typed events.

        Yields events from :mod:`tract.llm.anthropic_client` event types
        (TextDelta, ToolCallStart, ToolCallDelta, UsageEvent, MessageDone)
        for a provider-agnostic streaming interface.
        """
        from tract.llm.anthropic_client import (
            MessageDone,
            TextDelta,
            ToolCallDelta,
            ToolCallStart,
            UsageEvent,
        )

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
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
            text_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}  # index -> {id, name, args_parts}
            full_response: dict = {}
            finish_reason: str | None = None
            model_name: str = ""

            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                model_name = chunk.get("model", model_name)

                choices = chunk.get("choices", [])
                if not choices:
                    # Usage-only chunk (some providers send this)
                    usage = chunk.get("usage")
                    if usage:
                        yield UsageEvent(usage=usage)
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason", finish_reason)

                # Text content
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield TextDelta(text=content)

                # Tool calls (incremental)
                tcs = delta.get("tool_calls", [])
                for tc_delta in tcs:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        tc_id = tc_delta.get("id", "")
                        tc_name = tc_delta.get("function", {}).get("name", "")
                        tool_calls_acc[idx] = {
                            "id": tc_id,
                            "name": tc_name,
                            "args_parts": [],
                        }
                        if tc_id or tc_name:
                            yield ToolCallStart(
                                index=idx, id=tc_id, name=tc_name,
                            )
                    args_fragment = tc_delta.get("function", {}).get("arguments", "")
                    if args_fragment:
                        tool_calls_acc[idx]["args_parts"].append(args_fragment)
                        yield ToolCallDelta(
                            index=idx, partial_json=args_fragment,
                        )

            # Build full response
            text = "".join(text_parts) or None
            message: dict[str, Any] = {"role": "assistant", "content": text}

            if tool_calls_acc:
                oai_tcs = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    raw_args = "".join(tc["args_parts"])
                    oai_tcs.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": raw_args,
                        },
                    })
                message["tool_calls"] = oai_tcs

            full_response = {
                "id": "",
                "object": "chat.completion",
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason or "stop",
                    }
                ],
            }

            yield MessageDone(response=full_response)

    @staticmethod
    def extract_reasoning(response: dict) -> tuple[str, str] | None:
        """Extract reasoning/thinking content from a response dict.

        Checks multiple provider formats in priority order:

        1. Parsed field: ``response["choices"][0]["message"]["reasoning"]``
           (Cerebras parsed mode)
        2. OpenAI reasoning_content:
           ``response["choices"][0]["message"]["reasoning_content"]`` (o1/o3)
        3. Anthropic thinking blocks: ``response["content"]`` list with
           ``type="thinking"`` blocks
        4. ``<think>`` tags: Regex extraction from content text

        Returns:
            Tuple of (reasoning_text, format_name) if reasoning found,
            or None if no reasoning detected.
        """
        import re

        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            message = None

        if message is not None:
            # 1. Cerebras parsed field
            reasoning = message.get("reasoning")
            if reasoning:
                return (reasoning, "parsed")

            # 2. OpenAI o1/o3 reasoning_content
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                return (reasoning_content, "reasoning_content")

        # 3. Anthropic thinking blocks
        content_blocks = response.get("content")
        if isinstance(content_blocks, list):
            thinking_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    thinking_text = block.get("thinking", "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)
            if thinking_parts:
                return ("\n\n".join(thinking_parts), "anthropic")

        # 4. <think> tags in content
        if message is not None:
            content = message.get("content", "") or ""
            think_match = re.search(
                r"<think>(.*?)</think>", content, re.DOTALL
            )
            if think_match:
                return (think_match.group(1).strip(), "think_tags")

        return None

    # ------------------------------------------------------------------
    # Async support
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Async chat completion with retry.

        Uses an internal ``httpx.AsyncClient`` (created lazily on first call).
        Same retry logic as :meth:`chat`.
        """
        retryer = tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception(_is_retryable),
            wait=(
                tenacity.wait_exponential(multiplier=1, min=1, max=30)
                + tenacity.wait_random(0, 2)
            ),
            stop=tenacity.stop_after_attempt(self._max_retries),
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        return await retryer(
            self._ado_chat,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def _ado_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Execute a single async chat completion request (no retry)."""
        client = self._get_async_client()
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        response = await client.post(
            f"{self._base_url}/chat/completions",
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

        # Detect tool_use_failed (e.g. Groq 400 when tool-call JSON truncated)
        if response.status_code == 400:
            try:
                err_body = response.json()
            except Exception:
                err_body = {}
            err_info = err_body.get("error") if isinstance(err_body, dict) else None
            if isinstance(err_info, dict) and err_info.get("code") == "tool_use_failed":
                raise LLMToolUseError(
                    f"Tool call truncated (max_tokens too low?): "
                    f"{err_info.get('message', response.text)}"
                )

        response.raise_for_status()

        data = response.json()
        if "choices" not in data:
            raise LLMResponseError(
                f"Unexpected response format: missing 'choices' key. "
                f"Response: {data}"
            )
        return data

    def _get_async_client(self) -> httpx.AsyncClient:
        """Lazily create the async httpx client."""
        if not hasattr(self, "_async_client") or self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
        return self._async_client

    async def astream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Async stream a chat completion, yielding typed events.

        Async version of :meth:`stream`. Yields the same event types.
        """
        from tract.llm.anthropic_client import (
            MessageDone,
            TextDelta,
            ToolCallDelta,
            ToolCallStart,
            UsageEvent,
        )

        client = self._get_async_client()
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
        ) as response:
            if response.status_code in _AUTH_ERROR_STATUS_CODES:
                await response.aread()
                raise LLMAuthError(
                    f"Authentication failed: HTTP {response.status_code}"
                )
            if response.status_code == 429:
                await response.aread()
                raise LLMRateLimitError("Rate limited: HTTP 429")
            if response.status_code >= 400:
                await response.aread()
                response.raise_for_status()

            text_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            full_response: dict = {}
            finish_reason: str | None = None
            model_name: str = ""

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                model_name = chunk.get("model", model_name)
                choices = chunk.get("choices", [])
                if not choices:
                    usage = chunk.get("usage")
                    if usage:
                        yield UsageEvent(usage=usage)
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason", finish_reason)

                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield TextDelta(text=content)

                tcs = delta.get("tool_calls", [])
                for tc_delta in tcs:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        tc_id = tc_delta.get("id", "")
                        tc_name = tc_delta.get("function", {}).get("name", "")
                        tool_calls_acc[idx] = {
                            "id": tc_id,
                            "name": tc_name,
                            "args_parts": [],
                        }
                        if tc_id or tc_name:
                            yield ToolCallStart(
                                index=idx, id=tc_id, name=tc_name,
                            )
                    args_fragment = tc_delta.get("function", {}).get("arguments", "")
                    if args_fragment:
                        tool_calls_acc[idx]["args_parts"].append(args_fragment)
                        yield ToolCallDelta(
                            index=idx, partial_json=args_fragment,
                        )

            text = "".join(text_parts) or None
            message: dict[str, Any] = {"role": "assistant", "content": text}

            if tool_calls_acc:
                oai_tcs = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    raw_args = "".join(tc["args_parts"])
                    oai_tcs.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": raw_args,
                        },
                    })
                message["tool_calls"] = oai_tcs

            full_response = {
                "id": "",
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason or "stop",
                }],
            }
            yield MessageDone(response=full_response)

    async def aclose(self) -> None:
        """Close async resources."""
        if hasattr(self, "_async_client") and self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
