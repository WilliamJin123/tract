"""Tests for FallbackClient -- multi-client failover wrapper.

Covers:
- LLMClient protocol conformance
- Primary succeeds (fallback never called)
- Primary fails, falls back to secondary
- All clients fail (last exception re-raised)
- close() closes all clients
- extract_content/extract_usage delegate to last successful client
- Requires at least one client
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tract.llm.fallback import FallbackClient
from tract.llm.protocols import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: str = "Hello!", usage: dict | None = None) -> dict:
    """Build a minimal OpenAI-style response dict."""
    resp: dict[str, Any] = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        resp["usage"] = usage
    return resp


class StubClient:
    """Full LLMClient-conformant stub that records calls."""

    def __init__(self, name: str = "stub", response: dict | None = None):
        self.name = name
        self._response = response or _make_response(f"from-{name}")
        self.chat_calls: list[dict] = []
        self.closed = False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        self.chat_calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        })
        return self._response

    def close(self) -> None:
        self.closed = True

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict | None:
        return response.get("usage")


class FailStub(StubClient):
    """Stub that raises on chat()."""

    def __init__(self, name: str = "fail", error: Exception | None = None, **kw: Any):
        super().__init__(name=name, **kw)
        self._error = error or RuntimeError(f"{name} failed")

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        self.chat_calls.append({"messages": messages, **kwargs})
        raise self._error


# ===========================================================================
# Protocol conformance
# ===========================================================================

class TestProtocolConformance:
    """FallbackClient must satisfy isinstance(client, LLMClient)."""

    def test_isinstance_llm_client(self):
        client = FallbackClient(StubClient())
        assert isinstance(client, LLMClient)

    def test_has_all_protocol_methods(self):
        client = FallbackClient(StubClient())
        assert callable(getattr(client, "chat", None))
        assert callable(getattr(client, "close", None))
        assert callable(getattr(client, "extract_content", None))
        assert callable(getattr(client, "extract_usage", None))


# ===========================================================================
# Constructor validation
# ===========================================================================

class TestConstructor:
    """FallbackClient requires at least one client."""

    def test_no_clients_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one client"):
            FallbackClient()

    def test_single_client_accepted(self):
        client = FallbackClient(StubClient())
        assert len(client.clients) == 1

    def test_multiple_clients_accepted(self):
        client = FallbackClient(StubClient("a"), StubClient("b"), StubClient("c"))
        assert len(client.clients) == 3

    def test_last_client_index_initially_none(self):
        client = FallbackClient(StubClient())
        assert client.last_client_index is None


# ===========================================================================
# chat() fallback behavior
# ===========================================================================

class TestChatFallback:
    """Core fallback logic in chat()."""

    def test_primary_succeeds_returns_response(self):
        primary = StubClient("primary")
        fallback = StubClient("fallback")
        client = FallbackClient(primary, fallback)

        response = client.chat([{"role": "user", "content": "hi"}])

        assert response["choices"][0]["message"]["content"] == "from-primary"

    def test_primary_succeeds_never_calls_fallback(self):
        primary = StubClient("primary")
        fallback = StubClient("fallback")
        client = FallbackClient(primary, fallback)

        client.chat([{"role": "user", "content": "hi"}])

        assert len(primary.chat_calls) == 1
        assert len(fallback.chat_calls) == 0

    def test_primary_succeeds_sets_last_client_index_zero(self):
        client = FallbackClient(StubClient(), StubClient())
        client.chat([{"role": "user", "content": "hi"}])
        assert client.last_client_index == 0

    def test_primary_fails_falls_back_to_secondary(self):
        primary = FailStub("primary")
        secondary = StubClient("secondary")
        client = FallbackClient(primary, secondary)

        response = client.chat([{"role": "user", "content": "hi"}])

        assert response["choices"][0]["message"]["content"] == "from-secondary"
        assert len(primary.chat_calls) == 1
        assert len(secondary.chat_calls) == 1

    def test_primary_fails_sets_last_client_index_one(self):
        client = FallbackClient(FailStub(), StubClient())
        client.chat([{"role": "user", "content": "hi"}])
        assert client.last_client_index == 1

    def test_first_two_fail_third_succeeds(self):
        a = FailStub("a")
        b = FailStub("b")
        c = StubClient("c")
        client = FallbackClient(a, b, c)

        response = client.chat([{"role": "user", "content": "hi"}])

        assert response["choices"][0]["message"]["content"] == "from-c"
        assert client.last_client_index == 2
        assert len(a.chat_calls) == 1
        assert len(b.chat_calls) == 1
        assert len(c.chat_calls) == 1

    def test_all_fail_raises_last_exception(self):
        a = FailStub("a", error=ConnectionError("a down"))
        b = FailStub("b", error=TimeoutError("b timeout"))
        client = FallbackClient(a, b)

        with pytest.raises(TimeoutError, match="b timeout"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_all_fail_tried_every_client(self):
        a = FailStub("a")
        b = FailStub("b")
        c = FailStub("c")
        client = FallbackClient(a, b, c)

        with pytest.raises(RuntimeError):
            client.chat([{"role": "user", "content": "hi"}])

        assert len(a.chat_calls) == 1
        assert len(b.chat_calls) == 1
        assert len(c.chat_calls) == 1

    def test_single_client_success(self):
        only = StubClient("only")
        client = FallbackClient(only)
        response = client.chat([{"role": "user", "content": "hi"}])
        assert response["choices"][0]["message"]["content"] == "from-only"
        assert client.last_client_index == 0

    def test_single_client_failure_raises(self):
        only = FailStub("only", error=ValueError("nope"))
        client = FallbackClient(only)
        with pytest.raises(ValueError, match="nope"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_chat_forwards_kwargs(self):
        """All keyword arguments are forwarded to the successful client."""
        primary = StubClient("primary")
        client = FallbackClient(primary)

        client.chat(
            [{"role": "user", "content": "hi"}],
            model="gpt-4o",
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        )

        call = primary.chat_calls[0]
        assert call["model"] == "gpt-4o"
        assert call["temperature"] == 0.5
        assert call["max_tokens"] == 100
        assert call["top_p"] == 0.9

    def test_chat_forwards_kwargs_to_fallback(self):
        """When primary fails, kwargs are forwarded to fallback too."""
        primary = FailStub("primary")
        secondary = StubClient("secondary")
        client = FallbackClient(primary, secondary)

        client.chat(
            [{"role": "user", "content": "hi"}],
            model="claude-sonnet",
            temperature=0.3,
        )

        call = secondary.chat_calls[0]
        assert call["model"] == "claude-sonnet"
        assert call["temperature"] == 0.3


# ===========================================================================
# close()
# ===========================================================================

class TestClose:
    """close() must close ALL underlying clients."""

    def test_close_all_clients(self):
        a = StubClient("a")
        b = StubClient("b")
        c = StubClient("c")
        client = FallbackClient(a, b, c)

        client.close()

        assert a.closed is True
        assert b.closed is True
        assert c.closed is True

    def test_close_single_client(self):
        only = StubClient("only")
        client = FallbackClient(only)
        client.close()
        assert only.closed is True


# ===========================================================================
# extract_content / extract_usage delegation
# ===========================================================================

class TestExtractDelegation:
    """extract_content/extract_usage delegate to the last successful client."""

    def test_extract_content_delegates_to_last_successful(self):
        """After fallback succeeds, extract_content uses that client."""
        primary = FailStub("primary")
        secondary = StubClient("secondary")
        client = FallbackClient(primary, secondary)

        response = client.chat([{"role": "user", "content": "hi"}])
        content = client.extract_content(response)

        assert content == "from-secondary"

    def test_extract_usage_delegates_to_last_successful(self):
        """After fallback succeeds, extract_usage uses that client."""
        usage_data = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        resp = _make_response("hello", usage=usage_data)

        primary = FailStub("primary")
        secondary = StubClient("secondary", response=resp)
        client = FallbackClient(primary, secondary)

        response = client.chat([{"role": "user", "content": "hi"}])
        usage = client.extract_usage(response)

        assert usage == usage_data

    def test_extract_content_before_any_chat_uses_first_client(self):
        """Before any chat() call, extract_content falls back to first client."""
        first = StubClient("first")
        client = FallbackClient(first, StubClient("second"))

        resp = _make_response("test-content")
        content = client.extract_content(resp)

        assert content == "test-content"
        assert client.last_client_index is None  # no chat() yet

    def test_extract_usage_before_any_chat_uses_first_client(self):
        """Before any chat() call, extract_usage falls back to first client."""
        first = StubClient("first")
        client = FallbackClient(first, StubClient("second"))

        usage_data = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        resp = _make_response("x", usage=usage_data)
        usage = client.extract_usage(resp)

        assert usage == usage_data

    def test_extract_content_after_primary_success(self):
        """When primary succeeds, extract_content delegates to primary."""
        primary = StubClient("primary")
        secondary = StubClient("secondary")
        client = FallbackClient(primary, secondary)

        response = client.chat([{"role": "user", "content": "hi"}])
        content = client.extract_content(response)

        assert client.last_client_index == 0
        assert content == "from-primary"

    def test_extract_usage_returns_none_when_absent(self):
        """extract_usage returns None when response has no usage."""
        resp = _make_response("hi")  # no usage key
        client = FallbackClient(StubClient())
        client.chat([{"role": "user", "content": "x"}])
        assert client.extract_usage(resp) is None


# ===========================================================================
# last_client_index tracking
# ===========================================================================

class TestLastClientIndex:
    """last_client_index tracks which client succeeded last."""

    def test_updates_on_each_successful_chat(self):
        a = StubClient("a")
        b = StubClient("b")
        client = FallbackClient(a, b)

        client.chat([{"role": "user", "content": "1"}])
        assert client.last_client_index == 0

        # Replace primary with a failing one
        client.clients[0] = FailStub("a-broken")
        client.chat([{"role": "user", "content": "2"}])
        assert client.last_client_index == 1

        # Restore primary
        client.clients[0] = StubClient("a-fixed")
        client.chat([{"role": "user", "content": "3"}])
        assert client.last_client_index == 0

    def test_not_updated_when_all_fail(self):
        client = FallbackClient(FailStub("a"), FailStub("b"))
        client.last_client_index = 0  # set manually

        with pytest.raises(RuntimeError):
            client.chat([{"role": "user", "content": "hi"}])

        # Should NOT have been updated (no success)
        assert client.last_client_index == 0
