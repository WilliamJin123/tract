"""Comprehensive tests for the tract.llm package.

Tests cover:
- OpenAIClient: request formatting, retry behavior, auth errors, env config
- LLMClient protocol: conformance, custom implementations
- OpenAIResolver: resolution generation, prompt formatting, generation_config
- Error hierarchy: correct inheritance, error attributes
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tract.exceptions import TraceError
from tract.llm import (
    LLMAuthError,
    LLMClient,
    LLMClientError,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
    OpenAIClient,
    OpenAIResolver,
    Resolution,
    ResolverCallable,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _success_response(
    content: str = "Hello!",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    """Build a realistic OpenAI chat completion response dict."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_transport(handler):
    """Create an httpx.MockTransport from a handler function."""
    return httpx.MockTransport(handler)


def _make_client(
    transport=None,
    api_key: str = "test-key",
    base_url: str = "http://test-api",
    max_retries: int = 3,
    **kwargs,
) -> OpenAIClient:
    """Create an OpenAIClient with an optional mock transport."""
    client = OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        max_retries=max_retries,
        **kwargs,
    )
    if transport is not None:
        # Replace with mock transport but preserve original headers
        client._client = httpx.Client(
            transport=transport,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
    return client


class MockLLMClient:
    """A mock LLM client that records calls and returns canned responses."""

    def __init__(self, response: dict | None = None):
        self.calls: list[dict] = []
        self._response = response or _success_response("resolved content")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self._response

    def close(self) -> None:
        pass


# ===========================================================================
# Error hierarchy tests
# ===========================================================================

class TestErrorHierarchy:
    """Verify the LLM error hierarchy."""

    def test_llm_client_error_inherits_trace_error(self):
        assert issubclass(LLMClientError, TraceError)

    def test_llm_config_error_inherits_client_error(self):
        assert issubclass(LLMConfigError, LLMClientError)

    def test_llm_rate_limit_error_inherits_client_error(self):
        assert issubclass(LLMRateLimitError, LLMClientError)

    def test_llm_auth_error_inherits_client_error(self):
        assert issubclass(LLMAuthError, LLMClientError)

    def test_llm_response_error_inherits_client_error(self):
        assert issubclass(LLMResponseError, LLMClientError)

    def test_rate_limit_error_has_retry_after(self):
        err = LLMRateLimitError("rate limited", retry_after=30.0)
        assert err.retry_after == 30.0
        assert "30.0s" in str(err)

    def test_rate_limit_error_no_retry_after(self):
        err = LLMRateLimitError("rate limited")
        assert err.retry_after is None

    def test_all_errors_catchable_as_trace_error(self):
        """All LLM errors should be catchable with except TraceError."""
        for error_class in [LLMClientError, LLMConfigError, LLMRateLimitError,
                            LLMAuthError, LLMResponseError]:
            with pytest.raises(TraceError):
                raise error_class("test")


# ===========================================================================
# OpenAIClient tests
# ===========================================================================

class TestOpenAIClientChat:
    """Test OpenAIClient.chat() with mocked httpx transport."""

    def test_chat_success(self):
        """Successful chat returns parsed response dict."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler))
        response = client.chat([{"role": "user", "content": "Hello"}])

        assert "choices" in response
        assert response["choices"][0]["message"]["content"] == "Hello!"
        client.close()

    def test_chat_request_format(self):
        """Verify the request payload is correctly formatted."""
        captured_request = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured_request["payload"] = json.loads(request.content)
            captured_request["headers"] = dict(request.headers)
            captured_request["url"] = str(request.url)
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler))
        client.chat(
            [{"role": "user", "content": "Test"}],
            model="gpt-4o",
            temperature=0.5,
            max_tokens=100,
        )

        payload = captured_request["payload"]
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == [{"role": "user", "content": "Test"}]
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert "http://test-api/chat/completions" in captured_request["url"]
        assert "Bearer test-key" in captured_request["headers"]["authorization"]
        client.close()

    def test_chat_with_default_model(self):
        """When no model specified, uses default_model."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_success_response())

        client = _make_client(
            transport=_make_transport(handler),
            default_model="gpt-3.5-turbo",
        )
        client.chat([{"role": "user", "content": "Hello"}])

        assert captured["payload"]["model"] == "gpt-3.5-turbo"
        client.close()

    def test_chat_with_optional_params(self):
        """Optional params only included when provided."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler))
        # No temperature or max_tokens
        client.chat([{"role": "user", "content": "Hello"}])

        payload = captured["payload"]
        assert "temperature" not in payload
        assert "max_tokens" not in payload
        client.close()

    def test_chat_forwards_extra_kwargs(self):
        """Extra kwargs are forwarded to the API payload."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler))
        client.chat(
            [{"role": "user", "content": "Test"}],
            top_p=0.9,
            frequency_penalty=0.5,
        )

        payload = captured["payload"]
        assert payload["top_p"] == 0.9
        assert payload["frequency_penalty"] == 0.5
        client.close()


class TestOpenAIClientExtractors:
    """Test helper extract methods."""

    def test_extract_content(self):
        """extract_content returns message content string."""
        response = _success_response("Test content")
        assert OpenAIClient.extract_content(response) == "Test content"

    def test_extract_content_bad_format(self):
        """extract_content raises LLMResponseError on bad format."""
        with pytest.raises(LLMResponseError):
            OpenAIClient.extract_content({})

        with pytest.raises(LLMResponseError):
            OpenAIClient.extract_content({"choices": []})

        # A message with no content key returns "" (e.g. tool_calls response)
        assert OpenAIClient.extract_content({"choices": [{"message": {}}]}) == ""

    def test_extract_usage(self):
        """extract_usage returns usage dict."""
        response = _success_response()
        usage = OpenAIClient.extract_usage(response)
        assert usage is not None
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_extract_usage_missing(self):
        """extract_usage returns None when no usage in response."""
        response = {"choices": [{"message": {"content": "hi"}}]}
        assert OpenAIClient.extract_usage(response) is None


class TestOpenAIClientRetry:
    """Test retry behavior with different HTTP status codes."""

    def test_retry_on_429_then_success(self):
        """Client retries on 429 and succeeds on next attempt."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    429,
                    json={"error": "rate limited"},
                    headers={"Retry-After": "1"},
                )
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        response = client.chat([{"role": "user", "content": "Test"}])

        assert call_count == 2
        assert "choices" in response
        client.close()

    def test_retry_on_500_then_success(self):
        """Client retries on 500 and succeeds on next attempt."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(500, json={"error": "server error"})
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        response = client.chat([{"role": "user", "content": "Test"}])

        assert call_count == 2
        assert "choices" in response
        client.close()

    def test_retry_on_502(self):
        """Client retries on 502."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(502, json={"error": "bad gateway"})
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        response = client.chat([{"role": "user", "content": "Test"}])
        assert call_count == 2
        client.close()

    def test_retry_on_503(self):
        """Client retries on 503."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(503, json={"error": "service unavailable"})
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        response = client.chat([{"role": "user", "content": "Test"}])
        assert call_count == 2
        client.close()

    def test_retry_on_504(self):
        """Client retries on 504."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(504, json={"error": "gateway timeout"})
            return httpx.Response(200, json=_success_response())

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        response = client.chat([{"role": "user", "content": "Test"}])
        assert call_count == 2
        client.close()

    def test_no_retry_on_401(self):
        """Client raises LLMAuthError immediately on 401 (no retry)."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(401, json={"error": "unauthorized"})

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        with pytest.raises(LLMAuthError, match="Authentication failed"):
            client.chat([{"role": "user", "content": "Test"}])

        assert call_count == 1  # No retry
        client.close()

    def test_no_retry_on_403(self):
        """Client raises LLMAuthError immediately on 403."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(403, json={"error": "forbidden"})

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        with pytest.raises(LLMAuthError):
            client.chat([{"role": "user", "content": "Test"}])

        assert call_count == 1
        client.close()

    def test_no_retry_on_400(self):
        """Client raises HTTPStatusError on 400 (bad request, not retryable)."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"error": "bad request"})

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        with pytest.raises(httpx.HTTPStatusError):
            client.chat([{"role": "user", "content": "Test"}])

        assert call_count == 1
        client.close()

    def test_max_retries_exhausted(self):
        """After max_retries exhausted, raises the last error."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                429,
                json={"error": "rate limited"},
                headers={"Retry-After": "0.1"},
            )

        client = _make_client(transport=_make_transport(handler), max_retries=3)
        with pytest.raises(LLMRateLimitError):
            client.chat([{"role": "user", "content": "Test"}])

        assert call_count == 3  # Initial + 2 retries = 3 attempts
        client.close()

    def test_rate_limit_error_has_retry_after(self):
        """429 response with Retry-After header populates retry_after."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": "rate limited"},
                headers={"Retry-After": "42"},
            )

        client = _make_client(transport=_make_transport(handler), max_retries=1)
        with pytest.raises(LLMRateLimitError) as exc_info:
            client.chat([{"role": "user", "content": "Test"}])

        assert exc_info.value.retry_after == 42.0
        client.close()

    def test_response_missing_choices_raises(self):
        """Response without 'choices' key raises LLMResponseError."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "no choices"})

        client = _make_client(transport=_make_transport(handler))
        with pytest.raises(LLMResponseError, match="missing 'choices'"):
            client.chat([{"role": "user", "content": "Test"}])
        client.close()


class TestOpenAIClientConfig:
    """Test client configuration and environment variables."""

    def test_env_var_api_key(self, monkeypatch):
        """Client reads API key from TRACT_OPENAI_API_KEY env var."""
        monkeypatch.setenv("TRACT_OPENAI_API_KEY", "env-key-123")
        client = OpenAIClient()
        assert client._api_key == "env-key-123"
        client.close()

    def test_env_var_base_url(self, monkeypatch):
        """Client reads base URL from TRACT_OPENAI_BASE_URL env var."""
        monkeypatch.setenv("TRACT_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("TRACT_OPENAI_BASE_URL", "http://custom-api/v1")
        client = OpenAIClient()
        assert client._base_url == "http://custom-api/v1"
        client.close()

    def test_constructor_overrides_env(self, monkeypatch):
        """Constructor args take precedence over env vars."""
        monkeypatch.setenv("TRACT_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("TRACT_OPENAI_BASE_URL", "http://env-url/v1")
        client = OpenAIClient(api_key="arg-key", base_url="http://arg-url/v1")
        assert client._api_key == "arg-key"
        assert client._base_url == "http://arg-url/v1"
        client.close()

    def test_missing_api_key_raises(self, monkeypatch):
        """No key in env or constructor raises LLMConfigError."""
        monkeypatch.delenv("TRACT_OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMConfigError, match="No API key"):
            OpenAIClient()

    def test_empty_api_key_raises(self, monkeypatch):
        """Empty string key raises LLMConfigError."""
        monkeypatch.delenv("TRACT_OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMConfigError):
            OpenAIClient(api_key="")

    def test_context_manager(self):
        """OpenAIClient works as a context manager."""
        with OpenAIClient(api_key="test-key") as client:
            assert isinstance(client, OpenAIClient)
        # After exiting, the underlying httpx client should be closed
        assert client._client.is_closed

    def test_base_url_trailing_slash_stripped(self):
        """Trailing slash on base_url is stripped."""
        client = OpenAIClient(api_key="test-key", base_url="http://api/v1/")
        assert client._base_url == "http://api/v1"
        client.close()


# ===========================================================================
# Protocol conformance tests
# ===========================================================================

class TestProtocolConformance:
    """Test that classes conform to their protocols."""

    def test_openai_client_conforms_to_llm_client(self):
        """OpenAIClient is an instance of LLMClient protocol."""
        client = OpenAIClient(api_key="test-key")
        assert isinstance(client, LLMClient)
        client.close()

    def test_custom_client_conforms_to_protocol(self):
        """A custom class with all protocol methods conforms to LLMClient."""

        class FullClient:
            def chat(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs):
                return {"choices": [{"message": {"content": "hi"}}]}

            def close(self):
                pass

            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]

            def extract_usage(self, response):
                return response.get("usage")

        assert isinstance(FullClient(), LLMClient)

    def test_minimal_client_without_extract_methods(self):
        """A minimal chat+close client doesn't satisfy isinstance but works via fallback."""
        mock = MockLLMClient()
        # Protocol now requires extract_content/extract_usage for full conformance
        assert not isinstance(mock, LLMClient)

    def test_missing_method_fails_protocol(self):
        """A class without chat() does not conform to LLMClient."""

        class BadClient:
            def close(self) -> None:
                pass

        assert not isinstance(BadClient(), LLMClient)

    def test_resolver_callable_duck_typing(self):
        """OpenAIResolver can be called with an issue and returns Resolution."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace(
            conflict_type="both_edit",
            content_a_text="Version A",
            content_b_text="Version B",
        )
        result = resolver(issue)
        assert isinstance(result, Resolution)
        assert result.action == "resolved"

    def test_custom_resolver_callable(self):
        """A custom callable matching the protocol works."""

        class MyResolver:
            def __call__(self, issue: object) -> Resolution:
                return Resolution(action="skip", reasoning="custom skip")

        resolver = MyResolver()
        result = resolver(SimpleNamespace(conflict_type="test"))
        assert result.action == "skip"
        assert result.reasoning == "custom skip"


# ===========================================================================
# OpenAIResolver tests
# ===========================================================================

class TestOpenAIResolver:
    """Test the built-in OpenAIResolver."""

    def test_resolver_returns_resolution(self):
        """Resolver returns a Resolution with action='resolved'."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace(
            conflict_type="both_edit",
            content_a_text="Branch A says dark mode",
            content_b_text="Branch B says light mode",
        )
        result = resolver(issue)

        assert isinstance(result, Resolution)
        assert result.action == "resolved"
        assert result.content_text == "resolved content"

    def test_resolver_sends_correct_messages(self):
        """Resolver sends system prompt + user prompt to client."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace(
            conflict_type="both_edit",
            content_a_text="Version A",
            content_b_text="Version B",
        )
        resolver(issue)

        assert len(mock_client.calls) == 1
        messages = mock_client.calls[0]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_resolver_custom_system_prompt(self):
        """Custom system prompt is used instead of default."""
        mock_client = MockLLMClient()
        custom_prompt = "Always prefer the newer version."
        resolver = OpenAIResolver(mock_client, system_prompt=custom_prompt)
        issue = SimpleNamespace(conflict_type="test")
        resolver(issue)

        messages = mock_client.calls[0]["messages"]
        assert messages[0]["content"] == custom_prompt

    def test_resolver_default_system_prompt(self):
        """Default system prompt mentions context merge resolver."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace(conflict_type="test")
        resolver(issue)

        messages = mock_client.calls[0]["messages"]
        assert "context merge resolver" in messages[0]["content"]

    def test_resolver_forwards_model_params(self):
        """Resolver forwards model, temperature, max_tokens to client."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(
            mock_client,
            model="gpt-4o",
            temperature=0.1,
            max_tokens=4096,
        )
        issue = SimpleNamespace(conflict_type="test")
        resolver(issue)

        call = mock_client.calls[0]
        assert call["model"] == "gpt-4o"
        assert call["temperature"] == 0.1
        assert call["max_tokens"] == 4096

    def test_resolver_records_generation_config(self):
        """Resolution includes generation_config with model, temperature, source."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client, temperature=0.3)
        issue = SimpleNamespace(conflict_type="test")
        result = resolver(issue)

        assert result.generation_config is not None
        assert result.generation_config["temperature"] == 0.3
        assert result.generation_config["source"] == "infrastructure:merge"
        assert "model" in result.generation_config

    def test_resolver_records_usage_in_generation_config(self):
        """When API returns usage, it's included in generation_config."""
        response = _success_response("resolved")
        mock_client = MockLLMClient(response=response)
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace(conflict_type="test")
        result = resolver(issue)

        assert result.generation_config is not None
        assert "usage" in result.generation_config
        assert result.generation_config["usage"]["prompt_tokens"] == 10

    def test_resolver_reasoning_includes_model(self):
        """Resolution reasoning mentions the model used."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client, model="gpt-4o")
        issue = SimpleNamespace(conflict_type="test")
        result = resolver(issue)

        assert result.reasoning is not None
        assert "gpt-4o" in result.reasoning

    def test_format_issue_with_full_conflict(self):
        """_format_issue produces output with all conflict fields."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace(
            conflict_type="both_edit",
            content_a_text="Dark mode preferred",
            content_b_text="Light mode preferred",
            ancestor_content_text="No preference set",
        )
        formatted = resolver._format_issue(issue)

        assert "both_edit" in formatted
        assert "Dark mode preferred" in formatted
        assert "Light mode preferred" in formatted
        assert "No preference set" in formatted
        assert "Branch A content" in formatted
        assert "Branch B content" in formatted
        assert "Common ancestor content" in formatted

    def test_format_issue_minimal(self):
        """_format_issue works with minimal issue (just conflict_type)."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace(conflict_type="skip_vs_edit")
        formatted = resolver._format_issue(issue)

        assert "skip_vs_edit" in formatted
        assert "Branch A content" not in formatted

    def test_format_issue_unknown_type(self):
        """_format_issue handles objects without conflict_type attribute."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)
        issue = SimpleNamespace()  # No conflict_type
        formatted = resolver._format_issue(issue)

        assert "unknown" in formatted

    def test_format_issue_with_compiled_context(self):
        """_format_issue includes truncated compiled context."""
        mock_client = MockLLMClient()
        resolver = OpenAIResolver(mock_client)

        msg1 = SimpleNamespace(role="system", content="You are a helpful assistant.")
        msg2 = SimpleNamespace(role="user", content="What is the capital of France?")
        context = SimpleNamespace(messages=[msg1, msg2])

        issue = SimpleNamespace(
            conflict_type="test",
            compiled_context=context,
        )
        formatted = resolver._format_issue(issue)

        assert "Surrounding context" in formatted
        assert "[system]" in formatted
        assert "[user]" in formatted


# ===========================================================================
# Resolution dataclass tests
# ===========================================================================

class TestResolution:
    """Test the Resolution dataclass."""

    def test_resolution_defaults(self):
        """Resolution has sensible defaults."""
        r = Resolution(action="resolved")
        assert r.action == "resolved"
        assert r.content is None
        assert r.content_text is None
        assert r.reasoning is None
        assert r.generation_config is None

    def test_resolution_with_all_fields(self):
        """Resolution can be created with all fields."""
        r = Resolution(
            action="resolved",
            content_text="Merged content",
            reasoning="LLM chose version A",
            generation_config={"model": "gpt-4o", "temperature": 0.3},
        )
        assert r.action == "resolved"
        assert r.content_text == "Merged content"
        assert r.reasoning == "LLM chose version A"
        assert r.generation_config["model"] == "gpt-4o"

    def test_resolution_abort(self):
        """Resolution with abort action."""
        r = Resolution(action="abort", reasoning="Irreconcilable conflict")
        assert r.action == "abort"

    def test_resolution_skip(self):
        """Resolution with skip action."""
        r = Resolution(action="skip")
        assert r.action == "skip"
