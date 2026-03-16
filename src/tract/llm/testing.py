"""First-party test utilities for tract's LLM-dependent paths.

Provides drop-in LLMClient implementations that require no external
dependencies and no network access.  Use these to test any code that
calls ``Tract.chat()``, ``Tract.generate()``, ``Tract.run()``,
``SemanticGate``, ``SemanticMaintainer``, or LLM-backed compression
without spending tokens or dealing with flaky HTTP calls.

Three clients are provided, each targeting a different testing style:

* **MockLLMClient** -- cycles through canned string responses.
* **ReplayLLMClient** -- plays responses sequentially, then raises.
* **FunctionLLMClient** -- delegates to a user-supplied callable.

All three satisfy the ``LLMClient`` protocol, record every call in a
``.calls`` list, and expose a ``closed`` flag set by ``close()``.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "MockLLMClient",
    "ReplayLLMClient",
    "FunctionLLMClient",
]


# -------------------------------------------------------------------
# Shared helper
# -------------------------------------------------------------------

def _make_response(content: str, model: str) -> dict:
    """Wrap a plain string in an OpenAI-format response dict."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model": model,
    }


# -------------------------------------------------------------------
# MockLLMClient
# -------------------------------------------------------------------

class MockLLMClient:
    """LLM client that cycles through pre-configured string responses.

    When the response list is exhausted the index wraps around to the
    beginning, so tests that make more calls than there are responses
    will keep receiving answers.

    Every call is recorded in ``self.calls`` for later assertions::

        mock = MockLLMClient(["Hello!"])
        mock.chat([{"role": "user", "content": "Hi"}])
        assert mock.call_count == 1
        assert mock.calls[0]["messages"][0]["content"] == "Hi"
    """

    def __init__(self, responses: list[str], *, model: str = "mock-model") -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.call_count: int = 0
        self.closed: bool = False
        self._model = model
        self._index = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Return the next canned response, cycling when exhausted."""
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        })
        self.call_count += 1
        content = self.responses[self._index % len(self.responses)]
        self._index += 1
        return _make_response(content, model or self._model)

    def close(self) -> None:
        """Mark the client as closed."""
        self.closed = True

    def extract_content(self, response: dict) -> str:
        """Extract assistant message text from *response*."""
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict | None:
        """Extract the usage dict (or ``None``) from *response*."""
        return response.get("usage")


# -------------------------------------------------------------------
# ReplayLLMClient
# -------------------------------------------------------------------

class ReplayLLMClient:
    """LLM client that plays responses in order, then raises.

    Unlike :class:`MockLLMClient`, this client does **not** cycle.
    Once all responses have been consumed an ``IndexError`` is raised,
    making it easy to assert that your code makes exactly the expected
    number of LLM calls.

    Responses may be plain strings (auto-wrapped via :func:`_make_response`)
    or full OpenAI-format dicts (returned as-is).
    """

    def __init__(self, responses: list[str | dict], *, model: str = "replay-model") -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.call_count: int = 0
        self.closed: bool = False
        self._model = model
        self._index = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Return the next response or raise ``IndexError``."""
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        })
        self.call_count += 1
        if self._index >= len(self.responses):
            raise IndexError(
                f"ReplayLLMClient exhausted: {len(self.responses)} responses consumed"
            )
        raw = self.responses[self._index]
        self._index += 1
        if isinstance(raw, dict):
            return raw
        return _make_response(raw, model or self._model)

    def close(self) -> None:
        """Mark the client as closed."""
        self.closed = True

    def extract_content(self, response: dict) -> str:
        """Extract assistant message text from *response*."""
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict | None:
        """Extract the usage dict (or ``None``) from *response*."""
        return response.get("usage")


# -------------------------------------------------------------------
# FunctionLLMClient
# -------------------------------------------------------------------

class FunctionLLMClient:
    """LLM client that delegates to a user-supplied callable.

    The callable receives ``(messages, kwargs)`` and must return either:

    * A **string** -- automatically wrapped in OpenAI format.
    * A **dict** -- returned as-is (must be a valid OpenAI-format response).

    This is the most flexible option: your function can inspect the
    messages, count calls, return different answers based on content,
    or raise exceptions to simulate failures.
    """

    def __init__(
        self,
        response_fn: Callable[[list[dict], dict], str | dict],
        *,
        model: str = "function-model",
    ) -> None:
        self.response_fn = response_fn
        self.calls: list[dict] = []
        self.call_count: int = 0
        self.closed: bool = False
        self._model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Call the response function and return its result."""
        all_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        self.calls.append({"messages": messages, **all_kwargs})
        self.call_count += 1
        result = self.response_fn(messages, all_kwargs)
        if isinstance(result, dict):
            return result
        return _make_response(result, model or self._model)

    def close(self) -> None:
        """Mark the client as closed."""
        self.closed = True

    def extract_content(self, response: dict) -> str:
        """Extract assistant message text from *response*."""
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict | None:
        """Extract the usage dict (or ``None``) from *response*."""
        return response.get("usage")
