"""LLM client and resolver protocols.

Defines pluggable interfaces for LLM interaction and conflict resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for pluggable LLM clients.

    Any object with chat() and close() methods matching this signature works.
    The built-in OpenAIClient implements this protocol.

    Custom clients can override ``extract_content()`` and ``extract_usage()``
    to support non-OpenAI response formats.  The defaults assume OpenAI-style
    responses (``choices[0].message.content`` and ``.usage``).
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send messages, return response dict."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...

    def extract_content(self, response: dict) -> str:
        """Extract assistant message content from an LLM response.

        Override this for non-OpenAI response formats.
        Default assumes ``response["choices"][0]["message"]["content"]``.
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Cannot extract content from response: {exc}. "
                f"Override extract_content() for custom formats."
            ) from exc

    def extract_usage(self, response: dict) -> dict | None:
        """Extract usage dict from an LLM response.

        Override this for non-OpenAI response formats.
        Default assumes ``response.get("usage")``.
        """
        return response.get("usage")


@runtime_checkable
class AsyncLLMClient(Protocol):
    """Protocol for async LLM clients.

    Mirrors LLMClient but with async methods. Clients may implement
    both protocols. The built-in OpenAIClient and AnthropicClient do.
    """

    async def achat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send messages asynchronously, return response dict."""
        ...

    async def aclose(self) -> None:
        """Release underlying async resources."""
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
    (ConflictInfo, RebaseWarning, ImportIssue) vary by caller.
    """

    def __call__(self, issue: object) -> Resolution:
        """Resolve an issue and return a Resolution."""
        ...


@runtime_checkable
class AgentLoop(Protocol):
    """Protocol for pluggable agent loops.

    Any object implementing ``run()`` and ``stop()`` can serve as tract's
    agent loop.  External adapters (Agno, LangGraph, CrewAI) implement
    this protocol to plug into ``t.run_loop(agent_loop=...)``.

    The protocol parallels :class:`LLMClient`: LLMClient swaps the LLM
    transport layer, AgentLoop swaps the orchestration layer.

    Tract prepares everything (assessment, tools, executor) and hands off
    to the loop.  The loop runs however it wants (its own LLM, its own
    tool dispatch) and returns a structured result with optional provenance.
    """

    def run(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict],
        execute_tool: Any,
    ) -> Any:
        """Execute the agent loop.

        Args:
            messages: Initial conversation messages (typically a system prompt
                and a user message containing the context assessment).
            tools: Tool definitions in OpenAI function-calling format.
            execute_tool: Callable ``(tool_name: str, arguments: dict) -> ToolResult``
                that dispatches to tract's tool executor.

        Returns:
            A result object with steps taken and optional provenance data.
        """
        ...

    def stop(self) -> None:
        """Signal the loop to stop at the next safe point."""
        ...


async def acall_llm(
    client: Any,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> dict:
    """Call an LLM client asynchronously.

    Uses ``achat()`` if the client implements it (AsyncLLMClient),
    otherwise wraps the sync ``chat()`` in ``asyncio.to_thread()``.
    """
    import asyncio
    if hasattr(client, "achat"):
        return await client.achat(messages, **kwargs)
    return await asyncio.to_thread(client.chat, messages, **kwargs)
