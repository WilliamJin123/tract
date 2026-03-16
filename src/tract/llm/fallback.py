"""Fallback LLM client that tries multiple clients in sequence.

Provides production resilience by cascading through a priority-ordered
list of LLM clients.  If the primary client raises any exception, the
next client is tried, and so on.  If every client fails, the last
exception is re-raised.

Example::

    from tract.llm.fallback import FallbackClient

    client = FallbackClient(primary_openai, fallback_anthropic)
    response = client.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

from typing import Any

__all__ = ["FallbackClient"]


class FallbackClient:
    """LLM client that tries multiple clients in sequence.

    On ``chat()``, tries the primary client first.  If it raises any
    exception, tries the next client, and so on.  If all clients fail,
    re-raises the last exception.

    Useful for production resilience: try GPT-4o, fall back to Claude
    on error.

    The class satisfies the :class:`~tract.llm.protocols.LLMClient`
    protocol so it can be used anywhere a regular client is expected.
    """

    def __init__(self, *clients: Any) -> None:
        """Initialize with one or more LLM clients in priority order.

        Args:
            *clients: LLM client instances.  Each must implement
                ``chat()``, ``close()``, ``extract_content()``, and
                ``extract_usage()``.

        Raises:
            ValueError: If no clients are provided.
        """
        if not clients:
            raise ValueError("FallbackClient requires at least one client")
        self.clients: list[Any] = list(clients)
        self.last_client_index: int | None = None

    # -- LLMClient protocol methods ------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send messages, trying each client in order until one succeeds.

        Args:
            messages: Conversation messages.
            model: Optional model override.
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.
            **kwargs: Extra keyword arguments forwarded to the client.

        Returns:
            The response dict from the first client that succeeds.

        Raises:
            Exception: The last exception if all clients fail.
        """
        last_exc: BaseException | None = None
        for idx, client in enumerate(self.clients):
            try:
                response = client.chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self.last_client_index = idx
                return response
            except Exception as exc:
                last_exc = exc
                continue
        # All clients failed -- re-raise the last exception.
        # last_exc is guaranteed non-None because self.clients is non-empty.
        raise last_exc  # type: ignore[misc]

    def close(self) -> None:
        """Close all underlying clients."""
        for client in self.clients:
            client.close()

    def extract_content(self, response: dict) -> str:
        """Extract content using the client that produced the response.

        Delegates to the client at :attr:`last_client_index`.  If no
        successful call has been made yet, delegates to the first client.
        """
        idx = self.last_client_index if self.last_client_index is not None else 0
        return self.clients[idx].extract_content(response)

    def extract_usage(self, response: dict) -> dict | None:
        """Extract usage using the client that produced the response.

        Delegates to the client at :attr:`last_client_index`.  If no
        successful call has been made yet, delegates to the first client.
        """
        idx = self.last_client_index if self.last_client_index is not None else 0
        return self.clients[idx].extract_usage(response)
