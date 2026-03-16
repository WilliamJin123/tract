"""Fallback Clients -- resilient LLM access with automatic failover.

Demonstrates FallbackClient, a thin wrapper that tries multiple LLM
clients in priority order.  If the primary client raises any exception,
the next client is tried automatically.

Patterns shown:
  1. Basic Setup              -- FallbackClient with two mock clients
  2. Primary Succeeds         -- fallback is never called
  3. Primary Fails, Fallback  -- automatic failover on exception
  4. All Fail                 -- last exception re-raised
  5. Checking Which Client    -- inspect last_client_index after chat()

No LLM required.  All examples use mock clients.

Production usage (not shown because it requires API keys):

    from tract.llm.client import OpenAIClient
    from tract.llm.anthropic_client import AnthropicClient
    from tract.llm.fallback import FallbackClient

    client = FallbackClient(
        OpenAIClient(api_key="sk-..."),
        AnthropicClient(api_key="sk-ant-..."),
    )
    # Uses OpenAI by default; falls back to Anthropic on any error.
    response = client.chat([{"role": "user", "content": "Hello"}])
"""

from typing import Any

from tract.llm.fallback import FallbackClient
from tract.llm.protocols import LLMClient


# ---------------------------------------------------------------------------
# Mock clients for demonstration
# ---------------------------------------------------------------------------

class SuccessClient:
    """Mock client that always succeeds."""

    def __init__(self, name: str = "success"):
        self.name = name
        self.calls: list[dict] = []

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             **kwargs: Any) -> dict:
        self.calls.append({"messages": messages})
        return {
            "choices": [{"message": {"role": "assistant", "content": f"Hello from {self.name}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def close(self) -> None:
        pass

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict | None:
        return response.get("usage")


class FailingClient:
    """Mock client that always raises an exception."""

    def __init__(self, name: str = "failing", error: Exception | None = None):
        self.name = name
        self._error = error or ConnectionError(f"{name}: connection refused")
        self.calls: list[dict] = []

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             **kwargs: Any) -> dict:
        self.calls.append({"messages": messages})
        raise self._error

    def close(self) -> None:
        pass

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict | None:
        return response.get("usage")


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def primary_succeeds() -> None:
    """When the primary client works, the fallback is never called."""

    print("=" * 60)
    print("1. Primary Succeeds -- Fallback Never Called")
    print("=" * 60)
    print()

    primary = SuccessClient("openai")
    fallback = SuccessClient("anthropic")
    client = FallbackClient(primary, fallback)

    response = client.chat([{"role": "user", "content": "Hello"}])
    content = client.extract_content(response)

    assert content == "Hello from openai"
    assert client.last_client_index == 0
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0

    print(f"  Response: {content}")
    print(f"  Client used: index {client.last_client_index} (primary)")
    print(f"  Fallback calls: {len(fallback.calls)}")

    client.close()

    print()
    print("PASSED")


def primary_fails_fallback_succeeds() -> None:
    """Primary raises an exception; fallback handles the request."""

    print()
    print("=" * 60)
    print("2. Primary Fails -- Automatic Fallback")
    print("=" * 60)
    print()

    primary = FailingClient("openai")
    fallback = SuccessClient("anthropic")
    client = FallbackClient(primary, fallback)

    response = client.chat([{"role": "user", "content": "Hello"}])
    content = client.extract_content(response)

    assert content == "Hello from anthropic"
    assert client.last_client_index == 1
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1

    print(f"  Response: {content}")
    print(f"  Client used: index {client.last_client_index} (fallback)")
    print(f"  Primary attempted and failed, fallback succeeded")

    client.close()

    print()
    print("PASSED")


def all_clients_fail() -> None:
    """When every client fails, the last exception is re-raised."""

    print()
    print("=" * 60)
    print("3. All Clients Fail -- Last Exception Re-raised")
    print("=" * 60)
    print()

    client_a = FailingClient("openai", ConnectionError("OpenAI down"))
    client_b = FailingClient("anthropic", TimeoutError("Anthropic timeout"))
    client = FallbackClient(client_a, client_b)

    caught = None
    try:
        client.chat([{"role": "user", "content": "Hello"}])
    except TimeoutError as exc:
        caught = exc
        print(f"  Caught: {type(exc).__name__}: {exc}")

    assert caught is not None, "Should have raised the last exception"
    assert isinstance(caught, TimeoutError)
    assert len(client_a.calls) == 1
    assert len(client_b.calls) == 1

    print(f"  Both clients were tried ({len(client_a.calls)} + {len(client_b.calls)} calls)")
    print(f"  Last exception type: {type(caught).__name__}")

    client.close()

    print()
    print("PASSED")


def checking_which_client() -> None:
    """Inspect last_client_index to see which client handled the request."""

    print()
    print("=" * 60)
    print("4. Checking Which Client Was Used")
    print("=" * 60)
    print()

    primary = SuccessClient("gpt-4o")
    secondary = SuccessClient("claude-sonnet")
    tertiary = SuccessClient("llama-local")
    client = FallbackClient(primary, secondary, tertiary)

    # First call: primary succeeds
    client.chat([{"role": "user", "content": "First"}])
    assert client.last_client_index == 0
    print(f"  Call 1: used client index {client.last_client_index} ({primary.name})")

    # Simulate primary breaking by replacing it
    client.clients[0] = FailingClient("gpt-4o-broken")

    # Second call: primary fails, secondary succeeds
    client.chat([{"role": "user", "content": "Second"}])
    assert client.last_client_index == 1
    print(f"  Call 2: used client index {client.last_client_index} ({secondary.name})")

    client.close()

    print()
    print("PASSED")


def protocol_conformance() -> None:
    """FallbackClient satisfies the LLMClient protocol."""

    print()
    print("=" * 60)
    print("5. Protocol Conformance")
    print("=" * 60)
    print()

    client = FallbackClient(SuccessClient())
    assert isinstance(client, LLMClient), "FallbackClient must satisfy LLMClient protocol"

    print(f"  isinstance(FallbackClient(...), LLMClient) = True")

    client.close()

    print()
    print("PASSED")


def main() -> None:
    primary_succeeds()
    primary_fails_fallback_succeeds()
    all_clients_fail()
    checking_which_client()
    protocol_conformance()

    print()
    print("=" * 60)
    print("Summary: Fallback Client Patterns")
    print("=" * 60)
    print()
    print("  Pattern                     Description")
    print("  --------------------------  ------------------------------------")
    print("  Primary succeeds            Fallback never called")
    print("  Primary fails               Automatic failover to next client")
    print("  All fail                    Last exception re-raised")
    print("  Which client?               Check last_client_index after chat()")
    print("  Protocol conformance        isinstance(client, LLMClient) == True")
    print()
    print("  Production tip: FallbackClient(OpenAIClient(...), AnthropicClient(...))")
    print("  gives automatic cross-provider resilience with zero app code changes.")
    print()
    print("Done.")


# Alias for pytest discovery
test_fallback_clients = main


if __name__ == "__main__":
    main()


# --- See also ---
# Recovery strategies:     error_handling/01_recovery_strategies.py
# Graceful degradation:    error_handling/02_graceful_degradation.py
# LLM client reference:    reference/ (TBD)
