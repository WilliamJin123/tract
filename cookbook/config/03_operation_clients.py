"""Per-Operation LLM Clients

Use a different LLM *client* for each operation — e.g. OpenAI for chat,
a local Ollama for compression, Anthropic for merge conflict resolution.

LLMConfig controls *what settings* to use. LLMClient controls *where to
send the request*. They're fully decoupled: you can use temperature=0.1
with either OpenAI or Ollama.

Demonstrates: OpenAIClient, configure_clients(), OperationClients,
              client routing, client lifecycle, additive merging,
              combining per-operation clients with per-operation configs
"""

import os

from dotenv import load_dotenv

from tract import LLMConfig, OperationClients, Tract
from tract.llm import OpenAIClient

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 3: Per-Operation LLM Clients
# =============================================================================
# Use a different LLM *client* for each operation — e.g. OpenAI for chat,
# a local Ollama for compression, Anthropic for merge conflict resolution.
#
# LLMConfig controls *what settings* to use. LLMClient controls *where to
# send the request*. They're fully decoupled: you can use temperature=0.1
# with either OpenAI or Ollama.

def part3_per_operation_clients():
    print(f"\n{'=' * 60}")
    print("Part 3: PER-OPERATION LLM CLIENTS")
    print("=" * 60)
    print()

    # Two different LLM endpoints
    PRIMARY_KEY = TRACT_OPENAI_API_KEY
    PRIMARY_URL = TRACT_OPENAI_BASE_URL
    PRIMARY_MODEL = MODEL_ID

    # If you have a second endpoint (e.g. local Ollama), configure it here:
    # SECONDARY_URL = "http://localhost:11434/v1"
    # SECONDARY_KEY = "ollama"
    # SECONDARY_MODEL = "llama3"
    #
    # For this example, we use the same endpoint with different configs
    # to demonstrate the routing — in production, these would be different services.

    # --- Create clients manually ---
    # Each client is an independent HTTP transport with its own auth + base_url.

    chat_client = OpenAIClient(
        api_key=PRIMARY_KEY,
        base_url=PRIMARY_URL,
    )

    # In a real setup, this would point to a different service:
    #   compress_client = OpenAIClient(api_key=SECONDARY_KEY, base_url=SECONDARY_URL)
    compress_client = OpenAIClient(
        api_key=PRIMARY_KEY,
        base_url=PRIMARY_URL,
    )

    try:
        # Note: use default_config= (not model=) when not providing api_key,
        # because model= on open() only takes effect with api_key=.
        t = Tract.open(default_config=LLMConfig(model=PRIMARY_MODEL))

        # --- Assign clients per operation ---
        # No default client needed — each operation routes to its own.
        t.configure_clients(
            chat=chat_client,
            compress=compress_client,
        )

        # You can also use the typed OperationClients dataclass:
        #   t.configure_clients(OperationClients(
        #       chat=chat_client,
        #       compress=compress_client,
        #   ))

        t.system("You are a helpful assistant.")

        print("=== Chat: routed to chat_client ===\n")
        response = t.chat("What are Python generators?")
        # pprint() shows the full response — text, usage, and which config was used
        response.pprint()
        print()

        # --- Combine with per-operation configs ---
        # Clients and configs are independent: set both.
        t.configure_operations(
            chat=LLMConfig(temperature=0.8),
            compress=LLMConfig(temperature=0.1, seed=42),
        )

        print("=== Chat: creative config + chat_client ===\n")
        response = t.chat("Give a creative analogy for generators.")
        gc = response.generation_config
        print(f"  temperature={gc.temperature} (from operation config)")
        print(f"  Response: {response.text[:120]}...\n")

        # --- Inspect current routing ---
        print("=== Client routing ===\n")
        clients = t.operation_clients
        print(f"  chat:        {'configured' if clients.chat else 'default'}")
        print(f"  compress:    {'configured' if clients.compress else 'default'}")
        print(f"  merge:       {'configured' if clients.merge else 'default (will error if no default)'}")
        print(f"  orchestrate: {'configured' if clients.orchestrate else 'default'}")

        # --- Merge with existing (additive) ---
        # configure_clients() merges, it doesn't replace.
        # Adding a merge client doesn't remove the chat client.
        t.configure_clients(merge=chat_client)
        clients = t.operation_clients
        print(f"\n  After adding merge client:")
        print(f"  chat:  {'configured' if clients.chat else 'default'} (preserved)")
        print(f"  merge: {'configured' if clients.merge else 'default'} (added)")

        t.close()  # does NOT close operation clients (they're yours)

    finally:
        # --- Lifecycle: you manage your clients ---
        # Tract.close() only closes clients it created internally
        # (via api_key= on open). Per-operation clients are user-provided,
        # so you close them yourself.
        chat_client.close()
        compress_client.close()
        print("\n=== Clients closed by caller ===")


if __name__ == "__main__":
    part3_per_operation_clients()
