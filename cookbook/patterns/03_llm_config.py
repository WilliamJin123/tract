"""LLM Configuration Patterns

Four patterns for controlling LLM settings in tract, from quick per-call
tweaks to multi-client routing and the full resolution chain.

Part 1: Per-call config — sugar params + LLMConfig objects
Part 2: Defaults and per-operation config — tract-level defaults, operation overrides
Part 3: Per-operation clients — route different operations to different LLM services
Part 4: The full resolution chain + cross-framework config translation

The 4-level chain resolves each field independently:
  1. Sugar params (temperature=, model=) — highest priority
  2. llm_config= (LLMConfig per call)
  3. Operation config (via configure_operations)
  4. Tract default (via default_config= on open)

Every resolved config is auto-captured on assistant commits for provenance.

Demonstrates: LLMConfig, sugar params, generate() two-step, default_config=,
              configure_operations(), OperationConfigs, configure_clients(),
              OperationClients, OpenAIClient, 4-level resolution,
              LLMConfig.from_dict(), cross-framework aliases,
              generation_config provenance, query_by_config(),
              response.pprint(), print(entry) in log loop
"""

import os
import sys

from dotenv import load_dotenv

from tract import LLMConfig, OperationClients, OperationConfigs, Tract
from tract.llm import OpenAIClient

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: Per-Call Config (sugar params + LLMConfig)
# =============================================================================
# Override model, temperature, or any LLM setting for a single call —
# without changing your defaults. Two styles: sugar params for quick tweaks,
# LLMConfig for full control.

def part1_per_call_config():
    print("=" * 60)
    print("Part 1: PER-CALL CONFIG")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")

        # --- Style 1: Sugar params ---
        # Pass temperature=, model=, max_tokens= directly on chat().
        # Quick and readable for one-off tweaks.

        print("=== Sugar params ===\n")

        response = t.chat("What is Python?", temperature=0.2)
        gc = response.generation_config
        # Keep the field-level prints — they teach what's resolved where
        print(f"  temperature={gc.temperature}, model={gc.model}")
        # pprint() shows the full response panel (text + usage + config)
        response.pprint()
        print()

        # --- Style 2: LLMConfig object ---
        # For more settings or when you want to pass config around.

        print("=== LLMConfig object ===\n")

        creative = LLMConfig(temperature=0.9, top_p=0.95)
        response = t.chat("Give a creative analogy for Python.", llm_config=creative)
        gc = response.generation_config
        print(f"  temperature={gc.temperature}, top_p={gc.top_p}")
        print(f"  Response: {response.text[:100]}...\n")

        # --- Style 3: Both (sugar wins) ---
        # If you pass both llm_config= and a sugar param, sugar wins
        # for that specific field.

        print("=== Sugar overrides LLMConfig ===\n")

        response = t.chat(
            "Explain decorators.",
            llm_config=LLMConfig(temperature=0.3, max_tokens=200),
            temperature=0.8,  # overrides the 0.3 from llm_config
        )
        gc = response.generation_config
        print(f"  temperature={gc.temperature} (sugar won over 0.3)")
        print(f"  max_tokens={gc.max_tokens} (from llm_config)")
        print(f"  Response: {response.text[:100]}...\n")

        # --- generate(): two-step control ---
        # Commit the user message yourself, then call generate() separately.
        # Same config params, but you choose when to call.

        print("=== generate() two-step ===\n")

        t.user("What is the GIL?")
        response = t.generate(temperature=0.1, max_tokens=100)
        print(f"  temperature={response.generation_config.temperature}")
        print(f"  Response: {response.text[:100]}...")
        # Note: response.pprint() would show the same info as a rich panel


# =============================================================================
# Part 2: Defaults and Per-Operation Config
# =============================================================================
# Set a tract-level default (every call inherits it), then override per
# operation so chat is creative and compression is deterministic.

def part2_defaults_and_operations():
    print(f"\n{'=' * 60}")
    print("Part 2: DEFAULTS AND PER-OPERATION CONFIG")
    print("=" * 60)
    print()

    # --- Tract-level default ---
    # All operations inherit this unless overridden.
    tract_default = LLMConfig(
        model=MODEL_ID,
        temperature=0.5,
        top_p=0.95,
    )

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        default_config=tract_default,
    ) as t:
        t.system("You are a helpful assistant.")

        # --- Per-operation overrides ---
        # Chat gets higher temperature for creativity.
        # Compress gets low temperature + fixed seed for determinism.
        # Merge and orchestrate inherit the tract default.
        t.configure_operations(
            chat=LLMConfig(temperature=0.8),
            compress=LLMConfig(temperature=0.1, seed=42),
        )

        # Equivalent using the typed OperationConfigs dataclass:
        #   t.configure_operations(OperationConfigs(
        #       chat=LLMConfig(temperature=0.8),
        #       compress=LLMConfig(temperature=0.1, seed=42),
        #   ))

        # --- See it in action ---
        print("=== Chat: creative (temp=0.8 from operation config) ===\n")
        response = t.chat("Invent a metaphor for Python's import system.")
        gc = response.generation_config
        # Keep the field-level prints — they teach the resolution chain explicitly
        print(f"  model={gc.model}")
        print(f"  temperature={gc.temperature}  (operation override)")
        print(f"  top_p={gc.top_p}  (inherited from tract default)")
        # pprint() combines text + usage + resolved config in one panel
        response.pprint()
        print()

        # --- Inspect the operation configs ---
        print("=== Current operation configs ===\n")
        ops = t.operation_configs
        print(f"  chat:        {ops.chat}")
        print(f"  compress:    {ops.compress}")
        print(f"  merge:       {ops.merge}  (None = use default)")
        print(f"  orchestrate: {ops.orchestrate}  (None = use default)")

        # --- Per-call still overrides everything ---
        print("\n=== Per-call override (beats operation config) ===\n")
        response = t.chat("What is asyncio?", temperature=0.1)
        gc = response.generation_config
        print(f"  temperature={gc.temperature}  (per-call sugar won over 0.8)")
        print(f"  Response: {response.text[:120]}...")
        # Note: response.pprint() would show the full panel with all resolved fields


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


# =============================================================================
# Part 4: The Full Resolution Chain + Cross-Framework Config
# =============================================================================
# Demonstrates all 4 levels firing together, and how configs from other
# frameworks (OpenAI, Anthropic) are auto-translated via LLMConfig.from_dict().

def part4_resolution_chain():
    print(f"\n{'=' * 60}")
    print("Part 4: FULL RESOLUTION CHAIN + CROSS-FRAMEWORK CONFIG")
    print("=" * 60)
    print()

    # Ensure Unicode output works on Windows consoles (cp1252 can't encode
    # characters like \u2011 that LLMs may return).
    sys.stdout.reconfigure(encoding="utf-8")

    # --- Set up all 4 levels ---
    tract_default = LLMConfig(
        model=MODEL_ID,
        temperature=0.5,
        top_p=0.95,
    )

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        default_config=tract_default,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")

        # Level 3: operation config
        t.configure_operations(
            chat=LLMConfig(temperature=0.8),
        )

        # --- Call 1: Default + operation resolution ---
        print("=== Call 1: Default + operation resolution ===")
        print("  Level 4 (tract default): model=gpt-oss-120b, temp=0.5, top_p=0.95")
        print("  Level 3 (chat operation): temp=0.8 (overrides default)")
        print("  Effective: model=gpt-oss-120b, temp=0.8, top_p=0.95\n")

        response = t.chat("What is Python's GIL?")
        gc = response.generation_config
        print(f"  Captured: model={gc.model}, temp={gc.temperature}, top_p={gc.top_p}")
        print(f"  Response: {response.text[:120]}...\n")

        # --- Call 2: llm_config= override ---
        print("=== Call 2: llm_config= override ===")
        precise = LLMConfig(temperature=0.2, seed=123)
        print("  Level 3 (chat operation): temp=0.8")
        print("  Level 2 (llm_config=): temp=0.2, seed=123 (overrides operation)")
        print("  Effective: model=gpt-oss-120b, temp=0.2, top_p=0.95, seed=123\n")

        response = t.chat("Explain it in one sentence.", llm_config=precise)
        gc = response.generation_config
        print(f"  Captured: model={gc.model}, temp={gc.temperature}, seed={gc.seed}")
        print(f"  Response: {response.text[:120]}...\n")

        # --- Call 3: Sugar param (highest priority) ---
        print("=== Call 3: Sugar param override ===")
        print("  Level 2 (llm_config=): temp=0.2, seed=123")
        print("  Level 1 (sugar): temperature=0.9 (beats llm_config)")
        print("  Effective: model=gpt-oss-120b, temp=0.9, top_p=0.95, seed=123\n")

        response = t.chat(
            "Give a creative analogy for the GIL.",
            llm_config=precise,
            temperature=0.9,  # sugar beats llm_config for this field
        )
        gc = response.generation_config
        print(f"  Captured: model={gc.model}, temp={gc.temperature}, seed={gc.seed}")
        print(f"  Response: {response.text[:120]}...\n")

        # --- LLMConfig.from_dict(): Cross-framework aliases ---
        print("=== LLMConfig.from_dict(): Cross-framework aliases ===\n")

        # Config from an OpenAI-style dict
        openai_params = {
            "model": MODEL_ID,
            "temperature": 0.3,
            "max_completion_tokens": 200,  # alias -> max_tokens
            "stop": ["\n\n"],             # alias -> stop_sequences
            "messages": [...],            # API plumbing — auto-ignored
        }

        config = LLMConfig.from_dict(openai_params)
        print(f"  Input: max_completion_tokens=200, stop=['\\n\\n'], messages=[...]")
        print(f"  Parsed: max_tokens={config.max_tokens}, "
              f"stop_sequences={config.stop_sequences}")
        print(f"  (messages was auto-ignored as API plumbing)\n")

        response = t.chat("Summarize the GIL in 2 sentences.", llm_config=config)
        gc = response.generation_config
        print(f"  Captured: model={gc.model}, temp={gc.temperature}, max_tokens={gc.max_tokens}")
        # pprint() for the final call — shows resolved config + usage together
        response.pprint()
        print()

        # --- Provenance: every config is captured ---
        print("=== Generation configs across all calls ===\n")
        history = t.log(limit=20)
        for entry in reversed(history):
            if entry.generation_config:
                # print(entry) gives "hash message" — compact for the provenance loop
                print(f"  {entry} | {entry.generation_config.to_dict()}")


def main():
    part1_per_call_config()
    part2_defaults_and_operations()
    part3_per_operation_clients()
    part4_resolution_chain()


if __name__ == "__main__":
    main()
