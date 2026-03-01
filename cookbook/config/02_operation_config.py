"""Defaults and Per-Operation Config

Set a tract-level default (every call inherits it), then override per
operation so chat is creative and compression is deterministic.

The 4-level chain resolves each field independently:
  1. Sugar params (temperature=, model=) — highest priority
  2. llm_config= (LLMConfig per call)
  3. Operation config (via configure_operations)
  4. Tract default (via default_config= on open)

Every resolved config is auto-captured on assistant commits for provenance.

Demonstrates: LLMConfig, default_config=, configure_operations(),
              OperationConfigs, per-call override, response.pprint()
"""

import os

from dotenv import load_dotenv

from tract import LLMConfig, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


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


if __name__ == "__main__":
    part2_defaults_and_operations()
