"""Defaults and Per-Operation Config

Set a tract-level default (every call inherits it), then override per
operation so chat is creative and compression is deterministic.

Demonstrates: default_config=, configure_operations(), OperationConfigs,
              per-operation LLMConfig, field-level inheritance
"""

import os

from dotenv import load_dotenv

from tract import LLMConfig, OperationConfigs, Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]


def main():
    # --- Tract-level default ---
    # All operations inherit this unless overridden.
    tract_default = LLMConfig(
        model="gpt-oss-120b",
        temperature=0.5,
        top_p=0.95,
    )

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
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
        print(f"  model={gc.model}")
        print(f"  temperature={gc.temperature}  (operation override)")
        print(f"  top_p={gc.top_p}  (inherited from tract default)")
        print(f"  Response: {response.text[:120]}...\n")

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


if __name__ == "__main__":
    main()
