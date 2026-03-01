"""Full Resolution Chain + Cross-Framework Config

Demonstrates all 4 levels firing together, and how configs from other
frameworks (OpenAI, Anthropic) are auto-translated via LLMConfig.from_dict().

The 4-level chain resolves each field independently:
  1. Sugar params (temperature=, model=) — highest priority
  2. llm_config= (LLMConfig per call)
  3. Operation config (via configure_operations)
  4. Tract default (via default_config= on open)

Every resolved config is auto-captured on assistant commits for provenance.

Demonstrates: LLMConfig, 4-level resolution, LLMConfig.from_dict(),
              cross-framework aliases, generation_config provenance,
              query_by_config(), response.pprint(), print(entry) in log loop
"""

import os
import sys

from dotenv import load_dotenv

from tract import LLMConfig, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


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


if __name__ == "__main__":
    part4_resolution_chain()
