"""Per-Call Config (sugar params + LLMConfig)

Override model, temperature, or any LLM setting for a single call —
without changing your defaults. Two styles: sugar params for quick tweaks,
LLMConfig for full control.

The 4-level chain resolves each field independently:
  1. Sugar params (temperature=, model=) — highest priority
  2. llm_config= (LLMConfig per call)
  3. Operation config (via configure_operations)
  4. Tract default (via default_config= on open)

Every resolved config is auto-captured on assistant commits for provenance.

Demonstrates: LLMConfig, sugar params, generate() two-step,
              generation_config provenance, response.pprint()
"""

import os

from dotenv import load_dotenv

from tract import LLMConfig, Tract

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


if __name__ == "__main__":
    part1_per_call_config()
