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
# Part 1 -- Manual: Per-Call Config (sugar params + LLMConfig)
# =============================================================================
# Override model, temperature, or any LLM setting for a single call —
# without changing your defaults. Two styles: sugar params for quick tweaks,
# LLMConfig for full control.

def part1_per_call_config():
    print("=" * 60)
    print("PART 1 -- Manual: PER-CALL CONFIG")
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
# Part 2 -- Interactive: Config Presets (developer-defined)
# =============================================================================
# Define reusable config presets in code, run the same question with each,
# and compare the outputs and captured provenance side-by-side.

def part2_interactive():
    """Part 2: Interactive -- compare config presets side-by-side."""
    print("=" * 60)
    print("PART 2 -- Interactive: Config Presets (Compare Side-by-Side)")
    print("=" * 60)
    print()

    # Define presets — a developer tweaks these in code, not at runtime
    PRESETS = {
        "precise":  LLMConfig(temperature=0.1, max_tokens=150),
        "balanced": LLMConfig(temperature=0.5, max_tokens=300),
        "creative": LLMConfig(temperature=0.9, max_tokens=300, top_p=0.95),
    }

    question = "What is a good analogy for how a database index works?"

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")

        for name, config in PRESETS.items():
            response = t.chat(question, llm_config=config)
            gc = response.generation_config
            print(f"  [{name}] temp={gc.temperature}, max_tokens={gc.max_tokens}")
            print(f"    {response.text[:120]}...")
            print()


# =============================================================================
# Part 3 -- Agent: Self-Configuring via configure_model Tool
# =============================================================================
# The agent uses the built-in configure_model tool to change its own
# temperature between calls. High temp for creative work, low for factual.

def part3_agent():
    print(f"\n{'=' * 60}")
    print("PART 3 -- Agent: SELF-CONFIGURING VIA TOOLKIT")
    print("=" * 60)
    print()

    from tract.toolkit import ToolExecutor

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        executor = ToolExecutor(t)

        # Agent checks its baseline config
        status = executor.execute("status", {})
        print(f"  Baseline config:\n{status}\n")

        # Agent sets high temperature for creative work
        result = executor.execute("configure_model", {"temperature": 0.95})
        print(f"  configure_model(temperature=0.95): {result}\n")

        r1 = t.chat("Write a surreal one-sentence poem about a clock that melts.")
        gc1 = r1.generation_config
        print(f"  Creative call — captured temp={gc1.temperature}")
        print(f"    {r1.text[:120]}...\n")

        # Agent switches to low temperature for factual work
        result = executor.execute("configure_model", {"temperature": 0.1})
        print(f"  configure_model(temperature=0.1): {result}\n")

        r2 = t.chat("What is the speed of light in meters per second?")
        gc2 = r2.generation_config
        print(f"  Factual call — captured temp={gc2.temperature}")
        print(f"    {r2.text[:120]}...\n")

        # Both responses live in the same tract with different generation_configs
        print(f"  Temperature changed: {gc1.temperature} -> {gc2.temperature}")


def main():
    part1_per_call_config()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
