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

import click
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
# Part 2 -- Interactive: Prompt user for config before each call
# =============================================================================
# Show the user what config will be used, let them tweak temperature and
# max_tokens via click prompts, and confirm before sending.

def part2_interactive():
    """Part 2: Interactive -- prompt for temperature/max_tokens before chat."""
    print("=" * 60)
    print("PART 2 -- Interactive: Prompt for Config Before Each Call")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")

        question = click.prompt("\n  Your question", default="What is Python?")
        temp = click.prompt("  Temperature", type=float, default=0.7)
        max_tok = click.prompt("  Max tokens", type=int, default=300)

        config = LLMConfig(temperature=temp, max_tokens=max_tok)
        print(f"\n  Resolved config: temperature={temp}, max_tokens={max_tok}")

        if click.confirm("  Use this config?", default=True):
            response = t.chat(question, llm_config=config)
            gc = response.generation_config
            print(f"\n  Captured: temp={gc.temperature}, max_tokens={gc.max_tokens}")
            response.pprint()
        else:
            print("  Skipped -- no call made.")


# =============================================================================
# Part 3 -- Agent: Self-Configuring via Toolkit
# =============================================================================
# Agents introspect their own generation_config via status() and adjust
# temperature/max_tokens per-call using the toolkit.

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

        # Agent checks its current config via status()
        status = executor.execute("status", {})
        print(f"  Agent sees its own config via status():\n{status}\n")

        # Agent makes a creative call with high temperature
        r = t.chat("Invent a metaphor for recursion.", temperature=0.9)
        print(f"  Creative call (temp=0.9): {r.text[:80]}...")

        # Agent makes a precise call with low temperature
        r = t.chat("Define recursion in one sentence.", temperature=0.1)
        print(f"  Precise call (temp=0.1): {r.text[:80]}...")

    # Note: Agents introspect their own generation_config via status()
    # and adjust temperature/max_tokens per-call using the toolkit.
    # The toolkit's status tool surfaces the resolved config so agents
    # can reason about what settings are active before each call.


def main():
    part1_per_call_config()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
