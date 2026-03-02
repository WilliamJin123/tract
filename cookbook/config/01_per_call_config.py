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
              generation_config provenance, response.pprint(),
              ToolConfig description overrides for reliable tool selection
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
        gc = response.generation_config
        print(f"  temperature={gc.temperature}, max_tokens={gc.max_tokens}")
        response.pprint()


# =============================================================================
# Part 2 -- Agent: LLM Self-Configures via configure_model Tool
# =============================================================================
# The LLM receives only configure_model + status tools (custom ToolProfile)
# and autonomously decides to call configure_model to set its own temperature.
# We use generate() for the agentic loop — no custom LLM calls needed.
#
# The trick: ToolConfig description overrides tell the model *when* to call
# each tool. Without explicit trigger conditions in the description, models
# won't reliably self-select tools — they know what configure_model *does*
# but not that they should call it *before* answering creative vs. factual
# tasks. This scales to 20+ tools: each tool carries its own trigger signal.

def part2_agent():
    print(f"\n{'=' * 60}")
    print("PART 2 -- Agent: LLM SELF-CONFIGURES VIA TOOLKIT")
    print("=" * 60)
    print()

    from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

    # Custom profile: only expose config + status tools — the LLM can
    # change its own temperature but we handle the actual chat calls.
    #
    # Key insight: description overrides tell the LLM *when* to use a tool,
    # not just *what* it does. Without this, models won't reliably self-select
    # tools — especially in profiles with many tools. This scales to 20+ tools
    # because each tool carries its own trigger signal in the description,
    # no centralized system prompt needed.
    config_profile = ToolProfile(
        name="config-only",
        tool_configs={
            "configure_model": ToolConfig(
                enabled=True,
                description=(
                    "Set temperature BEFORE answering. Call this when the task "
                    "is creative (temperature 0.7-1.0) or requires precision "
                    "(temperature 0.0-0.3)."
                ),
            ),
            "status": ToolConfig(enabled=True),
        },
    )

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=config_profile)
        t.set_tools(tools)

        t.system("You are a helpful assistant.")

        # Ask tasks one at a time so each gets its own temperature.
        # The LLM can only return tool_calls OR text per turn — not
        # interleaved — so bundling both tasks would produce one response
        # at whatever the last configured temperature was.
        tasks = [
            "Write a surreal, creative, one-sentence poem about a clock that melts.",
            "What is the speed of light in meters per second?",
        ]

        for task in tasks:
            t.user(task)
            for turn in range(5):
                response = t.generate()

                if not response.tool_calls:
                    # Text response — print and move to next task
                    gc = response.generation_config
                    print(f"  [{gc.temperature or 'default'}] {response.text[:120]}...")
                    break

                for tc in response.tool_calls:
                    result = executor.execute(tc.name, tc.arguments)
                    t.tool_result(tc.id, tc.name, str(result))
                    print(f"  LLM called: {tc.name}({tc.arguments})")
                    print(f"    -> {result.output[:100]}")

        # --- Inspect the full conversation the agent built ---
        print(f"\n{'=' * 60}")
        print("COMPILED CONTEXT")
        print("=" * 60)
        ctx = t.compile()
        print(f"  {len(ctx.messages)} messages, {ctx.token_count} tokens\n")
        ctx.pprint(style="chat")

        # --- Show generation_config provenance per commit ---
        print(f"\n{'=' * 60}")
        print("GENERATION CONFIG PROVENANCE")
        print("=" * 60)
        for ci in t.log():
            if ci.generation_config:
                gc = ci.generation_config
                print(f"  {ci.commit_hash[:8]}  temp={gc.temperature}  "
                      f"({ci.content_type})")


def main():
    part1_per_call_config()
    part2_agent()


if __name__ == "__main__":
    main()
