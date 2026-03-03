"""Reasoning formatting and LLM integration.

Two tiers: deterministic formatting patterns and LLM-powered reasoning
extraction via generate().

  Part 1 -- Formatting:     pprint() styles for reasoning, to_dicts()/to_openai()
  Part 2 -- LLM Integration: generate() with reasoning, ChatResponse.reasoning,
                              reasoning=False, Tract.open(commit_reasoning=False)

Demonstrates: pprint() reasoning style (table, chat, compact),
              compile(include_reasoning=True), to_dicts(), to_openai(),
              generate() with reasoning, ChatResponse.reasoning,
              ChatResponse.reasoning_commit, reasoning=False,
              Tract.open(commit_reasoning=False)
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


# =====================================================================
# Part 1 -- Formatting
# =====================================================================

def part1_formatting():
    """pprint styles and format output for reasoning content."""
    print(f"\n{'=' * 60}")
    print("Part 1: FORMATTING")
    print("=" * 60)
    print()
    print("  Reasoning commits render in dim cyan, visually")
    print("  distinct from regular dialogue.\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What is the capital of France?")
    t.reasoning(
        "The user is asking about France's capital. This is a "
        "straightforward geography question. The answer is Paris."
    )
    t.assistant("The capital of France is Paris.")

    # Include reasoning so pprint() can show it
    ctx = t.compile(include_reasoning=True)

    print("  --- table style ---\n")
    ctx.pprint(style="table")

    print("\n  --- chat style ---\n")
    ctx.pprint(style="chat")

    print("\n  --- compact style ---\n")
    ctx.pprint(style="compact")

    t.close()


def part1b_format_output():
    """to_dicts() and to_openai() with reasoning content."""
    print("=" * 60)
    print("Part 1b: FORMAT OUTPUT FOR APIS")
    print("=" * 60)
    print()
    print("  to_dicts() and to_openai() include reasoning when compiled")
    print("  with include_reasoning=True. This is what agents consume.\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What is 2 + 2?")
    t.reasoning("Simple arithmetic: 2 + 2 = 4.")
    t.assistant("2 + 2 = 4.")

    # Without reasoning
    ctx_no = t.compile()
    dicts_no = ctx_no.to_dicts()
    print(f"  to_dicts() without reasoning: {len(dicts_no)} messages")
    for d in dicts_no:
        print(f"    [{d['role']}] {d['content'][:50]}")

    # With reasoning
    ctx_yes = t.compile(include_reasoning=True)
    dicts_yes = ctx_yes.to_dicts()
    print(f"\n  to_dicts() with reasoning: {len(dicts_yes)} messages")
    for d in dicts_yes:
        print(f"    [{d['role']}] {d['content'][:50]}")

    print()
    t.close()


# =====================================================================
# Part 2 -- LLM Integration
# =====================================================================

def part2_llm_integration():
    """generate() with reasoning extraction and commit control."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("Part 2: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("Part 2: LLM REASONING VIA GENERATE()")
    print("=" * 60)
    print()

    # --- 2a: generate() with reasoning ---

    print("  2a: generate() auto-commits reasoning traces\n")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("Think step by step before answering. Make sure your reasoning is thorough and clear, but your answers are concise")
        t.user("I want to wash my car and the car wash is 50 meters close. Should I drive there or walk?")

        resp = t.generate(reasoning_effort="high")

        if resp.reasoning_commit:
            print(f"  reasoning_commit hash: {resp.reasoning_commit.commit_hash[:8]}")
            print(f"  reasoning_commit type: {resp.reasoning_commit.content_type}")
        else:
            print("  (Model did not produce reasoning tokens)")

        # Compile with include_reasoning=True to see reasoning in pprint
        print(f"\n  compile(include_reasoning=True):\n")
        ctx = t.compile(include_reasoning=True)
        ctx.pprint(style="chat")

    # --- 2b: Per-call opt-out ---

    print(f"\n  2b: reasoning=False skips the commit\n")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("Think step by step.")
        t.user("What is 7 * 8?")

        resp = t.generate(reasoning=False, reasoning_effort="high")

        # Reasoning text is still extracted (if available)...
        print(f"  reasoning extracted: {resp.reasoning is not None}")
        # ...but NOT committed
        print(f"  reasoning committed: {resp.reasoning_commit is not None}")

        log_types = [e.content_type for e in t.log()]
        print(f"  content types in log: {log_types}")
        print(f"  'reasoning' in log: {'reasoning' in log_types}")

    # --- 2c: Global opt-out ---

    print(f"\n  2c: Tract.open(commit_reasoning=False) disables globally\n")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        commit_reasoning=False,
    ) as t:
        t.system("Think carefully.")
        t.user("What is 12 + 13?")

        resp = t.generate(reasoning_effort="high")

        print(f"  reasoning extracted: {resp.reasoning is not None}")
        print(f"  reasoning committed: {resp.reasoning_commit is not None}")
        print(f"  (t.reasoning() shorthand still works even with global opt-out)")

        # Manual reasoning is always allowed
        manual = t.reasoning("This was added manually.")
        print(f"  manual commit type:  {manual.content_type}")


# =====================================================================
# Main
# =====================================================================

def main():
    part1_formatting()
    part1b_format_output()
    part2_llm_integration()
    print("=" * 60)
    print("Done -- reasoning formatting and LLM integration demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
