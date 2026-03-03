"""LLM Integration

Two tiers: manual reasoning commits and LLM-powered reasoning
extraction via generate().

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 3 -- LLM-Powered       generate() with reasoning extraction

Demonstrates: t.reasoning() manual, generate() with reasoning,
              ChatResponse.reasoning, ChatResponse.reasoning_commit,
              generate(reasoning=False), Tract.open(commit_reasoning=False)
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =============================================================================
# Part 1: Manual Reasoning  (PART 1 — Manual)
# =============================================================================

def part1_manual_reasoning():
    """Commit reasoning manually — no LLM needed."""
    print("=" * 60)
    print("Part 1: MANUAL REASONING COMMITS  [Manual Tier]")
    print("=" * 60)
    print()
    print("  t.reasoning() commits chain-of-thought text without any LLM.")
    print("  It appears in log() but not in compile() by default.")
    print()

    t = Tract.open()
    t.system("You are a math tutor.")
    t.user("What is 15 * 13?")

    # Manual reasoning commit
    r_info = t.reasoning(
        "Let me think step by step about this problem...\n"
        "15 * 13 = 15 * 10 + 15 * 3 = 150 + 45 = 195",
        format="parsed",
    )
    t.assistant("15 x 13 = 195")

    # Reasoning is in log()
    print("  log() shows reasoning:")
    for entry in reversed(t.log()):
        print(f"    {entry}")

    # But excluded from compile()
    ctx = t.compile()
    print(f"\n  compile() -> {ctx.commit_count} messages (reasoning excluded)")
    for msg in ctx.messages:
        print(f"    [{msg.role}] {msg.content[:60]}")

    print()
    t.close()


# =============================================================================
# Part 3: LLM Reasoning via generate()
# =============================================================================

def part3_llm_integration():
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("Part 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("Part 3: LLM REASONING VIA GENERATE()  [LLM-Powered]")
    print("=" * 60)
    print()

    # --- 3a: generate() with reasoning ---

    print("  3a: generate() auto-commits reasoning traces\n")

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

    # --- 3b: Per-call opt-out ---

    print(f"\n  3b: reasoning=False skips the commit\n")

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

    # --- 3c: Global opt-out ---

    print(f"\n  3c: Tract.open(commit_reasoning=False) disables globally\n")

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


# =============================================================================
# Main
# =============================================================================

def main():
    part1_manual_reasoning()
    part3_llm_integration()
    print("=" * 60)
    print("Done -- both parts of LLM reasoning integration demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
