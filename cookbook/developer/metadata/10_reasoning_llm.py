"""LLM Integration

Three tiers: manual reasoning commits, interactive approval of generated
reasoning, and fully autonomous LLM reasoning extraction.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: t.reasoning() manual, generate() with reasoning,
              ChatResponse.reasoning, ChatResponse.reasoning_commit,
              generate(reasoning=False), click.confirm(),
              Tract.open(commit_reasoning=False)
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


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
# Part 2: Interactive Reasoning Approval  (PART 2 — Interactive)
# =============================================================================

def part2_interactive():
    """After generate(), confirm whether to keep reasoning."""
    if not TRACT_OPENAI_API_KEY:
        print("=" * 60)
        print("Part 2: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print("=" * 60)
    print("Part 2: INTERACTIVE REASONING APPROVAL  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  After generate(), inspect reasoning and decide whether to")
    print("  keep it committed. Show generate(reasoning=False) as opt-out.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("Think step by step before answering.")
        t.user("What is 23 * 17?")

        response = t.generate(reasoning_effort="high")

        if response.reasoning:
            print(f"  Reasoning extracted ({len(response.reasoning)} chars):")
            preview = response.reasoning[:120].replace("\n", " ")
            print(f"    {preview}...\n")

            if click.confirm("  Keep this reasoning committed?", default=True):
                print("  -> reasoning already committed by generate()")
            else:
                print("  -> To skip reasoning, use generate(reasoning=False)")
                print("     The reasoning text is still in ChatResponse.reasoning")
                print("     but won't be committed to history.")
        else:
            print("  (Model did not produce reasoning tokens)")

        print(f"\n  Answer: {response.text[:80]}")

    print()


# =============================================================================
# Part 3: Autonomous LLM Reasoning  (PART 3 — LLM / Agent)
# =============================================================================

def part3_llm_integration():
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("Part 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("Part 3: AUTONOMOUS LLM REASONING  [Agent Tier]")
    print("=" * 60)
    print()

    # --- 3a: generate() with reasoning ---

    print("  3a: generate() auto-commits reasoning traces\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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
    part2_interactive()
    part3_llm_integration()
    print("=" * 60)
    print("Done -- all 3 tiers of LLM reasoning integration demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
