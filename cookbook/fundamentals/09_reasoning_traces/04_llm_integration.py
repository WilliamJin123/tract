"""LLM Integration

Part 4 of Reasoning Traces: generate() auto-extracts reasoning from
provider responses (Cerebras, OpenAI o1/o3, Anthropic thinking, <think>
tags). Auto-committed before the assistant response. Per-call and global
opt-out available. Requires an LLM — skips if no API key.

Demonstrates: generate() with reasoning, ChatResponse.reasoning,
              ChatResponse.reasoning_commit, generate(reasoning=False),
              Tract.open(commit_reasoning=False), t.reasoning() manual
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


def part4_llm_integration():
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("Part 4: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("Part 4: LLM INTEGRATION (auto-extract reasoning)")
    print("=" * 60)
    print()

    # --- 4a: generate() with reasoning ---

    print("  4a: generate() auto-commits reasoning traces\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("Think step by step before answering. Make sure your reasoning is thorough and clear, but your answers are concise")
        t.user("I want to wash my car and the car wash is 50 meters close. Should I drive there or walk?")

        resp = t.generate()

        print(f"  ChatResponse.text:     {resp.text[:80]}...")
        print(f"  ChatResponse.reasoning: {repr(resp.reasoning)[:80] if resp.reasoning else 'None'}")

        if resp.reasoning_commit:
            print(f"  reasoning_commit hash: {resp.reasoning_commit.commit_hash[:8]}")
            print(f"  reasoning_commit type: {resp.reasoning_commit.content_type}")
        else:
            print("  (Model did not produce reasoning tokens)")

        # Compile with include_reasoning=True to see reasoning in pprint
        print(f"\n  compile(include_reasoning=True):\n")
        ctx = t.compile(include_reasoning=True)
        ctx.pprint(style="chat")

    # --- 4b: Per-call opt-out ---

    print(f"\n  4b: reasoning=False skips the commit\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("Think step by step.")
        t.user("What is 7 * 8?")

        resp = t.generate(reasoning=False)

        # Reasoning text is still extracted (if available)...
        print(f"  reasoning extracted: {resp.reasoning is not None}")
        # ...but NOT committed
        print(f"  reasoning committed: {resp.reasoning_commit is not None}")

        log_types = [e.content_type for e in t.log()]
        print(f"  content types in log: {log_types}")
        print(f"  'reasoning' in log: {'reasoning' in log_types}")

    # --- 4c: Global opt-out ---

    print(f"\n  4c: Tract.open(commit_reasoning=False) disables globally\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
        commit_reasoning=False,
    ) as t:
        t.system("Think carefully.")
        t.user("What is 12 + 13?")

        resp = t.generate()

        print(f"  reasoning extracted: {resp.reasoning is not None}")
        print(f"  reasoning committed: {resp.reasoning_commit is not None}")
        print(f"  (t.reasoning() shorthand still works even with global opt-out)")

        # Manual reasoning is always allowed
        manual = t.reasoning("This was added manually.")
        print(f"  manual commit type:  {manual.content_type}")


if __name__ == "__main__":
    part4_llm_integration()
