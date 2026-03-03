"""Message Reordering

PART 1 -- Manual           compile(order=), ReorderWarning, deterministic

Demonstrates: compile(order=), ReorderWarning, commit_hashes,
              pprint(style="chat")
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.small


def _build_conversation(t):
    """Helper: build a 3-exchange conversation and return hashes."""
    t.system("You are a concise nutrition expert.")
    t.chat("What are macronutrients?")
    t.chat("Explain intermittent fasting.")
    t.chat("What's the role of gut bacteria in digestion?")

    ctx = t.compile()
    return ctx, ctx.commit_hashes


# =============================================================================
# PART 1 -- Manual: compile(order=), ReorderWarning, deterministic
# =============================================================================

def main():
    print("=" * 60)
    print("PART 1 -- Manual: Message Reordering")
    print("=" * 60)
    print()
    print("  compile(order=) reorders messages by commit hash.")
    print("  Returns (CompiledContext, list[ReorderWarning]).")
    print("  Safety checks warn about structural issues like edits")
    print("  appearing before their targets.")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        ctx, hashes = _build_conversation(t)
        # [0]=system, [1]=r1_user, [2]=r1_asst, [3]=r2_user, [4]=r2_asst, [5]=r3_user, [6]=r3_asst

        print("\n  Original order:\n")
        ctx.pprint(style="chat")

        # Reorder: gut bacteria first, then macros, then fasting
        new_order = [
            hashes[0],              # system stays first
            hashes[5], hashes[6],   # r3: gut bacteria
            hashes[1], hashes[2],   # r1: macronutrients
            hashes[3], hashes[4],   # r2: fasting
        ]

        reordered, warnings = t.compile(order=new_order)

        print(f"\n  Reordered (gut bacteria first):\n")
        reordered.pprint(style="chat")

        print(f"\n  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    [{w.severity}] {w.warning_type}: {w.description}")

        if not warnings:
            print(f"    (none -- all APPEND-only, safe to reorder)")

        print(f"\n  Reordering rearranges the compiled context for better LLM")
        print(f"  flow without changing the underlying commit history.")


if __name__ == "__main__":
    main()
