"""Message Reordering

compile(order=) reorders messages by commit hash. Returns
(CompiledContext, list[ReorderWarning]). Safety checks warn about
structural issues like edits appearing before their targets.

Demonstrates: compile(order=), ReorderWarning, commit_hashes,
              pprint(style="chat"), reorder safety checks
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


def main():
    # =================================================================
    # Part 3: Message reordering with compile(order=)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 3: MESSAGE REORDERING (compile(order=))")
    print("=" * 60)
    print()
    print("  compile(order=) reorders messages by commit hash.")
    print("  Returns (CompiledContext, list[ReorderWarning]).")
    print("  Safety checks warn about structural issues like edits")
    print("  appearing before their targets.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a concise nutrition expert.")
        r1 = t.chat("What are macronutrients?")
        r2 = t.chat("Explain intermittent fasting.")
        r3 = t.chat("What's the role of gut bacteria in digestion?")

        print("\n  Original order:\n")
        ctx = t.compile()
        ctx.pprint(style="chat")

        # Get commit hashes (parallel to messages)
        hashes = ctx.commit_hashes
        # [0]=system, [1]=r1_user, [2]=r1_asst, [3]=r2_user, [4]=r2_asst, [5]=r3_user, [6]=r3_asst

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
