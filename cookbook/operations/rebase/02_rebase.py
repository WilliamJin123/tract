"""Rebase

Update a stale branch to include the latest main by replaying
the branch's commits on top of the target's tip. Rebase gives
the branch everything main has plus its own work, with new
hashes (new parents) but same content.

Demonstrates: rebase(), RebaseResult, replayed_commits,
              original_commits, new_head, branch(), switch(),
              pprint(style="compact"), pprint(style="chat"),
              before/after state visualization
"""

import os

from dotenv import load_dotenv

from tract import Tract, CompiledContext

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
    # =================================================================
    # Part 2: rebase
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: REBASE")
    print("=" * 60)
    print()
    print("  Scenario: 'examples' branch started from an older main.")
    print("  Main has since advanced with new content. Rebase replays")
    print("  the examples branch's commits on top of main's current tip,")
    print("  so the branch includes everything main has plus its own work.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # Step 1: Build shared base
        print("\n  Step 1: Build shared base on main\n")

        t.system("You are a concise music theory tutor. One paragraph max.")
        t.chat("What are major and minor scales?")

        print("  main (shared base):")
        t.compile().pprint(style="compact")

        # Step 2: Branch and add feature work
        print("\n  Step 2: Branch 'examples' and add 2 LLM exchanges\n")

        t.branch("examples")
        r1 = t.chat("Give me 3 chord progression examples, from simple to complex.")
        r2 = t.chat("Now explain the emotional feel of each progression.")

        print("  examples (2 extra exchanges beyond shared base):")
        t.compile().pprint(style="compact")

        # Step 3: Main advances independently
        print("\n  Step 3: Main advances -- examples is now stale\n")

        t.switch("main")
        t.chat("What are modes? How do they differ from standard scales?")

        print("  main (has modes content examples branch doesn't):")
        t.compile().pprint(style="compact")

        # Step 4: Show the stale examples branch
        print("\n  Step 4: examples branch is missing main's latest\n")

        t.switch("examples")
        ctx_before = t.compile()

        print(f"  examples BEFORE rebase ({len(ctx_before.messages)} messages):")
        ctx_before.pprint(style="compact")
        print("\n  (notice: no modes content)")

        # Step 5: Rebase
        print("\n  Step 5: Rebase examples onto main\n")

        result = t.rebase("main")

        print(f"  Rebase complete:")
        print(f"    replayed: {len(result.replayed_commits)} commits")
        print(f"    new HEAD: {result.new_head[:8]}")
        print(f"    warnings: {len(result.warnings)}")

        # Show hash changes (new parentage -> new hashes)
        print(f"\n  Hash changes (same content, new lineage):")
        for orig, replayed in zip(result.original_commits, result.replayed_commits):
            print(f"    {orig.commit_hash[:8]} -> {replayed.commit_hash[:8]}")

        # Step 6: Show the result
        ctx_after = t.compile()

        print(f"\n  examples AFTER rebase ({len(ctx_after.messages)} messages):")
        ctx_after.pprint(style="chat")

        print("\n  The examples branch now includes main's modes")
        print("  content PLUS its own chord progression work.")
        print("  Commits got new hashes (new parents) but same content.")


if __name__ == "__main__":
    main()
