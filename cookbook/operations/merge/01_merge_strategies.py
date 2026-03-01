"""Merge Strategies

Two merge modes — fast-forward (linear history) and clean merge (diverged
but no edit conflicts) — plus a bonus showing no_ff and auto-delete.

Demonstrates: merge(), merge_type, MergeResult, committed,
              merge_commit_hash, no_ff, delete_branch=True,
              pprint(style="compact"), MergeResult.pprint()
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 2: Merge Strategies
# =============================================================================

def part2_merge_strategies():
    print("=" * 60)
    print("Part 2: MERGE STRATEGIES")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Scenario 1: Fast-Forward
    # -------------------------------------------------------------------------

    print()
    print("Scenario 1: FAST-FORWARD")
    print("-" * 40)
    print()
    print("  Main hasn't moved since branching. Merge just slides")
    print("  main's pointer forward to feature's tip.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Keep answers to one sentence.")
        t.chat("What is recursion?")

        print(f"\n  BEFORE MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

        # Branch and add commits only on feature
        t.branch("feature")
        t.chat("Give an example of a base case in recursion.")

        print(f"\n  feature:")
        t.compile().pprint(style="compact")

        # Merge
        t.switch("main")
        result = t.merge("feature")

        result.pprint()

        print(f"  AFTER MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

    # -------------------------------------------------------------------------
    # Scenario 2: Clean Merge (diverged, APPEND-only)
    # -------------------------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("Scenario 2: CLEAN MERGE")
    print("-" * 40)
    print()
    print("  Both branches diverged with new messages, but nobody edited")
    print("  existing ones. All APPENDs -- Tract auto-merges cleanly.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Keep answers to one sentence.")
        t.chat("What is a linked list?")

        # Feature: stacks
        t.branch("feature")
        t.chat("What is a stack?")

        # Main: queues
        t.switch("main")
        t.chat("What is a queue?")

        print(f"\n  BEFORE MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

        t.switch("feature")
        print(f"\n  feature:")
        t.compile().pprint(style="compact")

        # Merge
        t.switch("main")
        result = t.merge("feature")

        result.pprint()

        print(f"  AFTER MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

    # -------------------------------------------------------------------------
    # Bonus: no_ff and delete_branch=True
    # -------------------------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("Bonus: no_ff + delete_branch")
    print("-" * 40)
    print()
    print("  This COULD fast-forward, but no_ff=True forces a merge commit.")
    print("  delete_branch=True auto-cleans the source branch after merge.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Keep answers to one sentence.")
        t.branch("quick-fix")
        t.chat("What is the DRY principle?")

        t.switch("main")

        branches_before = [b.name for b in t.list_branches()]
        print(f"\n  BEFORE MERGE")
        print(f"    branches: {branches_before}")
        print(f"  main:")
        t.compile().pprint(style="compact")

        result = t.merge("quick-fix", no_ff=True, delete_branch=True)

        result.pprint()

        branches_after = [b.name for b in t.list_branches()]
        print(f"  AFTER MERGE")
        print(f"    branches: {branches_after}  ('quick-fix' auto-deleted)")
        print(f"  main:")
        t.compile().pprint(style="compact")


if __name__ == "__main__":
    part2_merge_strategies()
