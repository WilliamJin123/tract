"""Branching and Merging

Try experimental approaches without affecting main. Part 1 covers the
branch lifecycle: create, switch, list, and delete. Part 2 covers two
merge modes — fast-forward (linear history) and clean merge (diverged
but no edit conflicts) — plus a bonus showing no_ff and auto-delete.

Part 1 — Branch Lifecycle (uses LLM):
  Demonstrates: branch(), switch(), list_branches(), current_branch,
                branch(switch=False), delete_branch(force=True)

Part 2 — Merge Strategies (uses LLM):
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
# Part 1: Branch Lifecycle — Create, Switch, List, Delete
# =============================================================================

def part1_branch_lifecycle():
    print("=" * 60)
    print("Part 1: BRANCH LIFECYCLE")
    print("=" * 60)
    print()
    print("  Try an experimental explanation style without affecting main.")
    print("  Branching is lightweight — it's just a pointer to a commit,")
    print("  not a copy.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # --- Build a conversation on main ---

        print("=== Main branch: start a conversation ===\n")

        t.system("You are a concise Python tutor. One paragraph max.")
        r1 = t.chat("Explain what a decorator is.")
        r1.pprint()

        main_messages = len(t.compile().messages)
        print(f"\n  Branch: {t.current_branch}  |  Messages: {main_messages}\n")

        # --- Branch: try a different explanation style ---
        # branch() creates a new branch at HEAD and switches to it by default.

        print("=== Branch 'analogy': try a different angle ===\n")

        t.branch("analogy")
        print(f"  Switched to: {t.current_branch}")

        r2 = t.chat("Re-explain decorators using a real-world analogy, like gift wrapping.")
        r2.pprint()

        analogy_messages = len(t.compile().messages)
        print(f"\n  Branch: {t.current_branch}  |  Messages: {analogy_messages}\n")

        # --- List branches: see what exists ---
        # list_branches() returns BranchInfo objects with is_current flag.

        print("=== All branches ===\n")

        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"  {marker} {b.name:12s}  @ {b.commit_hash[:8]}")

        # --- Switch back to main: experiment is isolated ---
        # Main still has only the original messages — the analogy chat isn't here.

        print("\n=== Switch back to main ===\n")

        t.switch("main")
        ctx_main = t.compile()
        print(f"  Branch: {t.current_branch}  |  Messages: {len(ctx_main.messages)}")
        print(f"  (analogy branch had {analogy_messages} — main is untouched)")

        # --- Peek at analogy from main ---
        # switch() to analogy and back to verify both are intact.

        print("\n=== Peek at analogy ===\n")

        t.switch("analogy")
        ctx_analogy = t.compile()
        print(f"  Branch: {t.current_branch}  |  Messages: {len(ctx_analogy.messages)}")
        ctx_analogy.pprint(style="chat")

        # --- Create a branch without switching ---
        # branch(switch=False) keeps HEAD on the current branch.

        t.switch("main")
        t.branch("draft", switch=False)
        print(f"\n=== Created 'draft' without switching ===")
        print(f"  Still on: {t.current_branch}")

        print("\n  All branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        # --- Clean up ---
        # delete_branch() removes the branch pointer. force=True allows
        # deleting branches with unmerged work (analogy was never merged).

        print("\n=== Clean up ===\n")

        t.delete_branch("analogy", force=True)
        t.delete_branch("draft", force=True)

        remaining = [b.name for b in t.list_branches()]
        print(f"  Remaining branches: {remaining}")


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


def main():
    part1_branch_lifecycle()
    print()
    part2_merge_strategies()


if __name__ == "__main__":
    main()
