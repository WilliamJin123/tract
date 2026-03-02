"""Merge Strategies

Three tiers of merge usage — manual FF/clean/no_ff merges, interactive
review with PendingMerge, and agent-driven auto-merge with hooks.

PART 1 -- Manual           Direct merge calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.confirm, human decides
PART 3 -- LLM / Agent      Hook auto-manages merge decisions

Demonstrates: merge(), merge_type, MergeResult, no_ff, delete_branch=True,
              merge(review=True), PendingMerge, approve/reject,
              t.on("merge", handler), ToolExecutor
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract, ToolExecutor
from tract.hooks.merge import PendingMerge

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# PART 1 -- Manual: Direct API calls, no LLM, deterministic
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Merge Strategies")
    print("=" * 60)

    # --- Scenario 1: Fast-Forward ---

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

        t.branch("feature")
        t.chat("Give an example of a base case in recursion.")

        print(f"\n  feature:")
        t.compile().pprint(style="compact")

        t.switch("main")
        result = t.merge("feature")

        result.pprint()

        print(f"  AFTER MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

    # --- Scenario 2: Clean Merge (diverged, APPEND-only) ---

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

        t.branch("feature")
        t.chat("What is a stack?")

        t.switch("main")
        t.chat("What is a queue?")

        print(f"\n  BEFORE MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

        t.switch("feature")
        print(f"\n  feature:")
        t.compile().pprint(style="compact")

        t.switch("main")
        result = t.merge("feature")

        result.pprint()

        print(f"  AFTER MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

    # --- Bonus: no_ff and delete_branch=True ---

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


# =============================================================================
# PART 2 -- Interactive: review=True, click.confirm, human decides
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Merge with Review")
    print("=" * 60)
    print()
    print("  merge(review=True) returns a PendingMerge instead of committing")
    print("  immediately. You inspect the preview and decide.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Keep answers to one sentence.")
        t.chat("What is a variable?")

        t.branch("feature")
        t.chat("What is a constant?")

        t.switch("main")

        print(f"\n  main before merge:")
        t.compile().pprint(style="compact")

        # review=True returns PendingMerge for human approval
        pending = t.merge("feature", review=True)

        if not isinstance(pending, PendingMerge):
            print(f"  Merge completed without review: {pending.merge_type}")
            return

        print(f"\n  PendingMerge returned:")
        print(f"    merge_type:  {pending.merge_type}")
        print(f"    conflicts:   {len(pending.conflicts)}")
        pending.pprint()

        if click.confirm("  Proceed with merge?", default=True):
            result = pending.approve()
            result.pprint()
            print(f"\n  AFTER MERGE")
            t.compile().pprint(style="compact")
        else:
            pending.reject("User declined merge.")
            print("  Merge rejected.")


# =============================================================================
# PART 3 -- LLM / Agent: Hook auto-manages merge decisions
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: Auto-Merge via Hook")
    print("=" * 60)
    print()
    print("  Register a merge hook that auto-approves fast-forwards")
    print("  and rejects non-FF merges. No human in the loop.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # Register hook: auto-approve FF, reject others
        def auto_merge(pending):
            if pending.merge_type == "fast_forward":
                print(f"    [hook] FF merge -> auto-approving")
                return pending.approve()
            else:
                print(f"    [hook] {pending.merge_type} -> rejecting")
                return pending.reject("Non-FF rejected by policy")

        t.on("merge", auto_merge)

        t.system("You are a helpful assistant.")
        t.chat("What is polymorphism?")

        # FF scenario: hook auto-approves
        t.branch("feature")
        t.chat("Give an example of polymorphism.")
        t.switch("main")

        print(f"\n  Merging 'feature' (FF eligible):")
        result = t.merge("feature")
        print(f"  Result: {result.merge_type}")
        t.compile().pprint(style="compact")

    # Toolkit approach
    print(f"\n  --- ToolExecutor approach ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.chat("What is encapsulation?")
        t.branch("toolkit-feature")
        t.chat("Give an example of encapsulation.")
        t.switch("main")

        executor = ToolExecutor(t)
        result = executor.execute("merge", {"branch": "toolkit-feature"})
        print(f"  ToolExecutor merge result: {result}")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
