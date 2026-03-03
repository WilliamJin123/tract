"""Merge Strategies

Three tiers of merge usage -- manual FF/clean/no_ff merges, hook-driven
merge decisions, and trigger-based completion detection.

PART 1 -- Manual           Direct merge calls, no LLM, deterministic
PART 2 -- Hooks            Hook auto-manages merge decisions
PART 3 -- Triggers         MergeTrigger fires on branch completion

Demonstrates: merge(), merge_type, MergeResult, no_ff, delete_branch=True,
              t.on("merge", handler), MergeTrigger, configure_triggers()
"""

import sys
from pathlib import Path

from tract import Tract, MergeTrigger
from tract.hooks.merge import PendingMerge
from tract.hooks.trigger import PendingTrigger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


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
        api_key=llm.api_key,
        base_url=llm.base_url,
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
        api_key=llm.api_key,
        base_url=llm.base_url,
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
        api_key=llm.api_key,
        base_url=llm.base_url,
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
# PART 2 -- Automated: Hook auto-manages merge decisions
# =============================================================================

def part2_automated():
    print("=" * 60)
    print("PART 2 -- Automated: Auto-Merge via Hook")
    print("=" * 60)
    print()
    print("  Register a merge hook that auto-approves fast-forwards")
    print("  and rejects non-FF merges. No human in the loop.")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # Register hook: auto-approve FF, reject others
        def auto_merge(pending: PendingMerge):
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



# =============================================================================
# PART 3 -- Triggers: MergeTrigger fires on branch completion
# =============================================================================

def part3_trigger():
    print("=" * 60)
    print("PART 3 -- Triggers: Completion-Based Auto-Merge")
    print("=" * 60)
    print()
    print("  MergeTrigger evaluates on every commit. When a feature branch")
    print("  has enough commits and has been idle long enough, it fires.")
    print("  idle_seconds=0 for demo purposes (normally 300s).")
    print()

    # Low thresholds for demo
    trigger = MergeTrigger(target_branch="main", completion_commits=3, idle_seconds=0)

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Initial work on main.")

        t.branch("feature-auth")
        t.switch("feature-auth")

        # Build up commits on the feature branch
        print("  Building feature branch...")
        for i in range(4):
            t.user(f"Auth feature work item {i}")
            print(f"    commit {i+1}: auth work")

        # Manual evaluate to show when trigger fires
        action = trigger.evaluate(t)
        if action:
            print(f"\n  Trigger fired: {action}")
        print()

        # Now show it working via configure_triggers + hook
        print("  --- With configure_triggers + hook ---")
        merged = []

        def on_trigger(pending: PendingTrigger):
            print(f"  [trigger] {pending.trigger_name}: {pending.reason}")
            merged.append(pending.action_params)
            pending.approve()

        t.on("trigger", on_trigger, name="auto-merge")
        t.configure_triggers([trigger])

        # One more commit to trip the trigger via configure_triggers
        t.user("Final auth cleanup")

        if merged:
            print(f"  Auto-merged: {merged[0]}")
        print(f"  Current branch: {t.current_branch}")
        print(f"  Branches: {[b.name for b in t.list_branches()]}")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_automated()
    part3_trigger()


if __name__ == "__main__":
    main()
