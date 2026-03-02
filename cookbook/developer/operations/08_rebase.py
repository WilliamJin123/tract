"""Rebase

Three tiers of rebase usage -- manual rebase with result inspection,
interactive review with PendingRebase, and agent-driven triggers/toolkit.

PART 1 -- Manual           Direct rebase(), inspect RebaseResult
PART 2 -- Interactive       rebase(review=True), PendingRebase, click.confirm
PART 3 -- LLM / Agent      RebaseTrigger + ToolExecutor auto-rebase

Demonstrates: rebase(), RebaseResult, replayed_commits, original_commits,
              new_head, rebase(review=True), PendingRebase, approve/reject,
              RebaseTrigger, ToolExecutor
"""

import os

import click
from dotenv import load_dotenv

from tract import RebaseTrigger, Tract, ToolExecutor
from tract.hooks.rebase import PendingRebase

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _build_diverged_branches(t):
    """Helper: build main + examples branch that diverge."""
    t.system("You are a concise music theory tutor. One paragraph max.")
    t.chat("What are major and minor scales?")

    t.branch("examples")
    t.chat("Give me 3 chord progression examples, from simple to complex.")
    t.chat("Now explain the emotional feel of each progression.")

    t.switch("main")
    t.chat("What are modes? How do they differ from standard scales?")

    t.switch("examples")


# =============================================================================
# PART 1 -- Manual: Direct rebase(), inspect RebaseResult
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Rebase")
    print("=" * 60)
    print()
    print("  Scenario: 'examples' branch started from an older main.")
    print("  Main has since advanced with new content. Rebase replays")
    print("  the examples branch's commits on top of main's current tip.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        _build_diverged_branches(t)

        ctx_before = t.compile()
        print(f"\n  examples BEFORE rebase ({len(ctx_before.messages)} messages):")
        ctx_before.pprint(style="compact")
        print("\n  (notice: no modes content)")

        # Rebase
        print("\n  Rebasing examples onto main...\n")
        result = t.rebase("main")

        print(f"  Rebase complete:")
        print(f"    replayed: {len(result.replayed_commits)} commits")
        print(f"    new HEAD: {result.new_head[:8]}")
        print(f"    warnings: {len(result.warnings)}")

        # Show hash changes
        print(f"\n  Hash changes (same content, new lineage):")
        for orig, replayed in zip(result.original_commits, result.replayed_commits):
            print(f"    {orig.commit_hash[:8]} -> {replayed.commit_hash[:8]}")

        ctx_after = t.compile()
        print(f"\n  examples AFTER rebase ({len(ctx_after.messages)} messages):")
        ctx_after.pprint(style="chat")

        print("\n  The examples branch now includes main's modes")
        print("  content PLUS its own chord progression work.")


# =============================================================================
# PART 2 -- Interactive: rebase(review=True), PendingRebase, click.confirm
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Rebase with Review")
    print("=" * 60)
    print()
    print("  rebase(review=True) returns a PendingRebase for human")
    print("  inspection before committing the replayed commits.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        _build_diverged_branches(t)

        print(f"\n  examples BEFORE rebase:")
        t.compile().pprint(style="compact")

        # review=True returns PendingRebase
        pending = t.rebase("main", review=True)

        if not isinstance(pending, PendingRebase):
            print(f"  Rebase completed without review.")
            return

        print(f"\n  PendingRebase returned:")
        print(f"    commits to replay: {len(pending.original_commits)}")
        pending.pprint()

        if click.confirm("  Proceed with rebase?", default=True):
            result = pending.approve()
            print(f"\n  Rebase approved:")
            print(f"    replayed: {len(result.replayed_commits)} commits")
            print(f"    new HEAD: {result.new_head[:8]}")
            t.compile().pprint(style="chat")
        else:
            pending.reject("User declined rebase.")
            print("  Rebase rejected. Branch unchanged.")


# =============================================================================
# PART 3 -- LLM / Agent: RebaseTrigger + ToolExecutor auto-rebase
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: Auto-Rebase via Trigger + Toolkit")
    print("=" * 60)
    print()
    print("  RebaseTrigger fires when branch diverges beyond a threshold.")
    print("  ToolExecutor can also rebase on demand.")

    # --- Trigger approach ---
    print(f"\n  --- RebaseTrigger approach ---")

    trigger = RebaseTrigger(target_branch="main", divergence_commits=3)
    print(f"  RebaseTrigger: fires_on={trigger.fires_on}, "
          f"divergence_commits=3")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.configure_triggers([trigger])

        t.system("You are a concise music theory tutor.")
        t.chat("What are scales?")

        t.branch("feature")
        t.chat("What are arpeggios?")

        # Main advances past the threshold
        t.switch("main")
        for i in range(4):
            t.user(f"Main content {i}")

        t.switch("feature")
        action = trigger.evaluate(t)
        if action:
            print(f"  Trigger fired: {action.action_type}")
            print(f"  Reason: {action.reason}")

    # --- ToolExecutor approach ---
    print(f"\n  --- ToolExecutor approach ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _build_diverged_branches(t)

        executor = ToolExecutor(t)
        result = executor.execute("rebase", {"target": "main"})
        print(f"  ToolExecutor rebase result: {result}")

        print(f"\n  AFTER REBASE:")
        t.compile().pprint(style="compact")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
