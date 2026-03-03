"""Rebase

Two tiers of rebase usage -- manual rebase with result inspection and
trigger-driven auto-rebase.

PART 1 -- Manual           Direct rebase(), inspect RebaseResult
PART 2 -- Trigger-Driven    RebaseTrigger auto-rebase on divergence

Demonstrates: rebase(), RebaseResult, replayed_commits, original_commits,
              new_head, RebaseTrigger
"""

import sys
from pathlib import Path

from tract import RebaseTrigger, Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


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
        api_key=llm.api_key,
        base_url=llm.base_url,
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
# PART 2 -- Trigger-Driven: RebaseTrigger auto-rebase on divergence
# =============================================================================

def part2_trigger_driven():
    print("=" * 60)
    print("PART 2 -- Trigger-Driven: Auto-Rebase via RebaseTrigger")
    print("=" * 60)
    print()
    print("  RebaseTrigger fires when branch diverges beyond a threshold.")

    # --- Trigger approach ---
    print(f"\n  --- RebaseTrigger approach ---")

    trigger = RebaseTrigger(target_branch="main", divergence_commits=3)
    print(f"  {trigger}")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
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
            print(f"  {action}")



# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_trigger_driven()


if __name__ == "__main__":
    main()
