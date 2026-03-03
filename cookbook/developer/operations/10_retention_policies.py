"""Retention Policies

Two tiers of GC retention -- manual conservative/aggressive policies and
automated GCTrigger with hooks.

PART 1 -- Manual           gc() with archive_retention_days, no interaction
PART 3 -- Automated         GCTrigger with auto-approve hook

Demonstrates: gc(), archive_retention_days, compress(),
              GCTrigger, t.on("gc", handler)
"""

import sys
from pathlib import Path

from tract import GCTrigger, Priority, Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.small


# =============================================================================
# PART 1 -- Manual: gc() with archive_retention_days
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Retention Policies")
    print("=" * 60)
    print()
    print("  orphan_retention_days: how long orphaned commits survive.")
    print("  archive_retention_days: how long compressed sources survive.")
    print("  None (default) = archives kept forever.")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise chess coach.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What's the Sicilian Defense?")
        t.chat("Explain the Queen's Gambit.")

        t.compress(target_tokens=80)

        # Conservative: keep archives forever
        print(f"\n  gc(archive_retention_days=None)  -- keep archives forever")
        gc1 = t.gc(archive_retention_days=None)
        print(f"    commits_removed: {gc1.commits_removed}  (archives preserved)")

        # Aggressive: remove archives immediately
        print(f"\n  gc(archive_retention_days=0)  -- remove archives now")
        gc2 = t.gc(archive_retention_days=0)
        print(f"    commits_removed: {gc2.commits_removed}")
        print(f"    tokens_freed:    {gc2.tokens_freed}")

        print(f"\n  In production, use archive_retention_days=30 (or similar)")
        print(f"  so you can still audit pre-compression history for a month.")


# =============================================================================
# PART 3 -- Automated: GCTrigger with auto-approve hook
# =============================================================================

def part3_automated():
    print("=" * 60)
    print("PART 3 -- Automated: Auto-GC via Trigger + Hook")
    print("=" * 60)
    print()
    print("  GCTrigger fires when dead commits exceed a threshold.")
    print("  A hook handler logs and auto-approves GC operations.")

    trigger = GCTrigger(archive_retention_days=30)
    print(f"\n  GCTrigger: fires_on={trigger.fires_on}, "
          f"archive_retention_days=30")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # Register hook: log and auto-approve GC
        def auto_gc(pending):
            print(f"    [hook] GC pending: {pending.commits_to_remove} commits")
            print(f"    [hook] Auto-approving...")
            return pending.approve()

        t.on("gc", auto_gc)

        sys_ci = t.system("You are a concise chess coach.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What's the Sicilian Defense?")
        t.chat("Explain the Queen's Gambit.")

        t.compress(target_tokens=80)

        # Evaluate trigger
        action = trigger.evaluate(t)
        if action:
            print(f"\n  Trigger fired: {action.reason}")
        else:
            print(f"\n  Trigger not yet fired (dead commits below threshold)")

        # Run gc -- hook auto-approves if registered
        print(f"\n  Running gc(archive_retention_days=0)...")
        gc_result = t.gc(archive_retention_days=0)
        print(f"  commits_removed: {gc_result.commits_removed}")
        print(f"  tokens_freed:    {gc_result.tokens_freed}")

        print(f"\n  Hook-based GC integrates with triggers for fully")
        print(f"  autonomous storage maintenance.")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part3_automated()


if __name__ == "__main__":
    main()
