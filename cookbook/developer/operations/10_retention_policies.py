"""Retention Policies

Three tiers of GC retention -- manual conservative/aggressive policies,
interactive review with PendingGC, and agent-driven GCTrigger automation.

PART 1 -- Manual           gc() with archive_retention_days, no interaction
PART 2 -- Interactive       gc(review=True), PendingGC, exclude(), click.confirm
PART 3 -- LLM / Agent      GCTrigger with auto-approve hook

Demonstrates: gc(), archive_retention_days, compress(),
              gc(review=True), PendingGC, exclude(), approve(),
              GCTrigger, t.on("gc", handler)
"""

import os

import click
from dotenv import load_dotenv

from tract import GCTrigger, Priority, Tract
from tract.hooks.gc import PendingGC

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


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
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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
# PART 2 -- Interactive: gc(review=True), PendingGC, exclude(), click.confirm
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: GC with Review")
    print("=" * 60)
    print()
    print("  gc(review=True) returns a PendingGC for human inspection.")
    print("  You can exclude specific commits before approving.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        sys_ci = t.system("You are a concise chess coach.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        r1 = t.chat("What's the Sicilian Defense?")
        r2 = t.chat("Explain the Queen's Gambit.")

        # Save a hash we might want to protect
        important_hash = r1.commit_info.commit_hash

        t.compress(target_tokens=80)

        # Review before gc
        pending = t.gc(archive_retention_days=0, review=True)

        if not isinstance(pending, PendingGC):
            print(f"  GC completed without review.")
            return

        print(f"\n  PendingGC returned:")
        print(f"    commits to remove: {pending.commits_to_remove}")
        pending.pprint()

        # Protect an important commit
        print(f"\n  Excluding {important_hash[:8]} from GC...")
        pending.exclude(important_hash)

        if click.confirm("  Proceed with GC?", default=True):
            gc_result = pending.approve()
            print(f"\n  GC approved:")
            print(f"    commits_removed: {gc_result.commits_removed}")
            print(f"    tokens_freed:    {gc_result.tokens_freed}")
        else:
            pending.reject("User declined GC.")
            print("  GC rejected.")

        print(f"\n  Context after GC:")
        t.compile().pprint(style="chat")


# =============================================================================
# PART 3 -- LLM / Agent: GCTrigger with auto-approve hook
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: Auto-GC via Trigger + Hook")
    print("=" * 60)
    print()
    print("  GCTrigger fires when dead commits exceed a threshold.")
    print("  A hook handler logs and auto-approves GC operations.")

    trigger = GCTrigger(archive_retention_days=30)
    print(f"\n  GCTrigger: fires_on={trigger.fires_on}, "
          f"archive_retention_days=30")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
