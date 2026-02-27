"""PendingGC hooks: intercept garbage collection to inspect what will be
removed, exclude specific commits from the removal plan, then approve.
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.gc import PendingGC
from tract.models.compression import GCResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def gc_hooks() -> None:
    print("=" * 60)
    print("PendingGC: Selective Garbage Collection")
    print("=" * 60)
    print()
    print("  GC removes orphaned commits (not reachable from any branch).")
    print("  Create orphans by branching, adding commits, then deleting the branch.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Build a main conversation
        t.system("You are a home cooking assistant specializing in quick weeknight meals.")
        t.chat("What's a good 30-minute pasta recipe for two people?", max_tokens=500)

        # Create a throwaway branch with several commits
        t.branch("throwaway")
        orphan_count = 0
        for i in range(4):
            t.user(f"Throwaway question {i + 1}")
            t.assistant(f"Throwaway answer {i + 1}")
            orphan_count += 2

        # Delete the branch — its commits become orphaned
        t.switch("main")
        t.delete_branch("throwaway", force=True)
        print(f"\n  Deleted 'throwaway' branch — {orphan_count} commits now orphaned")

        # --- review=True: get PendingGC without executing ---
        # orphan_retention_days=0 makes them immediately eligible
        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)

        # pprint shows status, commits_to_remove, tokens_to_free, actions
        pending.pprint()

        # --- Exclude one commit: keep it despite being orphaned ---
        if len(pending.commits_to_remove) > 1:
            keep_hash: str = pending.commits_to_remove[0]
            original_count: int = len(pending.commits_to_remove)
            pending.exclude(keep_hash)
            print(f"\n  Excluded {keep_hash[:12]} from removal")
            print(f"    commits_to_remove: {len(pending.commits_to_remove)} (was {original_count})")

        # --- Approve the reduced plan ---
        result: GCResult = pending.approve()
        print(f"\n  Approved! GC complete")
        pending.pprint()

    # --- Hook handler pattern: auto-exclude by token count ---
    print(f"\n  Hook pattern: protect high-value orphans")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a home cooking assistant specializing in quick weeknight meals.")
        t.chat("What can I make with chicken, rice, and broccoli?", max_tokens=500)

        # Create and delete two branches to make orphans
        for branch_name in ["experiment-a", "experiment-b"]:
            t.branch(branch_name)
            for i in range(3):
                t.user(f"{branch_name} Q{i}")
                t.assistant(f"{branch_name} A{i} — " + "x" * (50 * (i + 1)))
            t.switch("main")
            t.delete_branch(branch_name, force=True)

        def protect_large_orphans(pending: PendingGC) -> None:
            """Keep orphans that might have substantial content."""
            # Note: PendingGC doesn't expose per-commit token counts publicly.
            # In practice, you'd use your own tracking or inspect via t.log().
            # Here we demonstrate exclude() by keeping every other commit.
            for i, h in enumerate(list(pending.commits_to_remove)):
                if i % 2 == 0:
                    pending.exclude(h)
            pending.approve()

        t.on("gc", protect_large_orphans, name="protect-large")
        t.gc(orphan_retention_days=0)

        # print_hooks shows registered handlers and recent hook_log
        t.print_hooks()


if __name__ == "__main__":
    gc_hooks()
