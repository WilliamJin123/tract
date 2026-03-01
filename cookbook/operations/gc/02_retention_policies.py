"""Retention Policies

orphan_retention_days controls how long orphaned commits survive.
archive_retention_days controls how long compressed sources survive.
None (default) means archives are kept forever.

Demonstrates: gc(), archive_retention_days, compress(),
              Priority.PINNED, conservative vs aggressive retention
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


def main():
    # =================================================================
    # Part 2: Retention policies
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: RETENTION POLICIES")
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


if __name__ == "__main__":
    main()
