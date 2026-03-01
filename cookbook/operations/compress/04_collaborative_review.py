"""Collaborative Review (review=True, PendingCompress)

review=True returns a PendingCompress. You inspect the LLM's draft,
edit individual summaries with edit_summary(), then approve().
Nothing is committed until you call approve().

Demonstrates: compress(target_tokens=, review=True), PendingCompress,
              edit_summary(), approve(), CompressResult fields,
              click.edit(), click.confirm(), before/after visualization,
              pprint(style="chat"), pprint(style="table")
"""

import os

import click
from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 4: Collaborative Review (review=True, PendingCompress)
# =============================================================================
# review=True returns a PendingCompress. You inspect the LLM's draft,
# edit individual summaries with edit_summary(), then approve().
# Nothing is committed until you call approve().

def part4_collaborative_review():
    print(f"\n{'=' * 60}")
    print("Part 4: COLLABORATIVE REVIEW (review=True)")
    print("=" * 60)
    print()
    print("  review=True returns a PendingCompress. You inspect")
    print("  the LLM's draft, edit with edit_summary(), then approve().")
    print("  Nothing is committed until you say so.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise biology explainer.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What is CRISPR and how does it work?")
        t.chat("How does mRNA deliver instructions to cells?")
        t.chat("Explain epigenetics and why it matters.")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # Collaborative: LLM drafts, user reviews before committing
        # review=True replaces the old auto_commit=False parameter
        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        print(f"\n  PendingCompress (NOT yet committed):")
        print(f"    summaries:        {len(pending.summaries)} draft(s)")
        print(f"    source_commits:   {len(pending.source_commits)}")
        print(f"    preserved:        {len(pending.preserved_commits)}")
        print(f"    original_tokens:  {pending.original_tokens}")
        print(f"    estimated_tokens: {pending.estimated_tokens}")

        # Interactive review: open each draft in $EDITOR for real editing
        for i, summary in enumerate(pending.summaries):
            print(f"\n  Opening summary [{i}] in your editor...")
            edited = click.edit(summary)
            if edited is not None and edited.strip() != summary.strip():
                pending.edit_summary(i, edited.strip())
                print(f"  Summary [{i}] updated with your edits.")
            else:
                print(f"  Summary [{i}] kept as-is.")

        # Approve -- NOW it commits
        if not click.confirm("\n  Approve and commit?", default=True):
            print("  Cancelled. Nothing was committed.")
            return

        result = pending.approve()

        print(f"\n  Approved! CompressResult:")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    new_head:          {result.new_head[:8]}")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="table")

        print("\n  You reviewed and edited before it landed.")


if __name__ == "__main__":
    part4_collaborative_review()
