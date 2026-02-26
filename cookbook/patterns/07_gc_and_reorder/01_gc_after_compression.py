"""GC After Compression

After compression, archived source commits remain in the database.
gc() reclaims storage by removing unreachable commits beyond a retention
window. This example compresses a conversation, then runs gc() with
0-day archive retention for immediate cleanup.

Demonstrates: gc(), GCResult, compress(review=True), PendingCompress,
              edit_summary(), approve(), archive_retention_days,
              pprint(style="chat"), before/after visualization
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
    # Part 1: GC after compression
    # =================================================================

    print("=" * 60)
    print("Part 1: GARBAGE COLLECTION AFTER COMPRESSION")
    print("=" * 60)
    print()
    print("  After compression, original commits are archived but still")
    print("  in the database. gc() removes them to reclaim storage.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # Build a conversation and compress it
        sys_ci = t.system("You are a concise travel advisor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("Best time to visit Japan?")
        t.chat("What about Iceland?")
        t.chat("Tips for visiting Morocco.")
        t.chat("Best way to travel through Southeast Asia?")

        ctx_before = t.compile()
        print(f"\n  Built conversation: {len(ctx_before.messages)} messages, "
              f"{ctx_before.token_count} tokens")

        # Collaborative compression: LLM drafts, user reviews & edits
        pending = t.compress(target_tokens=100, review=True)

        print(f"\n  PendingCompress (NOT yet committed):")
        print(f"    summaries:        {len(pending.summaries)} draft(s)")
        print(f"    source_commits:   {len(pending.source_commits)}")
        print(f"    original_tokens:  {pending.original_tokens}")
        print(f"    estimated_tokens: {pending.estimated_tokens}")

        # Interactive review: open each draft in $EDITOR for real editing
        import click as _click

        for i, summary in enumerate(pending.summaries):
            print(f"\n  Opening summary [{i}] in your editor...")
            edited = _click.edit(summary)
            if edited is not None and edited.strip() != summary.strip():
                pending.edit_summary(i, edited.strip())
                print(f"  Summary [{i}] updated with your edits.")
            else:
                print(f"  Summary [{i}] kept as-is.")

        # Approve -- NOW it commits
        compress_result = pending.approve()

        print(f"\n  Approved! CompressResult:")
        print(f"    {len(compress_result.source_commits)} commits archived")
        print(f"    {compress_result.original_tokens} -> {compress_result.compressed_tokens} tokens")
        print(f"    compression_ratio: {compress_result.compression_ratio:.1%}")

        ctx_after = t.compile()
        print(f"\n  Post-compression context:")
        ctx_after.pprint(style="chat")

        # GC with 0-day archive retention (immediate cleanup for demo)
        print(f"\n  Running gc(archive_retention_days=0)...\n")

        gc_result = t.gc(archive_retention_days=0)

        print(f"  GCResult:")
        print(f"    commits_removed:        {gc_result.commits_removed}")
        print(f"    blobs_removed:          {gc_result.blobs_removed}")
        print(f"    tokens_freed:           {gc_result.tokens_freed}")
        print(f"    source_commits_removed: {gc_result.source_commits_removed}")
        print(f"    duration_seconds:       {gc_result.duration_seconds:.3f}s")

        # Context is unchanged -- GC only removes unreachable data
        print(f"\n  Context unchanged after GC:")
        t.compile().pprint(style="chat")
        print(f"\n  Archived source commits are gone. Reachable data is safe.")


if __name__ == "__main__":
    main()
