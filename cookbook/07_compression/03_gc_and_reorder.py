"""Garbage Collection and Message Reordering

After compression, archived source commits pile up in the database.
gc() reclaims storage by removing unreachable commits beyond a retention
window. compile(order=) reorders messages for better LLM context flow,
with safety checks that warn about structural issues.

Demonstrates: gc(), GCResult, orphan_retention_days,
              archive_retention_days, compile(order=), ReorderWarning,
              check_safety, before/after visualization
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    # =================================================================
    # Part 1: Garbage Collection
    # =================================================================

    print("=" * 60)
    print("Part 1: GARBAGE COLLECTION")
    print("=" * 60)
    print()
    print("  After compression, original commits are archived but still")
    print("  in the database. gc() removes unreachable commits beyond")
    print("  the retention window to reclaim storage.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # Build a conversation and compress it
        t.system("You are a concise Python tutor.")
        t.chat("What are decorators?")
        t.chat("What about context managers?")
        t.chat("Explain generators and yield.")
        t.chat("What's the difference between args and kwargs?")

        ctx_before = t.compile()
        print(f"\n  Built conversation: {len(ctx_before.messages)} messages, "
              f"{ctx_before.token_count} tokens")

        # Compress to create archived commits
        compress_result = t.compress(
            content=(
                "User asked about decorators (@syntax wrapping), context "
                "managers (with statements), generators (yield keyword), "
                "and args/kwargs (*args for positional, **kwargs for keyword)."
            ),
        )

        print(f"\n  Compressed: {len(compress_result.source_commits)} commits archived")
        print(f"    {compress_result.original_tokens} -> {compress_result.compressed_tokens} tokens")

        ctx_after = t.compile()
        print(f"\n  Post-compression: {len(ctx_after.messages)} messages")
        ctx_after.pprint(style="compact")

        # Also create orphaned commits by resetting
        # (reset moves HEAD back, leaving unreachable commits)
        t.chat("What about list comprehensions?")
        r_orphan = t.chat("And dict comprehensions?")
        orphan_head = t.head
        t.reset(ctx_after.commit_hashes[-1])  # reset back to post-compression

        print(f"\n  Created orphaned commits by adding 2 messages then resetting.")
        print(f"    Orphaned HEAD was: {orphan_head[:8]}")
        print(f"    Reset back to:     {t.head[:8]}")

        # Run GC with 0-day retention (immediate cleanup for demo)
        print(f"\n  Running gc(orphan_retention_days=0, archive_retention_days=0)...\n")

        gc_result = t.gc(
            orphan_retention_days=0,
            archive_retention_days=0,
        )

        print(f"  GCResult:")
        print(f"    commits_removed:        {gc_result.commits_removed}")
        print(f"    blobs_removed:          {gc_result.blobs_removed}")
        print(f"    tokens_freed:           {gc_result.tokens_freed}")
        print(f"    source_commits_removed: {gc_result.source_commits_removed}")
        print(f"    duration_seconds:       {gc_result.duration_seconds:.3f}s")

        # Context is unchanged -- GC only removes unreachable data
        print(f"\n  Context unchanged after GC:")
        t.compile().pprint(style="compact")

        print(f"\n  GC removed archived + orphaned commits. Reachable data is safe.")

    # =================================================================
    # Part 2: Retention policies
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: RETENTION POLICIES")
    print("=" * 60)
    print()
    print("  orphan_retention_days: how long orphaned commits survive.")
    print("  archive_retention_days: how long compressed source commits survive.")
    print("  None (default) means archives are never removed.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a concise Python tutor.")
        t.chat("What's a closure?")
        t.chat("What's a lambda?")

        # Compress
        t.compress(
            content="User asked about closures (functions capturing scope) and lambdas (anonymous functions).",
        )

        # GC with conservative retention -- keeps archives indefinitely
        print(f"\n  gc(orphan_retention_days=7, archive_retention_days=None)")
        print(f"  (7-day orphan window, archives kept forever)\n")

        gc_result = t.gc(
            orphan_retention_days=7,
            archive_retention_days=None,  # never remove archives
        )

        print(f"  commits_removed: {gc_result.commits_removed}")
        print(f"  (Archives survived because archive_retention_days=None)")

        # GC again with 0-day archive retention
        print(f"\n  gc(orphan_retention_days=0, archive_retention_days=0)")
        print(f"  (immediate cleanup of everything unreachable)\n")

        gc_result = t.gc(
            orphan_retention_days=0,
            archive_retention_days=0,
        )

        print(f"  commits_removed: {gc_result.commits_removed}")
        print(f"  tokens_freed:    {gc_result.tokens_freed}")

    # =================================================================
    # Part 3: Message reordering with compile(order=)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 3: MESSAGE REORDERING (compile(order=))")
    print("=" * 60)
    print()
    print("  compile(order=) reorders messages by commit hash.")
    print("  Returns (CompiledContext, list[ReorderWarning]) with")
    print("  safety checks for structural issues like edits")
    print("  appearing before their targets.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a concise Python tutor.")
        r1 = t.chat("What are decorators?")
        r2 = t.chat("What about generators?")
        r3 = t.chat("Explain the GIL.")

        print("\n  Original order:\n")
        ctx = t.compile()
        ctx.pprint(style="compact")

        # Reorder: put GIL discussion first, then decorators, then generators
        # We need the commit hashes for the user+assistant pairs
        hashes = ctx.commit_hashes  # parallel to messages
        # [0]=system, [1]=r1_user, [2]=r1_asst, [3]=r2_user, [4]=r2_asst, [5]=r3_user, [6]=r3_asst

        # Put GIL (r3) right after system, then decorators (r1), then generators (r2)
        new_order = [
            hashes[0],  # system
            hashes[5], hashes[6],  # r3: GIL
            hashes[1], hashes[2],  # r1: decorators
            hashes[3], hashes[4],  # r2: generators
        ]

        reordered, warnings = t.compile(order=new_order)

        print(f"\n  Reordered (GIL first):\n")
        reordered.pprint(style="compact")

        print(f"\n  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    [{w.severity}] {w.warning_type}: {w.description}")

        if not warnings:
            print(f"    (none -- all APPEND-only, safe to reorder)")

        print(f"\n  compile(order=) lets you rearrange context for better LLM flow")
        print(f"  without changing the underlying commit history.")


if __name__ == "__main__":
    main()
