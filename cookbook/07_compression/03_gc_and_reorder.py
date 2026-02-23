"""Garbage Collection and Message Reordering

After compression, archived source commits remain in the database.
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
    # Part 1: GC after compression
    # =================================================================

    print("=" * 60)
    print("Part 1: GARBAGE COLLECTION AFTER COMPRESSION")
    print("=" * 60)
    print()
    print("  After compression, original commits are archived but still")
    print("  in the database. gc() removes them to reclaim storage.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # Build a conversation and compress it
        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

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

        print(f"  Compressed: {len(compress_result.source_commits)} commits archived")
        print(f"    {compress_result.original_tokens} -> {compress_result.compressed_tokens} tokens")

        ctx_after = t.compile()
        print(f"\n  Post-compression context:")
        ctx_after.pprint(style="compact")

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
        t.compile().pprint(style="compact")
        print(f"\n  Archived source commits are gone. Reachable data is safe.")

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
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What's a closure?")
        t.chat("What's a lambda?")

        t.compress(
            content="User asked about closures (functions capturing scope) "
                    "and lambdas (anonymous functions).",
        )

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

    # =================================================================
    # Part 3: Message reordering with compile(order=)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 3: MESSAGE REORDERING (compile(order=))")
    print("=" * 60)
    print()
    print("  compile(order=) reorders messages by commit hash.")
    print("  Returns (CompiledContext, list[ReorderWarning]).")
    print("  Safety checks warn about structural issues like edits")
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

        # Get commit hashes (parallel to messages)
        hashes = ctx.commit_hashes
        # [0]=system, [1]=r1_user, [2]=r1_asst, [3]=r2_user, [4]=r2_asst, [5]=r3_user, [6]=r3_asst

        # Reorder: GIL first, then decorators, then generators
        new_order = [
            hashes[0],              # system stays first
            hashes[5], hashes[6],   # r3: GIL
            hashes[1], hashes[2],   # r1: decorators
            hashes[3], hashes[4],   # r2: generators
        ]

        reordered, warnings = t.compile(order=new_order)

        print(f"\n  Reordered (GIL first):\n")
        reordered.pprint(style="compact")

        print(f"\n  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    [{w.severity}] {w.warning_type}: {w.description}")

        if not warnings:
            print(f"    (none -- all APPEND-only, safe to reorder)")

        print(f"\n  Reordering rearranges the compiled context for better LLM")
        print(f"  flow without changing the underlying commit history.")


if __name__ == "__main__":
    main()
