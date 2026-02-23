"""LLM and Collaborative Compression

Let the LLM summarize your conversation automatically, guide it with
instructions=, or use a collaborative workflow where you inspect, edit,
and approve before committing. PINNED and SKIP annotations control what
enters the summary.

When PINNED commits interleave with normal ones, the compressor splits
them into separate groups, each getting its own LLM-generated summary.
Manual mode (content=) only handles a single group -- LLM mode handles
any number.

Demonstrates: compress(target_tokens=), instructions=, system_prompt=,
              auto_commit=False, PendingCompression, edit_summary(),
              approve(), PINNED/SKIP interaction, multi-group compression,
              before/after visualization
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
    # Part 1: Auto-commit LLM compression
    # =================================================================

    print("=" * 60)
    print("Part 1: AUTO-COMMIT LLM COMPRESSION")
    print("=" * 60)
    print()
    print("  compress(target_tokens=) uses the configured LLM to summarize.")
    print("  PINNED commits survive verbatim. SKIP commits are excluded.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What are decorators?")
        t.chat("What about context managers?")
        noise = t.user("[debug] latency=342ms | cache=miss")
        t.annotate(noise.commit_hash, Priority.SKIP)
        t.chat("What's the difference between a module and a package?")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="compact")
        print(f"\n  {ctx_before.token_count} tokens, {len(ctx_before.messages)} messages")
        print(f"  (system is PINNED, debug noise is SKIP)")

        # LLM generates the summary automatically
        result = t.compress(target_tokens=200)

        print(f"\n  CompressResult:")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    preserved:         {len(result.preserved_commits)} (PINNED)")
        print(f"    source_commits:    {len(result.source_commits)} archived")

        print("\n  AFTER compression:\n")
        ctx_after = t.compile()
        ctx_after.pprint(style="compact")
        print(f"\n  {ctx_after.token_count} tokens, {len(ctx_after.messages)} messages")
        print(f"\n  PINNED system survived. SKIP noise excluded. Rest summarized by LLM.")

    # =================================================================
    # Part 2: Guided summarization (instructions=)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: GUIDED SUMMARIZATION (instructions=)")
    print("=" * 60)
    print()
    print("  instructions= adds guidance to the summarization prompt.")
    print("  Control what the summary focuses on without replacing")
    print("  the entire prompt.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What are decorators? Give me a practical example.")
        t.chat("What about context managers? Show me a database example.")
        t.chat("Explain generators. When should I use them over lists?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Focus the summary on code examples
        result = t.compress(
            target_tokens=150,
            instructions=(
                "Focus on the practical code examples. "
                "Preserve any Python code snippets verbatim. "
                "Omit theoretical explanations."
            ),
        )

        print(f"\n  Compressed with instructions (code-focused):")
        print(f"    {result.original_tokens} -> {result.compressed_tokens} tokens")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  The summary should emphasize code examples per our instructions.")

    # =================================================================
    # Part 3: Collaborative compression (auto_commit=False)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 3: COLLABORATIVE COMPRESSION")
    print("=" * 60)
    print()
    print("  auto_commit=False returns a PendingCompression. You inspect")
    print("  the LLM's draft, edit with edit_summary(), then approve().")
    print("  Nothing is committed until you say so.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What are decorators?")
        t.chat("What about context managers?")
        t.chat("Explain the GIL and its implications for threading.")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Collaborative: LLM drafts, we review
        pending = t.compress(target_tokens=150, auto_commit=False)

        print(f"\n  PendingCompression (NOT yet committed):")
        print(f"    summaries:        {len(pending.summaries)} draft(s)")
        print(f"    source_commits:   {len(pending.source_commits)}")
        print(f"    preserved:        {len(pending.preserved_commits)}")
        print(f"    original_tokens:  {pending.original_tokens}")
        print(f"    estimated_tokens: {pending.estimated_tokens}")

        # Show the LLM's draft
        for i, summary in enumerate(pending.summaries):
            display = summary[:300] + "..." if len(summary) > 300 else summary
            print(f"\n  Draft summary [{i}]:")
            print(f"    {display}")

        # Edit before approving
        edited = pending.summaries[0].rstrip()
        if not edited.endswith("."):
            edited += "."
        edited += " Note: GIL limits true parallelism in CPython."
        pending.edit_summary(0, edited)

        print(f"\n  Edited summary [0] -- appended GIL note")

        # Approve -- NOW it commits
        result = pending.approve()

        print(f"\n  Approved! CompressResult:")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    new_head:          {result.new_head[:8]}")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  You reviewed and edited before it landed.")


if __name__ == "__main__":
    main()
