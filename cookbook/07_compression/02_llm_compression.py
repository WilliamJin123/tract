"""LLM and Collaborative Compression

Let the LLM summarize your conversation automatically, with optional
human review before committing. Guide the summary with instructions=
and system_prompt=. Use auto_commit=False for a collaborative workflow
where you inspect, edit, and approve.

Demonstrates: compress(target_tokens=), instructions=, system_prompt=,
              auto_commit=False, PendingCompression, edit_summary(),
              approve(), approve_compression(), CompressResult,
              PINNED/SKIP interaction, before/after visualization
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
    print("  compress(target_tokens=) uses the configured LLM to")
    print("  summarize commits. PINNED commits survive verbatim.")
    print("  SKIP commits are excluded from the summary.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # Build a conversation with varying importance
        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        r1 = t.chat("What are decorators?")
        r2 = t.chat("What about context managers?")
        r3 = t.chat("Explain list comprehensions.")
        noise = t.user("[debug] latency=342ms | cache=miss")
        t.annotate(noise.commit_hash, Priority.SKIP)
        r4 = t.chat("What's the difference between a module and a package?")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="compact")
        print(f"\n  {ctx_before.token_count} tokens, {len(ctx_before.messages)} messages")
        print(f"  (system is PINNED, debug noise is SKIP)")

        # Compress -- LLM generates the summary
        result = t.compress(target_tokens=200)

        print(f"\n  CompressResult:")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    preserved:         {len(result.preserved_commits)} (PINNED system prompt)")
        print(f"    source_commits:    {len(result.source_commits)} archived")

        print("\n  AFTER compression:\n")
        ctx_after = t.compile()
        ctx_after.pprint(style="compact")
        print(f"\n  {ctx_after.token_count} tokens, {len(ctx_after.messages)} messages")
        print(f"\n  PINNED system prompt survived. SKIP noise excluded. Rest summarized.")

    # =================================================================
    # Part 2: Guided summarization (instructions= / system_prompt=)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: GUIDED SUMMARIZATION")
    print("=" * 60)
    print()
    print("  instructions= adds guidance to the default summarization prompt.")
    print("  system_prompt= replaces the entire summarization prompt.")
    print("  Both let you control what the summary focuses on.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a concise Python tutor.")
        t.chat("What are decorators? Give me a practical example.")
        t.chat("What about context managers? Show me a database example.")
        t.chat("Explain generators. When should I use them over lists?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Compress with instructions that focus the summary on code examples
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

    # =================================================================
    # Part 3: Collaborative compression (auto_commit=False)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 3: COLLABORATIVE COMPRESSION")
    print("=" * 60)
    print()
    print("  auto_commit=False returns a PendingCompression instead of")
    print("  committing immediately. You can inspect the LLM's draft,")
    print("  edit it with edit_summary(), then approve() to finalize.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a concise Python tutor.")
        t.chat("What are decorators?")
        t.chat("What about context managers?")
        t.chat("Explain the GIL and its implications for threading.")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Request collaborative compression -- nothing committed yet
        pending = t.compress(target_tokens=150, auto_commit=False)

        print(f"\n  PendingCompression (NOT yet committed):")
        print(f"    summaries:        {len(pending.summaries)} draft(s)")
        print(f"    source_commits:   {len(pending.source_commits)}")
        print(f"    preserved:        {len(pending.preserved_commits)}")
        print(f"    original_tokens:  {pending.original_tokens}")
        print(f"    estimated_tokens: {pending.estimated_tokens}")

        # Show the LLM's draft summary
        for i, summary in enumerate(pending.summaries):
            print(f"\n  Draft summary [{i}]:")
            # Truncate for display
            display = summary[:300] + "..." if len(summary) > 300 else summary
            print(f"    {display}")

        # Edit the summary before approving
        # (In a real app, this could be user input or programmatic refinement)
        edited = pending.summaries[0].rstrip()
        if not edited.endswith("."):
            edited += "."
        edited += " Note: GIL discussion covered threading limitations."
        pending.edit_summary(0, edited)

        print(f"\n  Edited summary [{0}] (appended GIL note)")

        # Approve -- now it commits
        result = pending.approve()

        print(f"\n  Approved! CompressResult:")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    new_head:          {result.new_head[:8]}")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  The edited summary is now committed. You reviewed before it landed.")


if __name__ == "__main__":
    main()
