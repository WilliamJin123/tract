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

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


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
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise economics tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What is inflation and what causes it?")
        t.chat("Explain supply and demand.")
        noise = t.user("[debug] latency=342ms | cache=miss")
        t.annotate(noise.commit_hash, Priority.SKIP)
        t.chat("What causes a recession?")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="chat")
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
        ctx_after.pprint(style="table")
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
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise philosophy tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What is Stoicism? Give me a practical example.")
        t.chat("What about existentialism? Show me a real-life scenario.")
        t.chat("Explain utilitarianism. When does it fail?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # Focus the summary on practical examples
        guidance = (
            "Focus on the practical examples and real-life scenarios. "
            "Preserve any specific thought experiments verbatim. "
            "Omit historical background."
        )
        result = t.compress(target_tokens=150, instructions=guidance)

        print(f"\n  instructions= passed to compress():")
        print(f"    \"{guidance}\"")
        print(f"\n  How instructions= vs system_prompt= work:")
        print(f"    instructions=  -> appended to the USER MESSAGE (task prompt)")
        print(f"    system_prompt= -> replaces the SYSTEM MESSAGE (LLM persona)")
        print(f"    They target different parts of the LLM call, so both can")
        print(f"    be used together. Both are stored in provenance.")
        print(f"\n  Compressed with instructions (example-focused):")
        print(f"    {result.original_tokens} -> {result.compressed_tokens} tokens")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="table")

        print("\n  The summary should emphasize practical examples per our instructions.")

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

        # Collaborative: LLM drafts, user reviews interactively
        pending = t.compress(target_tokens=150, auto_commit=False)

        print(f"\n  PendingCompression (NOT yet committed):")
        print(f"    summaries:        {len(pending.summaries)} draft(s)")
        print(f"    source_commits:   {len(pending.source_commits)}")
        print(f"    preserved:        {len(pending.preserved_commits)}")
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
        if not _click.confirm("\n  Approve and commit?", default=True):
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
    main()
