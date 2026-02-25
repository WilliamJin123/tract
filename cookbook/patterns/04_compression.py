"""Compression Patterns

Four patterns for compressing conversation history, from manual text
replacement to LLM-generated summaries with collaborative review.

Part 1: Manual compression — you write the summary text (no LLM needed)
Part 2: LLM auto-compression — let the model summarize (target_tokens=)
Part 3: Guided summarization — steer the summary with instructions=
Part 4: Collaborative review — inspect and edit drafts before committing

PINNED commits survive every compression mode verbatim.
SKIP commits are excluded from LLM summarization.
Manual mode (content=) handles single-group scenarios.
LLM mode handles any number of groups (including interleaved PINNED commits).

Demonstrates: compress(content=), compress(target_tokens=), preserve=,
              instructions=, system_prompt=, review=True, PendingCompress,
              edit_summary(), approve(), CompressResult fields,
              PINNED/SKIP interaction, before/after visualization,
              pprint(style="compact"), pprint(style="chat"), pprint(style="table")
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
# Part 1: Manual Compression (content=)
# =============================================================================
# You know exactly what the summary should say — no LLM needed. Compress
# commits into a summary with your own text. Shows compress-all, preserve=
# for keeping specific messages, and CompressResult inspection.
#
# Manual mode (content=) works when the compression produces a single group
# of non-preserved commits. For multi-group scenarios (preserved commits in
# the middle), use LLM mode (see Parts 2-4).

def part1_manual_compression():
    print("=" * 60)
    print("Part 1: MANUAL COMPRESSION (content=)")
    print("=" * 60)
    print()

    # --- 1a: Compress everything (default) ---

    print("  compress(content=...) replaces all eligible commits with")
    print("  your text. PINNED commits survive verbatim.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # Pin the system prompt so it survives compression
        sys_ci = t.system("You are a concise astronomy guide.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("How do stars form?")
        t.chat("What are black holes?")
        t.chat("Explain neutron stars.")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="compact")
        print(f"\n  {ctx_before.token_count} tokens, {len(ctx_before.messages)} messages")

        # Compress everything -- PINNED system prompt survives
        result = t.compress(
            content=(
                "User learned about three stellar phenomena: "
                "star formation (nebulae collapsing under gravity), "
                "black holes (collapsed massive stars with event horizons), "
                "and neutron stars (ultra-dense remnants of supernovae)."
            ),
        )

        # Inspect the CompressResult
        print(f"\n  CompressResult:")
        print(f"    compression_id:    {result.compression_id[:8]}")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    source_commits:    {len(result.source_commits)} archived")
        print(f"    summary_commits:   {len(result.summary_commits)} created")
        print(f"    preserved_commits: {len(result.preserved_commits)} kept (PINNED)")
        print(f"    new_head:          {result.new_head[:8]}")

        print("\n  AFTER compression:\n")
        ctx_after = t.compile()
        ctx_after.pprint(style="compact")
        print(f"\n  {ctx_after.token_count} tokens, {len(ctx_after.messages)} messages")
        print(f"\n  3 Q&A pairs -> 1 summary. PINNED system prompt survived.")

    # --- 1b: preserve= keeps specific messages ---

    print(f"\n{'-' * 60}")
    print("  1b: TEMPORARY PIN (preserve=)")
    print(f"{'-' * 60}")
    print()
    print("  preserve= treats commits as PINNED for this one compression")
    print("  only, without permanently annotating them. They pass through")
    print("  verbatim while everything else gets summarized.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise history tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What caused the fall of Rome?")
        t.chat("Explain the Renaissance.")

        # This Q&A is the one we want to keep
        r3 = t.chat("What was the Space Race and who won?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Get the user + assistant hashes for the Space Race Q&A (last pair)
        all_entries = list(t.log(limit=20))
        all_entries.reverse()
        # [0]=system, [1,2]=r1, [3,4]=r2, [5]=r3_user, [6]=r3_asst
        r3_hashes = [all_entries[5].commit_hash, all_entries[6].commit_hash]

        print(f"\n  Preserving the Space Race Q&A (last pair): "
              f"[{r3_hashes[0][:8]}, {r3_hashes[1][:8]}]")

        result = t.compress(
            content=(
                "User learned about the fall of Rome (economic decline, "
                "military overreach, barbarian invasions) and the "
                "Renaissance (cultural rebirth in 14th-17th century Europe)."
            ),
            preserve=r3_hashes,
        )

        print(f"\n  Compressed: {result.original_tokens} -> {result.compressed_tokens} tokens")
        print(f"  Preserved:  {len(result.preserved_commits)} commits")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  Space Race Q&A passed through verbatim. No permanent annotation needed.")

    # --- 1c: Continue after compression ---

    print(f"\n{'-' * 60}")
    print("  1c: CONTINUE AFTER COMPRESSION")
    print(f"{'-' * 60}")
    print()
    print("  After compression, the conversation continues normally.")
    print("  New messages build on top of the compressed summary.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise fitness coach.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What is progressive overload?")
        t.chat("Explain compound vs isolation exercises.")
        t.chat("What are the best recovery strategies?")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="compact")

        # Compress
        t.compress(
            content="User learned about progressive overload, compound vs isolation exercises, and recovery strategies.",
        )

        print("\n  AFTER compression (before new messages):\n")
        t.compile().pprint(style="compact")

        # Continue chatting -- the LLM sees the compressed summary as context
        print("\n  Continuing the conversation...\n")
        r = t.chat("Based on what we discussed, what's the single best exercise for a beginner?")
        r.pprint()

        print("\n  FINAL context:\n")
        t.compile().pprint(style="compact")

        print("\n  The LLM built on the compressed summary seamlessly.")


# =============================================================================
# Part 2: LLM Auto-Compression (target_tokens=)
# =============================================================================
# Let the LLM summarize your conversation automatically.
# PINNED commits survive verbatim. SKIP commits are excluded.
# When PINNED commits interleave with normal ones, the compressor splits
# them into separate groups, each getting its own LLM-generated summary.

def part2_llm_compression():
    print(f"\n{'=' * 60}")
    print("Part 2: LLM AUTO-COMPRESSION (target_tokens=)")
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


# =============================================================================
# Part 3: Guided Summarization (instructions=)
# =============================================================================
# instructions= adds guidance to the summarization prompt.
# Control what the summary focuses on without replacing the entire prompt.
# system_prompt= replaces the LLM's persona for the compression call.
# Both are stored in provenance.

def part3_guided_summarization():
    print(f"\n{'=' * 60}")
    print("Part 3: GUIDED SUMMARIZATION (instructions=)")
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


def main():
    part1_manual_compression()
    part2_llm_compression()
    part3_guided_summarization()
    part4_collaborative_review()


if __name__ == "__main__":
    main()
