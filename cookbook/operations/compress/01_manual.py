"""Manual Compression (content=)

You know exactly what the summary should say — no LLM needed. Compress
commits into a summary with your own text. Shows compress-all, preserve=
for keeping specific messages, and CompressResult inspection.

Manual mode (content=) works when the compression produces a single group
of non-preserved commits. For multi-group scenarios (preserved commits in
the middle), use LLM mode (see Parts 2-4).

PINNED commits survive every compression mode verbatim.

Demonstrates: compress(content=), preserve=, CompressResult fields,
              PINNED interaction, before/after visualization,
              pprint(style="compact"), pprint(style="chat"),
              continue after compress
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


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
        ctx_before.pprint(style="chat")

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

        # No PINNED system here — preserve= is the focus of this demo.
        # (PINNED + preserve= would create multiple groups, requiring LLM mode.)
        t.system("You are a concise history tutor.")

        t.chat("What caused the fall of Rome?", reasoning=False)
        t.chat("Explain the Renaissance.", reasoning=False)

        # This Q&A is the one we want to keep
        r3 = t.chat("What was the Space Race and who won?", reasoning=False)

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # Get the user + assistant hashes for the Space Race Q&A (last pair)
        all_entries = list(t.log(limit=20))
        all_entries.reverse()
        # [0]=system, [1,2]=r1, [3,4]=r2, [5]=r3_user, [6]=r3_asst
        r3_hashes = [all_entries[5].commit_hash, all_entries[6].commit_hash]

        print(f"\n  Preserving the Space Race Q&A (last pair): "
              f"[{r3_hashes[0][:8]}, {r3_hashes[1][:8]}]")

        result = t.compress(
            content=(
                "User discussed history: the fall of Rome (economic decline, "
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
        ctx_before.pprint(style="chat")

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
        t.compile().pprint(style="chat")

        print("\n  The LLM built on the compressed summary seamlessly.")


if __name__ == "__main__":
    part1_manual_compression()
