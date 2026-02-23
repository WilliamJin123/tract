"""Manual Compression

You know exactly what the summary should say -- no LLM needed. Compress
commits into a summary with your own text. Shows compress-all, preserve=
for keeping specific messages, and CompressResult inspection.

Manual mode (content=) works when the compression produces a single group
of non-preserved commits. For multi-group scenarios (preserved commits in
the middle), use LLM mode (see 02_llm_compression.py).

Demonstrates: compress(content=), compress-all default, preserve=
              temporary pin, CompressResult fields, PINNED system prompt,
              before/after visualization with pprint(style="compact")
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
    # Part 1: Compress everything (default)
    # =================================================================

    print("=" * 60)
    print("Part 1: COMPRESS ALL (manual content)")
    print("=" * 60)
    print()
    print("  compress(content=...) replaces all eligible commits with")
    print("  your text. PINNED commits survive verbatim.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # Pin the system prompt so it survives compression
        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What are decorators?")
        t.chat("What about context managers?")
        t.chat("And generators?")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="compact")
        print(f"\n  {ctx_before.token_count} tokens, {len(ctx_before.messages)} messages")

        # Compress everything -- PINNED system prompt survives
        result = t.compress(
            content=(
                "User learned about three Python features: "
                "decorators (@syntax wrapping functions), "
                "context managers (with statements for setup/teardown), "
                "and generators (yield keyword for lazy iteration)."
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

    # =================================================================
    # Part 2: preserve= keeps specific messages
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: TEMPORARY PIN (preserve=)")
    print("=" * 60)
    print()
    print("  preserve= treats commits as PINNED for this one compression")
    print("  only, without permanently annotating them. They pass through")
    print("  verbatim while everything else gets summarized.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What are decorators?")
        t.chat("What about context managers?")

        # This Q&A is the one we want to keep
        r3 = t.chat("What's the GIL and how does it affect threading?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Get the user + assistant hashes for the GIL Q&A (last pair)
        all_entries = list(t.log(limit=20))
        all_entries.reverse()
        # [0]=system, [1,2]=r1, [3,4]=r2, [5]=r3_user, [6]=r3_asst
        r3_hashes = [all_entries[5].commit_hash, all_entries[6].commit_hash]

        print(f"\n  Preserving the GIL Q&A (last pair): "
              f"[{r3_hashes[0][:8]}, {r3_hashes[1][:8]}]")

        result = t.compress(
            content=(
                "User learned about decorators (@syntax wrapping) "
                "and context managers (with statements for setup/teardown)."
            ),
            preserve=r3_hashes,
        )

        print(f"\n  Compressed: {result.original_tokens} -> {result.compressed_tokens} tokens")
        print(f"  Preserved:  {len(result.preserved_commits)} commits")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  GIL Q&A passed through verbatim. No permanent annotation needed.")

    # =================================================================
    # Part 3: Continue after compression
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 3: CONTINUE AFTER COMPRESSION")
    print("=" * 60)
    print()
    print("  After compression, the conversation continues normally.")
    print("  New messages build on top of the compressed summary.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        sys_ci = t.system("You are a concise Python tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What are decorators?")
        t.chat("What about context managers?")
        t.chat("And generators?")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="compact")

        # Compress
        t.compress(
            content="User learned about decorators, context managers, and generators.",
        )

        print("\n  AFTER compression (before new messages):\n")
        t.compile().pprint(style="compact")

        # Continue chatting -- the LLM sees the compressed summary as context
        print("\n  Continuing the conversation...\n")
        r = t.chat("Based on what we discussed, which concept is most useful for file handling?")
        r.pprint()

        print("\n  FINAL context:\n")
        t.compile().pprint(style="compact")

        print("\n  The LLM built on the compressed summary seamlessly.")


if __name__ == "__main__":
    main()
