"""Manual Compression

You know exactly what the summary should say -- no LLM needed. Compress
a range of commits into a single summary commit with your own text.
Shows all three range selection modes and the preserve= temporary pin.

Demonstrates: compress(content=), from_commit/to_commit range,
              commits= explicit list, compress-all default,
              preserve= temporary pin, CompressResult fields,
              before/after visualization with pprint(style="compact")
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    # =================================================================
    # Part 1: Range compression (from_commit / to_commit)
    # =================================================================

    print("=" * 60)
    print("Part 1: RANGE COMPRESSION (from_commit / to_commit)")
    print("=" * 60)
    print()
    print("  Compress a specific range of commits using your own summary.")
    print("  content= bypasses the LLM summarizer entirely.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # Build a multi-turn conversation
        t.system("You are a concise Python tutor.")
        r1 = t.chat("What are decorators?")
        r2 = t.chat("What about context managers?")
        r3 = t.chat("And generators?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Compress the first two Q&A pairs (r1 + r2), keep the third (r3)
        # log() returns newest-first; reverse for chronological order
        all_entries = list(t.log(limit=20))
        all_entries.reverse()
        # [0]=system, [1]=r1_user, [2]=r1_asst, [3]=r2_user, [4]=r2_asst, ...
        from_hash = all_entries[1].commit_hash   # first user message
        to_hash = r2.commit_info.commit_hash     # second assistant response

        print(f"\n  Compressing range: {from_hash[:8]}..{to_hash[:8]}")
        print(f"  (first two Q&A pairs -> one summary)\n")

        result = t.compress(
            from_commit=from_hash,
            to_commit=to_hash,
            content=(
                "User asked about Python decorators and context managers. "
                "Decorators are functions that wrap other functions to extend "
                "behavior (using @syntax). Context managers handle setup/teardown "
                "via __enter__/__exit__ (using 'with' statements)."
            ),
        )

        # Inspect the CompressResult -- rich data about what happened
        print(f"  CompressResult:")
        print(f"    compression_id:    {result.compression_id[:8]}")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    source_commits:    {len(result.source_commits)} archived")
        print(f"    summary_commits:   {len(result.summary_commits)} created")
        print(f"    preserved_commits: {len(result.preserved_commits)} kept as-is")
        print(f"    new_head:          {result.new_head[:8]}")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  First two Q&A pairs -> one summary. Generators Q&A untouched.")

    # =================================================================
    # Part 2: Explicit commit list (commits=)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: EXPLICIT COMMIT LIST (commits=)")
    print("=" * 60)
    print()
    print("  Cherry-pick specific commits to compress by hash.")
    print("  Useful when the commits to compress aren't contiguous.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a concise Python tutor.")
        r1 = t.chat("What's a list comprehension?")
        r2 = t.chat("What's a dict comprehension?")
        r3 = t.chat("What's a set comprehension?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Compress r1 and r3 (skip r2) using explicit hashes
        all_entries = list(t.log(limit=20))
        all_entries.reverse()
        # [0]=system, [1,2]=r1 pair, [3,4]=r2 pair, [5,6]=r3 pair
        target_commits = [
            all_entries[1].commit_hash, all_entries[2].commit_hash,  # r1 pair
            all_entries[5].commit_hash, all_entries[6].commit_hash,  # r3 pair
        ]

        print(f"\n  Compressing 4 specific commits (r1 + r3, skipping r2):\n")
        for h in target_commits:
            print(f"    {h[:8]}")

        result = t.compress(
            commits=target_commits,
            content=(
                "User asked about list and set comprehensions. "
                "List: [expr for x in iter]. Set: {expr for x in iter} "
                "for unique values."
            ),
        )

        print(f"\n  Compressed {len(result.source_commits)} commits "
              f"({result.original_tokens} -> {result.compressed_tokens} tokens)")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  Dict comprehension Q&A (r2) survived untouched.")

    # =================================================================
    # Part 3: preserve= temporary pin
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 3: TEMPORARY PIN (preserve=)")
    print("=" * 60)
    print()
    print("  preserve= treats commits as PINNED for this one compression,")
    print("  without permanently annotating them. They pass through")
    print("  verbatim while everything else gets compressed.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a concise Python tutor.")
        r1 = t.chat("What are decorators?")
        r2 = t.chat("What's the GIL?")           # we want to keep this one
        r3 = t.chat("What are metaclasses?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Preserve r2's user+assistant commits
        all_entries = list(t.log(limit=20))
        all_entries.reverse()
        r2_hashes = [all_entries[3].commit_hash, all_entries[4].commit_hash]

        print(f"\n  Compressing all, but preserve= the GIL Q&A:")
        print(f"    [{r2_hashes[0][:8]}, {r2_hashes[1][:8]}]\n")

        result = t.compress(
            content=(
                "User asked about Python decorators and metaclasses. "
                "Decorators wrap functions with @syntax. Metaclasses "
                "control class creation via type() or __metaclass__."
            ),
            preserve=r2_hashes,
        )

        print(f"  Compressed: {result.original_tokens} -> {result.compressed_tokens} tokens")
        print(f"  Preserved:  {len(result.preserved_commits)} commits (GIL Q&A)")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")

        print("\n  GIL Q&A passed through verbatim -- no permanent annotation needed.")


if __name__ == "__main__":
    main()
