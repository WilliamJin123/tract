"""LLM Auto-Compression (target_tokens=)

Let the LLM summarize your conversation automatically.
PINNED commits survive verbatim. SKIP commits are excluded.
When PINNED commits interleave with normal ones, the compressor splits
them into separate groups, each getting its own LLM-generated summary.

Demonstrates: compress(target_tokens=), PINNED/SKIP interaction,
              CompressResult fields, before/after visualization,
              pprint(style="chat"), pprint(style="table")
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


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


if __name__ == "__main__":
    part2_llm_compression()
