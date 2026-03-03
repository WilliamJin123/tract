"""Core compression: manual compress with your own summary text.

  PART 1 -- Manual:      compress(content="your summary"), no LLM needed
"""

import sys
from pathlib import Path

from tract import Priority, Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


def main():
    print("=" * 60)
    print("PART 1 -- Manual: compress(content=), no LLM needed")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # Pin the system prompt so it survives compression
        sys_ci = t.system("You are a concise astronomy guide.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("How do stars form?")
        t.chat("What are black holes?")
        t.chat("Explain neutron stars.")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # Manual summary -- no LLM needed
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
        print(f"    preserved_commits: {len(result.preserved_commits)} kept (PINNED)")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")
        print(f"\n  3 Q&A pairs -> 1 summary. PINNED system prompt survived.")


if __name__ == "__main__":
    main()
