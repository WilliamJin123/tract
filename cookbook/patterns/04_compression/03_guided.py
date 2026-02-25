"""Guided Summarization (instructions=, system_prompt=)

instructions= adds guidance to the summarization prompt.
Control what the summary focuses on without replacing the entire prompt.
system_prompt= replaces the LLM's persona for the compression call.
Both are stored in provenance.

Demonstrates: compress(target_tokens=, instructions=), system_prompt=,
              PINNED interaction, before/after visualization,
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


if __name__ == "__main__":
    part3_guided_summarization()
