"""Rollback

Manual reset -- permanently rolls back, no interaction.

Demonstrates: reset(), compile(), pprint(style="chat")
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

MODEL_ID = llm.small


# =============================================================================
# PART 1 -- Manual: reset() permanently rolls back
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Permanent Rollback with reset()")
    print("=" * 60)
    print()
    print("  Build a conversation, then reset() to an earlier commit.")
    print("  Later turns become orphaned -- invisible to compile().")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise geography tutor. One sentence answers.")

        r1 = t.chat("What are the 3 largest countries by area?")
        early_hash = r1.commit_info.commit_hash

        t.chat("Which of those has the highest population density?")
        t.chat("What's the capital of that country?")

        print("\n  Full conversation (7 messages):")
        t.compile().pprint(style="chat")

        # Permanently roll back to turn 1
        print(f"\n  Resetting to turn 1 ({early_hash[:8]})...\n")
        t.reset(early_hash)

        print("  After reset:")
        ctx = t.compile()
        ctx.pprint(style="chat")
        print(f"\n  {len(ctx.messages)} messages -- turns 2-3 are orphaned.")


def main():
    part1_manual()


if __name__ == "__main__":
    main()
