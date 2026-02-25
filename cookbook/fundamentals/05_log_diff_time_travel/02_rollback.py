"""Rollback

Chat over several turns, then look back at what the context looked like
at earlier points. Use checkout to inspect past states and reset to
permanently roll back the conversation.

Demonstrates: compile(at_commit=), compile(at_time=), checkout(),
              reset(), pprint(style="chat") for session view
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


# =============================================================================
# Part 2: Rollback
# =============================================================================

def part2_time_travel():
    print("=" * 60)
    print("Part 2: Rollback")
    print("=" * 60)
    print()
    print("  Chat over several turns, then look back at what the context")
    print("  looked like at earlier points. Use checkout to inspect past")
    print("  states and reset to permanently roll back.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a concise geography tutor. One sentence answers.")

        # --- Build a 3-turn conversation ---

        print("=== Turn 1 ===\n")
        r1 = t.chat("What are the 3 largest countries by area?")
        r1.pprint()
        turn1_hash = r1.commit_info.commit_hash

        # Record the wall-clock time between turn 1 and 2
        midpoint = datetime.now(timezone.utc)

        print("=== Turn 2 ===\n")
        r2 = t.chat("Which of those has the highest population density?")
        r2.pprint()

        print("=== Turn 3 ===\n")
        r3 = t.chat("What's the capital of that country?")
        r3.pprint()

        # --- Show the full conversation as the LLM sees it now ---

        print("=== Full session (7 messages) ===\n")
        t.compile().pprint(style="chat")

        # --- Rollback: what did the LLM see after turn 1? ---
        # compile(at_commit=) rebuilds the context as it existed at that
        # commit — useful for debugging why the LLM gave a certain answer.

        print(f"\n=== Rollback: context as of turn 1 ({turn1_hash[:8]}) ===\n")
        past_ctx = t.compile(at_commit=turn1_hash)
        past_ctx.pprint(style="chat")
        print(f"\n  (turns 2-3 don't exist yet at this point)")

        # --- Rollback by timestamp ---
        # compile(at_time=) finds the latest commit before that timestamp.
        # Since midpoint is between turn 1 and turn 2, it matches turn 1.

        print(f"\n=== Rollback: context at midpoint timestamp ===\n")
        mid_ctx = t.compile(at_time=midpoint)
        print(f"  {len(mid_ctx.messages)} messages (same as turn 1 — "
              f"midpoint is between turn 1 and 2)")

        # --- Checkout: detached HEAD ---
        # checkout() moves HEAD to a past commit (like git checkout <hash>).
        # compile() now returns only messages up to that point.

        print(f"\n=== Checkout to turn 1 (detached HEAD) ===\n")
        t.checkout(turn1_hash)
        status = t.status()
        print(f"  HEAD:     {status.head_hash[:8]}")
        print(f"  Detached: {status.is_detached}")
        print(f"  Commits:  {status.commit_count} (turn 2-3 commits still exist, "
              f"just not reachable from HEAD)")

        # --- Return to main branch ---

        print(f"\n=== Checkout back to main ===\n")
        t.checkout("main")
        status = t.status()
        print(f"  HEAD:     {status.head_hash[:8]}")
        print(f"  Detached: {status.is_detached}")
        print(f"  Commits:  {status.commit_count} (all 3 turns visible again)")

        # --- Reset: permanently roll back ---
        # reset() moves the branch pointer backward. Turns 2-3 become
        # orphaned — still in storage, but invisible to compile().

        print(f"\n=== Reset to turn 1 — rolling back turns 2-3 ===\n")
        print("Before reset:")
        t.compile().pprint(style="chat")

        t.reset(turn1_hash)

        print("\nAfter reset:")
        t.compile().pprint(style="chat")


if __name__ == "__main__":
    part2_time_travel()
