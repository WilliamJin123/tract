"""Time Travel

Chat with an LLM over several turns, then reconstruct exactly what the
LLM was seeing at any past point. Useful for debugging bad answers —
compile(at_commit=) rebuilds the context as of any historical commit.
checkout() moves HEAD there for interactive inspection.

Demonstrates: compile(at_commit=), compile(at_time=), checkout(),
              reset(), detached HEAD, ORIG_HEAD
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a concise geography tutor.")

        # --- Build a multi-turn conversation ---

        print("=== Building conversation ===\n")

        r1 = t.chat("What are the 3 largest countries by area?")
        print(f"Turn 1: {r1.commit_info.commit_hash[:8]}")
        snapshot_after_turn1 = r1.commit_info.commit_hash

        # Record the time between turns for at_time demo
        midpoint = datetime.now(timezone.utc)

        r2 = t.chat("Which of those has the highest population density?")
        print(f"Turn 2: {r2.commit_info.commit_hash[:8]}")

        r3 = t.chat("What's the capital of that country?")
        print(f"Turn 3: {r3.commit_info.commit_hash[:8]}")

        current_ctx = t.compile()
        print(f"\nCurrent context: {len(current_ctx.messages)} messages, "
              f"{current_ctx.token_count} tokens\n")

        # --- Time travel: what did the LLM see at turn 1? ---

        print("=== Time travel: context at turn 1 ===\n")
        past_ctx = t.compile(at_commit=snapshot_after_turn1)
        print(f"At turn 1: {len(past_ctx.messages)} messages, "
              f"{past_ctx.token_count} tokens")
        past_ctx.pprint()

        # --- Time travel by timestamp ---

        print("\n=== Time travel: context at midpoint timestamp ===\n")
        mid_ctx = t.compile(at_time=midpoint)
        print(f"At midpoint: {len(mid_ctx.messages)} messages "
              f"(should match turn 1)\n")

        # --- Checkout: move HEAD to a past commit ---

        print("=== Checkout: detached HEAD at turn 1 ===\n")
        t.checkout(snapshot_after_turn1)
        status = t.status()
        print(f"HEAD: {status.head_hash[:8]}")
        print(f"Detached: {status.is_detached}")
        print(f"Commits visible: {status.commit_count}\n")

        # compile() at detached HEAD matches the time-travel view
        detached_ctx = t.compile()
        print(f"Compiled at detached HEAD: {len(detached_ctx.messages)} messages "
              f"(same as at_commit view)\n")

        # --- Return to latest ---

        print("=== Checkout back to main ===\n")
        t.checkout("main")
        status = t.status()
        print(f"HEAD: {status.head_hash[:8]}")
        print(f"Detached: {status.is_detached}")
        print(f"Commits visible: {status.commit_count}\n")

        # --- Reset: move HEAD backward permanently ---

        print("=== Reset to turn 1 ===\n")
        t.reset(snapshot_after_turn1)
        status = t.status()
        print(f"HEAD after reset: {status.head_hash[:8]}")
        print(f"Commits visible: {status.commit_count}")
        print("(turns 2-3 are orphaned — still in storage, invisible to compile)\n")

        # Compile confirms only turn 1 is visible
        reset_ctx = t.compile()
        print(f"Compiled after reset: {len(reset_ctx.messages)} messages")
        reset_ctx.pprint()


if __name__ == "__main__":
    main()
