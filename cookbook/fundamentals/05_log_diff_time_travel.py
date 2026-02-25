"""Log, Diff, and Time Travel

Chat with an LLM over several turns, then inspect the commit history,
diff two points in the conversation, and look back at what the context
looked like at earlier points. Finally, use checkout and reset to
navigate and roll back the conversation.

Part 1 — Log and Diff:
  Demonstrates: log(), show(), diff(), DiffResult.pprint(), stat_only mode,
                CommitInfo str/pprint, reversed() for chronological order

Part 2 — Time Travel:
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
# Part 1: Log and Diff
# =============================================================================

def part1_log_and_diff():
    print("=" * 60)
    print("Part 1: LOG AND DIFF")
    print("=" * 60)
    print()
    print("  Build a short conversation, walk the commit history, inspect")
    print("  individual commits, and diff two points in the conversation.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a concise geography tutor. One sentence answers.")

        # --- Build a short conversation with the LLM ---

        print("=== Chatting ===\n")
        r1 = t.chat("What is the capital of France?")
        r1.pprint()

        r2 = t.chat("And Germany?")
        r2.pprint()

        # --- Log: walk history from HEAD backward ---

        print("=== Log (newest first) ===\n")
        history = t.log()
        for entry in history:
            print(f"  {entry}")

        # --- Log reversed: chronological order ---

        print(f"\n=== Log (chronological, {len(history)} commits) ===\n")
        for entry in reversed(history):
            print(f"  {entry}")

        # --- Show: inspect a single commit with full content ---

        print("\n=== Show: first LLM response ===\n")
        t.show(r1.commit_info)

        # --- Edit the first answer ---
        # t.assistant(edit=...) replaces a previous response's content.
        # This creates a new EDIT commit; the original is still in history.

        print("\n=== Editing r1's response ===\n")
        fix = t.assistant(
            "The capital of France is Paris, also known as the City of Light.",
            edit=r1.commit_info.commit_hash,
            message="Add City of Light detail",
        )
        print(f"  Original (r1): {r1.commit_info.commit_hash[:8]}")
        print(f"  Edit commit:   {fix.commit_hash[:8]}")

        # --- Diff: compare compiled contexts at two commits ---
        # diff(A, B) compares the FULL compiled context at commit A vs B.
        # At r1: [system, user1, assistant(original)]  — 3 messages
        # At fix: [system, user1, assistant(edited), user2, assistant2] — 5 messages
        # So we see: 2 unchanged (system + user1), 1 modified (assistant text
        # changed by the edit), and 2 added (user2 + assistant2 exist in the
        # edit commit's chain but not in r1's).

        print("\n=== Diff: context at r1 vs context at edit commit ===\n")
        result = t.diff(r1.commit_info.commit_hash, fix.commit_hash)
        result.pprint()

        # --- Diff: stat-only mode (like git diff --stat) ---

        print("=== Same diff, stat-only (like git diff --stat) ===\n")
        result.pprint(stat_only=True)

        # --- Diff: early commit vs HEAD ---
        # Comparing the context at the first user message (2 messages: system
        # + user1) against HEAD (5 messages) — shows everything added since.

        first_q_hash = history[-2].commit_hash  # first user message
        print(f"\n=== Diff: first user message ({first_q_hash[:8]}) → HEAD ===\n")
        full_diff = t.diff(first_q_hash)
        full_diff.pprint(stat_only=True)


# =============================================================================
# Part 2: Time Travel
# =============================================================================

def part2_time_travel():
    print("=" * 60)
    print("Part 2: TIME TRAVEL")
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

        # --- Time travel: what did the LLM see after turn 1? ---
        # compile(at_commit=) rebuilds the context as it existed at that
        # commit — useful for debugging why the LLM gave a certain answer.

        print(f"\n=== Time travel: context as of turn 1 ({turn1_hash[:8]}) ===\n")
        past_ctx = t.compile(at_commit=turn1_hash)
        past_ctx.pprint(style="chat")
        print(f"\n  (turns 2-3 don't exist yet at this point)")

        # --- Time travel by timestamp ---
        # compile(at_time=) finds the latest commit before that timestamp.
        # Since midpoint is between turn 1 and turn 2, it matches turn 1.

        print(f"\n=== Time travel: context at midpoint timestamp ===\n")
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


def main():
    part1_log_and_diff()
    print()
    part2_time_travel()


if __name__ == "__main__":
    main()
