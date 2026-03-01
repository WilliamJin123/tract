"""Log and Diff

Chat with an LLM over several turns, then inspect the commit history,
diff two points in the conversation, and view individual commits.

Demonstrates: log(), show(), diff(), DiffResult.pprint(), stat_only mode,
              CommitInfo str/pprint, reversed() for chronological order
"""

import os

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


if __name__ == "__main__":
    part1_log_and_diff()
