"""Branch Lifecycle — Create, Switch, List, Delete

Try an experimental explanation style without affecting main. Branching
is lightweight — it's just a pointer to a commit, not a copy. Create
branches, switch between them, list what exists, and clean up.

Demonstrates: branch(), switch(), list_branches(), current_branch,
              branch(switch=False), delete_branch(force=True)
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: Branch Lifecycle — Create, Switch, List, Delete
# =============================================================================

def part1_branch_lifecycle():
    print("=" * 60)
    print("Part 1: BRANCH LIFECYCLE")
    print("=" * 60)
    print()
    print("  Try an experimental explanation style without affecting main.")
    print("  Branching is lightweight — it's just a pointer to a commit,")
    print("  not a copy.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # --- Build a conversation on main ---

        print("=== Main branch: start a conversation ===\n")

        t.system("You are a concise Python tutor. One paragraph max.")
        r1 = t.chat("Explain what a decorator is.")
        r1.pprint()

        main_messages = len(t.compile().messages)
        print(f"\n  Branch: {t.current_branch}  |  Messages: {main_messages}\n")

        # --- Branch: try a different explanation style ---
        # branch() creates a new branch at HEAD and switches to it by default.

        print("=== Branch 'analogy': try a different angle ===\n")

        t.branch("analogy")
        print(f"  Switched to: {t.current_branch}")

        r2 = t.chat("Re-explain decorators using a real-world analogy, like gift wrapping.")
        r2.pprint()

        analogy_messages = len(t.compile().messages)
        print(f"\n  Branch: {t.current_branch}  |  Messages: {analogy_messages}\n")

        # --- List branches: see what exists ---
        # list_branches() returns BranchInfo objects with is_current flag.

        print("=== All branches ===\n")

        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"  {marker} {b.name:12s}  @ {b.commit_hash[:8]}")

        # --- Switch back to main: experiment is isolated ---
        # Main still has only the original messages — the analogy chat isn't here.

        print("\n=== Switch back to main ===\n")

        t.switch("main")
        ctx_main = t.compile()
        print(f"  Branch: {t.current_branch}  |  Messages: {len(ctx_main.messages)}")
        print(f"  (analogy branch had {analogy_messages} — main is untouched)")

        # --- Peek at analogy from main ---
        # switch() to analogy and back to verify both are intact.

        print("\n=== Peek at analogy ===\n")

        t.switch("analogy")
        ctx_analogy = t.compile()
        print(f"  Branch: {t.current_branch}  |  Messages: {len(ctx_analogy.messages)}")
        ctx_analogy.pprint(style="chat")

        # --- Create a branch without switching ---
        # branch(switch=False) keeps HEAD on the current branch.

        t.switch("main")
        t.branch("draft", switch=False)
        print(f"\n=== Created 'draft' without switching ===")
        print(f"  Still on: {t.current_branch}")

        print("\n  All branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        # --- Clean up ---
        # delete_branch() removes the branch pointer. force=True allows
        # deleting branches with unmerged work (analogy was never merged).

        print("\n=== Clean up ===\n")

        t.delete_branch("analogy", force=True)
        t.delete_branch("draft", force=True)

        remaining = [b.name for b in t.list_branches()]
        print(f"  Remaining branches: {remaining}")


if __name__ == "__main__":
    part1_branch_lifecycle()
