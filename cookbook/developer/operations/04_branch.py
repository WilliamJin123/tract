"""Branch Lifecycle -- Create, Switch, List, Delete

PART 1 -- Manual           Direct branch/switch/list/delete calls

Demonstrates: branch(), switch(), list_branches(), current_branch,
              branch(switch=False), delete_branch(force=True)
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


# =============================================================================
# PART 1 -- Manual: Direct API calls, no LLM, deterministic
# =============================================================================

def main():
    print("=" * 60)
    print("PART 1 -- Manual: Branch Lifecycle")
    print("=" * 60)
    print()
    print("  Try an experimental explanation style without affecting main.")
    print("  Branching is lightweight -- it's just a pointer to a commit,")
    print("  not a copy.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
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

        print("=== Branch 'analogy': try a different angle ===\n")

        t.branch("analogy")
        print(f"  Switched to: {t.current_branch}")

        r2 = t.chat("Re-explain decorators using a real-world analogy, like gift wrapping.")
        r2.pprint()

        analogy_messages = len(t.compile().messages)
        print(f"\n  Branch: {t.current_branch}  |  Messages: {analogy_messages}\n")

        # --- List branches ---

        print("=== All branches ===\n")

        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"  {marker} {b.name:12s}  @ {b.commit_hash[:8]}")

        # --- Switch back to main ---

        print("\n=== Switch back to main ===\n")

        t.switch("main")
        ctx_main = t.compile()
        print(f"  Branch: {t.current_branch}  |  Messages: {len(ctx_main.messages)}")
        print(f"  (analogy branch had {analogy_messages} -- main is untouched)")

        # --- Peek at analogy from main ---

        print("\n=== Peek at analogy ===\n")

        t.switch("analogy")
        ctx_analogy = t.compile()
        print(f"  Branch: {t.current_branch}  |  Messages: {len(ctx_analogy.messages)}")
        ctx_analogy.pprint(style="chat")

        # --- Create a branch without switching ---

        t.switch("main")
        t.branch("draft", switch=False)
        print(f"\n=== Created 'draft' without switching ===")
        print(f"  Still on: {t.current_branch}")

        print("\n  All branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        # --- Clean up ---

        print("\n=== Clean up ===\n")

        t.delete_branch("analogy", force=True)
        t.delete_branch("draft", force=True)

        remaining = [b.name for b in t.list_branches()]
        print(f"  Remaining branches: {remaining}")


if __name__ == "__main__":
    main()
