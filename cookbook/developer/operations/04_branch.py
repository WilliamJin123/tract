"""Branch Lifecycle -- Create, Switch, List, Delete

Three tiers of branch management -- manual API calls, interactive prompts,
and agent-driven toolkit execution.

PART 1 -- Manual           Direct branch/switch/list/delete calls
PART 2 -- Interactive       click.confirm, click.prompt, human decides
PART 3 -- LLM / Agent      ToolExecutor dispatches branch operations

Demonstrates: branch(), switch(), list_branches(), current_branch,
              branch(switch=False), delete_branch(force=True),
              click.confirm, click.prompt, ToolExecutor
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract, ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# PART 1 -- Manual: Direct API calls, no LLM, deterministic
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Branch Lifecycle")
    print("=" * 60)
    print()
    print("  Try an experimental explanation style without affecting main.")
    print("  Branching is lightweight -- it's just a pointer to a commit,")
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


# =============================================================================
# PART 2 -- Interactive: click.confirm, click.prompt, human decides
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Branch Management with Prompts")
    print("=" * 60)
    print()
    print("  Use click prompts to let the user decide branch operations.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise Python tutor.")
        t.chat("What is a list comprehension?")

        # Create branch with confirmation
        if click.confirm("  Create branch 'experiment'?", default=True):
            t.branch("experiment")
            print(f"  Created and switched to: {t.current_branch}")
            t.chat("Explain generator expressions.")
        else:
            print("  Skipped branch creation.")
            return

        t.switch("main")

        # List branches with numbered display
        branches = t.list_branches()
        print(f"\n  Available branches:")
        for i, b in enumerate(branches):
            marker = "*" if b.is_current else " "
            print(f"    [{i}] {marker} {b.name:15s} @ {b.commit_hash[:8]}")

        # Switch with interactive selection
        choice = click.prompt(
            "  Switch to which branch? (number)",
            type=int,
            default=0,
        )
        if 0 <= choice < len(branches):
            t.switch(branches[choice].name)
            print(f"  Switched to: {t.current_branch}")
            t.compile().pprint(style="compact")

        # Delete with force confirmation
        t.switch("main")
        if click.confirm("  Force delete unmerged branch 'experiment'?", default=False):
            t.delete_branch("experiment", force=True)
            print(f"  Deleted 'experiment'.")
        else:
            print("  Kept 'experiment'.")

        remaining = [b.name for b in t.list_branches()]
        print(f"\n  Remaining branches: {remaining}")


# =============================================================================
# PART 3 -- LLM / Agent: ToolExecutor dispatches branch operations
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: ToolExecutor Branch Operations")
    print("=" * 60)
    print()
    print("  An LLM agent uses ToolExecutor to manage branches")
    print("  programmatically -- no human prompts needed.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise Python tutor.")
        t.chat("What is a class?")

        executor = ToolExecutor(t)

        # Agent creates a branch
        result = executor.execute("branch", {"name": "auto-research"})
        print(f"\n  branch('auto-research'): {result}")

        # Agent adds work on the new branch
        t.chat("Explain inheritance in Python.")

        # Agent lists branches
        result = executor.execute("list_branches", {})
        print(f"\n  list_branches(): {result}")

        # Agent switches back to main
        result = executor.execute("switch", {"branch": "main"})
        print(f"\n  switch('main'): {result}")
        print(f"  Current branch: {t.current_branch}")

        # Agent cleans up
        t.delete_branch("auto-research", force=True)
        remaining = [b.name for b in t.list_branches()]
        print(f"\n  After cleanup: {remaining}")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
