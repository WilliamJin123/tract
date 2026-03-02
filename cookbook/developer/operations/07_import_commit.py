"""Import Commit (Cherry-Pick Across Branches)

Three tiers of import_commit usage -- manual cherry-pick, interactive
confirmation with commit preview, and agent-driven toolkit execution.

PART 1 -- Manual           Direct import_commit(), inspect ImportResult
PART 2 -- Interactive       show() preview, click.confirm before import
PART 3 -- LLM / Agent      ToolExecutor dispatches import_commit

Demonstrates: import_commit(), ImportResult, branch(), switch(),
              show(), click.confirm, ToolExecutor,
              pprint(style="compact"), pprint(style="chat")
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract, ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _build_cherry_pick_scenario(t):
    """Helper: build main + deep-dive branch with a Q&A to cherry-pick."""
    t.system("You are a concise cooking instructor. One paragraph max.")
    t.chat("What's the difference between braising and roasting?")

    t.branch("deep-dive")
    user_ci = t.user("Explain when to use baking soda vs baking powder.")
    r_deep = t.generate()

    deep_user_hash = user_ci.commit_hash
    deep_asst_hash = r_deep.commit_info.commit_hash

    t.switch("main")
    t.chat("Give me a quick tip for caramelizing onions.")

    return deep_user_hash, deep_asst_hash


# =============================================================================
# PART 1 -- Manual: Direct import_commit(), inspect ImportResult
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Import Commit (Cherry-Pick)")
    print("=" * 60)
    print()
    print("  Scenario: deep-dive branch has a great Q&A pair about")
    print("  baking soda vs baking powder. We want BOTH the question")
    print("  and the answer on main, without merging the whole branch.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        deep_user_hash, deep_asst_hash = _build_cherry_pick_scenario(t)

        print("\n  main (diverged -- no deep-dive content):")
        t.compile().pprint(style="compact")

        # Cherry-pick both commits onto main
        print("\n  Importing both the question and answer from deep-dive\n")

        result_user = t.import_commit(deep_user_hash)
        result_asst = t.import_commit(deep_asst_hash)

        print(f"  Imported user question:")
        print(f"    original: {result_user.original_commit.commit_hash[:8]}")
        print(f"    new:      {result_user.new_commit.commit_hash[:8]}")
        print(f"    same content: {result_user.new_commit.content_hash == result_user.original_commit.content_hash}")
        print(f"    issues:   {len(result_user.issues)}")

        print(f"\n  Imported assistant answer:")
        print(f"    original: {result_asst.original_commit.commit_hash[:8]}")
        print(f"    new:      {result_asst.new_commit.commit_hash[:8]}")
        print(f"    same content: {result_asst.new_commit.content_hash == result_asst.original_commit.content_hash}")
        print(f"    issues:   {len(result_asst.issues)}")

        # Show the result
        print("\n  AFTER IMPORT -- main now includes the deep-dive Q&A:\n")
        ctx = t.compile()
        print(f"  main ({len(ctx.messages)} messages):")
        ctx.pprint(style="chat")

        print("\n  Key: new hashes, same content. deep-dive branch is untouched.")


# =============================================================================
# PART 2 -- Interactive: show() preview, click.confirm before import
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Preview and Confirm Import")
    print("=" * 60)
    print()
    print("  Show commit details before importing. User confirms each one.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        deep_user_hash, deep_asst_hash = _build_cherry_pick_scenario(t)

        for label, hash in [("user question", deep_user_hash), ("assistant answer", deep_asst_hash)]:
            # Preview the commit
            entry = t.show(hash)
            print(f"\n  Commit {hash[:8]} ({label}):")
            print(f"    role:    {entry.role}")
            print(f"    content: {str(entry.content)[:80]}...")

            if click.confirm(f"  Import commit {hash[:8]}?", default=True):
                result = t.import_commit(hash)
                print(f"  Imported: {result.original_commit.commit_hash[:8]} -> {result.new_commit.commit_hash[:8]}")
                assert result.new_commit.commit_hash != result.original_commit.commit_hash
            else:
                print(f"  Skipped {hash[:8]}.")

        print(f"\n  AFTER IMPORT:")
        t.compile().pprint(style="compact")


# =============================================================================
# PART 3 -- LLM / Agent: ToolExecutor dispatches import_commit
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: ToolExecutor Import Commit")
    print("=" * 60)
    print()
    print("  An LLM agent uses ToolExecutor to cherry-pick commits")
    print("  across branches without human interaction.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        deep_user_hash, deep_asst_hash = _build_cherry_pick_scenario(t)

        executor = ToolExecutor(t)

        print(f"\n  Importing user question via ToolExecutor:")
        result = executor.execute("import_commit", {"commit_hash": deep_user_hash})
        print(f"  Result: {result}")

        print(f"\n  Importing assistant answer via ToolExecutor:")
        result = executor.execute("import_commit", {"commit_hash": deep_asst_hash})
        print(f"  Result: {result}")

        print(f"\n  AFTER IMPORT:")
        t.compile().pprint(style="chat")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
