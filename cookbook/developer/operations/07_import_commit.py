"""Import Commit (Cherry-Pick Across Branches)

PART 1 -- Manual           Direct import_commit(), inspect ImportResult

Demonstrates: import_commit(), ImportResult, branch(), switch(),
              show(), pprint(style="compact"), pprint(style="chat")
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


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

def main():
    print("=" * 60)
    print("PART 1 -- Manual: Import Commit (Cherry-Pick)")
    print("=" * 60)
    print()
    print("  Scenario: deep-dive branch has a great Q&A pair about")
    print("  baking soda vs baking powder. We want BOTH the question")
    print("  and the answer on main, without merging the whole branch.")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
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


if __name__ == "__main__":
    main()
