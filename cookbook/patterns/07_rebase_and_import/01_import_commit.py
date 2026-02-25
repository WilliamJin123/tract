"""Import Commit (Cherry-Pick Across Branches)

Grab one useful commit from another branch using import_commit —
Tract's cherry-pick. Import creates a new commit with a new hash
but same content, without merging the whole branch.

Demonstrates: import_commit(), ImportResult, branch(), switch(),
              pprint(style="compact"), pprint(style="chat"),
              before/after state visualization
"""

import os

from dotenv import load_dotenv

from tract import Tract, CompiledContext

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
    # =================================================================
    # Part 1: import_commit (cherry-pick)
    # =================================================================

    print("=" * 60)
    print("Part 1: IMPORT COMMIT (cherry-pick)")
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

        # Step 1: Build a shared base conversation
        print("\n  Step 1: Build shared base on main\n")

        t.system("You are a concise cooking instructor. One paragraph max.")
        t.chat("What's the difference between braising and roasting?")

        print("  main (shared base):")
        t.compile().pprint(style="compact")

        # Step 2: Branch and build a deep-dive conversation
        print("\n  Step 2: Branch 'deep-dive' and add a Q&A pair\n")

        t.branch("deep-dive")

        # Use user() + generate() separately to capture both commit hashes
        user_ci = t.user("Explain when to use baking soda vs baking powder.")
        r_deep = t.generate()

        deep_user_hash = user_ci.commit_hash
        deep_asst_hash = r_deep.commit_info.commit_hash

        print("  deep-dive (has the Q&A we want to cherry-pick):")
        t.compile().pprint(style="compact")

        # Step 3: Main advances independently
        print("\n  Step 3: Switch to main -- it advances on its own\n")

        t.switch("main")
        t.chat("Give me a quick tip for caramelizing onions.")

        print("  main (diverged -- no deep-dive content):")
        t.compile().pprint(style="compact")

        # Step 4: Cherry-pick both commits onto main
        print("\n  Step 4: Import both the question and answer from deep-dive\n")

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

        # Step 5: Show the result
        print("\n  AFTER IMPORT -- main now includes the deep-dive Q&A:\n")

        ctx = t.compile()
        print(f"  main ({len(ctx.messages)} messages):")
        ctx.pprint(style="chat")

        print("\n  Key: new hashes, same content. deep-dive branch is untouched.")


if __name__ == "__main__":
    main()
