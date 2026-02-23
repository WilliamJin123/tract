"""Import Commit and Rebase

Grab one useful commit from another branch (import_commit -- Tract's
cherry-pick), and update a stale branch to include the latest main
(rebase). Import creates a new commit with a new hash but same content.
Rebase replays your branch's commits on top of the target's tip.

Demonstrates: import_commit(), ImportResult, rebase(), RebaseResult,
              replayed_commits, original_commits, new_head,
              pprint(style="compact"), before/after state visualization
"""

import os

from dotenv import load_dotenv

from tract import Tract, CompiledContext

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    # =================================================================
    # Part 1: import_commit (cherry-pick)
    # =================================================================

    print("=" * 60)
    print("Part 1: IMPORT COMMIT (cherry-pick)")
    print("=" * 60)
    print()
    print("  Scenario: deep-dive branch has a great Q&A pair about")
    print("  @staticmethod vs @classmethod. We want BOTH the question")
    print("  and the answer on main, without merging the whole branch.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # Step 1: Build a shared base conversation
        print("\n  Step 1: Build shared base on main\n")

        t.system("You are a concise Python tutor. One paragraph max.")
        t.chat("What's a decorator?")

        print("  main (shared base):")
        t.compile().pprint(style="compact")

        # Step 2: Branch and build a deep-dive conversation
        print("\n  Step 2: Branch 'deep-dive' and add a Q&A pair\n")

        t.branch("deep-dive")

        # Use user() + generate() separately to capture both commit hashes
        user_ci = t.user("Explain the difference between @staticmethod and @classmethod.")
        r_deep = t.generate()

        deep_user_hash = user_ci.commit_hash
        deep_asst_hash = r_deep.commit_info.commit_hash

        print("  deep-dive (has the Q&A we want to cherry-pick):")
        t.compile().pprint(style="compact")

        # Step 3: Main advances independently
        print("\n  Step 3: Switch to main -- it advances on its own\n")

        t.switch("main")
        t.chat("Show me a simple decorator example.")

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

    # =================================================================
    # Part 2: rebase
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Part 2: REBASE")
    print("=" * 60)
    print()
    print("  Scenario: 'examples' branch started from an older main.")
    print("  Main has since advanced with new content. Rebase replays")
    print("  the examples branch's commits on top of main's current tip,")
    print("  so the branch includes everything main has plus its own work.")

    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # Step 1: Build shared base
        print("\n  Step 1: Build shared base on main\n")

        t.system("You are a concise Python tutor. One paragraph max.")
        t.chat("What are list comprehensions?")

        print("  main (shared base):")
        t.compile().pprint(style="compact")

        # Step 2: Branch and add feature work
        print("\n  Step 2: Branch 'examples' and add 2 LLM exchanges\n")

        t.branch("examples")
        r1 = t.chat("Give me 3 list comprehension examples, from simple to complex.")
        r2 = t.chat("Now show the equivalent for-loop for each one.")

        print("  examples (2 extra exchanges beyond shared base):")
        t.compile().pprint(style="compact")

        # Step 3: Main advances independently
        print("\n  Step 3: Main advances -- examples is now stale\n")

        t.switch("main")
        t.chat("What are generator expressions? How do they differ from list comprehensions?")

        print("  main (has generator content examples branch doesn't):")
        t.compile().pprint(style="compact")

        # Step 4: Show the stale examples branch
        print("\n  Step 4: examples branch is missing main's latest\n")

        t.switch("examples")
        ctx_before = t.compile()

        print(f"  examples BEFORE rebase ({len(ctx_before.messages)} messages):")
        ctx_before.pprint(style="compact")
        print("\n  (notice: no generator expressions content)")

        # Step 5: Rebase
        print("\n  Step 5: Rebase examples onto main\n")

        result = t.rebase("main")

        print(f"  Rebase complete:")
        print(f"    replayed: {len(result.replayed_commits)} commits")
        print(f"    new HEAD: {result.new_head[:8]}")
        print(f"    warnings: {len(result.warnings)}")

        # Show hash changes (new parentage -> new hashes)
        print(f"\n  Hash changes (same content, new lineage):")
        for orig, replayed in zip(result.original_commits, result.replayed_commits):
            print(f"    {orig.commit_hash[:8]} -> {replayed.commit_hash[:8]}")

        # Step 6: Show the result
        ctx_after = t.compile()

        print(f"\n  examples AFTER rebase ({len(ctx_after.messages)} messages):")
        ctx_after.pprint(style="chat")

        print("\n  The examples branch now includes main's generator")
        print("  expressions content PLUS its own list comprehension work.")
        print("  Commits got new hashes (new parents) but same content.")


if __name__ == "__main__":
    main()
