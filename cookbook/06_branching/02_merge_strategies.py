"""Merge Strategies

Three merge modes, from trivial to complex: fast-forward (linear history),
clean (diverged but no overlap), and conflict (both branches edit the same
message). Scenarios 1, 2, and Bonus are deterministic. Scenario 3 uses an
LLM call after conflict resolution to prove the resolved persona works.

Run interactively to resolve conflicts yourself, or pipe/redirect to use
the default resolution automatically.

Demonstrates: merge(), merge_type, MergeResult, committed, merge_commit_hash,
              edit_resolution(), commit_merge(), no_ff, delete_branch=True,
              CommitOperation.EDIT, ConflictInfo, pprint(style="compact"),
              ConflictInfo.pprint(), MergeResult.pprint()
"""

import os
import sys

from dotenv import load_dotenv

from tract import CommitOperation, InstructionContent, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
    # =================================================================
    # Scenario 1: Fast-Forward
    # =================================================================

    print("=" * 60)
    print("Scenario 1: FAST-FORWARD")
    print("=" * 60)
    print()
    print("  Main hasn't moved since branching. Merge just slides")
    print("  main's pointer forward to feature's tip.")

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("What's Python?")
        t.assistant("A programming language.")

        print(f"\n  BEFORE MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

        # Branch and add commits only on feature
        t.branch("feature")
        t.user("What about type hints?")
        t.assistant("Type hints add optional static typing to Python.")

        print(f"\n  feature:")
        t.compile().pprint(style="compact")

        # Merge
        t.switch("main")
        result = t.merge("feature")

        result.pprint()

        print(f"  AFTER MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

    # =================================================================
    # Scenario 2: Clean Merge (diverged, APPEND-only)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Scenario 2: CLEAN MERGE")
    print("=" * 60)
    print()
    print("  Both branches diverged with new messages, but nobody edited")
    print("  existing ones. All APPENDs -- Tract auto-merges cleanly.")

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("What's Python?")
        t.assistant("A programming language.")

        # Feature: decorators
        t.branch("feature")
        t.user("Tell me about decorators.")
        t.assistant("Decorators wrap functions to extend behavior.")

        # Main: generators
        t.switch("main")
        t.user("Tell me about generators.")
        t.assistant("Generators yield values lazily using the yield keyword.")

        print(f"\n  BEFORE MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

        t.switch("feature")
        print(f"\n  feature:")
        t.compile().pprint(style="compact")

        # Merge
        t.switch("main")
        result = t.merge("feature")

        result.pprint()

        print(f"  AFTER MERGE")
        print(f"  main:")
        t.compile().pprint(style="compact")

    # =================================================================
    # Scenario 3: Conflict (both EDIT the same system prompt)
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Scenario 3: CONFLICT (manual resolution)")
    print("=" * 60)
    print()
    print("  Both branches EDIT the same system prompt. Tract can't")
    print("  auto-resolve -- you inspect the conflict and write a resolution.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        sys_ci = t.system("You are a helpful assistant.")
        t.user("What's Python?")
        t.assistant("A programming language.")

        # Feature branch: formal persona
        t.branch("formal")
        t.commit(
            InstructionContent(text="You are a formal, academic assistant. Use precise terminology."),
            operation=CommitOperation.EDIT,
            edit_target=sys_ci.commit_hash,
            message="persona: formal academic",
        )

        # Main branch: casual persona
        t.switch("main")
        t.commit(
            InstructionContent(text="You are a casual, friendly assistant. Use everyday language."),
            operation=CommitOperation.EDIT,
            edit_target=sys_ci.commit_hash,
            message="persona: casual friendly",
        )

        # Show both versions before merge
        print(f"\n  BEFORE MERGE")
        print(f"  main (casual persona):")
        t.compile().pprint(style="compact")

        t.switch("formal")
        print(f"\n  formal (academic persona):")
        t.compile().pprint(style="compact")

        # Merge -- conflict detected
        t.switch("main")
        result = t.merge("formal")

        print(f"\n  merge('formal') -> {result.merge_type}")
        print(f"    committed: {result.committed}  (needs resolution)")
        print(f"    conflicts: {len(result.conflicts)}\n")

        # Show the conflict with the SDK pprint helper
        conflict = result.conflicts[0]
        conflict.pprint()

        # Interactive resolution (falls through in non-interactive contexts)
        default_resolution = (
            "You are a knowledgeable assistant. Be precise when discussing "
            "technical concepts, but keep your tone approachable and friendly."
        )

        if sys.stdin.isatty():
            print(f"\n  Default resolution:")
            print(f"    \"{default_resolution}\"")
            user_input = input("\n  Your resolution (Enter to accept default): ").strip()
            resolution = user_input if user_input else default_resolution
        else:
            resolution = default_resolution

        print(f"\n  Resolution: \"{resolution}\"\n")

        result.edit_resolution(conflict.target_hash, resolution)
        result = t.commit_merge(result, message="merge: combined personas")

        # Show committed result with MergeResult.pprint()
        result.pprint()

        # Apply the resolution as an EDIT commit so compile() reflects it.
        # (The merge commit stores resolutions as metadata, but the compiler
        # doesn't extract them automatically -- you apply them explicitly.)
        for target_hash, resolved_text in result.resolutions.items():
            t.commit(
                InstructionContent(text=resolved_text),
                operation=CommitOperation.EDIT,
                edit_target=target_hash,
                message="apply merge resolution",
            )

        print(f"  AFTER MERGE (resolution applied)")
        print(f"  main:")
        t.compile().pprint(style="compact")

        # Prove it works -- chat with the resolved persona
        print(f"\n  VERIFY: chat with the resolved persona")
        r = t.chat("Explain Python decorators in two sentences.")
        r.pprint()

        print(f"\n  FINAL CONTEXT")
        t.compile().pprint(style="compact")

    # =================================================================
    # Bonus: no_ff and delete_branch=True
    # =================================================================

    print(f"\n{'=' * 60}")
    print("Bonus: no_ff + delete_branch")
    print("=" * 60)
    print()
    print("  This COULD fast-forward, but no_ff=True forces a merge commit.")
    print("  delete_branch=True auto-cleans the source branch after merge.")

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.branch("quick-fix")
        t.user("Fix: use snake_case for variable names.")
        t.assistant("Done -- all variables now use snake_case.")

        t.switch("main")

        branches_before = [b.name for b in t.list_branches()]
        print(f"\n  BEFORE MERGE")
        print(f"    branches: {branches_before}")
        print(f"  main:")
        t.compile().pprint(style="compact")

        result = t.merge("quick-fix", no_ff=True, delete_branch=True)

        result.pprint()

        branches_after = [b.name for b in t.list_branches()]
        print(f"  AFTER MERGE")
        print(f"    branches: {branches_after}  ('quick-fix' auto-deleted)")
        print(f"  main:")
        t.compile().pprint(style="compact")


if __name__ == "__main__":
    main()
