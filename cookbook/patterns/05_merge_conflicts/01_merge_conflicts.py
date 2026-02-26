"""Merge Conflict Resolution

Both branches EDIT the same system prompt simultaneously, creating a
conflict that Tract can't auto-resolve. You inspect the conflict,
write a resolution, and commit it — then verify with a live LLM call
that the resolved persona works correctly.

Run interactively to write your own resolution, or pipe/redirect to use
the default resolution automatically.

Demonstrates: merge(), merge_type, MergeResult, committed, conflicts,
              ConflictInfo, ConflictInfo.pprint(), edit_resolution(),
              commit_merge(), MergeResult.pprint(), CommitOperation.EDIT,
              InstructionContent, apply resolution via EDIT commit,
              chat() to verify the resolved persona, pprint(style="compact")
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
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Scenario: CONFLICT (manual resolution)")
    print("=" * 60)
    print()
    print("  Both branches EDIT the same system prompt. Tract can't")
    print("  auto-resolve -- you inspect the conflict and write a resolution.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # --- Shared base: a conversation both branches diverge from ---
        sys_ci = t.system("You are a helpful assistant.")
        t.user("What's Python?")
        t.assistant("A programming language.")

        # --- Feature branch: formal persona ---
        # EDIT the system prompt to use academic language
        t.branch("formal")
        t.commit(
            InstructionContent(text="You are a formal, academic assistant. Use precise terminology."),
            operation=CommitOperation.EDIT,
            edit_target=sys_ci.commit_hash,
            message="persona: formal academic",
        )

        # --- Main branch: casual persona ---
        # EDIT the same system prompt to be friendly
        t.switch("main")
        t.commit(
            InstructionContent(text="You are a casual, friendly assistant. Use everyday language."),
            operation=CommitOperation.EDIT,
            edit_target=sys_ci.commit_hash,
            message="persona: casual friendly",
        )

        # --- Show both versions before merge ---
        print(f"\n  BEFORE MERGE")
        print(f"  main (casual persona):")
        t.compile().pprint(style="compact")

        t.switch("formal")
        print(f"\n  formal (academic persona):")
        t.compile().pprint(style="compact")

        # --- Attempt the merge — conflict detected ---
        t.switch("main")
        result = t.merge("formal")

        print(f"\n  merge('formal') -> {result.merge_type}")
        print(f"    committed: {result.committed}  (needs resolution)")
        print(f"    conflicts: {len(result.conflicts)}\n")

        # --- Inspect the conflict with the SDK pprint helper ---
        conflict = result.conflicts[0]
        conflict.pprint()

        # --- Resolve the conflict ---
        # The default blends both personas into one balanced description.
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

        # --- Commit the merge with resolution ---
        result.edit_resolution(conflict.target_hash, resolution)
        result = t.commit_merge(result, message="merge: combined personas")

        # Show committed result with MergeResult.pprint()
        result.pprint()

        # --- Apply the resolution as an EDIT commit ---
        # The merge commit stores resolutions as metadata, but the compiler
        # doesn't extract them automatically — you apply them explicitly.
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

        # --- Verify: chat with the resolved persona ---
        # Proves the merged context is coherent and LLM-ready.
        print(f"\n  VERIFY: chat with the resolved persona")
        r = t.chat("Explain Python decorators in two sentences.")
        r.pprint()

        print(f"\n  FINAL CONTEXT")
        t.compile().pprint(style="compact")


if __name__ == "__main__":
    main()
