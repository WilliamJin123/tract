"""Merge Conflict Resolution

Both branches EDIT the same system prompt simultaneously, creating a
conflict that Tract can't auto-resolve. You inspect the conflict,
write a resolution, and commit it — then verify with a live LLM call
that the resolved persona works correctly.

Run interactively to resolve via a quick-pick menu (accept current,
accept incoming, accept both, or edit in $EDITOR). In non-interactive
mode the default resolution is applied automatically.

Demonstrates: merge(review=True), PendingMerge, edit_interactive(),
              set_resolution(), approve(), to_marker_text(),
              parse_conflict_markers(), chat(), pprint(style="compact")
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

        # --- Attempt the merge with review=True — conflict detected ---
        # review=True returns a PendingMerge even without a resolver,
        # letting us build resolutions interactively.
        t.switch("main")
        pending = t.merge("formal", review=True)

        from tract.hooks.merge import PendingMerge

        if not isinstance(pending, PendingMerge):
            print(f"\n  No conflicts — merge completed: {pending.merge_type}")
            pending.pprint()
            return

        print(f"\n  merge('formal', review=True) -> PendingMerge")
        print(f"    conflicts: {len(pending.conflicts)}")
        print(f"    resolutions: {len(pending.resolutions)} (empty — no resolver)\n")

        # --- Resolve the conflict ---
        default_resolution = (
            "You are a knowledgeable assistant. Be precise when discussing "
            "technical concepts, but keep your tone approachable and friendly."
        )

        if sys.stdin.isatty():
            # Interactive: quick-pick menu for each conflict
            # Options: accept current, accept incoming, accept both, $EDITOR
            pending.edit_interactive()

            # Check if user skipped — fall back to default
            for conflict in pending.conflicts:
                key = conflict.target_hash
                if key and key not in pending.resolutions:
                    print(f"  Using default resolution for unresolved conflict.")
                    pending.set_resolution(key, default_resolution)
        else:
            # Non-interactive: apply default resolution
            for conflict in pending.conflicts:
                if conflict.target_hash:
                    pending.set_resolution(conflict.target_hash, default_resolution)

        result = pending.approve()

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
