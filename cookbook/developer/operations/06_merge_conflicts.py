"""Merge Conflict Resolution

Three tiers of conflict resolution — programmatic resolution, interactive
editor-based resolution, and LLM-driven auto-resolution.

PART 1 -- Manual           Detect conflicts, set_resolution(), commit_merge()
PART 2 -- Interactive       review=True, edit_interactive(), click.confirm
PART 3 -- LLM / Agent      resolver="llm" for automatic conflict resolution

Demonstrates: merge(), MergeResult, conflicts, set_resolution(),
              merge(review=True), PendingMerge, edit_interactive(),
              approve(), merge(resolver="llm"), ToolExecutor
"""

import os
import sys

import click
from dotenv import load_dotenv

from tract import CommitOperation, InstructionContent, Tract, ToolExecutor
from tract.hooks.merge import PendingMerge

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _build_diverged_branches(t):
    """Helper: create two branches that both EDIT the same system prompt."""
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
    return sys_ci


# =============================================================================
# PART 1 -- Manual: Detect conflicts, resolve programmatically
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Programmatic Conflict Resolution")
    print("=" * 60)
    print()
    print("  Both branches EDIT the same system prompt. Detect the conflict,")
    print("  set a resolution programmatically, and commit -- no user input.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _build_diverged_branches(t)

        print(f"\n  BEFORE MERGE")
        print(f"  main (casual persona):")
        t.compile().pprint(style="compact")

        t.switch("formal")
        print(f"\n  formal (academic persona):")
        t.compile().pprint(style="compact")

        # Attempt merge with review=True to inspect conflicts
        t.switch("main")
        pending = t.merge("formal", review=True)

        if not isinstance(pending, PendingMerge):
            print(f"  No conflicts -- merge completed: {pending.merge_type}")
            return

        print(f"\n  Conflicts detected: {len(pending.conflicts)}")

        # Resolve each conflict programmatically
        resolved_text = (
            "You are a knowledgeable assistant. Be precise when discussing "
            "technical concepts, but keep your tone approachable and friendly."
        )
        for conflict in pending.conflicts:
            if conflict.target_hash:
                pending.set_resolution(conflict.target_hash, resolved_text)
                print(f"  Set resolution for {conflict.target_hash[:8]}")

        result = pending.approve()
        result.pprint()

        # Apply resolution as EDIT commit
        for target_hash, text in result.resolutions.items():
            t.commit(
                InstructionContent(text=text),
                operation=CommitOperation.EDIT,
                edit_target=target_hash,
                message="apply merge resolution",
            )

        print(f"\n  AFTER MERGE (resolution applied)")
        t.compile().pprint(style="compact")


# =============================================================================
# PART 2 -- Interactive: review=True, editor, click.confirm
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Editor-Based Conflict Resolution")
    print("=" * 60)
    print()
    print("  Both branches EDIT the same system prompt. Tract can't")
    print("  auto-resolve -- you inspect the conflict and write a resolution.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _build_diverged_branches(t)

        print(f"\n  BEFORE MERGE")
        print(f"  main (casual persona):")
        t.compile().pprint(style="compact")

        t.switch("formal")
        print(f"\n  formal (academic persona):")
        t.compile().pprint(style="compact")

        # review=True returns PendingMerge for interactive resolution
        t.switch("main")
        pending = t.merge("formal", review=True)

        if not isinstance(pending, PendingMerge):
            print(f"\n  No conflicts -- merge completed: {pending.merge_type}")
            pending.pprint()
            return

        print(f"\n  merge('formal', review=True) -> PendingMerge")
        print(f"    conflicts: {len(pending.conflicts)}")
        print(f"    resolutions: {len(pending.resolutions)} (empty -- no resolver)\n")

        # Resolve the conflict
        default_resolution = (
            "You are a knowledgeable assistant. Be precise when discussing "
            "technical concepts, but keep your tone approachable and friendly."
        )

        if sys.stdin.isatty():
            # Interactive: quick-pick menu for each conflict
            pending.edit_interactive()

            # Fall back to default for any unresolved
            for conflict in pending.conflicts:
                key = conflict.target_hash
                if key and key not in pending.resolutions:
                    print(f"  Using default resolution for unresolved conflict.")
                    pending.set_resolution(key, default_resolution)
        else:
            for conflict in pending.conflicts:
                if conflict.target_hash:
                    pending.set_resolution(conflict.target_hash, default_resolution)

        result = pending.approve()
        result.pprint()

        # Apply the resolution as an EDIT commit
        for target_hash, resolved_text in result.resolutions.items():
            t.commit(
                InstructionContent(text=resolved_text),
                operation=CommitOperation.EDIT,
                edit_target=target_hash,
                message="apply merge resolution",
            )

        print(f"  AFTER MERGE (resolution applied)")
        t.compile().pprint(style="compact")

        # Verify: chat with the resolved persona
        print(f"\n  VERIFY: chat with the resolved persona")
        r = t.chat("Explain Python decorators in two sentences.")
        r.pprint()

        print(f"\n  FINAL CONTEXT")
        t.compile().pprint(style="compact")


# =============================================================================
# PART 3 -- LLM / Agent: LLM-driven conflict resolution
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: LLM-Driven Conflict Resolution")
    print("=" * 60)
    print()
    print("  Use resolver='llm' to let the LLM auto-resolve conflicts,")
    print("  or use ToolExecutor for agent-driven merges.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _build_diverged_branches(t)

        print(f"\n  Attempting merge with resolver='llm'...")
        t.switch("main")
        result = t.merge("formal", resolver="llm")

        print(f"\n  MergeResult:")
        print(f"    merge_type: {result.merge_type}")
        print(f"    conflicts:  {len(result.conflicts)}")
        print(f"    resolutions: {len(result.resolutions)}")
        result.pprint()

        # Apply LLM-generated resolutions
        for target_hash, resolved_text in result.resolutions.items():
            t.commit(
                InstructionContent(text=resolved_text),
                operation=CommitOperation.EDIT,
                edit_target=target_hash,
                message="apply LLM merge resolution",
            )

        print(f"\n  AFTER LLM-RESOLVED MERGE")
        t.compile().pprint(style="compact")

    # ToolExecutor approach
    print(f"\n  --- ToolExecutor approach ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _build_diverged_branches(t)
        t.switch("main")

        executor = ToolExecutor(t)
        result = executor.execute("merge", {"branch": "formal", "resolver": "llm"})
        print(f"  ToolExecutor merge result: {result}")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
