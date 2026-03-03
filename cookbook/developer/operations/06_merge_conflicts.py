"""Merge Conflict Resolution

Two tiers of conflict resolution -- programmatic resolution and
automated LLM-driven resolution.

PART 1 -- Manual           Detect conflicts, set_resolution(), commit_merge()
PART 3 -- Automated         resolver="llm" for automatic conflict resolution

Demonstrates: merge(), MergeResult, conflicts, set_resolution(),
              merge(review=True), PendingMerge, approve(),
              merge(resolver="llm")
"""

import sys
from pathlib import Path

from tract import CommitOperation, InstructionContent, Tract
from tract.hooks.merge import PendingMerge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


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
        api_key=llm.api_key,
        base_url=llm.base_url,
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
# PART 3 -- Automated: LLM-driven conflict resolution
# =============================================================================

def part3_automated():
    print("=" * 60)
    print("PART 3 -- Automated: LLM-Driven Conflict Resolution")
    print("=" * 60)
    print()
    print("  Use resolver='llm' to let the LLM auto-resolve conflicts.")
    print("  No human input needed -- the LLM generates resolutions.")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
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



# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part3_automated()


if __name__ == "__main__":
    main()
