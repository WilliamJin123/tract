"""Branch Lifecycle -- Create, Switch, List, Delete

Two tiers of branch usage: manual lifecycle and automated tangent detection.

PART 1 -- Manual           Direct branch/switch/list/delete calls
PART 2 -- Automated        BranchTrigger detects content type tangents

Demonstrates: branch(), switch(), list_branches(), current_branch,
              branch(switch=False), delete_branch(force=True),
              BranchTrigger, configure_triggers(), t.on("trigger", handler)
"""

import sys
from pathlib import Path

from tract import Tract, ArtifactContent, BranchTrigger
from tract.hooks.trigger import PendingTrigger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


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
        api_key=llm.api_key,
        base_url=llm.base_url,
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
# PART 2 -- Automated: BranchTrigger detects content type tangents
# =============================================================================

def part2_automated():
    print("\n" + "=" * 60)
    print("PART 2 -- Automated: Tangent Detection via BranchTrigger")
    print("=" * 60)
    print()
    print("  BranchTrigger watches for rapid content type switching.")
    print("  When the conversation veers through dialogue -> artifact ->")
    print("  dialogue, it proposes a branch to isolate the tangent.")
    print()

    # Low threshold so our short demo triggers it.
    # Default ignore_transitions skips dialogue<->tool_io (normal agent chatter),
    # so we need transitions between dialogue, artifact, and reasoning types.
    trigger = BranchTrigger(content_type_window=6, switch_threshold=2)

    with Tract.open() as t:
        t.configure_triggers([trigger])

        # Hook to intercept the trigger proposal
        proposals = []

        def on_trigger(pending: PendingTrigger):
            proposals.append(pending)
            print(f"  [trigger] {pending.trigger_name}: {pending.reason}")
            if len(proposals) == 1:
                print(f"  [trigger] approved -> branch: {pending.action_params.get('name', '?')}")
                pending.approve()
            else:
                print(f"  [trigger] skipped (already branched)")
                pending.reject("Already branched for this tangent")

        t.on("trigger", on_trigger, name="tangent-detector")

        # Build a conversation that tangents through content types:
        #   instruction -> dialogue -> artifact -> dialogue
        # The instruction->dialogue and dialogue->artifact transitions
        # are both non-ignored, hitting the threshold of 2.
        print("=== Building conversation with mixed content types ===\n")

        t.system("You are a Python tutor.")
        t.user("Write me a fibonacci function.")
        t.assistant("Sure, here's a fibonacci implementation.")
        # Tangent: agent produces a code artifact mid-conversation
        t.commit(
            ArtifactContent(
                artifact_type="code",
                content="def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
                language="python",
            ),
            message="Generated fibonacci code artifact",
        )
        # Back to dialogue -- this is the transition that should tip the trigger
        t.user("Great, now explain how recursion works in general.")

        print()
        t.compile().pprint(style="compact")

        print(f"\n  Trigger fired {len(proposals)} time(s)")
        print(f"  Branches: {[b.name for b in t.list_branches()]}")

        # Manual evaluate to show the API
        print("\n  --- Manual evaluate() ---")
        action = trigger.evaluate(t)
        if action:
            print(f"  {action}")
        else:
            print("  No tangent detected (already handled or below threshold)")


# --- Tier notes ---
# Commits and compile are primitives; they don't have review=True variants.
# For HITL patterns, see: hooks/ (t.on(), review=True)
# For agent automation, see: agentic/sidecar/ (triggers, orchestrator)


def main():
    part1_manual()
    part2_automated()


if __name__ == "__main__":
    main()
