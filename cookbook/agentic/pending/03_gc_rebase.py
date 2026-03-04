"""Agentic GC and Rebase -- LLM Autonomously Controls Pending Operations

An LLM agent inspects and controls PendingGC and PendingRebase objects,
deciding which commits to exclude and when to approve. Both operations
share the same action set (approve, reject, exclude), so the agent
pattern is identical -- only the data shape differs.

Part 1: PendingGC -- orphan cleanup with selective exclusion
Part 2: PendingRebase -- replay plan review with commit skipping

Demonstrates: to_dict() inspection, exclude() for selective control,
              approve() for execution, tract's built-in LLM client
"""

import json
import sys
from pathlib import Path

from tract import Tract
from tract.hooks.gc import PendingGC
from tract.hooks.rebase import PendingRebase

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


def ask_agent(pending, instruction: str) -> dict:
    """Send a Pending's state to the LLM and get a structured decision back.

    Builds a prompt from to_dict() context and to_tools() schemas, sends
    it via tract's built-in LLM client, and returns the parsed decision dict.
    """
    state = pending.to_dict()
    tools = pending.to_tools()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a context management agent. You receive the state of "
                "a pending operation and must decide what to do using the "
                "provided tools. Always call exactly one tool per response."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"Current state:\n{json.dumps(state, indent=2)}"
            ),
        },
    ]

    # Use tract's built-in LLM client (configured via Tract.open())
    client = pending.tract._llm_client
    raw = client.chat(messages, tools=tools)
    tc_list = raw["choices"][0]["message"].get("tool_calls", [])

    if tc_list:
        tc = tc_list[0]
        return {
            "action": tc["function"]["name"],
            "args": json.loads(tc["function"].get("arguments", "{}")),
        }
    # Fallback if the LLM responds without a tool call
    return {"action": "approve", "args": {}}


# =====================================================================
# PART 1 -- PendingGC: Orphan Cleanup with Selective Exclusion
# =====================================================================

def part1_gc():
    """Agent inspects orphaned commits and selectively excludes one before approving GC."""
    if not llm.api_key:
        print("=" * 60)
        print("PART 1: SKIPPED (no API key)")
        print("=" * 60)
        return

    print("=" * 60)
    print("PART 1 -- PendingGC: Agent-Driven Orphan Cleanup")
    print("=" * 60)
    print()
    print("  The agent will inspect orphaned commits via to_dict(),")
    print("  exclude one worth keeping, then approve the rest for removal.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Build a main conversation
        t.system("You are a software architecture advisor.")
        t.user("What are the tradeoffs between monolith and microservices?")
        t.assistant(
            "Monoliths are simpler to deploy and debug but harder to scale "
            "independently. Microservices offer independent scaling and "
            "deployment but add network complexity and operational overhead."
        )

        # Create a work branch with several commits, then delete it
        t.branch("work")
        t.user("Draft a migration plan from monolith to microservices.")
        t.assistant(
            "Step 1: Identify bounded contexts. Step 2: Extract the "
            "least-coupled module first. Step 3: Set up service mesh."
        )
        t.user("What about database splitting strategies?")
        t.assistant(
            "Start with shared database, then use database-per-service "
            "pattern. Strangler fig for gradual migration."
        )
        t.user("How do we handle distributed transactions?")
        t.assistant(
            "Use the Saga pattern with compensating transactions. "
            "Avoid two-phase commit across services."
        )

        # Switch back to main and delete the work branch -> orphans
        t.switch("main")
        t.delete_branch("work", force=True)
        print("  Deleted 'work' branch -- commits are now orphaned")

        # Get PendingGC with review=True
        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)

        print(f"\n  PendingGC state:")
        print(f"    commits_to_remove: {len(pending.commits_to_remove)}")
        print(f"    tokens_to_free:    {pending.tokens_to_free}")
        pending.pprint()

        # Step 1: Agent inspects via to_dict()
        state = pending.to_dict()
        print(f"\n  to_dict() keys: {list(state.keys())}")
        print(f"  to_dict() fields: {list(state.get('fields', {}).keys())}")

        # Step 2: Agent excludes the first commit (saving it from removal)
        if len(pending.commits_to_remove) > 1:
            save_hash = pending.commits_to_remove[0]
            print(f"\n  Agent decides to save commit {save_hash[:12]}...")

            decision = ask_agent(
                pending,
                f"This GC will remove {len(pending.commits_to_remove)} orphaned "
                f"commits. The first commit ({save_hash[:12]}...) contains a "
                f"valuable migration plan. Please exclude it from removal using "
                f"the exclude tool with commit_hash='{save_hash}'.",
            )
            print(f"    LLM decision: {json.dumps(decision)}")
            pending.apply_decision(decision)
            print(f"    After exclude: {len(pending.commits_to_remove)} commits remain")

        # Step 3: Agent approves the remaining removals
        decision = ask_agent(
            pending,
            "The valuable commit has been saved. Please approve the GC "
            "to remove the remaining orphaned commits.",
        )
        print(f"\n  Agent approves:")
        print(f"    LLM decision: {json.dumps(decision)}")
        result = pending.apply_decision(decision)
        print(f"    status: {pending.status}")
        pending.pprint()

        print("\n  Main branch after GC:")
        t.compile().pprint(style="chat")


# =====================================================================
# PART 2 -- PendingRebase: Replay Plan Review with Commit Skipping
# =====================================================================

def part2_rebase():
    """Agent reviews a rebase replay plan and skips one commit before approving."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 2: SKIPPED (no API key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- PendingRebase: Agent-Driven Replay Review")
    print("=" * 60)
    print()
    print("  The agent will review the replay plan and warnings,")
    print("  exclude a commit it deems redundant, then approve.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Build main with a base conversation
        t.system("You are a fitness coach designing workout programs.")
        t.user("I want to build a 4-day strength program.")
        t.assistant(
            "A solid 4-day split: Mon upper push, Tue lower, "
            "Thu upper pull, Fri lower. Rest Wed/weekends."
        )

        # Create feature branch with several commits
        t.branch("feature")
        t.user("What exercises for the upper push day?")
        t.assistant(
            "Bench press 4x6, overhead press 3x8, incline dumbbell "
            "press 3x10, lateral raises 3x12, tricep pushdowns 3x12."
        )
        t.user("Actually, scratch that. What about a full-body approach instead?")
        t.assistant(
            "Full-body 4x/week: squat, bench, row, overhead press "
            "rotating as primary lifts with accessories."
        )
        t.user("How should I periodize over 12 weeks?")
        t.assistant(
            "Weeks 1-4: hypertrophy (3x10-12). Weeks 5-8: strength "
            "(4x5-6). Weeks 9-12: peaking (5x3, then deload)."
        )

        # Add commits on main so feature is behind
        t.switch("main")
        t.user("What about nutrition for strength training?")
        t.assistant(
            "Aim for 1g protein per pound bodyweight, surplus of "
            "300-500 calories on training days, maintenance on rest days."
        )
        t.switch("feature")

        print("  Feature branch BEFORE rebase:")
        t.compile().pprint(style="compact")

        # Get PendingRebase with review=True
        pending: PendingRebase = t.rebase("main", review=True)

        print(f"\n  PendingRebase state:")
        print(f"    replay_plan:  {len(pending.replay_plan)} commits")
        print(f"    target_base:  {pending.target_base[:12]}...")
        print(f"    warnings:     {len(pending.warnings)}")
        pending.pprint()

        # Step 1: Agent inspects via to_dict()
        state = pending.to_dict()
        print(f"\n  to_dict() keys: {list(state.keys())}")
        print(f"  to_dict() fields: {list(state.get('fields', {}).keys())}")

        # Step 2: Agent excludes the "scratch that" commit (user changed their mind)
        if len(pending.replay_plan) > 2:
            skip_hash = pending.replay_plan[2]
            original_count = len(pending.replay_plan)
            print(f"\n  Agent decides to skip commit {skip_hash[:12]}...")

            decision = ask_agent(
                pending,
                f"This rebase will replay {len(pending.replay_plan)} commits "
                f"onto a new base. The third commit ({skip_hash[:12]}...) "
                f"is outdated and should be dropped. Please exclude it using "
                f"the exclude tool with commit_hash='{skip_hash}'.",
            )
            print(f"    LLM decision: {json.dumps(decision)}")
            pending.apply_decision(decision)
            print(f"    After exclude: {len(pending.replay_plan)} commits (was {original_count})")

        # Step 3: Agent checks warnings then approves
        if pending.warnings:
            print(f"\n  Warnings for agent to consider:")
            for w in pending.warnings:
                print(f"    - {w}")

        decision = ask_agent(
            pending,
            "The outdated commit has been excluded from the replay plan. "
            "Please approve the rebase to replay the remaining commits "
            "onto the updated main branch.",
        )
        print(f"\n  Agent approves:")
        print(f"    LLM decision: {json.dumps(decision)}")
        result = pending.apply_decision(decision)
        print(f"    status: {pending.status}")
        pending.pprint()

        print("\n  Feature branch AFTER rebase:")
        t.compile().pprint(style="compact")


def main():
    part1_gc()
    part2_rebase()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/hooks/02_pending/04_gc.py                -- PendingGC manual hooks
# cookbook/hooks/02_pending/05_rebase.py            -- PendingRebase manual hooks
# cookbook/hooks/03_agent_interface/04_dispatch.py  -- apply_decision() full pipeline
# cookbook/hooks/03_agent_interface/01_serialization.py -- to_dict() / to_tools() details
