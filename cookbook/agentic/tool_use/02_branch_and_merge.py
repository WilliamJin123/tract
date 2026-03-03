"""Branch and Merge via Tools

An LLM agent explores alternative conversation paths by creating branches,
switching between them, and merging results — all through genuine tool calls.

Scenario: The agent is asked to explore two different approaches to a problem
on separate branches, then merge the best one back. It uses branch to create
alternatives, switch to move between them, list_branches to see what exists,
and merge to combine work.

Tools exercised: branch, switch, list_branches, merge, status, compile, log

Demonstrates: LLM-driven branch creation, switching, and merging;
              agent reasoning about which branch to keep
"""

import json
import sys
from pathlib import Path

import httpx

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


# Tool profile: branching and merging tools
BRANCH_PROFILE = ToolProfile(
    name="branch-manager",
    tool_configs={
        "branch": ToolConfig(
            enabled=True,
            description=(
                "Create a new branch from the current position. Use this to "
                "explore alternative conversation paths without affecting the "
                "main branch. Set switch=true to immediately work on it."
            ),
        ),
        "switch": ToolConfig(
            enabled=True,
            description=(
                "Switch to a different branch. Changes the active context to "
                "that branch's history. Use list_branches first to see options."
            ),
        ),
        "list_branches": ToolConfig(
            enabled=True,
            description=(
                "List all branches with their HEAD commits. Shows which "
                "branch is currently active with a * marker."
            ),
        ),
        "merge": ToolConfig(
            enabled=True,
            description=(
                "Merge a source branch into the current branch. Combines "
                "context from both branches. Provide a descriptive message."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description="Check current branch, HEAD, and token count.",
        ),
        "compile": ToolConfig(
            enabled=True,
            description="View current compiled context to verify branch state.",
        ),
        "log": ToolConfig(
            enabled=True,
            description="View recent commits on the current branch.",
        ),
    },
)


def run_agent_loop(t, executor, tools, task, max_turns=12):
    """Generic agentic loop: user task -> tool calls -> final response."""
    t.user(task)

    for turn in range(max_turns):
        response = t.generate()

        if not response.tool_calls:
            print(f"\n  Agent: {response.text[:200]}")
            if len(response.text) > 200:
                print(f"         ...({len(response.text)} chars total)")
            return response

        for tc in response.tool_calls:
            result = executor.execute(tc.name, tc.arguments)
            t.tool_result(tc.id, tc.name, str(result))
            args_short = json.dumps(tc.arguments)[:60]
            print(f"    -> {tc.name}({args_short})")
            output_short = str(result.output)[:80]
            print(f"       {output_short}")

    print("  (max turns reached)")
    return None


# =====================================================================
# PART 1 -- Manual: Show branching tools, execute directly
# =====================================================================

def part1_manual():
    """Show branch/merge tools and execute them manually."""
    print("=" * 60)
    print("PART 1 -- Manual: Branch and Merge Tools")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a research assistant.")
        t.user("What are the pros of solar energy?")
        t.assistant("Solar energy is renewable, reduces electricity bills, "
                    "and has low maintenance costs.")

        executor = ToolExecutor(t)
        executor.set_profile(BRANCH_PROFILE)

        # Show available tools
        tools = t.as_tools(profile=BRANCH_PROFILE, format="openai")
        print(f"\n  Branch profile: {len(tools)} tools")
        for tool in tools:
            print(f"    - {tool['function']['name']}")

        print("\n  Conversation on main:")
        t.compile().pprint(style="chat")

        # Create a branch
        result = executor.execute("branch", {"name": "wind-research", "switch": True})
        print(f"\n  branch('wind-research'):\n    {result.output}")

        # List branches
        result = executor.execute("list_branches", {})
        print(f"\n  list_branches():\n    {result.output}")

        print()


# =====================================================================
# PART 2 -- Agent: LLM manages branches autonomously
# =====================================================================

def part2_agent():
    """LLM agent creates branches, explores alternatives, and merges."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 2: SKIPPED (no API key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- Agent: LLM Manages Branches")
    print("=" * 60)
    print()
    print("  The agent will: create branches for different approaches,")
    print("  add content to each, then merge the chosen one back to main.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=BRANCH_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a research assistant with branch management tools. "
            "You can create branches to explore different approaches, "
            "then merge the best one back to main."
        )

        # Seed the main branch
        t.user("I need to compare two database options for my project.")
        t.assistant("I'll create separate branches to research each option.")

        # Let the agent manage branches
        print("  --- Task: Explore options on branches ---")
        run_agent_loop(
            t, executor, tools,
            "Create a branch called 'postgres-research' and switch to it. "
            "Then switch back to main and create 'sqlite-research'. "
            "Use list_branches to verify both exist, then switch back to main "
            "and merge 'postgres-research' into main with a descriptive message."
        )

        # Verify final state
        print("\n  --- Final state ---")
        result = executor.execute("list_branches", {})
        print(f"  Branches: {result.output}")
        result = executor.execute("status", {})
        print(f"  Status: {result.output}")

        print("\n  Final context after merge:")
        t.compile().pprint(style="compact")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/operations/04_branch.py          -- Manual branch lifecycle
# cookbook/developer/operations/05_merge_strategies.py -- FF, clean, no_ff merge
# cookbook/developer/operations/06_merge_conflicts.py  -- Conflict resolution
