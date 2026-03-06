"""Agent-Driven Branching and Merging

An LLM agent explores alternative conversation paths by creating branches,
switching between them, and merging results -- all through genuine tool calls.
The agent decides which branches to create, what to explore, and when to merge.

Tools exercised: branch, switch, list_branches, merge, status, compile, log

Demonstrates: LLM-driven branch creation, switching, and merging;
              agent reasoning about which branch to keep
"""

import io
import json
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def _log_step(step_num, response):
    """on_step callback -- print step number."""
    print(f"    [step {step_num}]")


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 60)
    print("Agent-Driven Branching and Merging")
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
        # Register tools from the profile
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
        result = t.run(
            "Create a branch called 'postgres-research' and switch to it. "
            "Then switch back to main and create 'sqlite-research'. "
            "Use list_branches to verify both exist, then switch back to main "
            "and merge 'postgres-research' into main with a descriptive message.",
            max_steps=12, on_step=_log_step,
        )
        print(f"\n  Loop result: {result.status} ({result.steps} steps, "
              f"{result.tool_calls} tool calls)")
        if result.final_response:
            print(f"  Agent: {result.final_response[:200]}")

        # Verify final state
        print("\n  --- Final state ---")
        executor = ToolExecutor(t)
        executor.set_profile(BRANCH_PROFILE)
        br = executor.execute("list_branches", {})
        print(f"  Branches: {br.output}")
        st = executor.execute("status", {})
        print(f"  Status: {st.output}")

        print("\n  Final context after merge:")
        t.compile().pprint(style="compact")


if __name__ == "__main__":
    main()
