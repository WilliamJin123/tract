"""Agent-Driven Staged Workflow (Implicit)

The developer pre-creates stage branches (design, implementation, validation)
with config metadata. The agent is given a single comprehensive task and
must discover the stage infrastructure, then navigate through it autonomously.

Tools available: get_config, transition, commit, switch, list_branches

Demonstrates: Can the model navigate pre-built stages to complete a
              multi-phase task in a single run with concise deliverables?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm
from _logging import StepLogger

MODEL_ID = llm.xlarge


PROFILE = ToolProfile(
    name="architect",
    tool_configs={
        "get_config": ToolConfig(enabled=True),
        "transition": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
        "switch": ToolConfig(enabled=True),
        "list_branches": ToolConfig(enabled=True),
    },
)


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 70)
    print("Agent-Driven Staged Workflow (Implicit)")
    print("=" * 70)
    print()
    print("  Pre-built stages: design, implementation, validation")
    print("  Agent must discover and navigate them in a single run.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
        tool_profile=PROFILE,
    ) as t:
        # Developer pre-creates the stage infrastructure
        t.system(
            "You are a software architect. Complete the task by working "
            "through each stage of the workflow.\n"
            "IMPORTANT: You MUST complete ALL three stages (design, "
            "implementation, validation). Keep each stage's output concise "
            "(2-3 key points) and transition promptly. Budget your steps."
        )

        for stage, temp in [("design", 0.9), ("implementation", 0.3), ("validation", 0.5)]:
            t.branch(stage, switch=True)
            t.configure(stage=stage, temperature=temp)
            t.switch("main")

        # Start the agent on the design branch
        t.switch("design")

        print(f"  Branches: {[b.name for b in t.list_branches()]}")
        print(f"  Starting on: {t.current_branch}")

        log = StepLogger()

        # Single task — agent must navigate all stages autonomously
        print("\n  --- Task ---")
        result = t.run(
            "Design a task management REST API (title, status, assignee).\n\n"
            "For EACH stage, you MUST commit() output then transition():\n"
            "1. DESIGN: commit your API URLs + data models, then transition('implementation')\n"
            "2. IMPLEMENTATION: commit module structure, then transition('validation')\n"
            "3. VALIDATION: commit a pre-ship checklist, then STOP\n\n"
            "Keep each commit concise (2-3 key points). Use commit() for every deliverable.",
            max_steps=18, max_tokens=4096,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Report — check what the agent actually did
        print("\n\n=== Final State ===\n")
        branches_visited = set()
        for stage in ["design", "implementation", "validation"]:
            t.switch(stage)
            ctx = t.compile()
            cfg = t.get_config("stage")
            if ctx.token_count > 50:  # has content beyond just config
                branches_visited.add(stage)
            print(f"  [{stage}] stage={cfg}, {len(ctx.messages)} messages, "
                  f"{ctx.token_count} tokens")

        if len(branches_visited) >= 3:
            print(f"\n  Agent navigated all 3 stages.")
        elif len(branches_visited) > 0:
            print(f"\n  Agent visited {len(branches_visited)} stage(s): "
                  f"{sorted(branches_visited)}")
        else:
            print("\n  Agent did not navigate to any stages.")


if __name__ == "__main__":
    main()


# --- See also ---
# Branching basics (no LLM):  getting_started/04_branches.py
# Config per branch:           config_and_middleware/01_config_basics.py
