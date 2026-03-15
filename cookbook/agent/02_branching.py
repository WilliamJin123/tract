"""Agent-Driven Branching and Merging (Implicit)

The agent is asked to draft two independent, potentially contradictory
technical proposals. If one analysis influences the other, the proposals
won't be genuine. The agent has branching tools but is never told to
use them.

Tools available: branch, switch, list_branches, merge, commit

Demonstrates: Does the model use branches to isolate conflicting analyses
              so they don't cross-pollinate?
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
    name="researcher",
    tool_configs={
        "branch": ToolConfig(enabled=True),
        "switch": ToolConfig(enabled=True),
        "list_branches": ToolConfig(enabled=True),
        "merge": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
    },
)


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 60)
    print("Agent-Driven Branching and Merging (Implicit)")
    print("=" * 60)
    print()
    print("  Two conflicting proposals that must be developed independently.")
    print("  Will the agent use branches for isolation?")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
        tool_profile=PROFILE,
    ) as t:
        t.system(
            "You are a senior solutions architect evaluating backend "
            "architecture options for a new product."
        )

        # Seed with project requirements
        t.user(
            "Building a SaaS analytics platform. 10k concurrent users, "
            "50TB warehouse, sub-second queries. 12 engineers. "
            "Need to pick: microservices vs monolith."
        )
        t.assistant("I'll evaluate both options.")

        # Task: two independent, conflicting proposals
        print("  --- Task ---")
        log = StepLogger()
        result = t.run(
            "Write two short proposals:\n"
            "A) Why microservices is the right call\n"
            "B) Why monolith is the right call\n\n"
            "Each must stand on its own — if one influences the other "
            "they won't be genuine independent evaluations.",
            max_steps=12, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Report — show each branch's compiled context
        print("\n  --- Final state ---")
        branches = t.list_branches()
        print(f"  Branches: {[b.name for b in branches]}")
        print(f"  Current: {t.current_branch}")

        original = t.current_branch
        for branch in branches:
            t.switch(branch.name)
            print(f"\n  [{branch.name}]:")
            t.compile().pprint(style="compact")
        t.switch(original)

        if len(branches) > 1:
            print(f"\n  Agent created {len(branches) - 1} branch(es) for isolation.")
        else:
            print("\n  Agent did not use branches.")


if __name__ == "__main__":
    main()


# --- See also ---
# Branching basics (no LLM):  reference/04_branching.py
