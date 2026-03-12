"""Tangent Isolation (Implicit)

During an API design conversation with limited context budget, the user
asks a completely unrelated question. The tangent content would waste
valuable context space if left in the main thread. The agent has branching
tools but no "tangent protocol."

Tools available: branch, switch, merge, compress, commit, status

Demonstrates: Can the model isolate an off-topic interruption on a branch
              to protect the main conversation context under budget pressure?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


PROFILE = ToolProfile(
    name="architect",
    tool_configs={
        "commit": ToolConfig(enabled=True),
        "branch": ToolConfig(enabled=True),
        "switch": ToolConfig(enabled=True),
        "merge": ToolConfig(enabled=True),
        "compress": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
    },
)


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 70)
    print("Tangent Isolation (Implicit)")
    print("=" * 70)
    print()
    print("  API design conversation with budget pressure.")
    print("  Off-topic interruption — will the agent isolate it?")
    print()

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2500))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
        tool_profile=PROFILE,
    ) as t:
        # System: role only
        t.system(
            "You are a senior API architect helping design a REST API "
            "for a task management app.\n"
            "IMPORTANT: You have a tight 2500-token context budget. Use your "
            "branch and switch tools to isolate off-topic conversations so they "
            "don't pollute the main API design thread. Always return to main "
            "after handling tangents."
        )

        log = StepLogger()

        # Phase 1: Substantial design conversation (fills ~60% of budget)
        print("=== Phase 1: API design ===\n")
        result = t.run(
            "Design CRUD endpoints for tasks (title, status, assignee). "
            "What URL structure and HTTP methods?",
            max_steps=5, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        status = t.status()
        pct = status.token_count / status.token_budget_max * 100
        print(f"\n  Context: {status.token_count} tokens ({pct:.0f}% of budget)")
        print(f"  Branch: {t.current_branch}")

        # Phase 2: Completely unrelated tangent
        print("\n\n=== Phase 2: Off-topic interruption ===\n")
        result = t.run(
            "Wait, different topic — plan a team offsite for 15 people "
            "next Friday, $2000 budget. Quick suggestions?",
            max_steps=8, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()
        print(f"\n  Branch: {t.current_branch}")
        print(f"  All branches: {[b.name for b in t.list_branches()]}")

        # Phase 3: Resume API design
        if t.current_branch != "main":
            print(f"  (agent left us on {t.current_branch} — switching to main)")
            t.switch("main")

        print(f"\n\n=== Phase 3: Resume API design ===\n")
        result = t.run(
            "Back to the API — what status codes for each endpoint?",
            max_steps=5, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Report — show each branch's compiled context
        print("\n\n=== Final State ===\n")
        branches = t.list_branches()
        print(f"  Branches: {[b.name for b in branches]}")
        print(f"  Current: {t.current_branch}")

        status = t.status()
        pct = status.token_count / status.token_budget_max * 100
        print(f"  Context: {status.token_count} tokens ({pct:.0f}% of budget)")

        original = t.current_branch
        for branch in branches:
            t.switch(branch.name)
            print(f"\n  [{branch.name}]:")
            t.compile().pprint(style="compact")
        t.switch(original)

        if len(branches) > 1:
            print(f"\n  Agent created {len(branches) - 1} branch(es) for tangent isolation.")
        else:
            print("\n  Agent did not branch for the tangent.")


if __name__ == "__main__":
    main()


# --- See also ---
# Branching basics (no LLM):  getting_started/04_branches.py
# Multi-agent delegation:     agent/08_multi_agent.py
