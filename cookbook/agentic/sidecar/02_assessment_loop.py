"""Orchestrator loop: autonomous context management.

  PART 1 -- Manual:      Manual assessment: status() + if/else decision tree
  PART 2 -- Interactive:  OrchestratorConfig(autonomy_ceiling=COLLABORATIVE) + hooks
  PART 3 -- LLM / Agent:  OrchestratorConfig(autonomy_ceiling=AUTONOMOUS, max_steps=20) + triggers
"""

import os

import click
from dotenv import load_dotenv

from tract import (
    CompressTrigger,
    GCTrigger,
    Tract,
    TractConfig,
    TokenBudgetConfig,
)
from tract.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    AutonomyLevel,
    ToolCallDecision,
    ToolCallReview,
)

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


# =====================================================================
# PART 1 -- Manual: if/else decision tree, no orchestrator
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Decision Tree")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=300))
    with Tract.open(config=config) as t:
        t.system("You are a planetary science assistant.")
        for i in range(8):
            t.user(f"Observation {i}: detected methane signatures on Titan.")
            t.assistant(f"Analysis {i}: methane lakes suggest active geology.")

        status = t.status()
        budget_max = status.token_budget_max or 1
        usage_pct = status.token_count / budget_max
        print(f"\n  Tokens: {status.token_count}/{budget_max} ({usage_pct:.0%})")

        # Manual decision tree
        if usage_pct > 0.8:
            print("  Action: compressing (over 80% budget)")
            t.compress(content="Titan observations: methane signatures detected, "
                       "suggesting active geological processes and liquid lakes.")
        elif len(t.log(limit=100)) > 30:
            print("  Action: running GC (>30 commits)")
            t.gc()
        else:
            print("  Action: no maintenance needed")

        status = t.status()
        print(f"  After: {status.token_count}/{budget_max} tokens, "
              f"{status.commit_count} commits")


# =====================================================================
# PART 2 -- Interactive: collaborative orchestrator with hooks
# =====================================================================

def part2_interactive():
    print("\n" + "=" * 60)
    print("PART 2 -- Interactive: Collaborative Orchestrator")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=300))
    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a space exploration planner.")
        for i in range(6):
            t.user(f"Mission requirement {i}: radiation shielding specs.")

        # Hook: user confirms compress operations
        def compress_hook(pending):
            if click.confirm(f"  Compress? reason={pending.reason}"):
                pending.approve()
            else:
                pending.reject("user declined")

        t.on("compress", compress_hook, name="interactive-compress")

        # Collaborative orchestrator: on_tool_call reviews each action
        def review_tool_call(tc):
            label = f"{tc.name}({tc.arguments})"
            if click.confirm(f"  Approve tool call: {label}?", default=True):
                return ToolCallReview(decision=ToolCallDecision.APPROVED)
            return ToolCallReview(
                decision=ToolCallDecision.REJECTED,
                reason="user declined",
            )

        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.COLLABORATIVE,
            max_steps=5,
            on_tool_call=review_tool_call,
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()
        print(f"\n  Orchestrator completed: {result.total_tool_calls} tool calls, "
              f"state={result.state.value}")


# =====================================================================
# PART 3 -- LLM / Agent: fully autonomous orchestrator with triggers
# =====================================================================

def part3_agent():
    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Autonomous Orchestrator")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=400))
    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are an astrophysics research coordinator.")

        # Configure triggers for auto-maintenance
        t.configure_triggers([
            CompressTrigger(threshold=0.7, summary_content="Astrophysics session summary."),
            GCTrigger(max_dead_commits=5),
        ])

        # Seed conversation
        for i in range(10):
            t.user(f"Data point {i}: quasar luminosity measurement at z={i * 0.5:.1f}.")
            t.assistant(f"Recorded measurement {i}. Luminosity consistent with models.")

        # Autonomous orchestrator: no human review, max 20 steps
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=20,
            task_context="Manage a long astrophysics research session. "
                         "Compress when near budget, GC dead commits.",
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Steps executed: {result.total_tool_calls}")
        print(f"  State: {result.state.value}")
        for step in result.steps[:5]:
            status = "OK" if step.success else "FAIL"
            print(f"    [{status}] {step.tool_call.name}: "
                  f"{(step.result_output or step.result_error or '')[:60]}")

        status = t.status()
        print(f"\n  Final: {status.token_count} tokens, {status.commit_count} commits")


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
