"""Orchestrator loop: autonomous context management.

  PART 1 -- Manual:      Manual assessment: status() + if/else decision tree
  PART 2 -- LLM / Agent:  OrchestratorConfig(autonomy_ceiling=AUTONOMOUS, max_steps=20) + triggers
  PART 3 -- LLM / Agent:  Agent adapts triggers mid-run via register/unregister tools
"""

import sys
from pathlib import Path

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
)
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


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

        print("\n  Conversation before maintenance:")
        t.compile().pprint(style="compact")

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

        print("\n  AFTER maintenance:")
        t.compile().pprint(style="compact")

        status = t.status()
        print(f"  After: {status.token_count}/{budget_max} tokens, "
              f"{status.commit_count} commits")


# =====================================================================
# PART 2 -- LLM / Agent: fully autonomous orchestrator with triggers
# =====================================================================

def part2_agent():
    print("\n" + "=" * 60)
    print("PART 2 -- LLM / Agent: Autonomous Orchestrator")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=400))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
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

        print("\n  Context after orchestrator:")
        t.compile().pprint(style="compact")

        status = t.status()
        print(f"\n  Final: {status.token_count} tokens, {status.commit_count} commits")


# =====================================================================
# PART 3 -- LLM / Agent: agent adapts its own trigger policies
# =====================================================================
# The agent can register, unregister, and toggle triggers during a run.
# This shows the key pattern: LLM reasoning creates deterministic rules.

def part3_adaptive_triggers():
    if not llm.api_key:
        print("\n" + "=" * 60)
        print("PART 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Adaptive Trigger Policies")
    print("=" * 60)
    print()
    print("  The agent starts with no triggers, assesses the context,")
    print("  and sets up appropriate policies using register_trigger.")
    print("  LLM judgment -> deterministic rules that fire automatically.")
    print()

    adaptive_profile = ToolProfile(
        name="adaptive",
        tool_configs={
            "status": ToolConfig(
                enabled=True,
                description="Check context health and budget.",
            ),
            "log": ToolConfig(
                enabled=True,
                description="View recent commits.",
            ),
            "register_trigger": ToolConfig(
                enabled=True,
                description=(
                    "Register a trigger. Types: compress, pin, gc, branch, "
                    "merge, rebase, archive. Each takes a config dict. "
                    "Example: register_trigger('compress', {'threshold': 0.8})"
                ),
            ),
            "unregister_trigger": ToolConfig(
                enabled=True,
                description="Remove a trigger by name (e.g., 'auto-compress').",
            ),
            "toggle_triggers": ToolConfig(
                enabled=True,
                description="Pause (false) or resume (true) all triggers.",
            ),
            "compress": ToolConfig(
                enabled=True,
                description="Compress context to free budget.",
            ),
        },
    )

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=500))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a context management agent.")
        for i in range(8):
            t.user(f"Data entry {i}: experiment results batch {i}.")
            t.assistant(f"Batch {i} recorded.")

        status = t.status()
        budget_max = status.token_budget_max or 1
        print(f"  Before: {status.token_count}/{budget_max} tokens, no triggers")

        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=12,
            profile=adaptive_profile,
            task_context=(
                "Assess this context and set up maintenance triggers. "
                "Register a compress trigger and a gc trigger with "
                "thresholds appropriate for the current budget. "
                "Also register a pin trigger for instructions."
            ),
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Orchestrator: {result.total_tool_calls} tool calls")
        for step in result.steps:
            s = "OK" if step.success else "FAIL"
            args_short = str(step.tool_call.arguments)[:70]
            print(f"    [{s}] {step.tool_call.name}({args_short})")

        # Show registered triggers
        print(f"\n  Triggers now active:")
        for info in t.list_triggers():
            print(f"    - {info['name']} (fires_on={info['fires_on']})")
        if not t.list_triggers():
            print("    (none)")


def main():
    part1_manual()
    part2_agent()
    part3_adaptive_triggers()


if __name__ == "__main__":
    main()
