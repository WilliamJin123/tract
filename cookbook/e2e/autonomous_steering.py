"""Autonomous steering: triggers + orchestrator + hooks over many turns.

  PART 1 -- Manual:      Manual trigger evaluation: trigger.evaluate(t) -> inspect -> execute
  PART 3 -- LLM / Agent:  All triggers active + autonomous orchestrator over 20+ turns
"""

import sys
from pathlib import Path

from tract import (
    CompressTrigger,
    GCTrigger,
    PinTrigger,
    Tract,
    TractConfig,
    TokenBudgetConfig,
)
from tract.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    AutonomyLevel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: evaluate triggers by hand, inspect, execute
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Trigger Evaluation")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=300))
    compress_trigger = CompressTrigger(threshold=0.6, summary_content="Session summary.")
    gc_trigger = GCTrigger(max_dead_commits=3)
    pin_trigger = PinTrigger(pin_types={"instruction"})

    with Tract.open(config=config) as t:
        t.system("You are a geology research assistant.")
        for i in range(8):
            t.user(f"Sample {i}: granite composition analysis from site Alpha.")
            t.assistant(f"Analysis {i}: 60% feldspar, 25% quartz, 15% mica.")

        # Evaluate each trigger manually -- no auto-fire
        for trigger in [compress_trigger, gc_trigger, pin_trigger]:
            action = trigger.evaluate(t)
            if action:
                print(f"\n  {type(trigger).__name__} FIRED:")
                print(f"    action_type: {action.action_type}")
                print(f"    autonomy:    {action.autonomy}")
                print(f"    reason:      {action.reason}")
                print(f"    params:      {action.params}")
            else:
                print(f"\n  {type(trigger).__name__}: no action needed")

        # Manually execute based on evaluation
        if compress_trigger.evaluate(t):
            print("\n  Manually executing compression...")
            t.compress(content="Geology session: 8 granite samples analyzed. "
                       "Composition: ~60% feldspar, 25% quartz, 15% mica.")
            print(f"  After compress: {t.status().token_count} tokens")


# =====================================================================
# PART 3 -- LLM / Agent: all triggers + autonomous orchestrator
# =====================================================================

def part3_agent():
    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Autonomous Steering Over 20+ Turns")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=500))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # All triggers active
        t.configure_triggers([
            CompressTrigger(threshold=0.8, summary_content="Auto-compressed session."),
            PinTrigger(pin_types={"instruction"}),
            GCTrigger(max_dead_commits=10),
        ])

        t.system("You are a seismology research assistant.")

        # Simulate 20+ turns of conversation
        regions = ["Pacific Ring of Fire", "Mid-Atlantic Ridge", "San Andreas Fault",
                   "Himalayan Belt", "East African Rift", "Cascadia Subduction Zone",
                   "Japan Trench", "Chilean Subduction Zone", "New Madrid Zone",
                   "Reykjanes Ridge"]
        for i, region in enumerate(regions):
            t.user(f"Seismic activity report for {region}, year 202{i % 5}.")
            t.assistant(f"Analysis: {region} shows typical tectonic activity. "
                        f"Magnitude range 2.0-6.{i % 10} recorded this period.")

        status = t.status()
        print(f"\n  Before orchestrator: {status.token_count} tokens, "
              f"{status.commit_count} commits")

        # Autonomous orchestrator manages everything
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=15,
            task_context="Manage a long seismology research session. "
                         "Keep context under budget, pin instructions.",
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Orchestrator: {result.total_tool_calls} tool calls")
        for step in result.steps[:5]:
            ok = "OK" if step.success else "FAIL"
            print(f"    [{ok}] {step.tool_call.name}")

        status = t.status()
        print(f"\n  After orchestrator: {status.token_count} tokens, "
              f"{status.commit_count} commits")


def main():
    part1_manual()
    part3_agent()


if __name__ == "__main__":
    main()
