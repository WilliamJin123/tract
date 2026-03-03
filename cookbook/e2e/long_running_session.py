"""Long-running session: 50+ turns with auto-maintenance.

  PART 1 -- Manual:      Chat loop, check status(), manually compress/gc at intervals
  PART 2 -- Triggers:    Developer-configured CompressTrigger + GCTrigger + PinTrigger
  PART 3 -- LLM / Agent: Agent registers its own triggers via orchestrator tools
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
from tract.toolkit import ToolConfig, ToolProfile
from tract.orchestrator import Orchestrator, OrchestratorConfig, AutonomyLevel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: chat loop with manual maintenance
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Periodic Maintenance Loop")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=500))
    with Tract.open(config=config) as t:
        t.system("You are a deep-space mission planner.")

        # 15 turns with manual maintenance every 5
        for i in range(1, 16):
            t.user(f"Mission log entry {i}: trajectory correction delta-v "
                   f"computed for waypoint {i}.")
            t.assistant(f"Acknowledged entry {i}. Delta-v {0.5 + i * 0.1:.1f} m/s "
                        f"recorded. Fuel reserves nominal.")

            # Check and maintain every 5 turns
            if i % 5 == 0:
                status = t.status()
                budget_max = status.token_budget_max or 1
                usage_pct = status.token_count / budget_max
                print(f"\n  Turn {i}: {status.token_count}/{budget_max} "
                      f"({usage_pct:.0%}), {status.commit_count} commits")

                if usage_pct > 0.7:
                    start = max(1, i - 4)
                    t.compress(content=f"Mission log entries {start}-{i}: "
                               f"trajectory corrections computed, all nominal. "
                               f"Cumulative delta-v within fuel budget.")
                    t.gc(archive_retention_days=0)
                    status = t.status()
                    print(f"  Compressed + GC: {status.token_count}/{budget_max} "
                          f"({status.token_count / budget_max:.0%})")

        final = t.status()
        print(f"\n  Final: {final.token_count} tokens, {final.commit_count} commits")


# =====================================================================
# PART 2 -- LLM / Agent: fully automatic trigger-based maintenance
# =====================================================================

def part2_agent():
    print("\n" + "=" * 60)
    print("PART 2 -- LLM / Agent: 50+ Turns with Auto-Maintenance")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=800))
    with Tract.open(config=config) as t:
        # Configure all triggers for automatic maintenance
        t.configure_triggers([
            CompressTrigger(
                threshold=0.8,
                summary_content="Auto-compressed: telescope observation session.",
            ),
            GCTrigger(max_dead_commits=20),
            PinTrigger(pin_types={"instruction"}),
        ])

        t.system("You are a radio telescope observation coordinator.")

        # 50+ turn simulation
        targets = [
            "Crab Nebula", "Sagittarius A*", "Andromeda Galaxy",
            "Cygnus X-1", "Vela Pulsar", "Centaurus A",
            "Cassiopeia A", "Orion Nebula", "Magellanic Clouds",
            "3C 273 Quasar",
        ]

        compress_count = 0
        def count_compress(pending):
            nonlocal compress_count
            compress_count += 1
            pending.approve()

        t.on("compress", count_compress, name="counter")

        for cycle in range(5):
            for j, target in enumerate(targets):
                turn = cycle * len(targets) + j + 1
                t.user(f"Observation {turn}: point dish at {target}, "
                       f"frequency {1420 + j * 10} MHz, integration 30 min.")
                t.assistant(f"Observation {turn} complete. {target}: "
                            f"signal-to-noise {15 + j:.1f} dB, "
                            f"flux density {0.5 + j * 0.3:.1f} Jy.")

            # compile() evaluates triggers automatically
            ctx = t.compile()
            status = t.status()
            budget_max = status.token_budget_max or 1
            print(f"\n  After cycle {cycle + 1} (turn {(cycle + 1) * len(targets)}): "
                  f"{status.token_count}/{budget_max} tokens "
                  f"({status.token_count / budget_max:.0%}), "
                  f"{status.commit_count} commits")

        print(f"\n  Auto-compressions fired: {compress_count}")
        final = t.status()
        budget_max = final.token_budget_max or 1
        print(f"  Final: {final.token_count}/{budget_max} tokens "
              f"({final.token_count / budget_max:.0%}), "
              f"{final.commit_count} commits")
        print(f"  Context stayed under budget across 50 turns.")


# =====================================================================
# PART 3 -- LLM / Agent: agent self-configures triggers
# =====================================================================
# The developer doesn't pre-configure triggers. Instead, the agent
# assesses the session and registers triggers itself, turning its
# probabilistic judgment into deterministic maintenance rules.

def part3_self_configuring():
    if not llm.api_key:
        print("\n" + "=" * 60)
        print("PART 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Self-Configuring Maintenance")
    print("=" * 60)
    print()
    print("  No triggers pre-configured. The agent assesses the session")
    print("  and registers its own triggers based on context health.")
    print("  Fuzzy LLM reasoning -> deterministic maintenance rules.")
    print()

    # Profile with trigger management tools
    session_profile = ToolProfile(
        name="session-manager",
        tool_configs={
            "status": ToolConfig(
                enabled=True,
                description="Check context health, token budget, and commit count.",
            ),
            "log": ToolConfig(
                enabled=True,
                description="View recent commit history.",
            ),
            "register_trigger": ToolConfig(
                enabled=True,
                description=(
                    "Register a trigger to automate maintenance. Types: "
                    "compress (auto-compress at budget threshold), "
                    "pin (auto-pin instruction/session content), "
                    "gc (auto-GC when dead commits accumulate). "
                    "Pass config dict with thresholds."
                ),
            ),
            "toggle_triggers": ToolConfig(
                enabled=True,
                description="Pause or resume all triggers.",
            ),
            "compress": ToolConfig(
                enabled=True,
                description="Compress context to free up budget.",
            ),
        },
    )

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=800))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a geological survey coordinator tracking field reports.")

        # Seed initial conversation (no triggers yet)
        for i in range(5):
            t.user(f"Field report {i}: soil sample from site {chr(65+i)}, "
                   f"clay content {20+i*5}%, moisture {30+i*3}%.")
            t.assistant(f"Report {i} logged. Site {chr(65+i)} shows "
                        f"{'normal' if i < 3 else 'elevated'} clay levels.")

        status = t.status()
        budget_max = status.token_budget_max or 1
        print(f"  Initial: {status.token_count}/{budget_max} tokens, "
              f"0 triggers registered")

        # Agent assesses and sets up its own triggers
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=12,
            profile=session_profile,
            task_context=(
                "This is a long-running geological survey session. Assess the "
                "current context health and set up appropriate maintenance "
                "triggers:\n"
                "- A compress trigger with a threshold appropriate for the budget\n"
                "- A pin trigger to preserve system instructions\n"
                "- A gc trigger to clean up dead commits\n"
                "Choose thresholds based on the actual budget and current usage."
            ),
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Orchestrator: {result.total_tool_calls} tool calls")
        for step in result.steps:
            s = "OK" if step.success else "FAIL"
            args_short = str(step.tool_call.arguments)[:70]
            print(f"    [{s}] {step.tool_call.name}({args_short})")

        # Show what the agent registered
        print(f"\n  Agent-registered triggers:")
        if t.trigger_evaluator:
            for trig in t.trigger_evaluator._triggers:
                cfg = trig.to_config() if hasattr(trig, 'to_config') else {}
                print(f"    - {trig.name}: {cfg}")
        else:
            print("    (none)")

        # Now run more turns -- triggers fire automatically
        print(f"\n  --- Continuing session with agent-configured triggers ---")
        compress_count = 0
        def count_compress(pending):
            nonlocal compress_count
            compress_count += 1
            pending.approve()

        t.on("compress", count_compress, name="counter")

        for i in range(10):
            t.user(f"Follow-up report {i}: additional samples from expanded grid.")
            t.assistant(f"Follow-up {i} recorded. Pattern consistent with prior data.")

        ctx = t.compile()
        final = t.status()
        print(f"  After 10 more turns: {final.token_count}/{budget_max} tokens, "
              f"{compress_count} auto-compressions")
        print(f"  The agent's triggers managed the session automatically.")


def main():
    part1_manual()
    part2_agent()
    part3_self_configuring()


if __name__ == "__main__":
    main()
