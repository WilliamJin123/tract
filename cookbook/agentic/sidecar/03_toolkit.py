"""Toolkit: expose tract operations as LLM-callable tools.

  PART 1 -- Manual:      as_tools(profile="self"), 3 profiles, ToolExecutor.execute()
  PART 2 -- LLM / Agent:  Orchestrator with custom ToolProfile for autonomous tool use
  PART 3 -- LLM / Agent:  Agent sets its own triggers via register_trigger tool
"""

import sys
from pathlib import Path

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile
from tract.orchestrator import Orchestrator, OrchestratorConfig, AutonomyLevel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: profiles, tool listing, direct execution
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Profiles and ToolExecutor")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are an astronomy research assistant.")
        t.user("What causes a solar eclipse?")
        t.assistant("A solar eclipse occurs when the Moon passes between "
                    "Earth and the Sun, casting a shadow on Earth.")

        # Three built-in profiles control which tools are exposed
        for profile in ["self", "supervisor", "full"]:
            tools = t.as_tools(profile=profile, format="openai")
            names = [tool["function"]["name"] for tool in tools]
            print(f"\n  Profile '{profile}': {len(tools)} tools")
            print(f"    {', '.join(names[:6])}{'...' if len(names) > 6 else ''}")

        # ToolExecutor dispatches tool calls against a tract
        executor = ToolExecutor(t)
        result = executor.execute("status", {})
        print(f"\n  executor.execute('status', {{}}):")
        print(f"    success={result.success}")
        print(f"    output={result.output[:100]}...")

        # Profile filtering on the executor
        executor.set_profile("supervisor")
        print(f"\n  Supervisor tools: {executor.available_tools()}")


# =====================================================================
# PART 2 -- LLM / Agent: Orchestrator with custom profile
# =====================================================================

def part2_agent():
    if not llm.api_key:
        print("\n" + "=" * 60)
        print("Part 2: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("PART 2 -- LLM / Agent: Orchestrator-Driven Tool Loop")
    print("=" * 60)

    # Custom profile: only expose inspection + maintenance tools
    maintenance_profile = ToolProfile(
        name="maintenance",
        tool_configs={
            "status": ToolConfig(
                enabled=True,
                description="Check context health: token count, budget, commits.",
            ),
            "log": ToolConfig(
                enabled=True,
                description="View recent commit history with hashes and types.",
            ),
            "annotate": ToolConfig(
                enabled=True,
                description=(
                    "Pin important commits (priority='pinned') or skip "
                    "irrelevant ones (priority='skip')."
                ),
            ),
            "compress": ToolConfig(
                enabled=True,
                description=(
                    "Compress context when budget usage is high. Provide "
                    "a concise summary of the conversation so far."
                ),
            ),
        },
    )

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2000))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a context management agent. Use the provided tools "
                 "to inspect and manage the conversation history.")
        for i in range(5):
            t.user(f"Research note {i}: stellar nucleosynthesis produces "
                   f"elements heavier than hydrogen in star cores.")

        # Autonomous orchestrator with our custom profile
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=15,
            profile=maintenance_profile,
            task_context="Inspect the conversation context. Check status and log, "
                         "then take any maintenance actions needed (pin important "
                         "content, compress if budget is high).",
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Orchestrator completed: {result.total_tool_calls} tool calls, "
              f"state={result.state.value}")
        for step in result.steps:
            status = "OK" if step.success else "FAIL"
            args_short = str(step.tool_call.arguments)[:60]
            print(f"    [{status}] {step.tool_call.name}({args_short})")

        final = t.status()
        print(f"\n  Final: {final.token_count} tokens, {final.commit_count} commits")


# =====================================================================
# PART 3 -- LLM / Agent: Agent creates its own triggers via tools
# =====================================================================
# Key insight: triggers are NOT just developer-time configuration.
# An LLM can use register_trigger to create deterministic "rules"
# from its probabilistic reasoning -- turning fuzzy judgment into
# repeatable, automatic policies for its own workflow.

def part3_agent_triggers():
    if not llm.api_key:
        print("\n" + "=" * 60)
        print("Part 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Self-Configuring Trigger Policies")
    print("=" * 60)
    print()
    print("  The agent assesses its context, then uses register_trigger")
    print("  to create deterministic rules for its own maintenance.")
    print("  This turns probabilistic LLM judgment into repeatable policies.")
    print()

    # SUPERVISOR profile includes register_trigger + unregister_trigger
    # SELF profile only has toggle_triggers (pause/resume)
    trigger_profile = ToolProfile(
        name="trigger-manager",
        tool_configs={
            "status": ToolConfig(
                enabled=True,
                description="Check context health and token budget usage.",
            ),
            "log": ToolConfig(
                enabled=True,
                description="View recent commits to understand context shape.",
            ),
            "register_trigger": ToolConfig(
                enabled=True,
                description=(
                    "Register a built-in trigger to automate maintenance. "
                    "Available types: compress, pin, gc, branch, merge, rebase, archive. "
                    "Each accepts a config dict with type-specific thresholds. "
                    "Example: register_trigger('compress', {'threshold': 0.7})"
                ),
            ),
            "unregister_trigger": ToolConfig(
                enabled=True,
                description="Remove a registered trigger by name.",
            ),
            "toggle_triggers": ToolConfig(
                enabled=True,
                description="Pause or resume all trigger evaluation.",
            ),
            "compress": ToolConfig(
                enabled=True,
                description="Compress context to free up token budget.",
            ),
        },
    )

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=1500))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a context management agent. Your job is to set up "
                 "automatic maintenance policies for a long-running session.")

        # Seed some conversation context
        for i in range(6):
            t.user(f"Research note {i}: quantum error correction using "
                   f"surface codes with distance {3 + i}.")
            t.assistant(f"Recorded. Logical error rate scales as p^{(3+i)//2}.")

        status_before = t.status()
        budget_max = status_before.token_budget_max or 1
        print(f"  Before: {status_before.token_count}/{budget_max} tokens, "
              f"0 triggers registered")

        # Let the agent assess and register its own triggers
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=15,
            profile=trigger_profile,
            task_context=(
                "Assess this context's health and set up automatic maintenance "
                "policies using register_trigger. Consider:\n"
                "1. A compress trigger to auto-compress when budget fills up\n"
                "2. A pin trigger to preserve important instructions\n"
                "3. A gc trigger to clean up dead commits\n"
                "Choose appropriate thresholds based on the current budget and "
                "context size. Then verify by checking status again."
            ),
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Orchestrator: {result.total_tool_calls} tool calls, "
              f"state={result.state.value}")
        for step in result.steps:
            s = "OK" if step.success else "FAIL"
            args_short = str(step.tool_call.arguments)[:70]
            print(f"    [{s}] {step.tool_call.name}({args_short})")

        # Show what triggers the agent registered
        print(f"\n  Agent-registered triggers:")
        for info in t.list_triggers():
            print(f"    - {info['name']} (fires_on={info['fires_on']}, "
                  f"priority={info['priority']})")
        if not t.list_triggers():
            print("    (none)")

        # Now the triggers will fire automatically on future operations
        print(f"\n  These triggers will now fire automatically on every")
        print(f"  commit() and compile() -- the agent created its own rules.")


def main():
    part1_manual()
    part2_agent()
    part3_agent_triggers()


if __name__ == "__main__":
    main()
