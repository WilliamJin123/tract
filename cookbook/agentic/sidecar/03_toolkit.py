"""Toolkit: expose tract operations as LLM-callable tools.

  PART 1 -- Manual:      as_tools(profile="self"), 3 profiles, ToolExecutor.execute()
  PART 3 -- LLM / Agent:  Orchestrator with custom ToolProfile for autonomous tool use
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
# PART 3 -- LLM / Agent: Orchestrator with custom profile
# =====================================================================

def part3_agent():
    if not llm.api_key:
        print("\n" + "=" * 60)
        print("Part 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Orchestrator-Driven Tool Loop")
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


def main():
    part1_manual()
    part3_agent()


if __name__ == "__main__":
    main()
