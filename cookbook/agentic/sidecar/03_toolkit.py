"""Toolkit: expose tract operations as LLM-callable tools.

  PART 1 -- Manual:      as_tools(profile="self"), 3 profiles, ToolExecutor.execute()
  PART 2 -- Interactive:  Human-gated tool execution with click.confirm per call
  PART 3 -- LLM / Agent:  Full LLM-driven tool loop: compile + tools -> LLM -> execute
"""

import json
import os

import click
import httpx
from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


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
# PART 2 -- Interactive: human-gated tool execution
# =====================================================================

def part2_interactive():
    print("\n" + "=" * 60)
    print("PART 2 -- Interactive: Human-Gated Tool Execution")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a research assistant.")
        t.user("Summarize the Drake equation.")
        t.assistant("The Drake equation estimates the number of active "
                    "civilizations in the Milky Way.")

        executor = ToolExecutor(t)

        # Simulate a list of tool calls an LLM might request
        planned_calls = [
            ("status", {}),
            ("log", {"limit": 5}),
            ("compile", {}),
        ]

        print("\n  Simulated LLM tool calls (human gates each one):\n")
        for name, args in planned_calls:
            if click.confirm(f"  Execute {name}({args})?", default=True):
                result = executor.execute(name, args)
                status = "OK" if result.success else "FAIL"
                output = (result.output or result.error or "")[:80]
                print(f"    [{status}] {output}...\n")
            else:
                print(f"    [SKIPPED] {name}\n")


# =====================================================================
# PART 3 -- LLM / Agent: full LLM-driven tool loop
# =====================================================================

def part3_agent():
    if not TRACT_OPENAI_API_KEY:
        print("\n" + "=" * 60)
        print("Part 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: LLM-Driven Tool Loop")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2000))
    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a context management agent. Use the provided tools "
                 "to inspect and manage the conversation history.")
        for i in range(5):
            t.user(f"Research note {i}: stellar nucleosynthesis produces "
                   f"elements heavier than hydrogen in star cores.")

        tools = t.as_tools(profile="self", format="openai")
        executor = ToolExecutor(t)
        print(f"\n  {len(tools)} tools available for LLM")

        # Use the tract's own context as the message history
        ctx = t.compile()
        messages = ctx.to_dicts()
        messages.append({
            "role": "user",
            "content": "Check the current context status and log. "
                       "Take any maintenance action if needed.",
        })

        # Call LLM with toolkit tools via HTTP (no private API access)
        llm_response = httpx.post(
            f"{TRACT_OPENAI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {TRACT_OPENAI_API_KEY}"},
            json={"model": MODEL_ID, "messages": messages, "tools": tools},
            timeout=120,
        )
        llm_response.raise_for_status()
        raw = llm_response.json()
        tool_calls = raw["choices"][0]["message"].get("tool_calls", [])

        if tool_calls:
            print(f"\n  LLM requested {len(tool_calls)} tool call(s):")
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = json.loads(tc["function"].get("arguments", "{}"))
                result = executor.execute(name, args)
                status = "OK" if result.success else "FAIL"
                print(f"    {name}({args}) -> [{status}]")
                print(f"      {(result.output or result.error or '')[:100]}")
        else:
            content = raw["choices"][0]["message"].get("content", "")
            print(f"\n  LLM responded without tools: {content[:120]}...")


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
