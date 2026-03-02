"""Tool Description Hints -- Description-Driven Behavior

The simplest self-managing pattern: good tool descriptions tell the agent
when to act. No system prompt crutches needed for simple meta-decisions.

The insight: "call BEFORE answering when creative vs precise" in a tool
description reliably triggers the right behavior. The agent reads the
tool schema, understands the hint, and self-configures.

Demonstrates: ToolProfile customization, description-driven tool selection,
              inline agent configuration without system prompts
"""

import json
import os

import click
import httpx
from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =====================================================================
# The hint profile: description IS the instruction
# =====================================================================
# Key insight: the description tells the model *when* to use the tool,
# not just what it does. Without this, models know configure_model
# exists but won't reliably call it before answering. With the hint,
# they self-configure every time.

HINT_PROFILE = ToolProfile(
    name="hint-driven",
    tool_configs={
        "configure_model": ToolConfig(
            enabled=True,
            description=(
                "Set temperature BEFORE answering. Call this when the task "
                "requires creative output (use temperature=0.7-1.0) or when "
                "it requires precise/factual output (use temperature=0.0-0.3). "
                "Always call this before your first response to a new question."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check context health. Call this AFTER every 3-5 exchanges "
                "to monitor token budget usage. If budget exceeds 70%, "
                "mention it to the user."
            ),
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Pin or skip a commit. Call with priority='pinned' when the "
                "user says something is important, a key decision, or should "
                "never be lost. Call with priority='skip' when content is "
                "clearly superseded or irrelevant."
            ),
        ),
    },
)


# =====================================================================
# PART 1 -- Manual: Show the profile, execute tools directly
# =====================================================================

def part1_manual():
    """Show the hint-driven ToolProfile and execute tools via ToolExecutor."""
    print("=" * 60)
    print("PART 1 -- Manual: Hint-Driven Tool Definitions")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Write a poem about rain.")
        t.assistant("Silver threads from grey above, rain descends with gentle love.")

        # Show what the LLM sees: tool schemas with embedded hints
        tools = t.as_tools(profile=HINT_PROFILE, format="openai")
        print(f"\n  Hint profile exposes {len(tools)} tools:\n")
        for tool in tools:
            fn = tool["function"]
            # The description IS the instruction -- no system prompt needed
            print(f"  {fn['name']}:")
            print(f"    \"{fn['description']}\"\n")

        # Execute manually via ToolExecutor
        executor = ToolExecutor(t)
        executor.set_profile(HINT_PROFILE)

        # The agent would call configure_model before a creative task
        result = executor.execute("configure_model", {"temperature": 0.9})
        print(f"  Manual configure_model(temperature=0.9):")
        print(f"    success={result.success}, output={result.output}\n")

        # The agent would call status to check budget
        result = executor.execute("status", {})
        print(f"  Manual status():")
        print(f"    {result.output}\n")

        # Compare: without the hint, the default description is generic
        default_tools = t.as_tools(profile="self", format="openai")
        for tool in default_tools:
            if tool["function"]["name"] == "configure_model":
                print(f"  Default description (no hint):")
                print(f"    \"{tool['function']['description']}\"")
                break

        print(f"\n  With hint:")
        for tool in tools:
            if tool["function"]["name"] == "configure_model":
                print(f"    \"{tool['function']['description']}\"")
                break

        print("\n  The hint tells the model WHEN to call, not just WHAT it does.")


# =====================================================================
# PART 2 -- Interactive: Agent suggests, human confirms
# =====================================================================

def part2_interactive():
    """Agent suggests tool calls based on task type; human confirms each one."""
    print(f"\n{'=' * 60}")
    print("PART 2 -- Interactive: Human-Gated Tool Hints")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        executor = ToolExecutor(t)
        executor.set_profile(HINT_PROFILE)

        # Simulate what a hint-driven agent would do for different tasks
        tasks = [
            ("Write a haiku about stars.", "creative", 0.9),
            ("What is 2 + 2?", "precise", 0.1),
            ("Invent a name for a new color.", "creative", 0.95),
        ]

        for task, task_type, suggested_temp in tasks:
            print(f"\n  Task: \"{task}\"")
            print(f"  Hint analysis: {task_type} -> temperature={suggested_temp}")

            # The agent would call configure_model based on the hint
            if click.confirm(
                f"  Execute configure_model(temperature={suggested_temp})?",
                default=True,
            ):
                result = executor.execute(
                    "configure_model", {"temperature": suggested_temp}
                )
                print(f"    -> {result.output}")
            else:
                print(f"    -> Skipped (human override)")

            # Commit the task as a user message
            t.user(task)

        # Check status at the end
        if click.confirm("\n  Check context status?", default=True):
            result = executor.execute("status", {})
            print(f"    -> {result.output}")


# =====================================================================
# PART 3 -- Agent: Full LLM loop with hint-driven tool selection
# =====================================================================

def part3_agent():
    """Full agentic loop where the LLM reads tool hints and self-configures."""
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("PART 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 3 -- Agent: LLM Self-Configures from Tool Hints")
    print("=" * 60)
    print()
    print("  No system prompt about when to configure.")
    print("  The tool description IS the instruction.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=HINT_PROFILE)
        t.set_tools(tools)

        # Minimal system prompt -- no mention of configure_model
        t.system("You are a helpful assistant.")

        # Each task should trigger different temperature selection
        # based purely on the tool description hint
        tasks = [
            "Write a surreal one-sentence poem about a clock that melts.",
            "What is the speed of light in meters per second?",
        ]

        for task in tasks:
            print(f"  Task: \"{task[:60]}...\"")
            t.user(task)

            # Agentic loop: let the LLM call tools or respond
            for turn in range(5):
                response = t.generate()

                if not response.tool_calls:
                    # Text response -- the agent is done with this task
                    gc = response.generation_config
                    temp = gc.temperature if gc else "default"
                    print(f"  [{temp}] {response.text[:120]}...")
                    print()
                    break

                # Execute each tool call the LLM requested
                for tc in response.tool_calls:
                    result = executor.execute(tc.name, tc.arguments)
                    t.tool_result(tc.id, tc.name, str(result))
                    print(f"    -> {tc.name}({tc.arguments})")
                    print(f"       {result.output[:80]}")

        # Show provenance: which commits got which temperature
        print(f"  {'=' * 50}")
        print("  Generation Config Provenance:")
        for ci in t.log():
            if ci.generation_config:
                gc = ci.generation_config
                print(f"    {ci.commit_hash[:8]}  temp={gc.temperature}  "
                      f"({ci.content_type})")


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/config/01_per_call.py       -- Per-call config with sugar params
# cookbook/agentic/sidecar/03_toolkit.py          -- ToolExecutor and profiles
# cookbook/agentic/self_managing/04_profiles.py   -- Profile scoping (self/supervisor/observer)
