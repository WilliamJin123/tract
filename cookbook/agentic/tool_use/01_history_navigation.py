"""History Navigation via Tools

An LLM agent navigates conversation history using tract's inspection and
navigation tools. The agent decides when to look back, compare states, and
rewind — all through genuine tool calls, not hardcoded logic.

Scenario: The agent builds a multi-turn conversation, then is asked to
investigate what changed between two points and undo a bad turn. It uses
log to find commits, diff to compare them, get_commit for details, reset
to rewind, and ORIG_HEAD to recover.

Tools exercised: log, diff, get_commit, reset, checkout, compile, status

Demonstrates: LLM-driven history inspection, reset + ORIG_HEAD undo,
              checkout for read-only time-travel, agent reasoning about
              its own commit history
"""

import json
import sys
from pathlib import Path

import httpx

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


# Tool profile: only history navigation tools
HISTORY_PROFILE = ToolProfile(
    name="history-navigator",
    tool_configs={
        "log": ToolConfig(
            enabled=True,
            description=(
                "View recent commit history. Call this to find commit hashes, "
                "understand what content was added and when. Returns hashes, "
                "messages, content types, and token counts."
            ),
        ),
        "diff": ToolConfig(
            enabled=True,
            description=(
                "Compare two commits. Call this to see what changed between "
                "two points in history — messages added, removed, modified, "
                "and token delta."
            ),
        ),
        "get_commit": ToolConfig(
            enabled=True,
            description=(
                "Get full details about a specific commit: content type, "
                "operation, tokens, metadata, tags. Use when you need to "
                "inspect a particular commit found via log."
            ),
        ),
        "reset": ToolConfig(
            enabled=True,
            description=(
                "Reset HEAD to a previous commit. Moves the branch pointer "
                "backward, making later commits invisible to compile(). "
                "The original HEAD is saved as ORIG_HEAD for recovery. "
                "Pass 'ORIG_HEAD' as target to undo the most recent reset."
            ),
        ),
        "checkout": ToolConfig(
            enabled=True,
            description=(
                "Checkout a commit for read-only inspection without modifying "
                "the branch. Use this to peek at a historical state. Pass '-' "
                "to return to your previous position."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile current context into messages. Call this to see "
                "what the LLM would receive right now — useful after reset "
                "or checkout to verify the state."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check current state: branch, HEAD commit, token count. "
                "Call after navigation operations to confirm position."
            ),
        ),
    },
)


def run_agent_loop(t, executor, tools, task, max_turns=10):
    """Generic agentic loop: user task -> tool calls -> final response."""
    t.user(task)

    for turn in range(max_turns):
        response = t.generate()

        if not response.tool_calls:
            print(f"\n  Agent: {response.text[:200]}")
            if len(response.text) > 200:
                print(f"         ...({len(response.text)} chars total)")
            return response

        for tc in response.tool_calls:
            result = executor.execute(tc.name, tc.arguments)
            t.tool_result(tc.id, tc.name, str(result))
            args_short = json.dumps(tc.arguments)[:60]
            print(f"    -> {tc.name}({args_short})")
            output_short = str(result.output)[:80]
            print(f"       {output_short}")

    print("  (max turns reached)")
    return None


# =====================================================================
# PART 1 -- Manual: Show the tools, execute directly
# =====================================================================

def part1_manual():
    """Show history navigation tools and execute them manually."""
    print("=" * 60)
    print("PART 1 -- Manual: History Navigation Tools")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a concise history tutor.")
        r1 = t.user("What caused World War I?")
        t.assistant("WWI was triggered by the assassination of Archduke "
                    "Franz Ferdinand, combined with alliance systems and "
                    "imperial rivalries.")
        r2 = t.user("What about World War II?")
        t.assistant("WWII began with Nazi Germany's invasion of Poland in "
                    "1939, driven by territorial expansion and ideology.")

        executor = ToolExecutor(t)
        executor.set_profile(HISTORY_PROFILE)

        # Show available tools
        tools = t.as_tools(profile=HISTORY_PROFILE, format="openai")
        print(f"\n  History profile: {len(tools)} tools")
        for tool in tools:
            print(f"    - {tool['function']['name']}")

        print("\n  Conversation:")
        t.compile().pprint(style="chat")

        # Log
        result = executor.execute("log", {"limit": 5})
        print(f"\n  log(limit=5):\n    {result.output[:200]}")

        # Status
        result = executor.execute("status", {})
        print(f"\n  status():\n    {result.output}")

        print()


# =====================================================================
# PART 2 -- Agent: LLM navigates history autonomously
# =====================================================================

def part2_agent():
    """LLM agent uses history tools to investigate and navigate."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 2: SKIPPED (no API key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- Agent: LLM Navigates History")
    print("=" * 60)
    print()
    print("  The agent will: inspect history with log, compare with diff,")
    print("  reset to undo, and use ORIG_HEAD to recover.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=HISTORY_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a context management assistant. You have tools to "
            "inspect and navigate conversation history. When asked to "
            "investigate or fix history, use the tools — don't guess."
        )

        # Build some history for the agent to work with
        t.user("The capital of France is Berlin.")
        t.assistant("Noted: capital of France is Berlin.")
        t.user("Actually, what's the capital of Germany?")
        t.assistant("The capital of Germany is Berlin.")
        t.user("And the capital of Japan?")
        t.assistant("The capital of Japan is Tokyo.")

        print("  Conversation before agent acts:")
        t.compile().pprint(style="chat")

        # Now ask the agent to investigate and fix
        print("  --- Task: Find and undo the bad information ---")
        run_agent_loop(
            t, executor, tools,
            "Look at the conversation history. There's incorrect information "
            "early on (France's capital was recorded as Berlin). Use log to "
            "find the commits, then reset HEAD to just before that incorrect "
            "exchange. After resetting, check status to confirm."
        )

        print("\n  After reset:")
        t.compile().pprint(style="compact")

        # Ask it to recover
        print("\n  --- Task: Undo the reset ---")
        run_agent_loop(
            t, executor, tools,
            "Actually, I want all the history back. Use ORIG_HEAD to undo "
            "the reset you just did, then verify with status and compile."
        )

        print("\n  After recovery:")
        t.compile().pprint(style="compact")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/history/01_log_and_diff.py   -- Manual log, diff, time-travel
# cookbook/developer/history/02_reset.py          -- Manual reset + ORIG_HEAD
# cookbook/developer/history/03_edit_history.py   -- Edit chain tracking
