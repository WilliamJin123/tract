"""Context Management via Tools

An LLM agent monitors and maintains its own context health — checking status,
compressing when budget is high, pinning important content, configuring models,
and running GC. All decisions are made by the LLM through tool calls.

Scenario: A long conversation fills up the token budget. The agent is asked
to assess and maintain the context: check status, pin key information,
compress older turns, adjust model settings, and clean up.

Tools exercised: status, compile, compress, annotate, gc, configure_model

Demonstrates: LLM-driven context maintenance, budget-aware compression,
              priority annotations, model switching, garbage collection
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


# Tool profile: context management tools
CONTEXT_PROFILE = ToolProfile(
    name="context-manager",
    tool_configs={
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check context health: branch, HEAD, token count, budget "
                "percentage. Call this first to understand the current state "
                "before taking any maintenance action."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile current context into messages. Shows message count "
                "and total tokens. Use to verify state after operations."
            ),
        ),
        "compress": ToolConfig(
            enabled=True,
            description=(
                "Compress a range of commits into a summary to reduce token "
                "usage. Pinned commits are preserved verbatim. Optionally "
                "set target_tokens and provide instructions for the summary."
            ),
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Set priority on a commit. Use 'pinned' to protect important "
                "context from compression. Use 'skip' to hide irrelevant "
                "content. Use 'normal' to reset. Requires a commit hash "
                "(find hashes via log)."
            ),
        ),
        "gc": ToolConfig(
            enabled=True,
            description=(
                "Run garbage collection to remove orphaned commits. Frees "
                "storage space. Set min_age_days to control retention."
            ),
        ),
        "configure_model": ToolConfig(
            enabled=True,
            description=(
                "Change model or temperature. Use a smaller model for "
                "compression tasks, a larger model for complex reasoning. "
                "Set operation='compress' to configure only compression."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description="View recent commits to find hashes for annotate/compress.",
        ),
    },
)


def run_agent_loop(t, executor, tools, task, max_turns=12):
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
# PART 1 -- Manual: Show context management tools
# =====================================================================

def part1_manual():
    """Show context management tools and execute them manually."""
    print("=" * 60)
    print("PART 1 -- Manual: Context Management Tools")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2000))
    with Tract.open(config=config) as t:
        t.system("You are a helpful assistant.")
        for i in range(5):
            t.user(f"Research note {i}: topic details here.")
            t.assistant(f"Acknowledged research note {i}.")

        executor = ToolExecutor(t)
        executor.set_profile(CONTEXT_PROFILE)

        # Show available tools
        tools = t.as_tools(profile=CONTEXT_PROFILE, format="openai")
        print(f"\n  Context profile: {len(tools)} tools")
        for tool in tools:
            print(f"    - {tool['function']['name']}")

        print("\n  Conversation:")
        t.compile().pprint(style="compact")

        # Status
        result = executor.execute("status", {})
        print(f"\n  status():\n    {result.output}")

        # Log
        result = executor.execute("log", {"limit": 3})
        print(f"\n  log(limit=3):\n    {result.output[:200]}")

        print()


# =====================================================================
# PART 2 -- Agent: LLM manages context autonomously
# =====================================================================

def part2_agent():
    """LLM agent assesses and maintains context health."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 2: SKIPPED (no API key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- Agent: LLM Manages Context Health")
    print("=" * 60)
    print()
    print("  The agent will: check status, pin important content,")
    print("  compress old turns, and run GC — all autonomously.")
    print()

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2000))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=CONTEXT_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a context management assistant. You have tools to "
            "inspect, compress, annotate, and clean up conversation history. "
            "When asked to maintain the context, use these tools to keep it "
            "healthy and within budget."
        )

        # Fill up the context with research notes
        for i in range(8):
            t.user(f"Research note {i}: Quantum computing uses qubits which "
                   f"can be in superposition states. Error rate: {0.1 / (i+1):.3f}")
            t.assistant(f"Recorded note {i}. Key finding: error rate {0.1/(i+1):.3f}.")

        # Pin the most important note manually (so agent can find it)
        important = t.log(limit=2)[0]
        t.annotate(important.commit_hash, "pinned", reason="Key error rate data")

        print(f"  Context filled: {t.status().token_count} tokens")
        print(f"  Budget: {t.status().token_budget_max} max")

        print("\n  BEFORE maintenance:")
        t.compile().pprint(style="compact")

        # Ask the agent to maintain the context
        print("\n  --- Task: Assess and maintain context ---")
        run_agent_loop(
            t, executor, tools,
            "The context is getting large. Please:\n"
            "1. Check status to see budget usage\n"
            "2. Use log to find commits, then pin any important ones\n"
            "3. Compress older turns to free up space\n"
            "4. Run GC to clean up orphaned commits\n"
            "5. Check status again to confirm improvement"
        )

        # Show final state
        print("\n  AFTER maintenance:")
        t.compile().pprint(style="compact")

        status = t.status()
        print(f"\n  Final: {status.token_count} tokens, "
              f"{status.commit_count} commits")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/operations/01_compress.py         -- Manual compression
# cookbook/developer/metadata/02_priority.py           -- Priority annotations
# cookbook/developer/operations/09_gc.py               -- Garbage collection
# cookbook/agentic/self_managing/03_budget_awareness.py -- Budget-driven decisions
