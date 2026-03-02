"""Message Reordering

Three tiers of message reordering -- manual compile(order=), interactive
hash selection, and agent-driven toolkit execution.

PART 1 -- Manual           compile(order=), ReorderWarning, deterministic
PART 2 -- Interactive       Show order, click.prompt for new order
PART 3 -- LLM / Agent      ToolExecutor compiles with custom order

Demonstrates: compile(order=), ReorderWarning, commit_hashes,
              click.prompt, ToolExecutor, pprint(style="chat")
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract, ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


def _build_conversation(t):
    """Helper: build a 3-exchange conversation and return hashes."""
    t.system("You are a concise nutrition expert.")
    t.chat("What are macronutrients?")
    t.chat("Explain intermittent fasting.")
    t.chat("What's the role of gut bacteria in digestion?")

    ctx = t.compile()
    return ctx, ctx.commit_hashes


# =============================================================================
# PART 1 -- Manual: compile(order=), ReorderWarning, deterministic
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Message Reordering")
    print("=" * 60)
    print()
    print("  compile(order=) reorders messages by commit hash.")
    print("  Returns (CompiledContext, list[ReorderWarning]).")
    print("  Safety checks warn about structural issues like edits")
    print("  appearing before their targets.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        ctx, hashes = _build_conversation(t)
        # [0]=system, [1]=r1_user, [2]=r1_asst, [3]=r2_user, [4]=r2_asst, [5]=r3_user, [6]=r3_asst

        print("\n  Original order:\n")
        ctx.pprint(style="chat")

        # Reorder: gut bacteria first, then macros, then fasting
        new_order = [
            hashes[0],              # system stays first
            hashes[5], hashes[6],   # r3: gut bacteria
            hashes[1], hashes[2],   # r1: macronutrients
            hashes[3], hashes[4],   # r2: fasting
        ]

        reordered, warnings = t.compile(order=new_order)

        print(f"\n  Reordered (gut bacteria first):\n")
        reordered.pprint(style="chat")

        print(f"\n  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    [{w.severity}] {w.warning_type}: {w.description}")

        if not warnings:
            print(f"    (none -- all APPEND-only, safe to reorder)")

        print(f"\n  Reordering rearranges the compiled context for better LLM")
        print(f"  flow without changing the underlying commit history.")


# =============================================================================
# PART 2 -- Interactive: Show order, click.prompt for new order
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Choose Your Order")
    print("=" * 60)
    print()
    print("  View the current message order, then rearrange by entering")
    print("  comma-separated indices.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        ctx, hashes = _build_conversation(t)

        print("\n  Current message order:")
        for i, (h, msg) in enumerate(zip(hashes, ctx.messages)):
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:50]
            print(f"    [{i}] {h[:8]}  {role:10s}  {content}...")

        # Prompt for new order
        raw = click.prompt(
            "\n  New order (comma-separated indices, e.g. 0,5,6,1,2,3,4)",
            default=",".join(str(i) for i in range(len(hashes))),
        )

        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            new_order = [hashes[i] for i in indices]
        except (ValueError, IndexError):
            print("  Invalid input. Using original order.")
            new_order = list(hashes)

        reordered, warnings = t.compile(order=new_order)

        print(f"\n  Reordered context:\n")
        reordered.pprint(style="chat")

        if warnings:
            print(f"\n  Warnings: {len(warnings)}")
            for w in warnings:
                print(f"    [{w.severity}] {w.warning_type}: {w.description}")
        else:
            print(f"\n  No warnings -- safe reorder.")


# =============================================================================
# PART 3 -- LLM / Agent: ToolExecutor compiles with custom order
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: ToolExecutor Reordering")
    print("=" * 60)
    print()
    print("  An LLM agent uses ToolExecutor to compile with a custom")
    print("  message order -- optimizing flow without human input.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        ctx, hashes = _build_conversation(t)

        executor = ToolExecutor(t)

        # Agent decides to put gut bacteria first for topical relevance
        agent_order = [
            hashes[0],              # system stays first
            hashes[5], hashes[6],   # gut bacteria
            hashes[1], hashes[2],   # macronutrients
            hashes[3], hashes[4],   # fasting
        ]

        print(f"\n  Agent-chosen order: system -> gut bacteria -> macros -> fasting")
        result = executor.execute("compile", {"order": agent_order})
        print(f"  ToolExecutor compile result: {result}")

        # Also compile directly to show the context
        reordered, warnings = t.compile(order=agent_order)
        print(f"\n  Reordered context:\n")
        reordered.pprint(style="chat")

        print(f"\n  Agent can optimize message ordering for better LLM")
        print(f"  performance based on topic relevance or recency.")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
