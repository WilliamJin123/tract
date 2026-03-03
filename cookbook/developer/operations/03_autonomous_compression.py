"""Autonomous compression: toolkit and trigger-driven automation.

Two ways to automate compression in pipelines:

  PART 1 -- Manual:      ToolExecutor(t).execute("compress", {...}) direct tool call
  PART 2 -- Trigger-Driven:  CompressTrigger(threshold=0.7) + budget auto-manages compression
"""

import sys
from pathlib import Path

from tract import CompressTrigger, Priority, TokenBudgetConfig, Tract, TractConfig
from tract.toolkit import ToolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: ToolExecutor direct tool call")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise philosophy tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What is Stoicism? Give me a practical example.")
        t.chat("Explain utilitarianism. When does it fail?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # ToolExecutor: the programmatic way to invoke what agents call
        executor = ToolExecutor(t)
        print(f"\n  Available tools: {executor.available_tools()}")

        result = executor.execute("compress", {
            "content": (
                "User discussed two philosophies: Stoicism (focus on what you "
                "can control) and utilitarianism (maximize overall good, fails "
                "when individual rights are trampled)."
            ),
        })

        print(f"\n  ToolResult:")
        print(f"    success: {result.success}")
        print(f"    output:  {result.output}" if result.success else f"    error:   {result.error}")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")


def part2_trigger_driven():
    print("=" * 60)
    print("PART 2 -- Trigger-Driven: CompressTrigger auto-manages budget")
    print("=" * 60)

    # Open with a token budget so the trigger has a threshold to watch
    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        config=TractConfig(token_budget=TokenBudgetConfig(max_tokens=800)),
    ) as t:

        sys_ci = t.system("You are a concise science tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        # Register a CompressTrigger that fires at 70% budget usage
        trigger = CompressTrigger(threshold=0.7)
        t.configure_triggers([trigger])

        print(f"  Budget: {t.config.token_budget.max_tokens} tokens")
        print(f"  Trigger fires at: 70% ({int(800 * 0.7)} tokens)\n")

        # Add messages until the budget fills and trigger fires
        questions = [
            "What is photosynthesis?",
            "How does gravity work?",
            "Explain the water cycle.",
            "What are tectonic plates?",
            "How do vaccines work?",
        ]

        for q in questions:
            status = t.status()
            print(f"  [{status.token_count}/{800} tokens] Asking: {q[:40]}...")
            t.chat(q)

        print(f"\n  Final context:\n")
        ctx = t.compile()
        ctx.pprint(style="table")
        print(f"\n  {ctx.token_count} tokens, {len(ctx.messages)} messages")
        print(f"  CompressTrigger auto-managed the budget. PINNED system survived.")


def main():
    part1_manual()
    part2_trigger_driven()


if __name__ == "__main__":
    main()
