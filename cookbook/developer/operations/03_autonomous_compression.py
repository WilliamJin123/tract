"""Agent-driven compression: toolkit, hooks, and orchestrator automation.

Three ways to automate compression in agent pipelines:

  PART 1 -- Manual:      ToolExecutor(t).execute("compress", {...}) direct tool call
  PART 2 -- Interactive:  t.on("compress", handler) with pending.pprint() + click.confirm
  PART 3 -- LLM / Agent:  CompressTrigger(threshold=0.7) + budget auto-manages compression
"""

import os

import click
from dotenv import load_dotenv

from tract import CompressTrigger, Priority, TokenBudgetConfig, Tract, TractConfig
from tract.hooks.compress import PendingCompress
from tract.toolkit import ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: ToolExecutor direct tool call")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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
        print(f"    output:  {result.output[:120]}...")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")


def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: hook handler with human approval")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise history tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What caused the fall of Rome?")
        t.chat("Explain the Renaissance.")
        t.chat("What was the Space Race and who won?")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # Register a hook handler -- fires when compress() is called
        def review_handler(pending: PendingCompress):
            pending.pprint(verbose=True)
            if click.confirm("\n  Approve compression?", default=True):
                pending.approve()
            else:
                pending.reject("User declined")

        t.on("compress", review_handler)

        # Now compress() fires the handler instead of auto-committing
        result = t.compress(target_tokens=200)

        print(f"\n  Hook log ({len(t.hook_log)} events):")
        for event in t.hook_log:
            print(f"    {event.operation}: {event.outcome}")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")


def part3_agent():
    print("=" * 60)
    print("PART 3 -- LLM / Agent: CompressTrigger auto-manages budget")
    print("=" * 60)

    # Open with a token budget so the trigger has a threshold to watch
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
