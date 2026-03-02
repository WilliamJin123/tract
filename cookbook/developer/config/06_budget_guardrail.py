"""Budget Guardrail Loop

A chatbot that checks its token budget before every LLM call and stops
when it's running hot. chat() automatically records the API's actual
token usage, so tracking reflects reality — not just tiktoken estimates.

Migrated from: 01_foundations/token_budget_guardrail.py

Demonstrates: status() in a loop, chat() with auto usage recording,
              record_usage() for manual calls, print(status) compact,
              print(response) for assistant text, status.pprint() final summary
"""

import os

from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: Manual Budget Guardrail Loop
# =============================================================================
# Check token budget before every call. Stop when usage exceeds threshold.

def part1_manual():
    print("=" * 60)
    print("Part 1: MANUAL BUDGET GUARDRAIL")
    print("=" * 60)
    print()

    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=4096),
    )

    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise assistant. Keep answers under 3 sentences.")

        questions = [
            "What is a Python decorator?",
            "How does asyncio work?",
            "Explain the GIL in Python.",
            "What are metaclasses?",
            "How does garbage collection work in CPython?",
        ]

        for i, question in enumerate(questions, 1):
            status = t.status()
            budget_max = status.token_budget_max or float("inf")
            usage_pct = (status.token_count / budget_max * 100) if budget_max else 0

            print(f"\n--- Turn {i} ---")
            print(f"  {status}")

            if usage_pct > 90:
                print(f"  STOPPING: Budget nearly exhausted ({usage_pct:.0f}%).")
                print(f"  (In production, you'd compress or branch here.)")
                break

            response = t.chat(question)

            if response.usage:
                print(f"  API usage: {response.usage.prompt_tokens} prompt + "
                      f"{response.usage.completion_tokens} completion")

            print(f"  Assistant: {str(response)[:100]}...")

        print("\n=== Final Status ===")
        t.status().pprint()


# =============================================================================
# Part 2: Interactive Budget Guardrail
# =============================================================================
# Status check in loop. When budget hits 80%, ask the human whether to
# compress. Show status.pprint() between turns for full visibility.

def part2_interactive():
    import click

    print(f"\n{'=' * 60}")
    print("Part 2: INTERACTIVE BUDGET GUARDRAIL")
    print("=" * 60)
    print()

    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=4096),
    )

    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise assistant. Keep answers under 3 sentences.")

        questions = [
            "What is a Python decorator?",
            "How does asyncio work?",
            "Explain the GIL in Python.",
            "What are metaclasses?",
            "How does garbage collection work in CPython?",
        ]

        for i, question in enumerate(questions, 1):
            status = t.status()
            budget_max = status.token_budget_max or float("inf")
            usage_pct = (status.token_count / budget_max * 100) if budget_max else 0

            print(f"\n--- Turn {i} ---")
            status.pprint()

            # At 80%, offer human the choice to compress
            if usage_pct > 80:
                if click.confirm(
                    f"  Budget at {usage_pct:.0f}%. Compress now?",
                    default=True,
                ):
                    result = t.compress(target_tokens=int(budget_max * 0.4))
                    print(f"  Compressed: {result.original_tokens} -> "
                          f"{result.compressed_tokens} tokens")
                    t.status().pprint()

            if usage_pct > 95:
                print("  STOPPING: Budget nearly exhausted.")
                break

            response = t.chat(question)
            print(f"  Assistant: {str(response)[:100]}...")

        print("\n=== Final Status ===")
        t.status().pprint()


# =============================================================================
# Part 3: Agent Auto-Manages Budget via Triggers
# =============================================================================
# CompressTrigger auto-compresses when budget fills past threshold.
# The agent's chat loop runs without manual intervention.

def part3_agent():
    print(f"\n{'=' * 60}")
    print("Part 3: AGENT AUTO-MANAGES BUDGET VIA TRIGGERS")
    print("=" * 60)
    print()

    from tract.orchestrator import CompressTrigger

    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=4096),
    )

    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise assistant. Keep answers under 3 sentences.")

        # Auto-compress when budget hits 80%
        trigger = CompressTrigger(threshold=0.8)
        t.configure_triggers([trigger])

        before_tokens = t.compile().token_count
        print(f"  Before chat loop: {before_tokens} tokens")

        questions = [
            "What is a Python decorator?",
            "How does asyncio work?",
            "Explain the GIL in Python.",
            "What are metaclasses?",
            "How does garbage collection work in CPython?",
        ]

        for i, question in enumerate(questions, 1):
            response = t.chat(question)
            status = t.status()
            budget_max = status.token_budget_max or float("inf")
            usage_pct = (status.token_count / budget_max * 100) if budget_max else 0
            print(f"  Turn {i}: {usage_pct:.0f}% budget | {str(response)[:60]}...")

        after_tokens = t.compile().token_count
        print(f"\n  After chat loop: {after_tokens} tokens")
        print(f"  Trigger auto-compressed to stay within budget.")
        t.status().pprint()


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
