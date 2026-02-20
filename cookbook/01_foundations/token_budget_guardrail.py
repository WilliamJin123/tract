"""Token Budget Guardrail

A chatbot with a tight token budget (4096 tokens) that checks remaining
capacity before every LLM call. chat() automatically records the API's actual
token usage so tracking reflects reality, not just tiktoken estimates.

Demonstrates: status(), TokenBudgetConfig, chat(), auto usage recording
"""

import os

from dotenv import load_dotenv

from tract import TokenBudgetConfig, Tract, TractConfig, ChatResponse

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    # Configure a budget of 4096 tokens — tight enough to see the guardrail fire
    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=4096),
    )

    with Tract.open(
        config=config,
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:
        # System prompt
        t.system("You are a concise assistant. Keep answers under 3 sentences.")

        # Simulate a multi-turn conversation
        questions = [
            "What is a Python decorator?",
            "How does asyncio work?",
            "Explain the GIL in Python.",
            "What are metaclasses?",
            "How does garbage collection work in CPython?",
        ]

        for i, question in enumerate(questions, 1):
            # --- Pre-call check: do we have budget left? ---
            status = t.status()
            budget_max = status.token_budget_max or float("inf")
            usage_pct = (status.token_count / budget_max * 100) if budget_max else 0

            print(f"\n--- Turn {i} ---")
            print(f"  Tokens: {status.token_count}/{budget_max} ({usage_pct:.0f}%)")

            if usage_pct > 90:
                print(f"  WARNING: Budget nearly exhausted ({usage_pct:.0f}%). Stopping.")
                print(f"  (In production, you'd compress or branch here.)")
                break

            # chat() does everything: commit user msg, compile, call LLM,
            # commit response, AND record API usage automatically
            response = t.chat(question)

            # Usage is available on the response — no manual record_usage() needed
            if response.usage:
                print(f"  API usage: {response.usage.prompt_tokens} prompt + "
                      f"{response.usage.completion_tokens} completion tokens")

            print(f"  Assistant: {response.text[:100]}...")

        # Final summary
        print("\n=== Final Status ===")
        status = t.status()
        budget_max = status.token_budget_max or 0
        print(f"Commits: {status.commit_count}")
        print(f"Tokens: {status.token_count}/{budget_max}")
        print(f"Token source: {status.token_source}")
        print(f"Branch: {status.branch_name}")


if __name__ == "__main__":
    main()
