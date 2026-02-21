"""Budget Guardrail Loop

A chatbot that checks its token budget before every LLM call and stops
when it's running hot. chat() automatically records the API's actual
token usage, so tracking reflects reality — not just tiktoken estimates.

Migrated from: 01_foundations/token_budget_guardrail.py

Demonstrates: status() in a loop, chat() with auto usage recording,
              record_usage() for manual calls
"""

import os

from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    # Tight budget — small enough to see the guardrail fire
    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=4096),
    )

    with Tract.open(
        config=config,
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
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
            # --- Pre-call check ---
            status = t.status()
            budget_max = status.token_budget_max or float("inf")
            usage_pct = (status.token_count / budget_max * 100) if budget_max else 0

            print(f"\n--- Turn {i} ---")
            print(f"  Tokens: {status.token_count}/{budget_max} ({usage_pct:.0f}%)")

            if usage_pct > 90:
                print(f"  STOPPING: Budget nearly exhausted ({usage_pct:.0f}%).")
                print(f"  (In production, you'd compress or branch here.)")
                break

            # chat() commits the user message, calls the LLM, commits the
            # response, AND records API-reported usage automatically
            response = t.chat(question)

            if response.usage:
                print(f"  API usage: {response.usage.prompt_tokens} prompt + "
                      f"{response.usage.completion_tokens} completion")

            print(f"  Assistant: {response.text[:100]}...")

        # --- Final summary ---
        print("\n=== Final Status ===")
        status = t.status()
        budget_max = status.token_budget_max or 0
        print(f"Commits:      {status.commit_count}")
        print(f"Tokens:       {status.token_count}/{budget_max}")
        print(f"Token source: {status.token_source}")


if __name__ == "__main__":
    main()
