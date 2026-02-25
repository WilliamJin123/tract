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


def main():
    # Tight budget — small enough to see the guardrail fire
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
            # --- Pre-call check ---
            status = t.status()
            budget_max = status.token_budget_max or float("inf")
            usage_pct = (status.token_count / budget_max * 100) if budget_max else 0

            print(f"\n--- Turn {i} ---")
            # print(status) gives a compact one-liner per turn — doesn't clutter the loop
            print(f"  {status}")

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

            # str(response) is the same as response.text — compact for loop output
            print(f"  Assistant: {str(response)[:100]}...")

        # --- Final summary ---
        print("\n=== Final Status ===")
        # pprint() gives the full panel with branch, HEAD, budget bar, and source
        t.status().pprint()


if __name__ == "__main__":
    main()
