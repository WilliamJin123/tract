"""Status and Token Budget

Check how many tokens are in the context window and how close you are to a
limit. With real LLM calls via chat(), status() reflects API-reported usage
alongside tiktoken estimates — giving you ground-truth tracking.

Demonstrates: status(), TractConfig(token_budget=), TokenBudgetConfig,
              budget tracking fields, status.pprint(), print(status),
              chat() with auto usage recording
"""

import os

from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


def main():
    # --- Without a budget: just counting tokens ---

    print("=== No budget: raw token counting ===\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Answer concisely.")

        # chat() commits the user message, calls the LLM, commits the response,
        # and records API-reported usage automatically
        response = t.chat("What are the three states of matter?")
        print(f"Assistant: {response.text}\n")

        status = t.status()
        # The individual fields are there if you need them — e.g. for conditional logic:
        print(f"Commits:     {status.commit_count}")
        print(f"Token source: {status.token_source}")
        print(f"Budget max:  {status.token_budget_max}")  # None — no budget set
        # pprint() shows all of the above in a rich panel at once
        print()
        status.pprint()

    # --- With a budget: tracking against a limit ---

    print("\n=== With budget: tracking against 500 tokens ===\n")

    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=500),
    )

    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Answer concisely.")

        response = t.chat("What is photosynthesis?")
        print(f"Assistant: {response.text}\n")

        status = t.status()
        budget_max = status.token_budget_max or 0
        usage_pct = (status.token_count / budget_max * 100) if budget_max else 0
        # str(status) gives a compact one-liner: "main @ abc1234f | N commits | X/Y (Z%) tokens"
        print(status)

        # Add more messages and watch the budget fill up
        response = t.chat("How does the water cycle work?")
        print(f"\nAssistant: {response.text}\n")

        status = t.status()
        usage_pct = (status.token_count / budget_max * 100) if budget_max else 0
        print(f"After follow-up:")
        # str(status) in a loop is compact and doesn't clutter the output
        print(status)

        if usage_pct > 80:
            print("WARNING: Over 80% budget — time to compress or take action!")


if __name__ == "__main__":
    main()
