"""Status and Token Budget

Check how many tokens are in the context window and how close you are to a
limit. No LLM needed — this is pure local tracking via tiktoken.

Demonstrates: status(), TractConfig(token_budget=), TokenBudgetConfig,
              budget tracking fields
"""

from tract import Tract, TractConfig, TokenBudgetConfig


def main():
    # --- Without a budget: just counting tokens ---

    print("=== No budget: raw token counting ===\n")

    t = Tract.open()

    t.system("You are a helpful assistant. Answer concisely.")
    t.user("What is Python?")
    t.assistant("Python is a high-level programming language.")

    status = t.status()
    print(f"Commits:     {status.commit_count}")
    print(f"Tokens:      {status.token_count}")
    print(f"Token source: {status.token_source}")
    print(f"Budget max:  {status.token_budget_max}")  # None — no budget set
    print(f"Branch:      {status.branch_name}")
    print(f"HEAD:        {status.head_hash[:8]}")

    t.close()

    # --- With a budget: tracking against a limit ---

    print("\n=== With budget: tracking against 500 tokens ===\n")

    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=500),
    )

    t = Tract.open(config=config)

    t.system("You are a helpful assistant. Answer concisely.")
    t.user("What is Python?")
    t.assistant(
        "Python is a high-level, interpreted programming language created by "
        "Guido van Rossum. It emphasizes code readability and supports multiple "
        "programming paradigms including procedural, object-oriented, and "
        "functional programming."
    )

    status = t.status()
    budget_max = status.token_budget_max or 0
    usage_pct = (status.token_count / budget_max * 100) if budget_max else 0

    print(f"Commits:     {status.commit_count}")
    print(f"Tokens:      {status.token_count} / {budget_max}")
    print(f"Usage:       {usage_pct:.1f}%")
    print(f"Token source: {status.token_source}")

    # Add more messages and watch the budget fill up
    t.user("Explain decorators in Python.")
    t.assistant(
        "Decorators are functions that modify the behavior of other functions. "
        "They use the @decorator syntax and are commonly used for logging, "
        "authentication, caching, and more. Under the hood, @decorator is "
        "syntactic sugar for func = decorator(func)."
    )

    status = t.status()
    usage_pct = (status.token_count / budget_max * 100) if budget_max else 0
    print(f"\nAfter 2 more commits:")
    print(f"Tokens:      {status.token_count} / {budget_max}")
    print(f"Usage:       {usage_pct:.1f}%")

    if usage_pct > 80:
        print("WARNING: Over 80% budget — time to compress or take action!")

    t.close()


if __name__ == "__main__":
    main()
