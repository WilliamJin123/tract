"""Shorthand Commits and Format Methods

Same conversation as 01, but using the convenience layer: system(), user(),
assistant() instead of manual content models, and format methods to produce
output ready for any LLM provider. Still no LLM call — just building and
formatting context.

Demonstrates: system(), user(), assistant(), to_dicts(), to_openai(),
              to_anthropic(), print(ctx) compact summary, ctx.pprint()
"""

from tract import Tract


def main():
    t = Tract.open()

    # --- Shorthand commits: one call per message, no imports needed ---

    t.system("You are a helpful assistant.")
    t.user("What is Python?")
    t.assistant(
        "Python is a high-level, interpreted programming language known for its readability."
    )

    # --- Compile and format for different providers ---

    ctx = t.compile()
    # str(ctx) gives a compact one-liner summary — handy for logging or quick checks
    print(ctx)
    print()

    # Generic format — list of {"role": ..., "content": ...} dicts
    print("=== to_dicts() ===")
    for msg in ctx.to_dicts():
        print(f"  {msg}")

    # OpenAI format — identical to to_dicts(), ready for openai.chat.completions.create()
    print("\n=== to_openai() ===")
    for msg in ctx.to_openai():
        print(f"  {msg}")

    # Anthropic format — system prompt extracted, ready for anthropic.messages.create()
    print("\n=== to_anthropic() ===")
    anthropic_fmt = ctx.to_anthropic()
    print(f"  system: {anthropic_fmt['system']}")
    print(f"  messages:")
    for msg in anthropic_fmt["messages"]:
        print(f"    {msg}")

    # pprint() gives the full rich table view of the compiled context
    print()
    ctx.pprint()

    t.close()


if __name__ == "__main__":
    main()
