"""Shorthand Commits and Format Methods

Same conversation as 01, but using the convenience layer: system(), user(),
generate() instead of manual content models, and format methods to produce
output ready for any LLM provider. Uses a real LLM call via generate() to
produce the assistant response.

Demonstrates: system(), user(), generate(), to_dicts(), to_openai(),
              to_anthropic(), print(ctx) compact summary, ctx.pprint()
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


def main():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # --- Shorthand commits: one call per message, no imports needed ---

        t.system("You are a helpful assistant.")
        t.user("What is Python?")

        # generate() compiles the context, calls the LLM, commits the response,
        # and records token usage — all in one call.
        r = t.generate()
        r.pprint()

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


if __name__ == "__main__":
    main()
