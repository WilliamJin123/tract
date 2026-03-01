"""Formatting

Part 3 of Reasoning Traces: pprint() renders reasoning commits in dim
cyan — visually distinct from dialogue. All three styles (table/chat/compact)
handle reasoning content. No LLM needed.

Demonstrates: pprint() reasoning style (table, chat, compact),
              compile(include_reasoning=True)
"""

from tract import Tract


def part3_formatting():
    print(f"\n{'=' * 60}")
    print("Part 3: FORMATTING (pprint with reasoning)")
    print("=" * 60)
    print()
    print("  Reasoning commits render in dim cyan, visually")
    print("  distinct from regular dialogue.\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What is the capital of France?")
    t.reasoning(
        "The user is asking about France's capital. This is a "
        "straightforward geography question. The answer is Paris."
    )
    t.assistant("The capital of France is Paris.")

    # Include reasoning so pprint() can show it
    ctx = t.compile(include_reasoning=True)

    print("  --- table style ---\n")
    ctx.pprint(style="table")

    print("\n  --- chat style ---\n")
    ctx.pprint(style="chat")

    print("\n  --- compact style ---\n")
    ctx.pprint(style="compact")

    t.close()


if __name__ == "__main__":
    part3_formatting()
