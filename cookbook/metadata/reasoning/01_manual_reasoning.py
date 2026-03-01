"""Manual Reasoning Commits

Part 1 of Reasoning Traces: use t.reasoning() to commit thinking text,
inspect it in log(), and verify it's excluded from compile(). No LLM needed.

t.reasoning() commits a ReasoningContent with SKIP priority by default.
The reasoning is in the commit chain (visible in log()) but excluded
from compile() — the LLM never sees it unless you ask for it.

Demonstrates: t.reasoning(), log(), compile(), get_content(), get_metadata()
"""

from tract import Tract


def part1_manual_reasoning():
    print("=" * 60)
    print("Part 1: MANUAL REASONING COMMITS")
    print("=" * 60)
    print()
    print("  t.reasoning() commits chain-of-thought text.")
    print("  Default priority is SKIP — excluded from compile().")
    print()

    t = Tract.open()

    # Build a conversation with reasoning between user and assistant
    t.system("You are a math tutor. Show your work.")
    t.user("What is 17 * 23?")

    # Reasoning: the model's internal thinking (committed manually here)
    r_info = t.reasoning(
        "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391",
        format="parsed",
    )

    t.assistant("17 × 23 = 391")

    # --- Reasoning is in log() ---

    print("  log() shows reasoning commits:\n")
    for entry in reversed(t.log()):
        print(f"    {entry}")

    print(f"\n  Reasoning commit: {r_info.commit_hash[:8]}")
    print(f"  Content type:     {r_info.content_type}")

    # --- But excluded from compile() ---

    ctx = t.compile()
    print(f"\n  compile() -> {ctx.commit_count} messages (reasoning excluded):")
    for msg in ctx.messages:
        print(f"    [{msg.role}] {msg.content[:60]}")

    # --- Format and metadata ---

    print(f"\n  t.reasoning() also accepts format= and metadata=:")
    t2 = Tract.open()
    info = t2.reasoning(
        "Let me think step by step...",
        format="think_tags",
        metadata={"source": "deepseek-r1"},
    )
    content = t2.get_content(info.commit_hash)
    print(f"    format:   {content['format']}")
    meta = t2.get_metadata(info.commit_hash)
    print(f"    metadata: {meta}")
    t2.close()

    t.close()


if __name__ == "__main__":
    part1_manual_reasoning()
