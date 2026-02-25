"""Manual Commit and Compile

The absolute basics: open a tract, commit three messages manually using
the content type models, compile them into a message list, and inspect
the result. No LLM, no shorthand — just the raw mechanics.

Demonstrates: Tract.open(), commit(), compile(), CompiledContext.messages,
              InstructionContent, DialogueContent, ctx.pprint()
"""

from tract import Tract, InstructionContent, DialogueContent


def main():
    # Open an in-memory tract (no file path = ephemeral, great for experimenting)
    t = Tract.open()

    # --- Commit three messages using content type models ---

    # 1. System prompt — InstructionContent has role "system" by default
    c1 = t.commit(InstructionContent(text="You are a helpful assistant."))
    print(f"Committed system:    {c1.commit_hash[:8]}  [{c1.content_type}]  {c1.message}")

    # 2. User message — DialogueContent needs an explicit role
    c2 = t.commit(DialogueContent(role="user", text="What is Python?"))
    print(f"Committed user:      {c2.commit_hash[:8]}  [{c2.content_type}]  {c2.message}")

    # 3. Assistant response — same model, different role
    c3 = t.commit(DialogueContent(
        role="assistant",
        text="Python is a high-level, interpreted programming language known for its readability.",
    ))
    print(f"Committed assistant: {c3.commit_hash[:8]}  [{c3.content_type}]  {c3.message}")

    # --- Compile: turn commit history into a message list ---

    ctx = t.compile()

    # pprint() shows all messages in a rich table with token totals —
    # the easiest way to inspect a compiled context at a glance.
    print()
    ctx.pprint()

    t.close()


if __name__ == "__main__":
    main()
