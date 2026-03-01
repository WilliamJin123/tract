"""Manual Commit and Compile

Part 1 — The absolute basics: open a tract, commit three messages manually
using the content type models, compile them into a message list, and inspect
the result. No LLM, no shorthand — just the raw mechanics.

Part 2 — The same flow but with a real LLM call: commit the system and user
messages manually, then call generate() to get the assistant response from
an LLM instead of writing it yourself.

Demonstrates: Tract.open(), commit(), compile(), generate(), ChatResponse,
              InstructionContent, DialogueContent, ctx.pprint()
"""

import os

from dotenv import load_dotenv

from tract import Tract, InstructionContent, DialogueContent

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


# ---------------------------------------------------------------------------
# Part 1: Pure static commits (no LLM)
# ---------------------------------------------------------------------------

def part1_manual_commits():
    """Commit three messages by hand and compile them."""
    print("=" * 60)
    print("PART 1 — Manual commits (no LLM)")
    print("=" * 60)

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
    print("\n" * 2)
    ctx.pprint(style="chat")
    print("\n" * 2)
    ctx.pprint(style="compact")
    print("\n" * 2)
    ctx.pprint(style="table")

    t.close()


# ---------------------------------------------------------------------------
# Part 2: Commit + generate (with LLM)
# ---------------------------------------------------------------------------

def part2_compile_and_chat():
    """Commit a system prompt and user message, then let the LLM reply."""
    print("\n" + "=" * 60)
    print("PART 2 — generate() with a real LLM call")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Same two manual commits — system + user
        t.commit(InstructionContent(text="You are a helpful assistant."))
        t.commit(DialogueContent(role="user", text="What is Python?"))

        # generate() compiles the context, calls the LLM, commits the
        # assistant response, and records token usage — all in one step.
        resp = t.generate()

        # --- Inspect the ChatResponse ---
        print(f"\nAssistant text (first 120 chars):\n  {resp.text[:120]}...")
        print(f"\nToken usage:  {resp.usage}")
        print(f"Commit hash:  {resp.commit_info.commit_hash[:8]}")
        print(f"Content type: {resp.commit_info.content_type}")

        # Compile and pprint the full conversation including the LLM reply
        ctx = t.compile()
        print()
        ctx.pprint(style="chat")


def main():
    part1_manual_commits()
    part2_compile_and_chat()


if __name__ == "__main__":
    main()
