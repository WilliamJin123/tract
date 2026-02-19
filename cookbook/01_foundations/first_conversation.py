"""First Conversation

A coding assistant that remembers its system prompt, takes a user question,
calls an LLM, and commits the response. Persists to disk so the conversation
survives a restart. Reopens the next day and picks up where it left off.

Uses Cerebras (OpenAI-compatible) as the LLM provider.
"""

import os
import tempfile

import httpx
from dotenv import load_dotenv

from tract import (
    DialogueContent,
    InstructionContent,
    Tract,
    TractConfig,
)

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def call_llm(messages: list[dict]) -> dict:
    """Call Cerebras chat completion and return the full response."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{CEREBRAS_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            },
            json={"model": CEREBRAS_MODEL, "messages": messages},
        )
        response.raise_for_status()
        return response.json()


def main():
    # --- Session 1: Start a new conversation and persist to disk ---

    db_path = os.path.join(os.path.curdir, "01_foundations.db")
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
        print("\nDB cleaned up.")
    
    config = TractConfig(db_path=db_path)

    print(f"=== Session 1: New conversation (db: {db_path}) ===\n")

    with Tract.open(db_path, config=config, tract_id="coding-assistant") as t:
        # 1. Commit the system prompt
        t.commit(
            InstructionContent(text="You are a helpful coding assistant. Be concise."),
            message="system prompt",
        )

        # 2. Commit a user question
        user_msg = "What's the difference between a list and a tuple in Python?"
        t.commit(
            DialogueContent(role="user", text=user_msg),
            message="user asks about lists vs tuples",
        )

        # 3. Compile context into LLM-ready messages
        compiled = t.compile()
        messages = [{"role": m.role, "content": m.content} for m in compiled.messages]

        print(f"Compiled {compiled.commit_count} commits, {compiled.token_count} tokens")
        print(f"Messages being sent to LLM:")
        for m in messages:
            print(f"  [{m['role']}] {m['content'][:80]}...")
        print()

        # 4. Call the LLM
        response = call_llm(messages)
        assistant_text = response["choices"][0]["message"]["content"]

        print(f"Assistant: {assistant_text[:200]}...\n")

        # 5. Commit the assistant's response
        t.commit(
            DialogueContent(role="assistant", text=assistant_text),
            message="assistant explains lists vs tuples",
        )

        # 6. Check status
        status = t.status()
        print(f"Status: {status.commit_count} commits, {status.token_count} tokens")
        print(f"Branch: {status.branch_name}, HEAD: {status.head_hash[:8]}")
        print(f"DB persisted to: {db_path}\n")

    # --- Session 2: Reopen the same tract and continue ---

    print("=== Session 2: Reopening persisted conversation ===\n")

    with Tract.open(db_path, config=config, tract_id="coding-assistant") as t:
        # The conversation is right where we left it
        status = t.status()
        print(f"Restored: {status.commit_count} commits, {status.token_count} tokens")

        # Walk the log to see what's there
        history = t.log()
        print(f"History ({len(history)} commits):")
        for entry in reversed(history):
            print(f"  {entry.commit_hash[:8]} [{entry.content_type}] {entry.message}")
        print()

        # Continue the conversation with a follow-up
        t.commit(
            DialogueContent(role="user", text="Show me a quick example of each."),
            message="user asks for examples",
        )

        compiled = t.compile()
        messages = [{"role": m.role, "content": m.content} for m in compiled.messages]

        print(f"Compiled {compiled.commit_count} commits for follow-up call")

        response = call_llm(messages)
        assistant_text = response["choices"][0]["message"]["content"]

        t.commit(
            DialogueContent(role="assistant", text=assistant_text),
            message="assistant shows examples",
        )

        print(f"Assistant: {assistant_text[:200]}...\n")

        # Final status
        status = t.status()
        print(f"Final: {status.commit_count} commits, {status.token_count} tokens")



if __name__ == "__main__":
    main()
