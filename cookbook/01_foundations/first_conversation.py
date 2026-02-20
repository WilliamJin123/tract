"""First Conversation

A coding assistant that remembers its system prompt, takes a user question,
calls an LLM, and commits the response. Persists to disk so the conversation
survives a restart. Reopens the next day and picks up where it left off.

Uses Cerebras (OpenAI-compatible) as the LLM provider.

Demonstrates: Tract.open(api_key=...), system(), chat(), ChatResponse,
              persistence across sessions
"""

import os

from dotenv import load_dotenv

from tract import Tract, TractConfig

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    # --- Session 1: Start a new conversation and persist to disk ---

    db_path = os.path.join(os.path.curdir, "01_foundations.db")
    if os.path.exists(db_path):
        os.unlink(db_path)
        print("\nDB cleaned up.")

    config = TractConfig(db_path=db_path)

    print(f"=== Session 1: New conversation (db: {db_path}) ===\n")

    with Tract.open(
        db_path,
        config=config,
        tract_id="coding-assistant",
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:
        # 1. System prompt — one call, no imports needed
        t.system("You are a helpful coding assistant. Be concise.")

        # 2. Ask a question — chat() commits the user message, calls the LLM,
        #    commits the response, and records token usage automatically
        response = t.chat("What's the difference between a list and a tuple in Python?")

        print(f"Assistant: {response.text[:200]}...")
        print(f"Model used: {response.generation_config.get('model')}")
        if response.usage:
            print(f"Tokens: {response.usage.prompt_tokens} prompt + "
                  f"{response.usage.completion_tokens} completion\n")

        # 3. Check status
        status = t.status()
        print(f"Status: {status.commit_count} commits, {status.token_count} tokens")
        print(f"Branch: {status.branch_name}, HEAD: {status.head_hash[:8]}")
        print(f"DB persisted to: {db_path}\n")

    # --- Session 2: Reopen the same tract and continue ---

    print("=== Session 2: Reopening persisted conversation ===\n")

    with Tract.open(
        db_path,
        config=config,
        tract_id="coding-assistant",
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:
        # The conversation is right where we left it
        status = t.status()
        print(f"Restored: {status.commit_count} commits, {status.token_count} tokens")

        # Walk the log
        history = t.log()
        print(f"History ({len(history)} commits):")
        for entry in reversed(history):
            print(f"  {entry.commit_hash[:8]} [{entry.content_type}] {entry.message}")
        print()

        # Continue with a follow-up — chat() includes all prior context automatically
        response = t.chat("Show me a quick example of each.")

        print(f"Assistant: {response.text[:200]}...\n")

        # Final status
        status = t.status()
        print(f"Final: {status.commit_count} commits, {status.token_count} tokens")


if __name__ == "__main__":
    main()
