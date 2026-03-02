"""Hello Chat -- The Developer On-Ramp

The simplest thing you can do with tract: persistent chat. chat() handles
everything in one call -- commits the user message, compiles context, calls
the LLM, commits the response, and records token usage. Close the tract,
reopen it, and the conversation continues where you left off.

Demonstrates: Tract.open(), system(), chat(), persistence, status()
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "getting_started_chat.db")
    if os.path.exists(db_path):
        os.unlink(db_path)

    # --- Session 1: Start a new conversation ---

    print("=== Session 1 ===\n")

    with Tract.open(
        db_path,
        tract_id="my-chat",
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")

        response = t.chat("What are Python decorators in one sentence?")
        response.pprint()

    # --- Session 2: Reopen -- conversation is restored ---

    print("\n=== Session 2 (reopened from disk) ===\n")

    with Tract.open(
        db_path,
        tract_id="my-chat",
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # All prior context is included automatically
        response = t.chat("Give me a short example of one.")
        response.pprint()

        # Status shows commit count, token budget, and HEAD position
        print()
        t.status().pprint()


if __name__ == "__main__":
    main()


# --- See also ---
# Full chat patterns: developer/conversations/04_chat_and_persist.py
# Under the hood (commit, compile, content types): developer/00_internals.py
# Agent on-ramp: getting_started/02_agent.py
