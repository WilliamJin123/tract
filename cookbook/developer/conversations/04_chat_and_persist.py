"""Chat and Persist

The real workflow: chat() handles everything in one call — commits the user
message, compiles context, calls the LLM, commits the response, and records
token usage. Close the tract, reopen it, and the conversation is right where
you left it.

Migrated from: 01_foundations/first_conversation.py

Demonstrates: chat(), ChatResponse, persistence, log(),
              response.pprint(), print(status), print(entry)
"""

import os
import sys
from pathlib import Path

from tract import Tract, TractConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm  

MODEL_ID = llm.large


def main():
    db_path = os.path.join(os.path.curdir, "conversation.db")
    if os.path.exists(db_path):
        os.unlink(db_path)

    # --- Session 1: Start a new conversation ---

    print("=== Session 1: New conversation ===\n")

    with Tract.open(
        db_path,
        tract_id="coding-assistant",
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # System prompt
        t.system("You are a helpful coding assistant. Be concise.")

        # chat() does everything: commit user msg -> compile -> call LLM ->
        # commit response -> record usage
        response = t.chat("What's the difference between a list and a tuple in Python?")

        # pprint() shows the response text, usage, and config in one rich panel
        response.pprint()
        print(f"DB: {db_path}\n")

    # --- Session 2: Reopen and continue ---

    print("=== Session 2: Reopening ===\n")

    with Tract.open(
        db_path,
        tract_id="coding-assistant",
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Everything is restored — print(status) gives a compact one-liner
        status = t.status()
        print(status)

        # Walk the log to see what's there — print(entry) gives "hash message" format
        history = t.log()
        print(f"\nHistory ({len(history)} commits):")
        for entry in reversed(history):
            print(f"  {entry}")

        # Continue — chat() includes all prior context automatically
        response = t.chat("Show me a quick example of each.")
        response.pprint()

        # pprint() for the final status gives the full rich panel
        print()
        t.status().pprint()


# --- Tier notes ---
# Commits and compile are primitives; they don't have review=True variants.
# For HITL patterns, see: hooks/ (t.on(), review=True)
# For agent automation, see: agentic/sidecar/ (triggers, orchestrator)

if __name__ == "__main__":
    main()
