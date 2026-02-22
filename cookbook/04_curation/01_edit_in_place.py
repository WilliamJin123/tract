"""Edit in Place

A support agent's system prompt says "60-day return policy" — but it's
actually 30 days. Chat with the LLM, see it parrot the wrong info, then
edit the system prompt in place and ask again. The LLM now sees the
corrected context and gives the right answer. Both versions stay in
history for audit.

Demonstrates: system(edit=hash), chat() before/after edit,
              compile() serves corrected content, log() preserves both versions
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # --- System prompt with a mistake baked in ---

        bad_prompt = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 60 days."
        )
        print(f"System prompt committed: {bad_prompt.commit_hash[:8]}\n")

        # --- Ask about returns — the LLM will cite the wrong policy ---

        print("=== Before edit ===\n")
        response = t.chat("What's your return policy?")
        response.pprint()

        # --- Fix the system prompt: 60 days -> 30 days ---
        # edit= replaces the target commit in compiled context.
        # Same shorthand, one extra param.

        fix = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 30 days.",
            edit=bad_prompt.commit_hash,
        )
        print(f"\nEdited system prompt: {fix.commit_hash[:8]}")

        # --- Ask again — the LLM now sees the corrected prompt ---

        print("\n=== After edit ===\n")
        response = t.chat("What's your return policy?")
        response.pprint()

        # --- Compiled context: only the corrected version appears ---

        print("\n=== Compiled context (what the LLM sees now) ===\n")
        t.compile().pprint()

        # --- Full history: both system prompts preserved for audit ---

        print("=== Full history (both versions preserved) ===\n")
        for entry in reversed(t.log()):
            print(f"  {entry}")


if __name__ == "__main__":
    main()
