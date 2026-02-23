"""Edit in Place

A support agent's system prompt says "60-day return policy" — but it's
actually 30 days. Chat with the LLM, see it parrot the wrong info, then
edit the system prompt in place and ask again. The LLM now sees the
corrected context and gives the right answer. Both versions stay in
history for audit.

Demonstrates: system(edit=hash), annotate(SKIP) for stale responses,
              chat() before/after edit, compile() serves corrected content,
              log() preserves both versions
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # --- System prompt with a mistake baked in ---

        bad_prompt = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 60 days.",
            message="Initial system prompt with incorrect return policy",
        )
        print(f"System prompt committed: {bad_prompt.commit_hash[:8]}\n")

        # --- Ask about returns — the LLM will cite the wrong policy ---

        print("=== Before edit ===\n")
        stale_q = t.user("What's your return policy?")
        stale_a = t.generate()
        stale_a.pprint()

        # --- Fix the system prompt: 60 days -> 30 days ---
        # edit= replaces the target commit in compiled context.
        # Same shorthand, one extra param.

        fix = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 30 days.",
            edit=bad_prompt.commit_hash,
            message="Fix return policy from 60 days to 30 days",
        )
        fix.pprint()

        # --- Skip the stale Q&A — it was based on the wrong prompt ---

        t.annotate(stale_q.commit_hash, Priority.SKIP, reason="based on wrong prompt")
        t.annotate(stale_a.commit_info.commit_hash, Priority.SKIP, reason="based on wrong prompt")
        print("Skipped stale Q&A (based on the old 60-day prompt)\n")

        # --- Ask again — the LLM now sees the corrected prompt ---

        print("=== After edit ===\n")
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
