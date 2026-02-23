"""IMPORTANT Priority and Retention Criteria

Load a real contract, discuss it with an LLM over several turns, then
compress the conversation. Critical facts (dollar amounts, dates, penalty
clauses) are annotated IMPORTANT with deterministic retain_match= patterns
so they survive compression verbatim. Noise is marked SKIP. After
compression, we verify the LLM still knows the key terms.

Demonstrates: Priority.IMPORTANT with retain= and retain_match=,
              Priority.SKIP, compress() with retention enforcement,
              post-compression verification via chat()
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"

CONTRACT_PATH = Path(__file__).parent / "sample_contract.md"


def main():
    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # --- Load the contract into context ---

        contract_text = CONTRACT_PATH.read_text(encoding="utf-8")

        t.system("You are a contract review assistant. Be concise and precise.")

        contract_msg = t.user(
            f"Please review this contract:\n\n{contract_text}",
            message="Full contract loaded for review",
        )

        # --- Turn 1: LLM summarizes the contract ---

        print("=== Turn 1: Initial review ===\n")
        review = t.chat("What are the most important financial terms?")
        review.pprint()

        # --- Turn 2: Ask about risks ---

        print("\n=== Turn 2: Risk analysis ===\n")
        risks = t.chat(
            "What are the biggest risks for the client? "
            "Focus on penalties, termination costs, and liability caps."
        )
        risks.pprint()

        # --- Turn 3: Clarifying question (verbose, low-value) ---

        print("\n=== Turn 3: Clarification (filler) ===\n")
        filler = t.chat("Can you explain what SOC 2 Type II means in plain English?")
        filler.pprint()

        # --- Turn 4: Debug noise ---

        noise = t.user("[debug] trace_id=x9f2-k3m1 | latency=342ms | cache=miss")

        # --- Annotate based on content ---
        # Critical financial data: retain specific values that must survive

        print("\n=== Annotating commits ===\n")

        # Use regex mode — LLMs rephrase freely, so exact substrings are
        # brittle. Regex lets us match "Net 45" / "Net-45" / "net 45", etc.

        t.annotate(
            contract_msg.commit_hash,
            Priority.IMPORTANT,
            retain="Preserve all dollar amounts, payment terms, dates, and penalty clauses",
            retain_match=[
                r"\$2[,.]?847[,.]?000",  # total contract value
                r"[Nn]et[\s\-]*45",       # payment terms
                r"1\.5\s*%",              # late payment penalty
                r"3\.5\s*%",              # annual escalation
                r"[Jj]une\s*15.*2026",    # go-live deadline
                r"99\.95\s*%",            # uptime SLA
            ],
            retain_match_mode="regex",
        )
        print(f"  {contract_msg.commit_hash[:8]}: IMPORTANT (6 regex retention patterns)")

        t.annotate(
            noise.commit_hash,
            Priority.SKIP,
            reason="debug trace, no value",
        )
        print(f"  {noise.commit_hash[:8]}: SKIP (debug noise)")

        # --- Show context size before compression ---

        ctx_before = t.compile()
        print(f"\n=== Before compression: {ctx_before.token_count} tokens, "
              f"{len(ctx_before.messages)} messages ===\n")

        # --- Compress the conversation ---
        # The contract and LLM responses are compressed, but retain_match
        # patterns must survive. PINNED system prompt is untouched.

        print("=== Compressing... ===\n")
        result = t.compress(max_retries=5)

        print(f"  Original:    {result.original_tokens} tokens")
        print(f"  Compressed:  {result.compressed_tokens} tokens")
        print(f"  Ratio:       {result.compression_ratio:.1%}")
        print(f"  Preserved:   {len(result.preserved_commits)} commits (PINNED)")
        print(f"  Summaries:   {len(result.summary_commits)} commits")

        # --- Show context after compression ---

        ctx_after = t.compile()
        print(f"\n=== After compression: {ctx_after.token_count} tokens, "
              f"{len(ctx_after.messages)} messages ===\n")
        ctx_after.pprint()

        # --- Verify: ask about key terms post-compression ---

        print("\n=== Verification: does the LLM still know the key facts? ===\n")
        verify = t.chat(
            "Quick: what's the total contract value, payment terms, "
            "late penalty rate, and go-live date?"
        )
        verify.pprint()


if __name__ == "__main__":
    main()
