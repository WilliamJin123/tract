"""Compression Validation (compress(validator=), retain_match=)

compress() accepts validator= and max_retries= just like chat(). The validator
checks summary quality after LLM generation. Combine with retain_match= for
a two-layer safety net: deterministic regex patterns that MUST appear in the
summary, plus a semantic validator for anything regex can't express.

Demonstrates: compress(validator=, max_retries=), compress + retain_match=,
              instructions= + validator=, post-compression verification
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: compress(validator=) — validate summary quality
# =============================================================================
# The validator ensures the LLM summary mentions specific technical terms.
# If validation fails, the LLM retries with a steering message injected
# into the summarization prompt.

def part1_compress_validator():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a database architecture assistant.")

        t.chat("Explain the difference between B-trees and LSM-trees for indexing.")
        t.chat("When should I use PostgreSQL vs ClickHouse for analytics?")
        t.chat("How does write-ahead logging work in crash recovery?")

        print("Before compression:")
        t.compile().pprint(style="compact")

        required_terms = ["B-tree", "LSM", "write-ahead"]

        def validate_summary(text: str) -> tuple[bool, str | None]:
            text_lower = text.lower()
            missing = [term for term in required_terms if term.lower() not in text_lower]
            if missing:
                return (False, f"Summary must mention: {missing}")
            if len(text.split()) < 20:
                return (False, "Summary too short — must be at least 20 words")
            return (True, None)

        result = t.compress(
            target_tokens=200,
            validator=validate_summary,
            max_retries=3,
        )

        print(f"\nCompressed: {result.original_tokens} -> {result.compressed_tokens} "
              f"tokens ({result.compression_ratio:.0%})\n")
        t.compile().pprint(style="chat")


# =============================================================================
# Part 2: retain_match= + validator= — two-layer safety net
# =============================================================================
# Layer 1 (retain_match): regex patterns — hard deterministic requirement.
# Layer 2 (validator): semantic check — soft quality control.
# Both must pass for the summary to be accepted.

def part2_combined_validation():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a financial analysis assistant.")

        report_ci = t.user(
            "Q3 Revenue Report:\n"
            "- Total revenue: $4,230,000\n"
            "- Operating margin: 23.5%\n"
            "- Customer churn: 2.1%\n"
            "- ARR growth: 47% YoY\n"
            "- Next board meeting: March 15, 2026"
        )
        t.chat("What are the key takeaways from this report?")
        t.chat("How does the 2.1% churn compare to industry benchmarks?")
        t.chat("What should we present to the board about ARR growth?")

        # Layer 1: deterministic regex retention
        t.annotate(
            report_ci.commit_hash,
            Priority.IMPORTANT,
            retain="Preserve all dollar amounts, percentages, and dates",
            retain_match=[
                r"\$4[,.]?230[,.]?000",   # revenue
                r"23\.5\s*%",              # margin
                r"2\.1\s*%",              # churn
                r"47\s*%",                # ARR growth
            ],
            retain_match_mode="regex",
        )

        print(f"Before: {t.compile().token_count} tokens")

        # Layer 2: semantic validator
        def validate_actionable(text: str) -> tuple[bool, str | None]:
            if "board" not in text.lower() and "present" not in text.lower():
                return (False, "Summary should mention the board presentation context")
            return (True, None)

        result = t.compress(
            target_tokens=200,
            validator=validate_actionable,
            max_retries=4,
        )

        print(f"After:  {result.compressed_tokens} tokens "
              f"({result.compression_ratio:.0%} compression)\n")
        t.compile().pprint(style="chat")

        # Verify: the LLM can still recall the retained figures
        print("\nVerification — can the LLM recall key figures?\n")
        t.chat(
            "Quick: what was Q3 revenue, operating margin, and churn rate?"
        ).pprint()


# =============================================================================
# Part 3: instructions= + validator= — guided and validated
# =============================================================================
# instructions= steers the summary focus. validator= confirms the summary
# actually followed those instructions.

def part3_guided_and_validated():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a security audit assistant.")

        t.chat("Review our authentication flow: we use JWT with RS256, "
               "refresh tokens with 7-day expiry, and rate limit to 100 req/min.")
        t.chat("What about our data encryption? We use AES-256-GCM at rest "
               "and TLS 1.3 in transit.")
        t.chat("For access control, we have RBAC with 4 roles: admin, editor, "
               "viewer, and auditor. Admin can modify roles.")

        print(f"Before: {t.compile().token_count} tokens")

        def validate_security_summary(text: str) -> tuple[bool, str | None]:
            text_lower = text.lower()
            categories = {
                "authentication": ["jwt", "token", "auth"],
                "encryption": ["aes", "tls", "encrypt"],
                "access control": ["rbac", "role", "access"],
            }
            missing = [cat for cat, kws in categories.items()
                       if not any(kw in text_lower for kw in kws)]
            if missing:
                return (False, f"Summary missing security categories: {missing}")
            return (True, None)

        result = t.compress(
            target_tokens=150,
            instructions="Organize by security domain: authentication, "
                        "encryption, access control. Keep specific protocol "
                        "names and configuration values.",
            validator=validate_security_summary,
            max_retries=3,
        )

        print(f"After:  {result.compressed_tokens} tokens "
              f"({result.compression_ratio:.0%} compression)\n")
        t.compile().pprint(style="chat")


def main():
    print("=== Part 1: compress(validator=) ===\n")
    part1_compress_validator()

    print(f"\n=== Part 2: retain_match= + validator= ===\n")
    part2_combined_validation()

    print(f"\n=== Part 3: instructions= + validator= ===\n")
    part3_guided_and_validated()


if __name__ == "__main__":
    main()
