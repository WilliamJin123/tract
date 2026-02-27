"""Basic token gate that rejects summaries exceeding a hard budget.

Uses a factory function to create a hook that checks each summary's token
count against a configurable max_tokens + tolerance limit.
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _seed_conversation(t):
    """Build a multi-turn code review conversation for tolerance demos."""
    sys_ci = t.system("You are a senior Python code reviewer focusing on correctness and performance.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("Review this function:\ndef calculate_discount(price, tier):\n    if tier == 'gold': return price * 0.8\n    if tier == 'silver': return price * 0.9\n    return price")
    t.chat("What about edge cases — can price be negative? What if tier is None?")
    t.chat("Should I add type hints and input validation? Here's what I'm thinking.")
    t.chat("Here's the updated version with your suggestions. Any final thoughts?")


def basic_token_gate():
    print("=" * 60)
    print("PART 1 — Basic Token Gate")
    print("=" * 60)

    def make_token_gate(max_tokens: int, tolerance: int = 100):
        """Factory: returns a hook that rejects over-budget summaries."""
        def token_gate(pending: PendingCompress):
            limit = max_tokens + tolerance
            for i, summary in enumerate(pending.summaries):
                actual = pending.tract._token_counter.count_text(summary)
                if actual > limit:
                    pending.reject(
                        f"Summary [{i}] is {actual} tokens "
                        f"(budget: {max_tokens} + {tolerance} tolerance = {limit})"
                    )
                    return
            pending.approve()
        return token_gate

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.on("compress", make_token_gate(max_tokens=150, tolerance=100))
        _seed_conversation(t)

        # token_tolerance=10000 disables built-in enforcement,
        # letting our hook be the sole gatekeeper
        result = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
        else:
            # result is PendingCompress (rejected by hook)
            result.pprint()

        t.print_hooks()


if __name__ == "__main__":
    basic_token_gate()
