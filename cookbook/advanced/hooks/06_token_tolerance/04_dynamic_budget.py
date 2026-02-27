"""Dynamic budget hook that scales tolerance with context size.

Tolerance is computed as 10% of original tokens (clamped to [50, 500]),
giving tighter enforcement for small contexts and looser for large ones.
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


def _seed_conversation(t: Tract) -> None:
    """Build a multi-turn code review conversation for tolerance demos."""
    sys_ci = t.system("You are a senior Python code reviewer focusing on correctness and performance.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("Review this function:\ndef calculate_discount(price, tier):\n    if tier == 'gold': return price * 0.8\n    if tier == 'silver': return price * 0.9\n    return price")
    t.chat("What about edge cases — can price be negative? What if tier is None?")
    t.chat("Should I add type hints and input validation? Here's what I'm thinking.")
    t.chat("Here's the updated version with your suggestions. Any final thoughts?")


def dynamic_budget() -> None:
    print("\n" + "=" * 60)
    print("PART 4 — Dynamic Budget")
    print("=" * 60)

    def dynamic_tolerance(pending: PendingCompress) -> None:
        """Tolerance = 10% of original tokens, clamped to [50, 500]."""
        original = pending.original_tokens
        tolerance = max(50, min(500, int(original * 0.10)))
        target = original // 2  # aim for 50% reduction
        limit = target + tolerance

        if pending.estimated_tokens <= limit:
            pending.approve()
        else:
            pending.reject(
                f"Estimated {pending.estimated_tokens} tokens exceeds "
                f"dynamic budget of {limit} ({target} target + {tolerance} tolerance)"
            )

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.on("compress", dynamic_tolerance)
        _seed_conversation(t)

        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()


if __name__ == "__main__":
    dynamic_budget()
