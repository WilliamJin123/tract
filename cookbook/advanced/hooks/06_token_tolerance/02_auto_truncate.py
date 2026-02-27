"""Auto-truncate hook that chops summaries to fit within a token budget.

Instead of rejecting over-budget summaries, this approach uses binary search
to find the longest truncation that fits, then edits the summary in place.
"""

import os

from dotenv import load_dotenv

from collections.abc import Callable

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult
from tract.protocols import CompiledContext

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _seed_conversation(t: Tract) -> None:
    """Build a multi-turn code review conversation for tolerance demos."""
    sys_ci = t.system("You are a senior Python code reviewer focusing on correctness and performance.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("Review this function:\ndef calculate_discount(price, tier):\n    if tier == 'gold': return price * 0.8\n    if tier == 'silver': return price * 0.9\n    return price", max_tokens=500)
    t.chat("What about edge cases — can price be negative? What if tier is None?", max_tokens=500)
    t.chat("Should I add type hints and input validation? Here's what I'm thinking.", max_tokens=500)
    t.chat("Here's the updated version with your suggestions. Any final thoughts?", max_tokens=500)


def auto_truncate() -> None:
    print("\n" + "=" * 60)
    print("PART 2 — Auto-Truncate")
    print("=" * 60)

    def make_truncator(max_tokens: int) -> Callable[[PendingCompress], None]:
        """Factory: truncate summaries to fit within max_tokens."""
        def truncate_to_budget(pending: PendingCompress) -> None:
            for i, summary in enumerate(pending.summaries):
                actual = pending.tract._token_counter.count_text(summary)
                if actual <= max_tokens:
                    continue

                # Binary search for the right truncation point
                words = summary.split()
                lo, hi = 0, len(words)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    candidate = " ".join(words[:mid]) + "..."
                    if pending.tract._token_counter.count_text(candidate) <= max_tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                pending.edit_summary(i, " ".join(words[:lo]) + "...")
            pending.approve()
        return truncate_to_budget

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.on("compress", make_truncator(max_tokens=200))
        _seed_conversation(t)

        result: CompressResult = t.compress(target_tokens=200, token_tolerance=10000)
        print(f"  Compressed: ratio={result.compression_ratio:.1%}")

        ctx: CompiledContext = t.compile()
        ctx.pprint(style="compact")


if __name__ == "__main__":
    auto_truncate()
