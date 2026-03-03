"""Token budget enforcement patterns for compression hooks.

Four patterns for controlling summary token budgets:

  Part 1 -- Basic Token Gate:       Reject summaries exceeding a hard budget
  Part 2 -- Auto-Truncate:          Binary-search truncation to fit within budget
  Part 3 -- Middleware + Enforcer:   pass_through() logger paired with budget enforcer
  Part 4 -- Dynamic Budget:          Tolerance scales with context size (10% clamped)

Demonstrates: make_token_gate(), make_truncator(), pass_through(),
              compression_logger, dynamic_tolerance, hook_log, print_hooks()
"""

import sys
from pathlib import Path

from collections.abc import Callable

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult
from tract.protocols import CompiledContext

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


def _seed_code_review(t: Tract) -> None:
    """Build a multi-turn code review conversation for tolerance demos."""
    sys_ci = t.system("You are a senior Python code reviewer focusing on correctness and performance.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("Review this function:\ndef calculate_discount(price, tier):\n    if tier == 'gold': return price * 0.8\n    if tier == 'silver': return price * 0.9\n    return price", max_tokens=500)
    t.chat("What about edge cases — can price be negative? What if tier is None?", max_tokens=500)
    t.chat("Should I add type hints and input validation? Here's what I'm thinking.", max_tokens=500)
    t.chat("Here's the updated version with your suggestions. Any final thoughts?", max_tokens=500)


# =====================================================================
# Part 1 -- Basic Token Gate
# =====================================================================

def basic_token_gate() -> None:
    """Reject summaries exceeding a hard budget using a factory hook."""
    print("=" * 60)
    print("PART 1 -- Basic Token Gate")
    print("=" * 60)

    # Without this gate, compress() uses its built-in token_tolerance
    # check (tier 3). This hook replaces that with a custom budget (tier 2).

    def make_token_gate(max_tokens: int, tolerance: int = 100) -> Callable[[PendingCompress], None]:
        """Factory: returns a hook that rejects over-budget summaries."""
        def token_gate(pending: PendingCompress) -> None:
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
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.on("compress", make_token_gate(max_tokens=150, tolerance=100))
        _seed_code_review(t)

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # token_tolerance=10000 disables built-in enforcement,
        # letting our hook be the sole gatekeeper
        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
        else:
            # result is PendingCompress (rejected by hook)
            result.pprint()

        t.print_hooks()


# =====================================================================
# Part 2 -- Auto-Truncate
# =====================================================================

def auto_truncate() -> None:
    """Binary search truncation to fit summaries within a token budget."""
    print("\n" + "=" * 60)
    print("PART 2 -- Auto-Truncate")
    print("=" * 60)

    # Instead of rejecting, this hook uses binary search to find the
    # longest truncation that fits, then edits the summary in place.

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
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.on("compress", make_truncator(max_tokens=200))
        _seed_code_review(t)

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        result: CompressResult = t.compress(target_tokens=200, token_tolerance=10000)
        print(f"  Compressed: ratio={result.compression_ratio:.1%}")

        print("\n  AFTER compression:\n")
        ctx: CompiledContext = t.compile()
        ctx.pprint(style="compact")


# =====================================================================
# Part 3 -- Middleware + Enforcer (pass_through)
# =====================================================================

def middleware_and_enforcer() -> None:
    """Logger middleware (pass_through) paired with a budget enforcer."""
    print("\n" + "=" * 60)
    print("PART 3 -- Middleware + Enforcer (pass_through)")
    print("=" * 60)
    print()
    print("  The logger calls pass_through() -- it inspects without deciding.")
    print("  The enforcer fires next and makes the approve/reject decision.")

    audit_log: list[dict[str, object]] = []

    def compression_logger(pending: PendingCompress) -> None:
        """Middleware: inspect and log, then pass through to next handler."""
        entry: dict[str, object] = {
            "summaries": len(pending.summaries),
            "original_tokens": pending.original_tokens,
            "estimated_tokens": pending.estimated_tokens,
        }
        if pending.original_tokens > 0:
            entry["ratio"] = f"{(1 - pending.estimated_tokens / pending.original_tokens):.0%}"
        audit_log.append(entry)
        pending.pass_through()  # explicit: "I'm not the decision-maker"

    def budget_enforcer(pending: PendingCompress) -> None:
        """Approve if estimated tokens are within budget, reject otherwise."""
        budget = 300
        if pending.estimated_tokens <= budget:
            pending.approve()
        else:
            pending.reject(f"Over budget: {pending.estimated_tokens} > {budget}")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Stack: logger (middleware) fires first, enforcer fires second
        t.on("compress", compression_logger, name="logger")
        t.on("compress", budget_enforcer, name="enforcer")
        _seed_code_review(t)

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
            print("\n  AFTER compression:\n")
            t.compile().pprint(style="compact")
        else:
            result.pprint()

        print(f"\n  Audit log: {audit_log}")

        # hook_log shows pass_through + approve/reject sequence
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")

        t.print_hooks()


# =====================================================================
# Part 4 -- Dynamic Budget
# =====================================================================

def dynamic_budget() -> None:
    """Tolerance scales with context size: 10% of original, clamped to [50, 500]."""
    print("\n" + "=" * 60)
    print("PART 4 -- Dynamic Budget")
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
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.on("compress", dynamic_tolerance)
        _seed_code_review(t)

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
            print("\n  AFTER compression:\n")
            t.compile().pprint(style="compact")
        else:
            result.pprint()


# =====================================================================
# Main
# =====================================================================

def main():
    basic_token_gate()
    auto_truncate()
    middleware_and_enforcer()
    dynamic_budget()


if __name__ == "__main__":
    main()
