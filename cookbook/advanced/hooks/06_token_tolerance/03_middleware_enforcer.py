"""Middleware logger paired with a budget enforcer hook.

Demonstrates the pass_through() pattern: a logging middleware inspects the
pending compression without deciding, then an enforcer makes the approve/reject
decision based on estimated token count.
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


def middleware_and_enforcer():
    print("\n" + "=" * 60)
    print("PART 3 — Middleware + Enforcer (pass_through)")
    print("=" * 60)
    print()
    print("  The logger calls pass_through() — it inspects without deciding.")
    print("  The enforcer fires next and makes the approve/reject decision.")

    audit_log = []

    def compression_logger(pending: PendingCompress):
        """Middleware: inspect and log, then pass through to next handler."""
        entry = {
            "summaries": len(pending.summaries),
            "original_tokens": pending.original_tokens,
            "estimated_tokens": pending.estimated_tokens,
        }
        if pending.original_tokens > 0:
            entry["ratio"] = f"{(1 - pending.estimated_tokens / pending.original_tokens):.0%}"
        audit_log.append(entry)
        pending.pass_through()  # explicit: "I'm not the decision-maker"

    def budget_enforcer(pending: PendingCompress):
        """Approve if estimated tokens are within budget, reject otherwise."""
        budget = 300
        if pending.estimated_tokens <= budget:
            pending.approve()
        else:
            pending.reject(f"Over budget: {pending.estimated_tokens} > {budget}")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Stack: logger (middleware) fires first, enforcer fires second
        t.on("compress", compression_logger, name="logger")
        t.on("compress", budget_enforcer, name="enforcer")
        _seed_conversation(t)

        result = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()

        print(f"\n  Audit log: {audit_log}")

        # hook_log shows pass_through + approve/reject sequence
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")

        t.print_hooks()


if __name__ == "__main__":
    middleware_and_enforcer()
