"""Handler ordering basics using name, before, and after parameters.

Shows how t.on() supports named handlers with explicit positioning via
before=True (prepend), after='name' (insert after), and default append.
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
    """Build a multi-turn research conversation for middleware demos."""
    sys_ci = t.system("You are a research assistant helping analyze technology adoption trends.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("What are the main factors driving enterprise adoption of LLMs in 2025?", max_tokens=500)
    t.chat("How does the cost-benefit analysis compare between fine-tuning and RAG approaches?", max_tokens=500)
    t.chat("Can you summarize the key risks companies should consider before deploying?", max_tokens=500)
    t.chat("What metrics should we track to measure ROI on LLM investments?", max_tokens=500)


def ordering_basics() -> None:
    """Named handlers, before=True, after='name' control execution order."""
    print("=" * 60)
    print("PART 1 — Handler Ordering Basics")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        # Track call order with a shared list
        call_order: list[str] = []

        def handler_a(pending: PendingCompress) -> None:
            """First registered handler."""
            call_order.append("validator")
            pending.pass_through()

        def handler_b(pending: PendingCompress) -> None:
            """Second registered handler — final approver."""
            call_order.append("formatter")
            pending.approve()

        # --- Registration order (default): append ---
        t.on("compress", handler_a, name="validator")
        t.on("compress", handler_b, name="formatter")

        print("\n  After registering validator, then formatter:")
        print(f"    hook_names = {t.hook_names}")

        # --- Prepend: before=True ---
        def rate_limiter(pending: PendingCompress) -> None:
            call_order.append("rate_limit")
            pending.pass_through()

        t.on("compress", rate_limiter, name="rate_limit", before=True)

        print("\n  After prepending rate_limit with before=True:")
        print(f"    hook_names = {t.hook_names}")

        # --- Insert relative to named handler: after='validator' ---
        def auditor(pending: PendingCompress) -> None:
            call_order.append("auditor")
            pending.pass_through()

        t.on("compress", auditor, name="auditor", after="validator")

        print("\n  After inserting auditor after='validator':")
        print(f"    hook_names = {t.hook_names}")

        # Show full hook table
        print()
        t.print_hooks()

        # --- Run compress to prove the firing order ---
        print("\n  Running compress()...")
        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()

        print(f"\n  Handler call order: {call_order}")
        print("  (rate_limit -> validator -> auditor -> formatter)")

        # Hook log confirms the pass_through chain
        print("\n  Hook log:")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    ordering_basics()
