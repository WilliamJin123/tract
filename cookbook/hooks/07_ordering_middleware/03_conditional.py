"""Conditional middleware that only intervenes for large compressions.

Uses pass_through() conditionally so the handler silently passes small
contexts but enforces strict budget rules on large ones.
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


def conditional_middleware() -> None:
    """Handlers that only intervene when they have an opinion."""
    print("\n" + "=" * 60)
    print("PART 3 — Conditional Middleware")
    print("=" * 60)
    print()
    print("  A handler can use pass_through() conditionally:")
    print("  small contexts pass through silently, large ones get scrutinized.")

    def size_gate(pending: PendingCompress) -> None:
        """Only intervene for large compressions."""
        if pending.original_tokens < 500:
            pending.pass_through()  # small context, don't care
            return
        # Large context: enforce strict budget
        if pending.estimated_tokens > pending.original_tokens * 0.5:
            pending.reject("Large context must achieve 50%+ reduction")
        else:
            pending.pass_through()  # meets the bar, let next handler decide

    def always_approve(pending: PendingCompress) -> None:
        """Safety net: approve if no earlier handler objected."""
        pending.approve()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.on("compress", size_gate, name="size_gate")
        t.on("compress", always_approve, name="approver")
        _seed_conversation(t)

        print(f"\n  Pipeline: {t.hook_names}")

        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"\n  Compressed: ratio={result.compression_ratio:.1%}")
            print(f"  Original tokens:  {result.original_tokens}")
            print(f"  Compressed tokens: {result.compressed_tokens}")
        else:
            print(f"\n  Rejected: {result.rejection_reason}")
            result.pprint()

        print("\n  Hook log:")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    conditional_middleware()
