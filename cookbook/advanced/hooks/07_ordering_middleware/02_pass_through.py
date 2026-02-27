"""The pass_through() pattern for non-resolving middleware handlers.

Three resolution actions exist: approve(), reject(), pass_through(). If every
handler calls pass_through(), the pending auto-approves (nobody objected).
"""

import os
import time

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _seed_conversation(t):
    """Build a multi-turn research conversation for middleware demos."""
    sys_ci = t.system("You are a research assistant helping analyze technology adoption trends.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("What are the main factors driving enterprise adoption of LLMs in 2025?")
    t.chat("How does the cost-benefit analysis compare between fine-tuning and RAG approaches?")
    t.chat("Can you summarize the key risks companies should consider before deploying?")
    t.chat("What metrics should we track to measure ROI on LLM investments?")


def pass_through_pipeline():
    """Three resolution actions: approve(), reject(), pass_through()."""
    print("\n" + "=" * 60)
    print("PART 2 — The pass_through() Pattern")
    print("=" * 60)
    print()
    print("  Three actions a handler can take:")
    print("    approve()      — accept the operation, stop the chain")
    print("    reject(reason) — deny the operation, stop the chain")
    print("    pass_through() — 'no opinion', let the next handler decide")
    print()
    print("  If ALL handlers call pass_through(), the pending auto-approves.")

    # Track the last compression time for the rate limiter
    last_compress_time = {"ts": 0.0}

    def rate_limiter(pending: PendingCompress):
        """Middleware: reject if compressed too recently, else pass through."""
        elapsed = time.time() - last_compress_time["ts"]
        if elapsed < 2.0 and last_compress_time["ts"] > 0:
            pending.reject(f"Rate limited: only {elapsed:.1f}s since last compress")
            return
        pending.pass_through()  # not rate limited, let next handler decide

    def quality_checker(pending: PendingCompress):
        """Middleware: inspect summary quality, pass through if OK."""
        for i, summary in enumerate(pending.summaries):
            if len(summary.strip()) < 10:
                pending.reject(f"Summary [{i}] too short ({len(summary)} chars)")
                return
        pending.pass_through()  # summaries look fine, let next handler decide

    def final_approver(pending: PendingCompress):
        """Catch-all: if we got this far, approve."""
        pending.approve()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.on("compress", rate_limiter, name="rate_limiter")
        t.on("compress", quality_checker, name="quality_checker")
        t.on("compress", final_approver, name="final_approver")
        _seed_conversation(t)

        print(f"\n  Pipeline: {t.hook_names}")

        # First compress — should pass through rate limiter and quality checker
        result = t.compress(target_tokens=150, token_tolerance=10000)
        last_compress_time["ts"] = time.time()

        if isinstance(result, CompressResult):
            print(f"\n  First compress: ratio={result.compression_ratio:.1%}")
        else:
            print(f"\n  First compress rejected: {result.rejection_reason}")
            result.pprint()

        # Show the pass_through chain in hook_log
        print("\n  Hook log (shows pass_through chain):")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    pass_through_pipeline()
