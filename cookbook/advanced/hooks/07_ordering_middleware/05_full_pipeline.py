"""Composing a full production pipeline with explicit handler ordering.

A realistic multi-handler pipeline: rate_limiter -> budget_checker ->
quality_validator -> final_approver, with compress() and hook_log tracing.
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


def full_pipeline():
    """A realistic multi-handler production pipeline."""
    print("\n" + "=" * 60)
    print("PART 5 — Composing a Full Pipeline")
    print("=" * 60)
    print()
    print("  Pipeline:  rate_limiter -> budget_checker -> quality_validator -> approver")
    print("  Each middleware uses pass_through() to delegate downstream.")

    last_compress = {"ts": 0.0}
    pipeline_trace = []

    # --- Handler 1: Rate limiter ---
    def rate_limiter(pending: PendingCompress):
        """Reject if compressed within last second, else pass through."""
        elapsed = time.time() - last_compress["ts"]
        pipeline_trace.append(f"rate_limiter (elapsed={elapsed:.1f}s)")
        if elapsed < 1.0 and last_compress["ts"] > 0:
            pending.reject(f"Rate limited ({elapsed:.1f}s since last)")
            return
        pending.pass_through()

    # --- Handler 2: Token budget checker ---
    def budget_checker(pending: PendingCompress):
        """Reject if estimated tokens exceed 80% of original."""
        pipeline_trace.append(f"budget_checker (est={pending.estimated_tokens})")
        if pending.original_tokens > 0:
            ratio = pending.estimated_tokens / pending.original_tokens
            if ratio > 0.80:
                pending.reject(
                    f"Insufficient reduction: {ratio:.0%} of original "
                    f"(need <80%)"
                )
                return
        pending.pass_through()

    # --- Handler 3: Quality validator ---
    def quality_validator(pending: PendingCompress):
        """Reject if any summary is suspiciously short."""
        pipeline_trace.append(f"quality_validator ({len(pending.summaries)} summaries)")
        for i, summary in enumerate(pending.summaries):
            word_count = len(summary.split())
            if word_count < 3:
                pending.reject(
                    f"Summary [{i}] has only {word_count} words — likely garbage"
                )
                return
        pending.pass_through()

    # --- Handler 4: Final approver (safety net) ---
    def final_approver(pending: PendingCompress):
        """Catch-all: approve if every middleware passed through."""
        pipeline_trace.append("final_approver")
        pending.approve()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        # Register with explicit ordering
        t.on("compress", rate_limiter, name="rate_limiter")
        t.on("compress", budget_checker, name="budget_checker", after="rate_limiter")
        t.on("compress", quality_validator, name="quality_validator", after="budget_checker")
        t.on("compress", final_approver, name="final_approver", after="quality_validator")

        print(f"\n  Registered pipeline: {t.hook_names}")

        # --- First compress ---
        result = t.compress(target_tokens=150, token_tolerance=10000)
        last_compress["ts"] = time.time()

        if isinstance(result, CompressResult):
            print(f"\n  Result: APPROVED (ratio={result.compression_ratio:.1%})")
        else:
            print(f"\n  Result: REJECTED ({result.rejection_reason})")
            result.pprint()

        print(f"\n  Pipeline trace: {pipeline_trace}")

        # --- Hook log: trace the full pipeline execution ---
        print("\n  Hook log:")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")

        t.print_hooks()


if __name__ == "__main__":
    full_pipeline()
