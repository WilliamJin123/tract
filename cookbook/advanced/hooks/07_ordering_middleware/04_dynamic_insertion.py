"""Inserting and removing named hooks dynamically at runtime.

Demonstrates adding handlers with before='name', before=True, and
removing them with t.off() -- all while inspecting hook_names.
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
    """Build a multi-turn research conversation for middleware demos."""
    sys_ci = t.system("You are a research assistant helping analyze technology adoption trends.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("What are the main factors driving enterprise adoption of LLMs in 2025?")
    t.chat("How does the cost-benefit analysis compare between fine-tuning and RAG approaches?")
    t.chat("Can you summarize the key risks companies should consider before deploying?")
    t.chat("What metrics should we track to measure ROI on LLM investments?")


def dynamic_insertion():
    """Add and remove named handlers at runtime."""
    print("\n" + "=" * 60)
    print("PART 4 — Inserting Hooks Dynamically")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        # --- Step 1: Start with a basic approver ---
        def basic_approver(pending: PendingCompress):
            pending.approve()

        t.on("compress", basic_approver, name="approver")
        print("\n  Step 1 — Start with a basic approver:")
        print(f"    hook_names = {t.hook_names}")

        # --- Step 2: Insert a logging middleware BEFORE the approver ---
        log_entries = []

        def logging_middleware(pending: PendingCompress):
            log_entries.append({
                "original": pending.original_tokens,
                "estimated": pending.estimated_tokens,
            })
            pending.pass_through()

        t.on("compress", logging_middleware, name="logger", before="approver")
        print("\n  Step 2 — Insert logger before='approver':")
        print(f"    hook_names = {t.hook_names}")

        # --- Step 3: Insert a rate limiter at the very front ---
        def rate_limiter(pending: PendingCompress):
            pending.pass_through()  # always pass in this demo

        t.on("compress", rate_limiter, name="rate_limiter", before=True)
        print("\n  Step 3 — Prepend rate_limiter with before=True:")
        print(f"    hook_names = {t.hook_names}")

        # --- Run compress to show all three fire ---
        result = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"\n  Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()

        print(f"  Logger captured: {log_entries}")

        # --- Step 4: Remove rate_limiter by name ---
        t.off("compress", "rate_limiter")
        print("\n  Step 4 — t.off('compress', 'rate_limiter'):")
        print(f"    hook_names = {t.hook_names}")

        t.print_hooks()


if __name__ == "__main__":
    dynamic_insertion()
