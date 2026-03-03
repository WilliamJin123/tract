"""Message ordering and handler pipeline patterns.

Five patterns for composing and ordering middleware handlers:

  Part 1 -- Ordering Basics:      Named handlers, before=True, after='name'
  Part 2 -- pass_through() Pattern: Three resolution actions, auto-approve fallback
  Part 3 -- Conditional Middleware: Intervene only for large compressions
  Part 4 -- Dynamic Insertion:      Add/remove named hooks at runtime
  Part 5 -- Full Pipeline:          Multi-handler production pipeline with tracing

Demonstrates: t.on(name=, before=, after=), pass_through(), approve(),
              reject(), t.off(), hook_names, hook_log, print_hooks()
"""

import sys
import time
from pathlib import Path

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


def _seed_research(t: Tract) -> None:
    """Build a multi-turn research conversation for middleware demos."""
    sys_ci = t.system("You are a research assistant helping analyze technology adoption trends.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("What are the main factors driving enterprise adoption of LLMs in 2025?", max_tokens=500)
    t.chat("How does the cost-benefit analysis compare between fine-tuning and RAG approaches?", max_tokens=500)
    t.chat("Can you summarize the key risks companies should consider before deploying?", max_tokens=500)
    t.chat("What metrics should we track to measure ROI on LLM investments?", max_tokens=500)


# =====================================================================
# Part 1 -- Handler Ordering Basics
# =====================================================================

def ordering_basics() -> None:
    """Named handlers, before=True, after='name' control execution order."""
    print("=" * 60)
    print("PART 1 -- Handler Ordering Basics")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        _seed_research(t)

        # Track call order with a shared list
        call_order: list[str] = []

        def handler_a(pending: PendingCompress) -> None:
            """First registered handler."""
            call_order.append("validator")
            pending.pass_through()

        def handler_b(pending: PendingCompress) -> None:
            """Second registered handler -- final approver."""
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
        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        print("\n  Running compress()...")
        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"  Compressed: ratio={result.compression_ratio:.1%}")
            print("\n  AFTER compression:\n")
            t.compile().pprint(style="compact")
        else:
            result.pprint()

        print(f"\n  Handler call order: {call_order}")
        print("  (rate_limit -> validator -> auditor -> formatter)")

        # Hook log confirms the pass_through chain
        print("\n  Hook log:")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


# =====================================================================
# Part 2 -- The pass_through() Pattern
# =====================================================================

def pass_through_pipeline() -> None:
    """Three resolution actions: approve(), reject(), pass_through()."""
    print("\n" + "=" * 60)
    print("PART 2 -- The pass_through() Pattern")
    print("=" * 60)
    print()
    print("  Three actions a handler can take:")
    print("    approve()      -- accept the operation, stop the chain")
    print("    reject(reason) -- deny the operation, stop the chain")
    print("    pass_through() -- 'no opinion', let the next handler decide")
    print()
    print("  If ALL handlers call pass_through(), the pending auto-approves.")

    # Track the last compression time for the rate limiter
    last_compress_time: dict[str, float] = {"ts": 0.0}

    def rate_limiter(pending: PendingCompress) -> None:
        """Middleware: reject if compressed too recently, else pass through."""
        elapsed = time.time() - last_compress_time["ts"]
        if elapsed < 2.0 and last_compress_time["ts"] > 0:
            pending.reject(f"Rate limited: only {elapsed:.1f}s since last compress")
            return
        pending.pass_through()  # not rate limited, let next handler decide

    def quality_checker(pending: PendingCompress) -> None:
        """Middleware: inspect summary quality, pass through if OK."""
        for i, summary in enumerate(pending.summaries):
            if len(summary.strip()) < 10:
                pending.reject(f"Summary [{i}] too short ({len(summary)} chars)")
                return
        pending.pass_through()  # summaries look fine, let next handler decide

    def final_approver(pending: PendingCompress) -> None:
        """Catch-all: if we got this far, approve."""
        pending.approve()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.on("compress", rate_limiter, name="rate_limiter")
        t.on("compress", quality_checker, name="quality_checker")
        t.on("compress", final_approver, name="final_approver")
        _seed_research(t)

        print(f"\n  Pipeline: {t.hook_names}")

        # First compress -- should pass through rate limiter and quality checker
        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)
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


# =====================================================================
# Part 3 -- Conditional Middleware
# =====================================================================

def conditional_middleware() -> None:
    """Handlers that only intervene when they have an opinion."""
    print("\n" + "=" * 60)
    print("PART 3 -- Conditional Middleware")
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
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.on("compress", size_gate, name="size_gate")
        t.on("compress", always_approve, name="approver")
        _seed_research(t)

        print(f"\n  Pipeline: {t.hook_names}")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"\n  Compressed: ratio={result.compression_ratio:.1%}")
            print(f"  Original tokens:  {result.original_tokens}")
            print(f"  Compressed tokens: {result.compressed_tokens}")
            print("\n  AFTER compression:\n")
            t.compile().pprint(style="compact")
        else:
            print(f"\n  Rejected: {result.rejection_reason}")
            result.pprint()

        print("\n  Hook log:")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


# =====================================================================
# Part 4 -- Dynamic Insertion
# =====================================================================

def dynamic_insertion() -> None:
    """Add and remove named handlers at runtime."""
    print("\n" + "=" * 60)
    print("PART 4 -- Inserting Hooks Dynamically")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        _seed_research(t)

        # --- Step 1: Start with a basic approver ---
        def basic_approver(pending: PendingCompress) -> None:
            pending.approve()

        t.on("compress", basic_approver, name="approver")
        print("\n  Step 1 -- Start with a basic approver:")
        print(f"    hook_names = {t.hook_names}")

        # --- Step 2: Insert a logging middleware BEFORE the approver ---
        log_entries: list[dict[str, int]] = []

        def logging_middleware(pending: PendingCompress) -> None:
            log_entries.append({
                "original": pending.original_tokens,
                "estimated": pending.estimated_tokens,
            })
            pending.pass_through()

        t.on("compress", logging_middleware, name="logger", before="approver")
        print("\n  Step 2 -- Insert logger before='approver':")
        print(f"    hook_names = {t.hook_names}")

        # --- Step 3: Insert a rate limiter at the very front ---
        def rate_limiter(pending: PendingCompress) -> None:
            pending.pass_through()  # always pass in this demo

        t.on("compress", rate_limiter, name="rate_limiter", before=True)
        print("\n  Step 3 -- Prepend rate_limiter with before=True:")
        print(f"    hook_names = {t.hook_names}")

        # --- Run compress to show all three fire ---
        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)

        if isinstance(result, CompressResult):
            print(f"\n  Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()

        print(f"  Logger captured: {log_entries}")

        # --- Step 4: Remove rate_limiter by name ---
        t.off("compress", "rate_limiter")
        print("\n  Step 4 -- t.off('compress', 'rate_limiter'):")
        print(f"    hook_names = {t.hook_names}")

        t.print_hooks()


# =====================================================================
# Part 5 -- Full Pipeline
# =====================================================================

def full_pipeline() -> None:
    """A realistic multi-handler production pipeline."""
    print("\n" + "=" * 60)
    print("PART 5 -- Composing a Full Pipeline")
    print("=" * 60)
    print()
    print("  Pipeline:  rate_limiter -> budget_checker -> quality_validator -> approver")
    print("  Each middleware uses pass_through() to delegate downstream.")

    last_compress: dict[str, float] = {"ts": 0.0}
    pipeline_trace: list[str] = []

    # --- Handler 1: Rate limiter ---
    def rate_limiter(pending: PendingCompress) -> None:
        """Reject if compressed within last second, else pass through."""
        elapsed = time.time() - last_compress["ts"]
        pipeline_trace.append(f"rate_limiter (elapsed={elapsed:.1f}s)")
        if elapsed < 1.0 and last_compress["ts"] > 0:
            pending.reject(f"Rate limited ({elapsed:.1f}s since last)")
            return
        pending.pass_through()

    # --- Handler 2: Token budget checker ---
    def budget_checker(pending: PendingCompress) -> None:
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
    def quality_validator(pending: PendingCompress) -> None:
        """Reject if any summary is suspiciously short."""
        pipeline_trace.append(f"quality_validator ({len(pending.summaries)} summaries)")
        for i, summary in enumerate(pending.summaries):
            word_count = len(summary.split())
            if word_count < 3:
                pending.reject(
                    f"Summary [{i}] has only {word_count} words -- likely garbage"
                )
                return
        pending.pass_through()

    # --- Handler 4: Final approver (safety net) ---
    def final_approver(pending: PendingCompress) -> None:
        """Catch-all: approve if every middleware passed through."""
        pipeline_trace.append("final_approver")
        pending.approve()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        _seed_research(t)

        # Register with explicit ordering
        t.on("compress", rate_limiter, name="rate_limiter")
        t.on("compress", budget_checker, name="budget_checker", after="rate_limiter")
        t.on("compress", quality_validator, name="quality_validator", after="budget_checker")
        t.on("compress", final_approver, name="final_approver", after="quality_validator")

        print(f"\n  Registered pipeline: {t.hook_names}")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # --- First compress ---
        result: CompressResult | PendingCompress = t.compress(target_tokens=150, token_tolerance=10000)
        last_compress["ts"] = time.time()

        if isinstance(result, CompressResult):
            print(f"\n  Result: APPROVED (ratio={result.compression_ratio:.1%})")
            print("\n  AFTER compression:\n")
            t.compile().pprint(style="compact")
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


# =====================================================================
# Main
# =====================================================================

def main():
    ordering_basics()
    pass_through_pipeline()
    conditional_middleware()
    dynamic_insertion()
    full_pipeline()


if __name__ == "__main__":
    main()
