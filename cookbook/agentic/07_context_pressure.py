"""Context Window Pressure: budgets, priority eviction, and auto-compression

Demonstrates how an agent handles context window pressure -- what happens
when token limits are hit, how priority-based eviction works, and the
full compression lifecycle.

Sections:
  1. Token Budget Enforcement  -- BudgetExceededError + manual compress
  2. Priority-Based Eviction   -- PINNED/IMPORTANT survive, NORMAL summarized
  3. Auto-Compression in Loop  -- LoopConfig.auto_compress_threshold mid-loop

Requires: LLM API key (section 3 uses the agent loop)
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import (
    Tract, TractConfig, TokenBudgetConfig,
    BudgetAction, BudgetExceededError, Priority,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


# =====================================================================
# 1. Token Budget Enforcement
# =====================================================================
# Hard token limit (500). Agent commits findings until BudgetExceededError
# fires, then compresses and retries.

def section_1_budget_enforcement() -> None:
    print("=" * 70)
    print("  1. Token Budget Enforcement (max_tokens=500, action=REJECT)")
    print("=" * 70)
    print()

    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=500, action=BudgetAction.REJECT),
    )

    with Tract.open(config=config) as t:
        t.system("You are a research assistant collecting findings.")

        findings = [
            ("HTTP/2 multiplexing",
             "HTTP/2 multiplexes streams over a single TCP connection, "
             "eliminating head-of-line blocking at the HTTP layer. Each stream "
             "has a unique ID and can carry headers and data frames independently."),
            ("QUIC transport",
             "QUIC runs over UDP with built-in TLS 1.3 encryption. It solves "
             "TCP head-of-line blocking by making streams independent at the "
             "transport layer, not just the application layer."),
            ("HTTP/3 adoption",
             "HTTP/3 uses QUIC as its transport. Major CDNs support it. Browser "
             "adoption is above 95%. Connection migration across networks is key."),
            ("Performance data",
             "Benchmarks show HTTP/3 reduces page load by 10-30% on lossy networks "
             "vs HTTP/2. Smaller gains (2-5%) on reliable networks."),
            ("Migration strategy",
             "Migrate incrementally: HTTP/3 on CDN edge first, then origin. "
             "Alt-Svc headers advertise HTTP/3. Fall back to HTTP/2 for old clients."),
        ]

        committed, compressions = 0, 0

        for topic, text in findings:
            try:
                t.commit(
                    content={"content_type": "freeform", "text": text},
                    message=topic,
                )
                committed += 1
                tokens = t.status().token_count
                print(f"  Committed: {topic} ({tokens} tokens)")

            except BudgetExceededError as e:
                print(f"\n  ** BudgetExceededError: {e.current_tokens}/{e.max_tokens} tokens")
                print(f"  ** Compressing to make room...\n")

                result = t.compress(
                    content="Research summary: HTTP/2 multiplexes over TCP. "
                    "QUIC/HTTP/3 improves with UDP-based streams, TLS 1.3, "
                    "and connection migration.",
                )
                compressions += 1
                after = t.status()
                print(f"  Compressed: {result.original_tokens} -> "
                      f"{result.compressed_tokens} tokens ({result.compression_ratio:.1%})")
                print(f"  Budget now: {after.token_count}/{e.max_tokens}\n")

                # Retry after compression
                t.commit(
                    content={"content_type": "freeform", "text": text},
                    message=topic,
                )
                committed += 1
                print(f"  Committed: {topic} ({t.status().token_count} tokens)")

        print(f"\n  Done: {committed} committed, {compressions} compression(s)")
        print(f"  Final: {t.status().token_count} tokens\n")


# =====================================================================
# 2. Priority-Based Eviction Under Pressure
# =====================================================================
# Different priorities: NORMAL dialogue, PINNED test cases, IMPORTANT
# decisions. Compression summarizes NORMAL, preserves the rest.

def section_2_priority_eviction() -> None:
    print("=" * 70)
    print("  2. Priority-Based Eviction Under Pressure")
    print("=" * 70)
    print()

    with Tract.open() as t:
        t.system("Architecture review context.")

        # NORMAL: routine dialogue (expendable)
        for text in [
            "User asked about deployment timeline for Q3.",
            "Discussed CI pipeline options: GitHub Actions vs CircleCI.",
            "Reviewed PR #142 comments on naming conventions.",
        ]:
            t.commit(content={"content_type": "freeform", "text": text}, message="dialogue")
            print(f"  [NORMAL]    {text[:60]}")

        # IMPORTANT: architecture decisions
        for text in [
            "DECISION: Use event sourcing for order service. Rationale: "
            "auditability, replay capability, natural fit for CQRS.",
            "DECISION: PostgreSQL primary, Redis cache. ACID guarantees "
            "needed; Redis for session/rate-limit.",
        ]:
            h = t.commit(content={"content_type": "freeform", "text": text},
                         message="architecture decision")
            t.annotate(h, Priority.IMPORTANT, reason="Architecture decision")
            print(f"  [IMPORTANT] {text[:60]}")

        # PINNED: test cases (must survive verbatim)
        for text in [
            "TEST: order_create -> event_stored -> projection_updated. "
            "Assert: OrderCreated event in store, read model has order.",
            "TEST: concurrent_writes -> conflict_detected -> retry_succeeds. "
            "Assert: optimistic lock raises, retry commits successfully.",
        ]:
            h = t.commit(content={"content_type": "freeform", "text": text},
                         message="test case")
            t.annotate(h, Priority.PINNED, reason="Test case -- must survive")
            print(f"  [PINNED]    {text[:60]}")

        before = t.status()
        print(f"\n  Before: {before.commit_count} commits, {before.token_count} tokens")

        # Compress -- NORMAL summarized, IMPORTANT/PINNED preserved
        result = t.compress(
            content="Team discussed deployment timeline, CI options, and PR comments.",
        )

        after = t.status()
        print(f"  After:  {after.commit_count} commits, {after.token_count} tokens")
        print(f"  Ratio: {result.compression_ratio:.1%}, "
              f"compressed {len(result.source_commits)} source commits, "
              f"preserved {len(result.preserved_commits)}")

        # Verify survivors
        print(f"\n  Surviving commits:")
        for entry in t.log(limit=20):
            if entry.content_type == "system":
                continue
            prio = entry.effective_priority or "normal"
            preview = (entry.text or "")[:65] if entry.text else "(summary)"
            print(f"    {entry.commit_hash[:8]}  [{prio:9s}]  {preview}")

        pinned = t.search.pinned()
        print(f"\n  Pinned survived: {len(pinned)} commit(s)")
        for p in pinned:
            print(f"    {p.commit_hash[:8]}  {(p.message or '')[:50]}")
        print()


# =====================================================================
# 3. Auto-Compression in Agent Loop
# =====================================================================
# LoopConfig.auto_compress_threshold triggers compression mid-loop when
# context exceeds 80% of max_tokens.

def section_3_auto_compression() -> None:
    print("=" * 70)
    print("  3. Auto-Compression in Agent Loop")
    print("=" * 70)
    print()

    if not llm.available:
        print("  SKIPPED (no LLM provider)\n")
        return

    from tract.loop import LoopConfig

    with Tract.open(
        config=TractConfig(
            token_budget=TokenBudgetConfig(max_tokens=2000, action=BudgetAction.WARN),
        ),
        **llm.tract_kwargs(MODEL_ID),
    ) as t:

        t.system(
            "You are a research agent investigating microservices patterns.\n"
            "Commit each finding. Topics: service discovery, circuit breakers,\n"
            "saga pattern, API gateway routing, event-driven communication."
        )

        # Pin a constraint so it survives auto-compression
        h = t.commit(
            content={"content_type": "freeform",
                     "text": "CONSTRAINT: gRPC for internal comms, REST external only."},
            message="Constraint: gRPC internal, REST external",
        )
        t.annotate(h, Priority.PINNED, reason="Hard architectural constraint")
        print(f"  Pinned constraint: {h[:8]}")

        log = StepLogger()
        print(f"\n  Running loop (max_tokens=2000, auto_compress at 80%)...\n")

        result = t.llm.run(
            "Research each microservices topic. Commit a detailed finding for "
            "each with implementation details and trade-offs.",
            max_steps=15,
            profile="full",
            tool_names=["commit", "status", "log"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
            loop_config=LoopConfig(
                max_tokens=2000,
                auto_compress_threshold=0.8,
            ),
        )

        # Auto-compression metrics
        print(f"\n  Result: {result.status} ({result.steps} steps, "
              f"{result.tool_calls} tool calls)")
        print(f"  Compressions triggered: {result.compressions_triggered}")

        print(f"\n  Step metrics:")
        for m in result.step_metrics:
            tag = " [COMPRESSED]" if m.compressed else ""
            print(f"    Step {m.step}: {m.context_tokens} tokens, "
                  f"{m.tool_count} tools{tag}")

        # Verify pinned constraint survived
        pinned = t.search.pinned()
        print(f"\n  Pinned after loop: {len(pinned)}")
        for p in pinned:
            print(f"    {p.commit_hash[:8]}  {(p.text or '')[:70]}")

        final = t.status()
        print(f"\n  Final: {final.commit_count} commits, {final.token_count} tokens\n")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    section_1_budget_enforcement()
    section_2_priority_eviction()
    section_3_auto_compression()

    print("=" * 70)
    print("  Key takeaways:")
    print("    Budget enforcement:  BudgetExceededError + compress() retry")
    print("    Priority eviction:   PINNED verbatim > IMPORTANT > NORMAL summarized")
    print("    Auto-compression:    LoopConfig(auto_compress_threshold=0.8)")
    print("=" * 70)


if __name__ == "__main__":
    main()


# --- See also ---
# Tool compaction:        agentic/03_tool_compaction.py
# Checkpoint/resume:      agentic/08_checkpoint_resume.py
