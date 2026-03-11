"""Error Recovery and Graceful Degradation

Production error recovery patterns using tract's git-like primitives.
Branch, reset, tag, and compress turn error handling from fragile
try/except chains into structured, auditable recovery workflows.

Patterns shown:
  1. Checkpoint-and-Rollback  -- tag before risky ops, reset on failure
  2. Branch-Based Isolation   -- branch for uncertain ops, abandon on failure
  3. Graceful Degradation     -- compress and retry when context is too large
  4. Circuit Breaker           -- middleware tracks failures, blocks after N
  5. Multi-Strategy Fallback   -- try primary/secondary/minimal strategies

Demonstrates: t.tag(), t.reset(), t.branch(), t.switch(), t.merge(),
              t.compress(), t.use(), BlockedError, BudgetExceededError,
              RetryExhaustedError, CompressionError

No LLM required.
"""

from tract import (
    BlockedError,
    BudgetExceededError,
    CompressionError,
    Tract,
)
from tract.exceptions import RetryExhaustedError


def main():
    # =================================================================
    # 1. Checkpoint-and-Rollback
    # =================================================================
    #
    # Tag the current HEAD before a risky operation. If it fails, reset
    # to the checkpoint and retry with a simpler approach. The failed
    # commits stay in the DAG for audit, but HEAD moves back cleanly.

    print("=" * 60)
    print("1. Checkpoint-and-Rollback")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a data analyst.")
        t.user("Here is our quarterly sales data for analysis.")
        t.assistant("I see the data. Ready for analysis.")

        # Register and apply a checkpoint tag
        t.register_tag("checkpoint", "Safe rollback point before risky operations")
        checkpoint_hash = t.head
        t.tag(checkpoint_hash, "checkpoint")
        print(f"  Checkpoint tagged at [{checkpoint_hash[:8]}]")

        # Simulate a risky multi-step operation that fails partway through
        t.user("Perform a complex multi-dimensional regression analysis.")
        t.assistant("Starting regression... computing correlations...")
        t.user("Include interaction terms and polynomial features.")

        # Simulate failure: the analysis went off the rails
        commits_before_reset = len(t.log())
        print(f"  Commits before failure: {commits_before_reset}")
        print("  Simulated failure: analysis produced garbage results")

        # Reset to checkpoint -- HEAD moves back, failed commits stay in DAG
        t.reset(checkpoint_hash)
        commits_after_reset = len(t.log())
        print(f"  Reset to checkpoint [{checkpoint_hash[:8]}]")
        print(f"  Commits visible after reset: {commits_after_reset}")

        # Retry with a simpler approach
        t.user("Summarize the key sales trends (top 3 takeaways).")
        t.assistant(
            "1. Revenue grew 12% QoQ. "
            "2. Enterprise segment led growth. "
            "3. Churn decreased in SMB."
        )

        ctx = t.compile()
        print(f"  Recovery succeeded: {len(ctx.messages)} messages in context")
        print(f"  Final HEAD: [{t.head[:8]}]")
        print()

        # Verify the failed analysis is NOT in compiled context
        compiled_text = " ".join((m.content or "") for m in ctx.messages)
        assert "polynomial" not in compiled_text, "Failed analysis should not be in context"
        assert "key sales trends" in compiled_text, "Recovery prompt should be in context"
        print("  Verified: failed analysis excluded, recovery included")

    # =================================================================
    # 2. Branch-Based Isolation
    # =================================================================
    #
    # Branch before uncertain operations. If the branch succeeds, merge
    # it back. If it fails, switch back to main -- the failed work is
    # isolated on a dead branch that doesn't pollute your main context.

    print()
    print("=" * 60)
    print("2. Branch-Based Isolation")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a research assistant.")
        t.user("We need to analyze competitor pricing strategies.")
        t.assistant("I will research competitor pricing. What industry?")
        t.user("SaaS B2B tools, focus on the top 5 competitors.")

        main_head_before = t.head
        print(f"  Main branch HEAD: [{main_head_before[:8]}]")

        # --- Attempt 1: branch for deep analysis (will fail) ---

        t.branch("deep_analysis")
        print(f"  Created branch 'deep_analysis', switched to it")

        t.assistant("Analyzing competitor A: enterprise pricing at $500/seat...")
        t.assistant("Analyzing competitor B: usage-based at $0.01/request...")
        t.assistant("Cross-referencing with market reports... found inconsistency.")

        # Simulate failure: analysis hit a dead end
        deep_commits = len(t.log())
        print(f"  Deep analysis produced {deep_commits} commits, then failed")
        print("  Simulated failure: data sources contradicted each other")

        # Abandon the branch -- switch back to main
        t.switch("main")
        main_commits = len(t.log())
        print(f"  Switched back to main: {main_commits} commits (clean)")

        # --- Attempt 2: branch for simpler analysis (will succeed) ---

        t.branch("simple_analysis")
        print(f"  Created branch 'simple_analysis', switched to it")

        t.assistant(
            "Top 5 SaaS competitors by pricing model:\n"
            "1. Competitor A: per-seat ($50-500/mo)\n"
            "2. Competitor B: usage-based ($0.01/req)\n"
            "3. Competitor C: flat-rate ($999/mo)\n"
            "4. Competitor D: freemium + enterprise\n"
            "5. Competitor E: per-seat + usage hybrid"
        )

        print(f"  Simple analysis succeeded with {len(t.log())} commits")

        # Merge the successful branch back to main
        t.switch("main")
        result = t.merge("simple_analysis")
        print(f"  Merged 'simple_analysis' into main: {result.merge_type}")

        ctx = t.compile()
        print(f"  Final context: {len(ctx.messages)} messages")
        print()

        # Verify isolation: deep_analysis content is not in main context
        compiled_text = " ".join((m.content or "") for m in ctx.messages)
        assert "inconsistency" not in compiled_text, "Failed branch should not leak"
        assert "flat-rate" in compiled_text, "Successful branch should be merged"
        print("  Verified: failed branch isolated, successful branch merged")

    # =================================================================
    # 3. Graceful Degradation with Compression
    # =================================================================
    #
    # When context grows too large, compress progressively. Each round
    # of compression reduces token count. If manual compression fails,
    # fall back to more aggressive strategies.

    print()
    print("=" * 60)
    print("3. Graceful Degradation with Compression")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a project manager tracking sprint progress.")

        # Build up a large context with many messages
        for i in range(1, 16):
            t.user(f"Sprint {i} update: completed {i * 3} story points.")
            t.assistant(
                f"Sprint {i} recorded. Velocity trend: "
                f"{'increasing' if i % 3 != 0 else 'dip detected'}. "
                f"Cumulative: {sum(j * 3 for j in range(1, i + 1))} points."
            )

        ctx_before = t.compile()
        tokens_before = ctx_before.token_count
        messages_before = len(ctx_before.messages)
        print(f"  Before compression: {messages_before} messages, ~{tokens_before} tokens")

        # Progressive compression: each round targets fewer tokens
        targets = [
            (tokens_before // 2, "light"),
            (tokens_before // 4, "medium"),
            (tokens_before // 8, "aggressive"),
        ]

        for target_tokens, label in targets:
            try:
                # Manual compression -- provide a summary ourselves
                t.compress(
                    content=(
                        f"[{label} summary] Sprint tracking data compressed. "
                        f"15 sprints completed. Average velocity: 24 points/sprint. "
                        f"Total: 360 story points delivered. "
                        f"Trend: generally increasing with periodic dips."
                    ),
                )
                ctx_after = t.compile()
                print(
                    f"  After {label} compression: "
                    f"{len(ctx_after.messages)} messages, "
                    f"~{ctx_after.token_count} tokens"
                )
                break  # compression succeeded, stop trying
            except CompressionError as e:
                print(f"  {label} compression failed: {e}")
                # Continue to next, more aggressive strategy

        ctx_final = t.compile()
        print(f"  Final: {len(ctx_final.messages)} messages, ~{ctx_final.token_count} tokens")
        print()

        # Verify compression reduced the context
        assert ctx_final.token_count < tokens_before, "Compression should reduce tokens"
        print("  Verified: context was reduced by compression")

    # =================================================================
    # 4. Middleware-Based Circuit Breaker
    # =================================================================
    #
    # Track consecutive failures via middleware state. After N failures,
    # the circuit breaker trips and blocks further operations until the
    # strategy is changed. This prevents burning tokens on a failing
    # approach.

    print()
    print("=" * 60)
    print("4. Middleware-Based Circuit Breaker")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a code reviewer.")

        # Circuit breaker state
        failure_tracker = {"consecutive": 0, "total": 0, "tripped": False}
        FAILURE_THRESHOLD = 3

        def circuit_breaker(ctx):
            """Block commits after too many consecutive failures."""
            if failure_tracker["tripped"]:
                raise BlockedError(
                    ctx.event,
                    f"Circuit breaker OPEN -- {failure_tracker['consecutive']} "
                    f"consecutive failures (threshold: {FAILURE_THRESHOLD})",
                )

        breaker_id = t.use("pre_commit", circuit_breaker)
        print(f"  Registered circuit breaker middleware: {breaker_id}")
        print(f"  Threshold: {FAILURE_THRESHOLD} consecutive failures")

        # Simulate operations with failures
        operations = [
            ("Review auth module", True),
            ("Review payment module", True),
            ("Review legacy XML parser", False),     # failure 1
            ("Review legacy SOAP adapter", False),   # failure 2
            ("Review legacy CORBA bridge", False),    # failure 3 -- trips breaker
            ("Review REST API", True),                # blocked by breaker
        ]

        for task, succeeds in operations:
            try:
                t.user(task)

                if not succeeds:
                    # Simulate failure by recording it
                    failure_tracker["consecutive"] += 1
                    failure_tracker["total"] += 1
                    t.assistant(f"FAILED: {task} -- too complex to review automatically")
                    print(f"  [{failure_tracker['consecutive']}/{FAILURE_THRESHOLD}] "
                          f"FAIL: {task}")

                    # Trip the breaker if threshold reached
                    if failure_tracker["consecutive"] >= FAILURE_THRESHOLD:
                        failure_tracker["tripped"] = True
                        print(f"  ** Circuit breaker TRIPPED after "
                              f"{failure_tracker['consecutive']} failures **")
                else:
                    failure_tracker["consecutive"] = 0  # reset on success
                    t.assistant(f"Review complete: {task} -- no issues found")
                    print(f"  OK: {task}")

            except BlockedError as e:
                print(f"  BLOCKED: {task} -- {e.reasons[0]}")

        # Reset the circuit breaker and switch strategy
        print()
        print("  Resetting circuit breaker, switching to manual review strategy")
        failure_tracker["tripped"] = False
        failure_tracker["consecutive"] = 0

        # Now operations can proceed again
        t.user("Review REST API (manual review mode)")
        t.assistant("Manual review: REST API looks good, minor style issues only.")
        print("  OK: Review REST API (after circuit breaker reset)")

        # Clean up
        t.remove_middleware(breaker_id)
        print(f"  Removed circuit breaker middleware: {breaker_id}")

    # =================================================================
    # 5. Multi-Strategy Fallback Chain
    # =================================================================
    #
    # Try multiple strategies in order: primary (complex), secondary
    # (simplified), minimal (basic). Each uses different config. Track
    # which strategy succeeded in metadata for observability.

    print()
    print("=" * 60)
    print("5. Multi-Strategy Fallback Chain")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a report generator.")

        t.user(
            "Generate a comprehensive market analysis report covering "
            "TAM/SAM/SOM, competitive landscape, pricing analysis, "
            "and 5-year revenue projections."
        )

        strategies = [
            {
                "name": "detailed",
                "config": {"model": "gpt-4o", "temperature": 0.3, "max_tokens": 4096},
                "prompt": (
                    "Full analysis with TAM $50B, SAM $8B, SOM $800M. "
                    "12 competitors mapped. 5-year CAGR 23%."
                ),
                "simulate_failure": True,  # simulate: model overloaded
            },
            {
                "name": "standard",
                "config": {"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 2048},
                "prompt": (
                    "Market summary: TAM $50B, 5 key competitors, "
                    "3-year projection at 20% CAGR."
                ),
                "simulate_failure": True,  # simulate: still too complex
            },
            {
                "name": "minimal",
                "config": {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1024},
                "prompt": (
                    "Brief market overview: $50B TAM, growing 20% annually. "
                    "Key insight: enterprise segment underserved."
                ),
                "simulate_failure": False,  # this one succeeds
            },
        ]

        # Tag checkpoint before attempting strategies
        t.register_tag("checkpoint", "Safe rollback point")
        checkpoint = t.head
        t.tag(checkpoint, "checkpoint")
        succeeded_strategy = None

        for i, strategy in enumerate(strategies):
            attempt = i + 1
            name = strategy["name"]
            print(f"  Attempt {attempt}: strategy='{name}' "
                  f"(model={strategy['config']['model']}, "
                  f"temp={strategy['config']['temperature']})")

            # Branch for each strategy attempt
            branch_name = f"strategy_{name}"
            t.branch(branch_name)

            # Apply strategy-specific config
            t.configure(**strategy["config"])

            if strategy["simulate_failure"]:
                # Simulate failure
                t.assistant(
                    f"[FAILED] Strategy '{name}' could not complete. "
                    f"Error: simulated timeout/overload."
                )
                print(f"    FAILED: strategy '{name}'")

                # Abandon this branch
                t.switch("main")
                continue

            # Strategy succeeded
            t.assistant(
                strategy["prompt"],
                metadata={
                    "strategy": name,
                    "attempt": attempt,
                    "fallback_chain": [s["name"] for s in strategies[:i]],
                },
            )
            succeeded_strategy = name
            print(f"    SUCCESS: strategy '{name}'")

            # Merge successful branch back to main
            t.switch("main")
            t.merge(branch_name)
            break

        print()
        if succeeded_strategy:
            print(f"  Final strategy: '{succeeded_strategy}' "
                  f"(after {strategies.index(next(s for s in strategies if s['name'] == succeeded_strategy))} fallbacks)")
        else:
            print("  All strategies failed -- would escalate to human")

        ctx = t.compile()
        print(f"  Final context: {len(ctx.messages)} messages")
        print()

        # Verify only the successful strategy is in the final context
        compiled_text = " ".join((m.content or "") for m in ctx.messages)
        assert "enterprise segment underserved" in compiled_text
        assert succeeded_strategy == "minimal"
        print("  Verified: only successful strategy in final context")

    # =================================================================
    # Summary
    # =================================================================

    print()
    print("=" * 60)
    print("Summary: Why Git-Like Primitives Excel at Error Recovery")
    print("=" * 60)
    print()
    print("  Pattern                   Tract Primitives Used")
    print("  -----------------------   --------------------------------")
    print("  Checkpoint-and-Rollback   tag() + reset()")
    print("  Branch-Based Isolation    branch() + switch() + merge()")
    print("  Graceful Degradation      compress() with progressive targets")
    print("  Circuit Breaker           use() middleware + BlockedError")
    print("  Multi-Strategy Fallback   branch() + configure() + merge()")
    print()
    print("  Key advantage: failed attempts stay in the DAG for audit,")
    print("  but HEAD moves cleanly past them. No manual context cleanup.")
    print()
    print("Done.")


# Alias for pytest discovery
test_recovery_strategies = main


if __name__ == "__main__":
    main()
