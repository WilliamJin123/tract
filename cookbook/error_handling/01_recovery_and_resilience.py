"""Recovery and Resilience

Production error-recovery patterns using tract's git-like primitives.
Branches isolate failures, tags checkpoint before risky ops, compression
manages growth, and middleware enforces circuit-breaker limits.

Patterns shown:
  1. Checkpoint Recovery        -- tag before risky ops, reset on failure
  2. Branch-Isolated Retries    -- branch per attempt, merge only on success
  3. Compression Fallback Chain -- progressively aggressive compression
  4. Circuit Breaker            -- middleware blocks after N consecutive failures

Demonstrates: t.tags.add(), t.branches.reset(), t.branches.create(),
              t.branches.switch(), t.merge(), t.compression.compress(),
              t.middleware.add(), BlockedError

No LLM required.
"""

from tract import (
    BlockedError,
    MiddlewareContext,
    Tract,
)


# ------------------------------------------------------------------
# 1. Checkpoint Recovery
# ------------------------------------------------------------------
# Tag HEAD before a risky operation.  If it fails, reset to the
# checkpoint and retry with a simpler approach.  Failed commits
# stay in the DAG for audit but HEAD moves back cleanly.

def checkpoint_recovery() -> None:
    print("=" * 60)
    print("1. Checkpoint Recovery")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a data analyst.")
        t.user("Here is our quarterly sales data for analysis.")
        t.assistant("I see the data. Ready for analysis.")

        # Tag a safe rollback point
        t.tags.register("checkpoint", "Safe rollback point before risky operations")
        checkpoint_hash = t.head
        t.tags.add(checkpoint_hash, "checkpoint")
        print(f"  Checkpoint tagged at [{checkpoint_hash[:8]}]")

        # Simulate a risky multi-step operation that goes off the rails
        t.user("Perform a complex multi-dimensional regression analysis.")
        t.assistant("Starting regression... computing correlations...")
        t.user("Include interaction terms and polynomial features.")

        commits_before = len(t.search.log())
        print(f"  Commits before failure: {commits_before}")
        print("  Simulated failure: analysis produced garbage results")

        # Reset HEAD back to the checkpoint
        t.branches.reset(checkpoint_hash)
        commits_after = len(t.search.log())
        print(f"  Reset to checkpoint [{checkpoint_hash[:8]}]")
        print(f"  Commits visible after reset: {commits_after}")

        # Retry with a simpler approach
        t.user("Summarize the key sales trends (top 3 takeaways).")
        t.assistant(
            "1. Revenue grew 12% QoQ. "
            "2. Enterprise segment led growth. "
            "3. Churn decreased in SMB."
        )

        ctx = t.compile()
        compiled_text = " ".join((m.content or "") for m in ctx.messages)
        assert "polynomial" not in compiled_text, "Failed analysis should be excluded"
        assert "key sales trends" in compiled_text, "Recovery prompt should be present"

        ctx.pprint(style="compact")
        print(f"  Final HEAD: [{t.head[:8]}]")
        print("  Verified: failed analysis excluded, recovery included")

    print()
    print("PASSED")


# ------------------------------------------------------------------
# 2. Branch-Isolated Retries
# ------------------------------------------------------------------
# Create a branch for each uncertain attempt.  On failure, switch
# back to main -- the failed work is isolated on a dead branch.
# On success, merge the winning branch back.

def branch_isolated_retries() -> None:
    print()
    print("=" * 60)
    print("2. Branch-Isolated Retries")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a financial modeler.")
        t.user("Build a revenue projection model.")
        t.assistant("Ready. I will try several modeling approaches.")

        main_head = t.head
        print(f"  Main branch: [{main_head[:8]}]")

        # Attempt 1: complex model (fails)
        t.branches.create("attempt_1", switch=True)
        t.assistant("Trying Monte Carlo simulation with 10k iterations...")
        t.assistant("ERROR: Variance too high, model diverged.")
        print(f"  Attempt 1 (branch): {len(t.search.log())} commits -- FAILED")
        t.branches.switch("main")

        # Attempt 2: moderate model (fails)
        t.branches.create("attempt_2", switch=True)
        t.assistant("Trying linear regression with seasonality...")
        t.assistant("ERROR: R-squared too low (0.23), unreliable.")
        print(f"  Attempt 2 (branch): {len(t.search.log())} commits -- FAILED")
        t.branches.switch("main")

        # Attempt 3: simple model (succeeds)
        t.branches.create("attempt_3", switch=True)
        t.assistant(
            "Using 3-month moving average: projected Q4 revenue $2.1M. "
            "Confidence: moderate. Method: simple, robust to noise."
        )
        print(f"  Attempt 3 (branch): {len(t.search.log())} commits -- SUCCESS")

        # Merge only the successful branch
        t.branches.switch("main")
        merge_result = t.merge("attempt_3")
        print(f"  Merged attempt_3: {merge_result.merge_type}")

        ctx = t.compile()
        text = " ".join((m.content or "") for m in ctx.messages)
        assert "diverged" not in text, "Failed attempt 1 should not leak"
        assert "R-squared" not in text, "Failed attempt 2 should not leak"
        assert "moving average" in text, "Successful attempt should be merged"

        ctx.pprint(style="compact")
        print("  Failed branches isolated, successful branch merged")

    print()
    print("PASSED")


# ------------------------------------------------------------------
# 3. Compression Fallback Chain
# ------------------------------------------------------------------
# When context grows large, try progressively more aggressive
# compression strategies until one succeeds.

def compression_fallback_chain() -> None:
    print()
    print("=" * 60)
    print("3. Compression Fallback Chain")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a data analyst tracking weekly metrics.")

        # Build up a large conversation
        for week in range(1, 21):
            t.user(f"Week {week} report: revenue ${week * 1000}, users {week * 50}.")
            t.assistant(
                f"Week {week} recorded. Revenue trend: "
                f"{'up' if week % 3 != 0 else 'flat'}. "
                f"Cumulative revenue: ${sum(i * 1000 for i in range(1, week + 1))}."
            )

        ctx_original = t.compile()
        original_tokens = ctx_original.token_count
        print(f"  Before compression: {len(ctx_original.messages)} messages, "
              f"~{original_tokens} tokens")

        # Try each strategy in order; stop on first success
        strategies = [
            ("sliding_window", "Keep only the last 5 weeks of detailed data."),
            ("manual_summary", (
                "20 weeks of metrics tracked. Revenue grew from $1k to $20k/week. "
                "Total cumulative: $210k. Users: 50 to 1000. "
                "Trend: generally increasing with periodic flat periods."
            )),
            ("aggressive", "Metrics: 20 weeks, $210k total revenue, 1000 users."),
        ]

        succeeded_strategy = None
        for name, summary_content in strategies:
            try:
                t.compression.compress(content=summary_content)
                ctx_after = t.compile()
                reduction = (1 - ctx_after.token_count / original_tokens) * 100
                print(f"  Strategy '{name}': {len(ctx_after.messages)} messages, "
                      f"~{ctx_after.token_count} tokens ({reduction:.0f}% reduction)")
                succeeded_strategy = name
                break
            except Exception as e:
                print(f"  Strategy '{name}' failed: {e}")

        assert succeeded_strategy is not None, "At least one strategy should succeed"
        ctx_final = t.compile()
        assert ctx_final.token_count < original_tokens, "Compression should reduce tokens"
        print(f"  Succeeded with: '{succeeded_strategy}'")

    print()
    print("PASSED")


# ------------------------------------------------------------------
# 4. Middleware Circuit Breaker
# ------------------------------------------------------------------
# A pre_commit middleware tracks consecutive failures.  After N
# failures the breaker trips, blocking further commits until the
# strategy is changed and the breaker is reset.

def circuit_breaker() -> None:
    print()
    print("=" * 60)
    print("4. Middleware Circuit Breaker")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are an API integration manager.")

        breaker = {"failures": 0, "threshold": 3, "open": False}

        def circuit_breaker_mw(ctx: MiddlewareContext):
            """Block commits when the breaker is open."""
            if breaker["open"]:
                raise BlockedError(
                    ctx.event,
                    f"Circuit breaker OPEN after {breaker['failures']} failures. "
                    f"Reset required before proceeding.",
                )

        mw_id = t.middleware.add("pre_commit", circuit_breaker_mw)
        print(f"  Circuit breaker registered (threshold={breaker['threshold']})")

        api_calls = [
            ("GET /users", True),
            ("GET /orders", True),
            ("POST /payment", False),       # failure 1
            ("POST /payment", False),       # failure 2
            ("POST /payment", False),       # failure 3 -> trips breaker
            ("GET /status", True),           # blocked!
        ]

        results: list[tuple[str, str]] = []
        for endpoint, succeeds in api_calls:
            try:
                if succeeds and not breaker["open"]:
                    t.user(f"Call {endpoint}")
                    t.assistant(f"200 OK: {endpoint}")
                    breaker["failures"] = 0
                    results.append(("OK", endpoint))
                    print(f"  OK: {endpoint}")
                elif not succeeds:
                    breaker["failures"] += 1
                    t.user(f"Call {endpoint}")
                    t.assistant(f"500 ERROR: {endpoint} -- server error")
                    results.append(("FAIL", endpoint))
                    print(f"  FAIL [{breaker['failures']}/{breaker['threshold']}]: "
                          f"{endpoint}")
                    if breaker["failures"] >= breaker["threshold"]:
                        breaker["open"] = True
                        print("  ** Circuit breaker TRIPPED **")
                else:
                    t.user(f"Call {endpoint}")
                    results.append(("OK", endpoint))
                    print(f"  OK: {endpoint}")
            except BlockedError:
                results.append(("BLOCKED", endpoint))
                print(f"  BLOCKED: {endpoint} -- circuit breaker open")

        # Verify the breaker tripped and blocked at least one call
        statuses = [r[0] for r in results]
        assert statuses.count("BLOCKED") >= 1, "At least one call should be blocked"
        assert breaker["open"] is True

        # Reset and verify calls work again
        breaker["open"] = False
        breaker["failures"] = 0
        print()
        print("  Circuit breaker reset")

        t.user("Call GET /health")
        t.assistant("200 OK: healthy")
        print("  OK: GET /health (after reset)")

        t.middleware.remove(mw_id)

    print()
    print("PASSED")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def main() -> None:
    checkpoint_recovery()
    branch_isolated_retries()
    compression_fallback_chain()
    circuit_breaker()

    print()
    print("=" * 60)
    print("Summary: Recovery and Resilience Patterns")
    print("=" * 60)
    print()
    print("  Pattern                     Tract Primitives Used")
    print("  --------------------------  ------------------------------------")
    print("  Checkpoint recovery         tags.add() + branches.reset()")
    print("  Branch-isolated retries     branches.create() + switch() + merge()")
    print("  Compression fallback        compression.compress() with strategies")
    print("  Circuit breaker             middleware.add('pre_commit') + BlockedError")
    print()
    print("  Key principle: branches isolate failures, tags checkpoint state,")
    print("  compression manages growth, middleware enforces limits.  Failed")
    print("  attempts stay in the DAG for audit but HEAD moves cleanly past them.")
    print()
    print("Done.")


# Alias for pytest discovery
test_recovery_and_resilience = main


if __name__ == "__main__":
    main()
