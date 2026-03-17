"""Observability and Monitoring

Production observability patterns using tract's middleware system and status
APIs. Middleware handlers fire on every significant operation, giving you
structured metrics, audit trails, and budget dashboards without wrappers.

Patterns shown:
  1. LLM Call Logging & Cost Tracking  -- per-commit model/tokens/latency/cost
  2. Token Budget Dashboard            -- per-stage usage with budget warnings
  3. Operation Audit Trail             -- timestamped log of every tract operation
  4. Context Growth Alerting           -- detect runaway context before it hurts

No LLM required.
"""

from datetime import datetime, timezone

from tract import Tract, MiddlewareContext, TractConfig, TokenBudgetConfig


# ===================================================================
# Pattern 1: LLM Call Logging & Cost Tracking
# ===================================================================

def llm_call_logging() -> None:
    """Track every LLM call: model, tokens, latency, cost via post_commit."""

    print("=" * 60)
    print("1. LLM Call Logging & Cost Tracking")
    print("=" * 60, "\n")

    with Tract.open() as t:
        metrics: dict = {"calls": [], "total_cost": 0.0}
        COST_PER_1K = {"input": 0.003, "output": 0.015}

        def track(ctx: MiddlewareContext) -> None:
            if ctx.commit is None:
                return
            meta = ctx.commit.metadata or {}
            inp = meta.get("input_tokens", ctx.commit.token_count)
            out = meta.get("output_tokens", ctx.commit.token_count)
            cost = (inp / 1000) * COST_PER_1K["input"] + (out / 1000) * COST_PER_1K["output"]
            metrics["calls"].append({
                "hash": ctx.commit.commit_hash[:8], "model": meta.get("model", "simulated"),
                "input": inp, "output": out, "latency_ms": meta.get("latency_ms", 0),
                "cost": round(cost, 6),
            })
            metrics["total_cost"] += cost

        tid = t.middleware.add("post_commit", track)

        t.system("You are a financial analyst.")
        t.user("Analyze Q3 revenue trends.")
        t.assistant(
            "Q3 revenue grew 18% YoY driven by enterprise expansion.",
            metadata={"model": "gpt-4o", "input_tokens": 45, "output_tokens": 32, "latency_ms": 820},
        )
        t.user("Break down by segment.")
        t.assistant(
            "Enterprise: +24% ($12M). SMB: +8% ($4M). Consumer: -2% ($1.5M).",
            metadata={"model": "gpt-4o", "input_tokens": 92, "output_tokens": 48, "latency_ms": 1150},
        )
        t.user("What is the forecast for Q4?")
        t.assistant(
            "Projecting 15% QoQ growth based on pipeline and seasonal patterns.",
            metadata={"model": "gpt-4o-mini", "input_tokens": 138, "output_tokens": 35, "latency_ms": 450},
        )

        real = [c for c in metrics["calls"] if c["model"] != "simulated"]
        print(f"  Tracked {len(real)} LLM calls, est. cost ${metrics['total_cost']:.4f}\n")
        for c in real:
            print(f"    [{c['hash']}] {c['model']:15s}  in={c['input']:>4}  out={c['output']:>4}"
                  f"  {c['latency_ms']:>5}ms  ${c['cost']:.4f}")

        assert len(real) == 3 and metrics["total_cost"] > 0
        t.middleware.remove(tid)

    print("\nPASSED")


# ===================================================================
# Pattern 2: Token Budget Dashboard
# ===================================================================

def token_budget_dashboard() -> None:
    """Track per-stage token usage against budgets using t.search.status()."""

    print("\n" + "=" * 60)
    print("2. Token Budget Dashboard")
    print("=" * 60, "\n")

    BUDGETS = {"research": 2000, "analysis": 1500, "synthesis": 1000}
    usage: dict[str, int] = {}

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=5000))
    with Tract.open(config=config) as t:
        t.system("You are a market research agent.")

        # --- Research stage ---
        start = t.search.status().token_count
        for item in [
            "Competitor A launched AI product at $99/mo targeting SMB.",
            "SaaS market growing 22% YoY, AI features are key differentiator.",
            "Customer survey: 68% want analytics, 45% want AI recommendations.",
            "Competitor B acquired data company for $50M.",
        ]:
            t.user(item)
            t.assistant(f"Noted: {item[:40]}...")
        usage["research"] = t.search.status().token_count - start

        t.compression.compress(content=(
            "Research: Competitor A AI $99/mo. SaaS +22% YoY. "
            "Customers want analytics (68%) and AI (45%). Competitor B acq $50M."
        ))

        # --- Analysis stage ---
        t.transition("analysis")
        start = t.search.status().token_count
        t.user("Analyze competitive positioning: pricing, features, momentum.")
        t.assistant(
            "We lead in NPS (52 vs 38). Pricing competitive at $79/mo. "
            "Feature gap in AI -- Competitor A has 6-month head start."
        )
        usage["analysis"] = t.search.status().token_count - start

        # --- Synthesis stage ---
        t.transition("synthesis")
        start = t.search.status().token_count
        t.user("Synthesize into executive brief.")
        t.assistant(
            "Strong NPS lead. Accelerate AI roadmap to close gap. "
            "Watch Competitor B data acquisition threat."
        )
        usage["synthesis"] = t.search.status().token_count - start

        # --- Dashboard ---
        total_budget = sum(BUDGETS.values())
        total_used = sum(usage.values())
        print(f"  {'Stage':<12} {'Budget':>7} {'Used':>7} {'Remain':>7} {'%':>6}")
        print(f"  {'-' * 42}")
        for name, budget in BUDGETS.items():
            used = usage[name]
            pct = used / budget * 100
            status = "" if pct < 80 else " WARNING" if pct < 100 else " OVER"
            print(f"  {name:<12} {budget:>7} {used:>7} {budget - used:>7} {pct:>5.0f}%{status}")
        print(f"  {'-' * 42}")
        print(f"  {'TOTAL':<12} {total_budget:>7} {total_used:>7} "
              f"{total_budget - total_used:>7} {total_used / total_budget * 100:>5.0f}%")
        print(f"\n  Final context: {t.search.status().token_count} tokens (compressed)")

        assert total_used > 0 and total_used < total_budget
        for name in BUDGETS:
            assert usage[name] > 0, f"Stage {name} should have tokens"

    print("\nPASSED")


# ===================================================================
# Pattern 3: Operation Audit Trail
# ===================================================================

def operation_audit_trail() -> None:
    """Log every tract operation via middleware on 6 events."""

    print("\n" + "=" * 60)
    print("3. Operation Audit Trail")
    print("=" * 60, "\n")

    with Tract.open() as t:
        log: list[dict] = []

        def record(ctx: MiddlewareContext) -> None:
            """Single handler for all auditable events."""
            entry: dict = {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": ctx.event, "branch": ctx.branch,
            }
            if ctx.commit:
                entry.update(op="commit", hash=ctx.commit.commit_hash[:8],
                             type=ctx.commit.content_type, tokens=ctx.commit.token_count)
            elif "transition" in ctx.event:
                entry.update(op="transition", target=ctx.target or "?")
            elif "compile" in ctx.event:
                entry.update(op="compile")
            elif "compress" in ctx.event:
                entry.update(op="compress")
            elif "merge" in ctx.event:
                entry.update(op="merge")
            log.append(entry)

        ids = [t.middleware.add(evt, record) for evt in [
            "post_commit", "pre_compile", "pre_compress",
            "pre_merge", "pre_transition", "post_transition",
        ]]

        # Generate varied operations
        t.system("You are a project assistant.")
        t.user("Start planning.")
        t.assistant("Planning initiated.")
        t.transition("implementation")
        t.user("Implement login.")
        t.assistant("Login module done with OAuth2.")
        t.compile()
        t.compression.compress(content="[Summary] Planning and implementation done.")
        t.branches.create("hotfix")
        t.user("Fix auth bypass.")
        t.assistant("Patched session validation.")
        t.branches.switch("implementation")
        t.merge("hotfix")

        # Display
        print(f"  {len(log)} audit entries:\n")
        print(f"  {'#':<3} {'Event':<18} {'Op':<11} {'Branch':<16} Details")
        print(f"  {'-' * 68}")
        for i, e in enumerate(log, 1):
            op = e.get("op", "?")
            if op == "commit":
                detail = f"[{e['hash']}] {e['type']} ({e['tokens']}tok)"
            elif op == "transition":
                detail = f"-> {e.get('target', '?')}"
            else:
                detail = ""
            print(f"  {i:<3} {e['event']:<18} {op:<11} {e['branch']:<16} {detail}")

        ops_found = {e.get("op") for e in log}
        assert {"commit", "transition", "compile", "compress", "merge"} <= ops_found
        print(f"\n  Verified: all 5 operation types captured")

        for mid in ids:
            t.middleware.remove(mid)

    print("\nPASSED")


# ===================================================================
# Pattern 4: Context Growth Alerting
# ===================================================================

def context_growth_alerting() -> None:
    """Alert when a single exchange adds more than 30% of the token ceiling."""

    print("\n" + "=" * 60)
    print("4. Context Growth Alerting")
    print("=" * 60, "\n")

    MAX_CTX = 2000
    ALERT_PCT = 30

    with Tract.open() as t:
        t.system("You are a technical architect.")
        prev = t.compile().token_count
        growth: list[dict] = []

        exchanges = [
            ("Review auth.", "JWT with 24h expiry and refresh rotation."),
            ("Database layer?", "PostgreSQL, connection pooling, read replicas, Alembic."),
            (
                "Full caching arch: Redis cluster, eviction, TTLs, warming, "
                "monitoring, failover, circuit breakers.",
                "3-node Redis Cluster. allkeys-lru for sessions, volatile-ttl "
                "for features. TTLs: 30min/5min/1h/4h. Cache warming on deploy. "
                "redis_exporter to Prometheus. Circuit breaker: 5 fails in 10s. "
                "Sentinel failover with 2/3 quorum.",
            ),
            ("API rate limiting?", "Token bucket 100 req/s, sliding window burst."),
        ]

        alerts = []
        for i, (q, a) in enumerate(exchanges):
            t.user(q)
            t.assistant(a)
            cur = t.compile().token_count
            delta = cur - prev
            gpct = delta / MAX_CTX * 100
            util = cur / MAX_CTX * 100
            entry = {"ex": i + 1, "tokens": cur, "delta": delta, "gpct": gpct, "util": util}
            growth.append(entry)
            if gpct > ALERT_PCT:
                alerts.append(entry)
            prev = cur

        print(f"  {'#':>3} {'Tokens':>7} {'Delta':>7} {'Growth%':>8} {'Util%':>7}")
        print(f"  {'-' * 38}")
        for e in growth:
            flag = " << ALERT" if e["gpct"] > ALERT_PCT else ""
            print(f"  {e['ex']:>3} {e['tokens']:>7} +{e['delta']:>5} "
                  f"{e['gpct']:>7.1f}% {e['util']:>6.1f}%{flag}")

        final = growth[-1]
        print(f"\n  Final: {final['tokens']}/{MAX_CTX} tokens ({final['util']:.0f}%)")
        if alerts:
            print(f"  {len(alerts)} alert(s) -- recommend compression before ceiling")
        else:
            print(f"  No alerts (all under {ALERT_PCT}% growth)")

        assert len(growth) == 4 and growth[-1]["tokens"] > growth[0]["tokens"]

    print("\nPASSED")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    llm_call_logging()
    token_budget_dashboard()
    operation_audit_trail()
    context_growth_alerting()

    print("\n" + "=" * 60)
    print("Summary: Observability and Monitoring")
    print("=" * 60)
    print("""
  Pattern                     Primitives Used
  --------------------------  ------------------------------------------
  LLM call logging & cost     post_commit middleware, commit.metadata
  Token budget dashboard      t.search.status(), t.transition(), budgets
  Operation audit trail       6 middleware events (commit/compile/compress
                              /merge/transition)
  Context growth alerting     t.compile().token_count, threshold math

  Key: instrument with middleware, not wrappers. Tract's event system
  gives structured data at every operation boundary.
""")
    print("Done.")


test_observability = main

if __name__ == "__main__":
    main()
