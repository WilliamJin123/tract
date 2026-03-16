"""Token Budget Optimization and Cost Management

Fine-grained control over context window size using tract's compile strategies,
compression, and priority annotations. Demonstrates that tract provides
precision budget management superior to simple truncation approaches.

Patterns covered:
  1. Per-stage token budgets (research / generation / review)
  2. Auto-compression triggers via threshold config
  3. Compile strategy optimization (full vs messages vs adaptive)
  4. Progressive compression for long workflows with pinned findings
  5. Token tracking and budget reporting

Demonstrates: TokenBudgetConfig, configure(), compile(strategy=), compress(),
              annotate(), Priority, status(), transition()

No LLM required -- all compression uses manual content= mode.
"""

from tract import Priority, Tract, TractConfig, TokenBudgetConfig


# ---------------------------------------------------------------------------
# Helpers: generate realistic long-form content for strategy demonstrations
# ---------------------------------------------------------------------------

def _research_report(quarter: int) -> str:
    """Generate a ~600-800 char research report for a given quarter."""
    revenue = 2.1 + quarter * 0.8
    users = 50 + quarter * 12
    churn = max(1.2, 4.5 - quarter * 0.3)
    nps = 32 + quarter * 3
    return (
        f"Q{quarter} Comprehensive Analysis Report\n"
        f"{'=' * 40}\n"
        f"Revenue Performance: ${revenue:.1f}M total revenue, representing "
        f"a {8 + quarter}% increase quarter-over-quarter. Enterprise segment "
        f"contributed {55 + quarter}% of revenue while SMB grew at {12 + quarter}% "
        f"month-over-month. Average contract value increased to ${45 + quarter * 5}K "
        f"from ${40 + quarter * 4}K in the prior quarter.\n\n"
        f"User Metrics: {users}K monthly active users with {churn:.1f}% monthly churn. "
        f"Net Promoter Score of {nps}. Feature adoption rate for the new analytics "
        f"dashboard reached {30 + quarter * 8}%. Power users (>20 sessions/month) "
        f"grew to {15 + quarter * 3}% of the base. Mobile usage now represents "
        f"{25 + quarter * 4}% of all sessions.\n\n"
        f"Competitive Intelligence: Primary competitor launched pricing changes "
        f"affecting {10 + quarter * 2}% of overlapping accounts. Secondary competitor "
        f"raised Series {'B' if quarter < 5 else 'C'} at ${80 + quarter * 20}M "
        f"valuation. Market consolidation continues with {2 + quarter // 3} "
        f"acquisitions this quarter in our segment."
    )


def _analysis_response(quarter: int) -> str:
    """Generate a ~600-800 char analysis response."""
    revenue = 2.1 + quarter * 0.8
    users = 50 + quarter * 12
    return (
        f"Analysis of Q{quarter} Results\n"
        f"{'=' * 40}\n"
        f"Key Findings:\n"
        f"1. Revenue trajectory is {'above' if quarter > 3 else 'on'} plan. "
        f"The ${revenue:.1f}M result {'exceeds' if quarter > 3 else 'meets'} the "
        f"board-approved target by {max(0, (quarter - 3) * 2)}%. Enterprise momentum "
        f"is strong with pipeline coverage at {2.5 + quarter * 0.2:.1f}x.\n\n"
        f"2. User growth at {users}K MAU is healthy but acquisition cost increased "
        f"to ${35 + quarter * 3} per user (up {5 + quarter}% QoQ). Recommend "
        f"shifting {10 + quarter}% of paid acquisition budget to content marketing "
        f"which shows {3 + quarter * 0.5:.1f}x better LTV:CAC ratio.\n\n"
        f"3. Competitive response needed in {'enterprise' if quarter < 5 else 'mid-market'} "
        f"segment. Recommend accelerating the {'API platform' if quarter < 4 else 'integration hub'} "
        f"roadmap by {4 - min(quarter, 3)} weeks to maintain differentiation.\n\n"
        f"4. Risk factors: regulatory changes in EU may impact {5 + quarter}% of "
        f"revenue. Recommend proactive compliance review within {8 - quarter} weeks. "
        f"Currency exposure on international contracts needs hedging strategy update."
    )


def main() -> None:

    # =================================================================
    # 1. Per-Stage Token Budgets
    # =================================================================
    # Different workflow stages need different context sizes:
    #   Research  -- generous budget, broad information gathering
    #   Generation -- moderate budget, focused on the task
    #   Review    -- tight budget, only critical context

    print("=" * 60)
    print("1. Per-Stage Token Budgets")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10000))
    with Tract.open(config=config) as t:

        # --- Stage: Research (generous 10K budget) ---
        t.system("You are a market research analyst.")
        t.directive("research", "Gather comprehensive information on competitors.")
        t.configure(compile_strategy="full")

        # Simulate research data gathering -- each finding is substantial
        research_data = [
            ("Competitor A: enterprise analytics",
             "Competitor A: $50M ARR, 30% market share, strong enterprise presence. "
             "Key product: unified analytics platform with 200+ integrations. "
             "Recently launched AI copilot feature. 500+ enterprise customers. "
             "Average deal size $120K. Primary market: North America and Europe. "
             "Founded 2015, 450 employees, Series D at $800M valuation."),
            ("Competitor B: SMB dashboards",
             "Competitor B: $25M ARR, 15% market share, SMB-focused self-serve model. "
             "Key product: self-serve dashboards with freemium tier. 50K+ free users "
             "with 3% conversion to paid. Average deal size $5K. Growing 80% YoY. "
             "Strong community and marketplace. Founded 2018, 120 employees."),
            ("Competitor C: data platform",
             "Competitor C: $80M ARR, 40% market share, full platform play. "
             "Key product: integrated data warehouse + BI + ML pipeline. "
             "Series D funded at $2B valuation. 200 enterprise accounts. "
             "Average deal size $400K. Lock-in through proprietary query language. "
             "Founded 2012, 800 employees, IPO planned for next year."),
            ("Market trends",
             "Market trends: consolidation toward platforms accelerating -- 5 acquisitions "
             "in last quarter alone. AI-assisted analysis tools growing 45% YoY. "
             "Privacy regulations (GDPR, CCPA) driving demand for on-prem and VPC "
             "deployment options. Total addressable market estimated at $15B by 2027."),
        ]
        for msg, data in research_data:
            t.user(data, message=msg)
            t.assistant(f"Recorded: {msg}.")

        research_status = t.status()
        print(f"\n  Research stage:")
        print(f"    Budget: {research_status.token_budget_max} tokens")
        print(f"    Used:   {research_status.token_count} tokens")
        print(f"    Usage:  {research_status.token_count / research_status.token_budget_max:.0%}")

        # --- Pin critical finding before compression ---
        # The market trends finding is essential -- pin it to survive compression
        log_entries = t.log()
        for entry in log_entries:
            if entry.message and "Market trends" in (entry.message or ""):
                t.annotate(entry.commit_hash, Priority.PINNED,
                           reason="Critical market trend data")
                break

        # --- Stage: Generation (compress research, focus budget) ---
        t.transition("generation")
        t.configure(compile_strategy="full")

        # Compress the research into a tight summary to free budget
        result = t.compress(
            content=(
                "Market research summary: Three main competitors. "
                "A ($50M, enterprise, AI copilot), B ($25M, SMB, freemium), "
                "C ($80M, platform, IPO planned). TAM $15B by 2027. "
                "Trends: platform consolidation, AI +45% YoY, on-prem demand."
            ),
        )
        print(f"\n  Generation stage (after compressing research):")
        result.pprint()
        gen_status = t.status()
        gen_status.pprint()

    # =================================================================
    # 2. Auto-Compression Threshold
    # =================================================================
    # Configure a token threshold so you know when to compress.
    # The threshold is an absolute token count stored in the DAG config.
    # Your application code checks status().token_count against it.

    print(f"\n{'=' * 60}")
    print("2. Auto-Compression Threshold Pattern")
    print("=" * 60)

    BUDGET = 600
    THRESHOLD = 450  # compress when context exceeds ~75% of budget

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=BUDGET))
    with Tract.open(config=config) as t:
        t.system("You are a senior software architect conducting a code review.")
        t.configure(auto_compress_threshold=THRESHOLD)

        threshold = t.get_config("auto_compress_threshold")
        print(f"\n  Budget: {BUDGET} tokens, threshold: {threshold} tokens "
              f"({threshold / BUDGET:.0%})")

        # Simulate a multi-file code review -- each review is substantial
        reviews = [
            ("auth module",
             "Reviewed auth/login.py (450 lines). Issues found: password hashing "
             "uses MD5 instead of bcrypt -- critical security vulnerability. Session "
             "tokens have no expiry -- tokens persist indefinitely after logout. "
             "Rate limiting is client-side only, easily bypassed. SQL queries "
             "concatenate user input directly -- injection risk. Recommend: migrate "
             "to bcrypt with work factor 12, add 24h token expiry with refresh "
             "rotation, implement server-side rate limiting with Redis, and use "
             "parameterized queries throughout.",
             "Auth: critical vulns (MD5, no expiry, SQL injection). Fix priority: P0."),
            ("payment module",
             "Reviewed payments/processor.py (380 lines). Issues: retry logic has "
             "no exponential backoff -- hammers payment provider on failure. "
             "Idempotency keys are timestamp-based -- collision risk at scale. "
             "PCI-sensitive card data logged at DEBUG level -- compliance violation. "
             "Webhook signature validation skipped in test mode but test mode flag "
             "is checked via query parameter -- attackable. Recommend: implement "
             "exponential backoff with jitter (1s, 2s, 4s, max 30s), use UUID v4 "
             "for idempotency, scrub all card data from logs, and validate webhook "
             "signatures unconditionally.",
             "Payments: retry, idempotency, PCI logging, webhook bypass. Priority: P0."),
            ("API layer",
             "Reviewed api/routes.py and api/middleware.py (620 lines total). "
             "Issues: CORS allows wildcard origin in production config. API "
             "versioning uses URL path but v1 and v2 share the same handler with "
             "if/else branches -- maintenance nightmare. Response serialization "
             "includes internal database IDs. Pagination uses offset-based approach "
             "which degrades at scale. Error responses leak stack traces. Recommend: "
             "restrict CORS to known origins, separate versioned handlers into "
             "distinct modules, add response DTOs to filter internal fields, migrate "
             "to cursor-based pagination, and sanitize error responses.",
             "API: CORS wildcard, version branching, ID leaks, pagination. Priority: P1."),
            ("data layer",
             "Reviewed data/models.py and data/queries.py (510 lines). Issues: "
             "N+1 query pattern in user list endpoint -- loads each user's profile "
             "separately. No database connection pooling -- new connection per request. "
             "Migrations are not idempotent -- re-running creates duplicate columns. "
             "Foreign key constraints missing on 4 tables. Soft deletes implemented "
             "inconsistently -- some use is_deleted flag, others use deleted_at timestamp. "
             "Recommend: add eager loading with joinedload(), configure connection pool "
             "(min=5, max=20), make migrations idempotent with IF NOT EXISTS, add "
             "foreign keys with migration, and standardize on deleted_at pattern.",
             "Data: N+1 queries, no pooling, non-idempotent migrations, missing FKs. Priority: P1."),
            ("caching layer",
             "Reviewed cache/redis_client.py and cache/strategies.py (290 lines). "
             "Issues: no cache invalidation strategy -- stale data served indefinitely. "
             "Cache keys use plain string concatenation -- collision between different "
             "entity types possible. TTL set globally at 1 hour regardless of data "
             "volatility. No circuit breaker -- Redis failure cascades to all requests. "
             "Cache warming on deploy not implemented -- cold start latency spike. "
             "Recommend: implement write-through invalidation for mutable data, use "
             "namespaced keys with entity type prefix, vary TTL by data type (5min "
             "for volatile, 1h for stable), add circuit breaker with fallback to DB, "
             "and implement background cache warming on deployment.",
             "Cache: no invalidation, key collisions, fixed TTL, no circuit breaker. Priority: P2."),
        ]

        compressed_count = 0
        for topic, review, summary in reviews:
            t.user(review, message=f"Code review: {topic}")
            t.assistant(summary, message=f"Summary: {topic}")

            status = t.status()
            pct = status.token_count / BUDGET * 100

            if status.token_count > threshold and compressed_count == 0:
                print(f"    [{topic}] {status.token_count} tokens ({pct:.0f}%) "
                      f"-- THRESHOLD ({threshold}) EXCEEDED")
                result = t.compress(
                    content=(
                        "Code review progress: auth module has critical P0 issues "
                        "(MD5 hashing, no session expiry, SQL injection). Payment "
                        "module has P0 issues (retry logic, PCI logging, webhook bypass). "
                        "API layer has P1 issues (CORS, versioning, pagination)."
                    ),
                )
                compressed_count += 1
                after = t.status()
                print(f"    Compressed: {result.original_tokens} -> "
                      f"{result.compressed_tokens} tokens")
                print(f"    Context after: {after.token_count} tokens "
                      f"({after.token_count / BUDGET:.0%})")
            elif status.token_count > threshold and compressed_count > 0:
                print(f"    [{topic}] {status.token_count} tokens ({pct:.0f}%) "
                      f"-- THRESHOLD EXCEEDED (2nd compression)")
                result = t.compress(
                    content=(
                        "Full code review: 5 modules reviewed. P0: auth (MD5, session, "
                        "SQLi), payments (retry, PCI, webhooks). P1: API (CORS, "
                        "versioning), data (N+1, pooling, migrations). P2: cache "
                        "(invalidation, TTL, circuit breaker). Total: 2250 lines."
                    ),
                )
                compressed_count += 1
                after = t.status()
                print(f"    Compressed: {result.original_tokens} -> "
                      f"{result.compressed_tokens} tokens")
                print(f"    Context after: {after.token_count} tokens "
                      f"({after.token_count / BUDGET:.0%})")
            else:
                print(f"    [{topic}] {status.token_count} tokens ({pct:.0f}%)")

        final = t.status()
        print(f"\n  Final: {final.token_count} / {BUDGET} tokens ({final.token_count / BUDGET:.0%})")
        print(f"  Compressions triggered: {compressed_count}")

    # =================================================================
    # 3. Compile Strategy Optimization
    # =================================================================
    # Three strategies control how tract builds the LLM context window:
    #   full     -- every commit at full detail (maximum fidelity)
    #   messages -- commit messages only (minimal tokens, for summaries)
    #   adaptive -- last K commits full, older ones as messages only
    #
    # The key: use explicit message= parameters on commits so that the
    # messages-only strategy has short summaries while full has the
    # complete content. This gives meaningful token savings.

    print(f"\n{'=' * 60}")
    print("3. Compile Strategy Comparison")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a senior data analyst.")

        # Build conversation with long-form content and short commit messages.
        # The message= parameter sets the commit message (used by messages strategy).
        # The full text is the actual content (used by full strategy).
        for i in range(8):
            q = i + 1
            t.user(
                _research_report(q),
                message=f"Q{q} research report: ${2.1 + q * 0.8:.1f}M revenue, "
                        f"{50 + q * 12}K MAU",
            )
            t.assistant(
                _analysis_response(q),
                message=f"Q{q} analysis: revenue {'above' if q > 3 else 'on'} plan, "
                        f"growth healthy, competitive response needed",
            )

        total_commits = len(t.log())
        print(f"\n  Built conversation: {total_commits} commits "
              f"(8 Q&A turns + system)")

        # Strategy: full -- every commit with complete content
        ctx_full = t.compile(strategy="full")
        print(f"\n  Strategy: full")
        ctx_full.pprint(style="compact")

        # Strategy: messages -- commit messages only (short summaries)
        ctx_msg = t.compile(strategy="messages")
        print(f"\n  Strategy: messages")
        ctx_msg.pprint(style="compact")

        # Strategy: adaptive -- last K full, rest as messages
        ctx_adaptive_5 = t.compile(strategy="adaptive", strategy_k=5)
        print(f"\n  Strategy: adaptive (k=5, last 5 commits full)")
        ctx_adaptive_5.pprint(style="compact")

        ctx_adaptive_3 = t.compile(strategy="adaptive", strategy_k=3)
        print(f"\n  Strategy: adaptive (k=3, tighter)")
        ctx_adaptive_3.pprint(style="compact")

        # Comparison table
        print(f"\n  {'Strategy':<25} {'Messages':>8} {'Tokens':>8} {'Savings':>10}")
        print(f"  {'-' * 53}")
        baseline = ctx_full.token_count
        for label, ctx in [
            ("full (baseline)", ctx_full),
            ("adaptive (k=5)", ctx_adaptive_5),
            ("adaptive (k=3)", ctx_adaptive_3),
            ("messages only", ctx_msg),
        ]:
            savings = baseline - ctx.token_count
            pct = (savings / baseline * 100) if baseline else 0
            sign = f"-{pct:.0f}%" if savings > 0 else "---"
            print(f"  {label:<25} {len(ctx.messages):>8} "
                  f"{ctx.token_count:>8} {sign:>10}")

        print(f"\n  Takeaway: 'adaptive' gives a sliding scale between fidelity")
        print(f"  and token cost. Recent context stays detailed while older")
        print(f"  context shrinks to summaries automatically.")

    # =================================================================
    # 4. Progressive Compression for Long Workflows
    # =================================================================
    # In long-running agent loops, progressively compress older work.
    # Pin critical findings so they survive compression verbatim.

    print(f"\n{'=' * 60}")
    print("4. Progressive Compression with Pinned Findings")
    print("=" * 60)

    BUDGET = 600

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=BUDGET))
    with Tract.open(config=config) as t:
        t.system("You are a research agent investigating supply chain risks.")

        # Workflow: investigation steps with detailed reports
        steps = [
            ("Supplier audit",
             "Comprehensive supplier audit completed across 15 tier-1 suppliers. "
             "Three flagged for elevated risk: Acme Corp (financial instability, "
             "risk score 7.2/10 -- declining revenue and recent credit downgrade), "
             "Beta Industries (compliance gaps, risk score 6.8/10 -- pending "
             "regulatory actions in EU and US), and Gamma Manufacturing (capacity "
             "constraints, risk score 8.1/10 -- near full utilization with no "
             "expansion plans). Remaining 12 suppliers scored below 4.0 risk threshold.",
             "Audit: 15 suppliers, 3 flagged (Acme 7.2, Beta 6.8, Gamma 8.1)"),
            ("Financial deep-dive",
             "Deep financial analysis of flagged suppliers. Acme Corp: cash reserves "
             "declined 40% YoY, debt-to-equity ratio at 3.2x (industry avg 1.5x), "
             "accounts receivable aging beyond 90 days at 35% of total. Beta Industries: "
             "financially stable with 18-month runway, but $4.2M provision for "
             "regulatory fines. Gamma Manufacturing: profitable with 15% margins "
             "but capex frozen by parent company. Working capital ratio 0.9 (below "
             "healthy threshold of 1.2).",
             "Financial: Acme cash -40%, Beta stable w/ fine risk, Gamma capex frozen"),
            ("Compliance review",
             "Regulatory compliance audit results. Beta Industries: 2 open FDA "
             "warning letters (21 CFR 820 and 21 CFR 211), 1 EPA consent decree "
             "for wastewater violations, EU MDR compliance deadline missed by 4 months. "
             "Estimated remediation cost: $6-8M. Timeline: 12-18 months. Gamma "
             "Manufacturing: passed all inspections, ISO 13485 and ISO 9001 current. "
             "Acme Corp: minor labeling nonconformities, corrective action underway.",
             "Compliance: Beta 2 FDA + 1 EPA violations, Gamma clean, Acme minor"),
            ("Capacity modeling",
             "Capacity and resilience analysis. Gamma Manufacturing operates at 95% "
             "utilization with no facility expansion approved. Single-source supplier "
             "for 3 critical components (part numbers GX-4401, GX-4402, GX-4410). "
             "Average lead time: 12 weeks. No buffer stock held. Geographic "
             "concentration risk: all production in one facility in Shenzhen. "
             "Natural disaster or trade policy change could halt supply for 8-16 weeks.",
             "Capacity: Gamma at 95%, single-source for 3 parts, Shenzhen risk"),
            ("Risk scoring",
             "Final consolidated risk scores. Acme Corp: HIGH (7.2 financial). "
             "Beta Industries: HIGH (6.8 regulatory, escalating to 8.0+ if EU fines "
             "materialize). Gamma Manufacturing: CRITICAL (8.1 capacity + single-source "
             "concentration). Immediate action required: dual-sourcing for Gamma "
             "components is the top priority. Estimated business impact of Gamma "
             "failure: $12M revenue at risk, 3-month production halt.",
             "RISK SCORES: Acme HIGH, Beta HIGH, Gamma CRITICAL. $12M at risk."),
            ("Mitigation plan",
             "Three-phase mitigation strategy. Phase 1 (immediate, 8 weeks): qualify "
             "alternate supplier for Gamma critical components GX-4401/4402/4410. "
             "Begin with Delta Corp (pre-qualified, 85% capability match). Build "
             "2-week buffer stock. Phase 2 (3 months): renegotiate Acme payment "
             "terms, require escrow for advance payments, add financial health "
             "monitoring clause. Phase 3 (6 months): monitor Beta EU regulatory "
             "outcome, prepare qualified alternate for Beta components if fines "
             "exceed $6M. Total mitigation budget: $2.4M.",
             "PLAN: Ph1 dual-source Gamma (8wk), Ph2 Acme escrow (3mo), Ph3 Beta monitor (6mo)"),
        ]

        compression_events = []
        pinned_findings = []

        for i, (step_name, finding, short_msg) in enumerate(steps):
            # Commit the finding with explicit short message
            ci = t.user(finding, message=short_msg)
            t.assistant(
                f"Acknowledged: {step_name} analysis complete. "
                f"Findings integrated into risk model.",
                message=f"Ack: {step_name}",
            )

            # Pin critical findings so they survive compression verbatim.
            # Early pins (supplier audit) test that compression preserves them.
            # Late pins (risk scoring, mitigation) test they persist in final context.
            if step_name in ("Supplier audit", "Risk scoring", "Mitigation plan"):
                t.annotate(ci.commit_hash, Priority.PINNED,
                           reason=f"Critical: {step_name}")
                pinned_findings.append(step_name)

            status = t.status()
            pct = status.token_count / BUDGET * 100

            # Compress when approaching budget
            if status.token_count > BUDGET * 0.80:
                before = status.token_count
                result = t.compress(
                    content=(
                        f"Supply chain investigation (steps 1-{i + 1}): "
                        f"Audited 15 suppliers, 3 flagged. Key risks: Acme (financial, "
                        f"HIGH), Beta (regulatory, HIGH), Gamma (capacity, CRITICAL). "
                        f"Gamma single-source for 3 components, $12M revenue at risk."
                    ),
                )
                after = t.status()
                compression_events.append({
                    "step": step_name,
                    "before": before,
                    "after": after.token_count,
                    "preserved": len(result.preserved_commits),
                })
                print(f"    Step {i + 1} [{step_name}]: {before} -> "
                      f"{after.token_count} tokens "
                      f"(compressed, {len(result.preserved_commits)} pinned preserved)")
            else:
                print(f"    Step {i + 1} [{step_name}]: "
                      f"{status.token_count} tokens ({pct:.0f}%)")

        final = t.status()
        print(f"\n  Final state:")
        print(f"    Tokens: {final.token_count} / {BUDGET} "
              f"({final.token_count / BUDGET:.0%})")
        print(f"    Commits: {final.commit_count}")
        print(f"    Compression events: {len(compression_events)}")
        print(f"    Pinned findings: {', '.join(pinned_findings)}")

        # Verify pinned content survived compression
        ctx = t.compile()
        pinned_count = sum(1 for p in ctx.priorities if p == "pinned")
        print(f"    Pinned messages in final context: {pinned_count}")

    # =================================================================
    # 5. Token Tracking and Budget Report
    # =================================================================
    # Build a budget report across a multi-stage workflow.
    # Track tokens consumed and saved at each stage.

    print(f"\n{'=' * 60}")
    print("5. Token Tracking and Budget Report")
    print("=" * 60)

    BUDGET = 5000

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=BUDGET))
    with Tract.open(config=config) as t:
        stages = {}

        # --- Stage: Planning ---
        t.system("You are a project planner for a recommendation engine.")
        t.user(
            "We need to build a recommendation engine for our e-commerce platform. "
            "Key requirements: real-time scoring with P99 under 50ms, A/B testing "
            "support for model variants, user segmentation across 12 behavioral "
            "cohorts, integration with our existing Snowflake data warehouse, "
            "support for 50M daily events, and compliance with GDPR for EU users. "
            "Budget: $500K for infrastructure, 6-month timeline, team of 5 engineers.",
            message="Requirements: rec engine, P99 <50ms, A/B, segments, Snowflake, GDPR",
        )
        t.assistant(
            "I'll break this into five phases with clear milestones. Phase 1: Data "
            "pipeline (Kafka -> Flink streaming, Snowflake batch, feature store). "
            "Phase 2: Model training (XGBoost baseline -> neural CF -> transformer "
            "ranker). Phase 3: Serving infrastructure (TF Serving, model registry, "
            "A/B framework). Phase 4: Integration (API gateway, client SDKs, "
            "monitoring). Phase 5: Optimization (latency tuning, cost optimization, "
            "automated retraining). Each phase has 4-6 week duration with 2-week "
            "buffer for the overall timeline.",
            message="Plan: 5 phases (pipeline, training, serving, integration, optimization)",
        )

        status = t.status()
        stages["planning"] = {
            "tokens": status.token_count,
            "commits": status.commit_count,
        }

        # --- Stage: Design ---
        t.user(
            "Design the data pipeline. We have 50M events/day in Kafka across 8 "
            "topics (page_view, add_to_cart, purchase, search, click, impression, "
            "wishlist, review). Need feature store with <100ms P99 read latency "
            "for online serving, batch retraining daily at 2am UTC. Features needed: "
            "user_click_rate_7d, item_popularity_30d, category_affinity_score, "
            "session_depth, cart_abandonment_rate, price_sensitivity_index, "
            "recency_weighted_purchases, cross_category_exploration_score.",
            message="Design: data pipeline, 50M events/day, 8 features, <100ms reads",
        )
        t.assistant(
            "Pipeline architecture: Kafka -> Flink (streaming feature computation) -> "
            "Redis Cluster (online feature store, 3 shards, 6 replicas). Daily batch: "
            "Kafka -> Spark on EMR (6x r5.2xlarge) -> S3 (Parquet, partitioned by date) "
            "-> training pipeline trigger. Feature store: Feast with Redis online store "
            "(TTL 48h) and S3 offline store. Feature freshness SLA: streaming features "
            "<5min, batch features <2h after daily run. Estimated infra cost: $8.5K/month "
            "for Flink (3x m5.xlarge), $3.2K/month for Redis, $1.8K/month for EMR batch.",
            message="Design: Kafka->Flink->Redis, Spark batch, Feast store, $13.5K/mo",
        )
        t.user(
            "Design the serving layer. Requirements: P99 <50ms end-to-end (including "
            "feature fetch + model inference + post-processing), 10K concurrent RPS at "
            "peak, model versioning with instant rollback, gradual rollout (canary) "
            "support, shadow mode for new models, and multi-armed bandit for A/B "
            "allocation. Need to serve 3 model variants simultaneously.",
            message="Design: serving layer, P99 <50ms, 10K RPS, canary + shadow + MAB",
        )
        t.assistant(
            "Serving stack: TensorFlow Serving behind Envoy proxy (L7 load balancing "
            "with header-based routing for model variants). Model registry: MLflow with "
            "S3 artifact store and PostgreSQL metadata. Deployment: Kubernetes with "
            "Istio service mesh for traffic splitting (canary: 5% -> 25% -> 50% -> 100% "
            "over 4 hours). Shadow mode: duplicate traffic to shadow deployment, log "
            "predictions without serving. A/B: Thompson Sampling bandit with 1-hour "
            "update intervals. Estimated serving cost: $12K/month (6x g4dn.xlarge for "
            "GPU inference, 3x c5.2xlarge for Envoy, managed K8s $2K).",
            message="Design: TF Serving + Envoy + Istio, MLflow registry, $12K/mo",
        )

        status = t.status()
        stages["design"] = {
            "tokens": status.token_count,
            "commits": status.commit_count,
            "delta": status.token_count - stages["planning"]["tokens"],
        }

        # Compress planning + early design to free budget for implementation
        result = t.compress(
            content=(
                "Project: recommendation engine for e-commerce. Budget $500K, 6mo, "
                "5 engineers. 5 phases: pipeline, training, serving, integration, "
                "optimization. Pipeline: Kafka->Flink->Redis (streaming), "
                "Kafka->Spark->S3 (batch). Feature store: Feast ($13.5K/mo). "
                "Serving: TF Serving + Envoy + Istio, MLflow, canary + shadow + "
                "Thompson Sampling. P99 <50ms, 10K RPS. Serving cost: $12K/mo."
            ),
        )

        post_compress = t.status()
        stages["design"]["compressed_to"] = post_compress.token_count
        stages["design"]["tokens_saved"] = status.token_count - post_compress.token_count

        # --- Stage: Implementation (multiple detailed exchanges) ---
        impl_exchanges = [
            (
                "Implement the feature store. Need these features computed and served: "
                "user_click_rate_7d (streaming, 5-min window), item_popularity_30d "
                "(batch, daily), category_affinity_score (batch, user x category matrix), "
                "session_depth (streaming, real-time). Include monitoring for feature "
                "freshness, data quality, and schema drift detection.",
                "Implement: feature store with 4 features + monitoring",
                "Feature store implementation complete. Feast feature definitions deployed "
                "to production repo. Streaming features (click_rate, session_depth) via "
                "Flink SQL with tumbling windows. Batch features (popularity, affinity) "
                "via daily Spark job. Redis online store TTL=48h. Monitoring stack: "
                "Prometheus metrics for freshness (alert if >10min stale for streaming, "
                ">3h for batch), Great Expectations for data quality checks (null rates, "
                "value ranges, distribution drift), and custom schema registry with "
                "backward compatibility enforcement. Feature backfill completed for "
                "90 days of historical data.",
                "Done: feature store deployed, monitoring active, 90-day backfill",
            ),
            (
                "Implement model training pipeline. XGBoost baseline first, then neural "
                "collaborative filtering. Use Optuna for hyperparameter tuning across "
                "learning rate, layer sizes, dropout, and batch size. Daily retraining "
                "at 2am UTC. Need model validation gates: AUC > 0.80, latency < 30ms, "
                "no feature drift beyond 2 standard deviations. MLflow for experiment "
                "tracking and model registry. Automated rollback if production metrics "
                "degrade more than 5% within 1 hour of deployment.",
                "Implement: model training pipeline, XGBoost + NCF, Optuna, MLflow",
                "Training pipeline deployed. Phase 1: XGBoost baseline achieves AUC 0.82 "
                "on holdout set (50K users, 7-day prediction window). Inference latency "
                "18ms at P99. Phase 2: Neural collaborative filtering with PyTorch, "
                "architecture [256, 128, 64] with 0.2 dropout. Optuna found optimal "
                "lr=0.001, batch_size=512 after 200 trials. AUC 0.87, latency 24ms P99. "
                "MLflow tracking active with 3 registered model versions. Validation "
                "gates configured: CI/CD blocks deployment if AUC < 0.80 or latency "
                "> 30ms. Automated canary rollback via Istio traffic rules.",
                "Done: XGBoost AUC 0.82, NCF AUC 0.87, MLflow + validation gates active",
            ),
            (
                "Implement the A/B testing framework. Need Thompson Sampling for "
                "multi-armed bandit allocation, support for 3 concurrent experiments, "
                "statistical significance calculator with Bayesian approach, and "
                "real-time dashboard for experiment monitoring. Integration with "
                "existing analytics pipeline for downstream metric tracking (revenue "
                "per user, conversion rate, session duration, cart value).",
                "Implement: A/B framework, Thompson Sampling, 3 concurrent experiments",
                "A/B framework deployed. Thompson Sampling bandit with Beta-Binomial "
                "conjugate prior, 1-hour update intervals. Supports 3 concurrent "
                "experiments with traffic isolation. Bayesian significance calculator "
                "with 95% credible interval reporting. Real-time Grafana dashboard "
                "showing: arm allocation percentages, cumulative reward curves, "
                "posterior distributions, and expected loss metrics. Analytics "
                "integration complete: revenue_per_user, conversion_rate, "
                "session_duration, and cart_value all flowing through Kafka to "
                "experiment attribution service. First experiment (NCF vs XGBoost) "
                "running with 10% traffic allocation.",
                "Done: A/B live, Thompson Sampling, Grafana dashboard, first experiment running",
            ),
        ]
        for user_text, user_msg, asst_text, asst_msg in impl_exchanges:
            t.user(user_text, message=user_msg)
            t.assistant(asst_text, message=asst_msg)

        status = t.status()
        stages["implementation"] = {
            "tokens": status.token_count,
            "commits": status.commit_count,
            "delta": status.token_count - post_compress.token_count,
        }

        # --- Budget Report ---
        print(f"\n  {'Stage':<20} {'Tokens':>8} {'Commits':>8}   {'Notes'}")
        print(f"  {'-' * 65}")
        for stage_name, info in stages.items():
            notes = ""
            if "tokens_saved" in info:
                notes = f"compressed, saved {info['tokens_saved']} tokens"
            elif "delta" in info:
                notes = f"+{info['delta']} tokens this stage"
            print(f"  {stage_name:<20} {info['tokens']:>8} {info['commits']:>8}   "
                  f"{notes}")

        final = t.status()
        print(f"\n  Budget summary:")
        print(f"    Total budget:     {BUDGET} tokens")
        print(f"    Current usage:    {final.token_count} tokens "
              f"({final.token_count / BUDGET:.0%})")
        print(f"    Remaining:        {BUDGET - final.token_count} tokens")
        print(f"    Total commits:    {final.commit_count}")
        if "tokens_saved" in stages.get("design", {}):
            print(f"    Tokens saved by compression: "
                  f"{stages['design']['tokens_saved']}")

        # --- Strategy comparison on the final context ---
        print(f"\n  Strategy efficiency on final context:")
        print(f"    {'Strategy':<25} {'Tokens':>8} {'Savings':>10}")
        print(f"    {'-' * 45}")
        for strategy, k_val in [
            ("full", None),
            ("adaptive", 5),
            ("adaptive", 3),
            ("messages", None),
        ]:
            kwargs = {"strategy": strategy}
            if k_val is not None:
                kwargs["strategy_k"] = k_val
            ctx = t.compile(**kwargs)
            label = strategy if k_val is None else f"{strategy} (k={k_val})"
            savings = final.token_count - ctx.token_count
            if savings > 0:
                print(f"    {label:<25} {ctx.token_count:>8} "
                      f"{f'-{savings} ({savings / final.token_count:.0%})':>10}")
            else:
                print(f"    {label:<25} {ctx.token_count:>8}   (baseline)")

    # =================================================================
    # 6. Adaptive Strategy with recent_ratio
    # =================================================================
    # Instead of a fixed strategy_k, use recent_ratio to keep a
    # percentage of recent commits at full detail. This scales
    # automatically as the conversation grows.
    #
    # Use ratio when: context length varies and you want a consistent
    #   proportion of full-detail commits (e.g., "always keep the last 70%").
    # Use fixed k when: you want an exact number of recent commits at
    #   full detail regardless of conversation length.

    print(f"\n{'=' * 60}")
    print("6. Adaptive Strategy with recent_ratio")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a senior data analyst.")

        # Build a conversation with 8 Q&A turns (16 dialogue commits + 1 system = 17)
        for i in range(8):
            q = i + 1
            t.user(
                _research_report(q),
                message=f"Q{q} research report: ${2.1 + q * 0.8:.1f}M revenue, "
                        f"{50 + q * 12}K MAU",
            )
            t.assistant(
                _analysis_response(q),
                message=f"Q{q} analysis: revenue {'above' if q > 3 else 'on'} plan, "
                        f"growth healthy, competitive response needed",
            )

        total_commits = len(t.log())
        print(f"\n  Built conversation: {total_commits} commits")

        # --- recent_ratio=0.7: keep last 70% of commits at full detail ---
        ctx_ratio_70 = t.compile(strategy="adaptive", recent_ratio=0.7)
        print(f"\n  Strategy: adaptive (recent_ratio=0.7, ~70% full detail)")
        ctx_ratio_70.pprint(style="compact")

        # --- Compare with fixed strategy_k ---
        # With 17 effective commits, ratio=0.7 -> k = max(1, int(17*0.7)) = 11
        # A fixed k=5 keeps only the last 5 at full detail
        ctx_fixed_k5 = t.compile(strategy="adaptive", strategy_k=5)
        print(f"\n  Strategy: adaptive (fixed strategy_k=5)")
        ctx_fixed_k5.pprint(style="compact")

        # --- recent_ratio=1.0: all commits at full detail ---
        ctx_ratio_100 = t.compile(strategy="adaptive", recent_ratio=1.0)
        ctx_full = t.compile(strategy="full")
        print(f"\n  Strategy: adaptive (recent_ratio=1.0) vs full")
        print(f"    ratio=1.0 tokens: {ctx_ratio_100.token_count}")
        print(f"    full tokens:      {ctx_full.token_count}")

        # --- recent_ratio=0.0: summarize everything (at least 1 stays full) ---
        ctx_ratio_0 = t.compile(strategy="adaptive", recent_ratio=0.0)
        ctx_k1 = t.compile(strategy="adaptive", strategy_k=1)
        print(f"\n  Strategy: adaptive (recent_ratio=0.0) vs strategy_k=1")
        print(f"    ratio=0.0 tokens: {ctx_ratio_0.token_count}")
        print(f"    k=1 tokens:       {ctx_k1.token_count}")

        # --- recent_ratio overrides strategy_k when both are set ---
        ctx_both = t.compile(strategy="adaptive", strategy_k=2, recent_ratio=0.7)
        print(f"\n  Both set: strategy_k=2, recent_ratio=0.7")
        print(f"    Tokens (recent_ratio wins): {ctx_both.token_count}")
        print(f"    Same as ratio=0.7 alone:    {ctx_ratio_70.token_count}")

        # Comparison table
        baseline = ctx_full.token_count
        print(f"\n  {'Strategy':<35} {'Tokens':>8} {'Savings':>10}")
        print(f"  {'-' * 55}")
        for label, ctx in [
            ("full (baseline)", ctx_full),
            ("adaptive (recent_ratio=1.0)", ctx_ratio_100),
            ("adaptive (recent_ratio=0.7)", ctx_ratio_70),
            ("adaptive (fixed k=5)", ctx_fixed_k5),
            ("adaptive (recent_ratio=0.0)", ctx_ratio_0),
        ]:
            savings = baseline - ctx.token_count
            pct = (savings / baseline * 100) if baseline else 0
            sign = f"-{pct:.0f}%" if savings > 0 else "---"
            print(f"  {label:<35} {ctx.token_count:>8} {sign:>10}")

        print(f"\n  Takeaway: recent_ratio scales with conversation length.")
        print(f"  Use it when you want a consistent proportion of full-detail")
        print(f"  commits rather than a fixed count.")

    print(f"\n{'=' * 60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
