"""Session Topology Patterns -- spawn, collapse, and multi-tract coordination.

Demonstrates how to structure multi-tract sessions using Session primitives:
spawn (create child tracts), collapse (summarize child back into parent),
and hierarchical coordination. All content is manually seeded -- no LLM
calls are made. These patterns show the *plumbing* for multi-agent
workflows; pair with real LLM calls (see agent/02_multi_agent.py
or workflows/06_adversarial_review.py) for genuine agent behavior.

All examples run locally -- no API keys needed.

Patterns:
  1. Basic Spawn-Collapse        -- parent spawns children, collapses results
  2. Parallel Workers + Selection -- 3 branches, compare, select best
  3. Pipeline (Sequential)       -- stage A -> B -> C with collapse handoff
  4. Debate Topology             -- opposing tracts, collapsed into judge
  5. Hierarchical Delegation     -- parent -> sub-parents -> workers
"""

import io
import sys

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Session


# =====================================================================
# Pattern 1: Basic Supervisor-Worker
# =====================================================================

def pattern_1_basic():
    """Supervisor spawns workers, workers do isolated work, supervisor collects."""
    print("=" * 60)
    print("Pattern 1: Basic Supervisor-Worker")
    print("=" * 60)

    with Session.open() as s:
        # Supervisor tract
        supervisor = s.create_tract(display_name="supervisor")
        supervisor.system("You are a project coordinator.")
        supervisor.user("Research three database options for our backend.")
        supervisor.assistant("I will delegate to specialist workers.")

        # Spawn 3 workers, each researching a different database
        databases = ["PostgreSQL", "MongoDB", "Redis"]
        results = {}

        for db_name in databases:
            worker = s.spawn(
                supervisor,
                purpose=f"research {db_name}",
                display_name=f"worker-{db_name.lower()}",
            )

            # Worker does isolated research (simulated via manual commits)
            worker.user(f"Research {db_name} for our use case.")
            worker.assistant(
                f"{db_name} analysis: "
                f"{'Relational, ACID compliant, mature ecosystem.' if db_name == 'PostgreSQL' else ''}"
                f"{'Document store, flexible schema, horizontal scaling.' if db_name == 'MongoDB' else ''}"
                f"{'In-memory, sub-ms latency, great for caching.' if db_name == 'Redis' else ''}"
            )

            # Collapse worker findings back to supervisor
            result = s.collapse(
                worker,
                into=supervisor,
                content=f"[{db_name} Report] Worker analyzed {db_name} and found it "
                        f"suitable for {'primary storage' if db_name == 'PostgreSQL' else 'specific use cases'}.",
                auto_commit=True,
            )
            results[db_name] = result
            print(f"  Worker {db_name}: {result.source_tokens} tokens -> "
                  f"{result.summary_tokens} token summary")

        # Supervisor now has all findings in its context
        compiled = supervisor.compile()
        text = compiled.to_text()

        # Verify all worker results are present
        for db_name in databases:
            assert db_name in text, f"Missing {db_name} in supervisor context"

        print(f"\n  Supervisor context: {compiled.commit_count} commits")
        print(f"  All 3 worker reports collected: PASSED")


# =====================================================================
# Pattern 2: Parallel Workers with Quality Gate
# =====================================================================

def pattern_2_quality_gate():
    """Spawn 3 workers for the same task with different approaches.
    Compare results and select the best one."""
    print("\n" + "=" * 60)
    print("Pattern 2: Parallel Workers with Quality Gate")
    print("=" * 60)

    with Session.open() as s:
        supervisor = s.create_tract(display_name="supervisor")
        supervisor.system("You evaluate multiple approaches to select the best one.")
        supervisor.user("Design an API endpoint for user authentication.")

        # Create branches for each approach
        approaches = {
            "jwt": "JWT token-based auth with refresh tokens and token rotation.",
            "session": "Server-side session with Redis store and CSRF protection.",
            "oauth": "OAuth2 delegated auth with PKCE flow for security.",
        }

        workers = {}
        for approach_name, description in approaches.items():
            branch_name = f"approach-{approach_name}"
            supervisor.branches.create(branch_name, switch=False)

            worker = s.spawn(
                supervisor,
                purpose=f"design {approach_name} auth",
                display_name=f"worker-{approach_name}",
            )

            # Worker designs its approach
            worker.user(f"Design authentication using {approach_name} approach.")
            worker.assistant(f"Proposed design: {description}")
            worker.user("What are the security considerations?")
            worker.assistant(
                f"Security for {approach_name}: "
                f"{'Token expiry, rotation, secure storage.' if approach_name == 'jwt' else ''}"
                f"{'Session fixation prevention, secure cookies.' if approach_name == 'session' else ''}"
                f"{'PKCE prevents interception, scoped permissions.' if approach_name == 'oauth' else ''}"
            )
            workers[approach_name] = worker

        # Supervisor compares approaches by examining each worker's context
        print("\n  Comparing approaches:")
        best_name = None
        best_depth = 0

        for name, worker in workers.items():
            compiled = worker.compile()
            depth = compiled.commit_count
            text = compiled.to_text()
            has_security = "security" in text.lower()
            print(f"    {name}: {depth} commits, "
                  f"security analysis: {'yes' if has_security else 'no'}")

            # Simple quality metric: commit depth + security coverage
            score = depth + (2 if has_security else 0)
            if score > best_depth:
                best_depth = score
                best_name = name

        print(f"\n  Selected best approach: {best_name}")

        # Collapse the winning approach into supervisor
        winner = workers[best_name]
        result = s.collapse(
            winner,
            into=supervisor,
            content=f"[Selected Approach: {best_name}] "
                    f"After comparing {len(approaches)} approaches, "
                    f"{best_name} was selected as the best design.",
            auto_commit=True,
        )

        # Verify
        final_text = supervisor.compile().to_text()
        assert best_name in final_text, "Winning approach not in supervisor context"
        assert "Selected Approach" in final_text

        print(f"  Winner collapsed into supervisor: {result.summary_tokens} tokens")
        print(f"  Quality gate applied: PASSED")


# =====================================================================
# Pattern 3: Pipeline (Sequential Workers)
# =====================================================================

def pattern_3_pipeline():
    """Worker A processes -> Worker B refines -> Worker C validates.
    Each stage collapses into the pipeline coordinator."""
    print("\n" + "=" * 60)
    print("Pattern 3: Pipeline (Sequential Workers)")
    print("=" * 60)

    with Session.open() as s:
        pipeline = s.create_tract(display_name="pipeline")
        pipeline.system("You coordinate a 3-stage data processing pipeline.")
        pipeline.user("Process customer feedback data through analysis pipeline.")

        stages = [
            {
                "name": "extractor",
                "purpose": "extract key themes from raw feedback",
                "input": "Raw feedback: 'Love the product but shipping is slow. "
                         "UI is great. Customer support needs improvement.'",
                "output": "Extracted themes: [product_quality: positive], "
                          "[shipping: negative], [ui: positive], [support: negative]",
                "summary": "Stage 1 (Extract): Identified 4 themes from raw feedback. "
                           "2 positive (product, UI), 2 negative (shipping, support).",
            },
            {
                "name": "analyzer",
                "purpose": "analyze sentiment and priority of extracted themes",
                "input": "Analyze priority of themes: product(+), shipping(-), ui(+), support(-)",
                "output": "Priority analysis: shipping(HIGH, -0.8 sentiment), "
                          "support(MEDIUM, -0.6 sentiment), product(LOW, +0.9 sentiment), "
                          "ui(LOW, +0.7 sentiment). Action items: fix shipping first.",
                "summary": "Stage 2 (Analyze): Shipping is highest priority (sentiment -0.8). "
                           "Support is medium priority. Product and UI are positive, no action needed.",
            },
            {
                "name": "validator",
                "purpose": "validate analysis and produce final recommendations",
                "input": "Validate analysis: shipping=HIGH, support=MEDIUM. Recommendations?",
                "output": "Validation passed. Recommendations: "
                          "1) Partner with faster logistics provider. "
                          "2) Expand support team by 20%. "
                          "3) Continue current product/UI trajectory.",
                "summary": "Stage 3 (Validate): Analysis validated. 3 recommendations produced: "
                           "logistics partner, expand support team, maintain product quality.",
            },
        ]

        for i, stage in enumerate(stages):
            worker = s.spawn(
                pipeline,
                purpose=stage["purpose"],
                display_name=stage["name"],
            )

            # Worker processes its stage
            worker.user(stage["input"])
            worker.assistant(stage["output"])

            # Collapse back to pipeline
            result = s.collapse(
                worker,
                into=pipeline,
                content=stage["summary"],
                auto_commit=True,
            )

            print(f"  Stage {i+1} ({stage['name']}): "
                  f"{result.source_tokens} -> {result.summary_tokens} tokens")

            # Verify pipeline accumulates context
            compiled = pipeline.compile()
            assert f"Stage {i+1}" in compiled.to_text(), \
                f"Stage {i+1} not found in pipeline context"

        # Final pipeline state
        final = pipeline.compile()
        text = final.to_text()
        assert "Stage 1" in text
        assert "Stage 2" in text
        assert "Stage 3" in text
        assert "recommendations" in text.lower()

        print(f"\n  Pipeline complete: {final.commit_count} commits in coordinator")
        print(f"  All 3 stages accumulated: PASSED")


# =====================================================================
# Pattern 4: Debate Pattern
# =====================================================================

def pattern_4_debate():
    """Two workers argue opposing positions. Supervisor synthesizes."""
    print("\n" + "=" * 60)
    print("Pattern 4: Debate Pattern")
    print("=" * 60)

    with Session.open() as s:
        judge = s.create_tract(display_name="judge")
        judge.system("You are an impartial judge synthesizing opposing arguments.")
        judge.user("Should we adopt microservices or keep our monolith?")

        # Spawn two debaters with opposing positions
        pro_micro = s.spawn(
            judge,
            purpose="argue FOR microservices",
            display_name="pro-microservices",
        )
        pro_mono = s.spawn(
            judge,
            purpose="argue FOR monolith",
            display_name="pro-monolith",
        )

        # Pro-microservices argues its case
        pro_micro.user("Present your strongest arguments for microservices.")
        pro_micro.assistant(
            "Arguments FOR microservices: "
            "1) Independent deployment reduces blast radius. "
            "2) Teams can use best-fit technology per service. "
            "3) Horizontal scaling of individual components. "
            "4) Failure isolation prevents cascade failures."
        )
        pro_micro.user("Address the counterarguments.")
        pro_micro.assistant(
            "Addressing concerns: "
            "Complexity is manageable with service mesh (Istio/Linkerd). "
            "Distributed tracing (Jaeger) solves observability. "
            "The upfront investment pays off at scale (>20 developers)."
        )

        # Pro-monolith argues its case
        pro_mono.user("Present your strongest arguments for keeping the monolith.")
        pro_mono.assistant(
            "Arguments FOR monolith: "
            "1) Simpler operational model, one deployment artifact. "
            "2) No network latency between components. "
            "3) Easier debugging with single process stack traces. "
            "4) Lower infrastructure cost, no service mesh overhead."
        )
        pro_mono.user("Address the counterarguments.")
        pro_mono.assistant(
            "Addressing concerns: "
            "Modular monolith gives you boundaries without network hops. "
            "Deployment coupling is solvable with feature flags. "
            "Most teams under 50 developers don't need microservices."
        )

        # Judge collapses both positions
        micro_result = s.collapse(
            pro_micro,
            into=judge,
            content="[PRO Microservices] Key arguments: independent deployment, "
                    "tech flexibility, horizontal scaling, failure isolation. "
                    "Acknowledges complexity but considers it manageable at scale.",
            auto_commit=True,
        )

        mono_result = s.collapse(
            pro_mono,
            into=judge,
            content="[PRO Monolith] Key arguments: operational simplicity, "
                    "no network latency, easier debugging, lower cost. "
                    "Proposes modular monolith as middle ground.",
            auto_commit=True,
        )

        # Judge synthesizes
        judge.assistant(
            "SYNTHESIS: Both positions have merit. For our team of 30 developers, "
            "a modular monolith with clear boundaries is the pragmatic choice now. "
            "Plan migration path to microservices for when team exceeds 50. "
            "KEY INSIGHT: Both sides agree on the importance of clear module boundaries."
        )

        # Verify
        final = judge.compile()
        text = final.to_text()
        assert "PRO Microservices" in text
        assert "PRO Monolith" in text
        assert "SYNTHESIS" in text

        print(f"  Pro-microservices: {micro_result.source_tokens} tokens collapsed")
        print(f"  Pro-monolith: {mono_result.source_tokens} tokens collapsed")
        print(f"  Judge synthesized from both positions")
        print(f"  Final context: {final.commit_count} commits")
        print(f"  Debate pattern complete: PASSED")


# =====================================================================
# Pattern 5: Hierarchical Delegation
# =====================================================================

def pattern_5_hierarchical():
    """Top supervisor -> sub-supervisors -> workers. Results flow up."""
    print("\n" + "=" * 60)
    print("Pattern 5: Hierarchical Delegation")
    print("=" * 60)

    with Session.open() as s:
        # Top-level supervisor
        director = s.create_tract(display_name="director")
        director.system("You are the project director overseeing the full system design.")
        director.user("Design a complete e-commerce backend: API, database, and security.")

        # Sub-supervisor for each domain
        domains = {
            "api": {
                "purpose": "design the REST API layer",
                "workers": [
                    {
                        "name": "endpoints",
                        "purpose": "define API endpoints",
                        "work": "Endpoints: GET /products, POST /cart, POST /checkout, "
                                "GET /orders/{id}. All return JSON, use pagination.",
                        "summary": "Defined 4 core endpoints with pagination and JSON responses.",
                    },
                    {
                        "name": "validation",
                        "purpose": "define input validation rules",
                        "work": "Validation: Pydantic models for all request bodies. "
                                "Rate limit: 100 req/min per user. Input sanitization on all strings.",
                        "summary": "Pydantic validation, rate limiting (100/min), input sanitization.",
                    },
                ],
                "synthesis": "API layer: 4 REST endpoints with Pydantic validation, "
                             "rate limiting, and input sanitization.",
            },
            "database": {
                "purpose": "design the database layer",
                "workers": [
                    {
                        "name": "schema",
                        "purpose": "design database schema",
                        "work": "Schema: users, products, orders, cart_items tables. "
                                "UUID primary keys, created_at/updated_at timestamps.",
                        "summary": "4 tables with UUID PKs and audit timestamps.",
                    },
                    {
                        "name": "indexing",
                        "purpose": "design database indexes",
                        "work": "Indexes: products(category, price), orders(user_id, status), "
                                "cart_items(user_id). Partial index on orders WHERE status='pending'.",
                        "summary": "Composite indexes on products, orders, cart_items. "
                                   "Partial index for pending orders.",
                    },
                ],
                "synthesis": "Database layer: 4-table schema with UUID PKs, "
                             "composite indexes, and partial indexes for performance.",
            },
            "security": {
                "purpose": "design the security layer",
                "workers": [
                    {
                        "name": "auth",
                        "purpose": "design authentication",
                        "work": "Auth: JWT with 15-min access tokens, 7-day refresh tokens. "
                                "bcrypt password hashing. Token rotation on refresh.",
                        "summary": "JWT auth with short-lived access + refresh token rotation.",
                    },
                    {
                        "name": "permissions",
                        "purpose": "design authorization",
                        "work": "Permissions: RBAC with roles (admin, seller, buyer). "
                                "Resource-level ACL for order access. Admin audit log.",
                        "summary": "RBAC with 3 roles, resource ACL, and admin audit log.",
                    },
                ],
                "synthesis": "Security layer: JWT auth with rotation, RBAC (3 roles), "
                             "resource ACL, and audit logging.",
            },
        }

        for domain_name, domain in domains.items():
            # Create sub-supervisor
            sub_sup = s.spawn(
                director,
                purpose=domain["purpose"],
                display_name=f"lead-{domain_name}",
            )
            sub_sup.user(f"Coordinate the {domain_name} design team.")

            # Sub-supervisor spawns its workers
            for worker_spec in domain["workers"]:
                worker = s.spawn(
                    sub_sup,
                    purpose=worker_spec["purpose"],
                    display_name=worker_spec["name"],
                )

                # Worker does its task
                worker.user(f"Complete your task: {worker_spec['purpose']}")
                worker.assistant(worker_spec["work"])

                # Collapse worker into sub-supervisor
                s.collapse(
                    worker,
                    into=sub_sup,
                    content=f"[{worker_spec['name']}] {worker_spec['summary']}",
                    auto_commit=True,
                )

            # Sub-supervisor synthesizes its domain
            sub_sup.assistant(f"Domain synthesis: {domain['synthesis']}")

            # Collapse sub-supervisor into director
            result = s.collapse(
                sub_sup,
                into=director,
                content=f"[{domain_name.upper()} DOMAIN] {domain['synthesis']}",
                auto_commit=True,
            )

            print(f"  {domain_name.upper()} domain: "
                  f"{len(domain['workers'])} workers -> "
                  f"sub-supervisor -> director "
                  f"({result.summary_tokens} tokens)")

        # Director has full picture
        final = director.compile()
        text = final.to_text()

        assert "API DOMAIN" in text
        assert "DATABASE DOMAIN" in text
        assert "SECURITY DOMAIN" in text

        # Count total agents created
        all_tracts = s.list_tracts()
        # director + 3 sub-supervisors + 6 workers = 10 tracts
        print(f"\n  Total agents in hierarchy: {len(all_tracts)}")
        print(f"  Director context: {final.commit_count} commits")
        print(f"  All 3 domains with 6 workers collapsed into director")
        print(f"  Hierarchical delegation: PASSED")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print()
    print("Session Topology Patterns (no LLM -- manual content)")
    print("All patterns run locally -- no API keys needed.")
    print()

    pattern_1_basic()
    pattern_2_quality_gate()
    pattern_3_pipeline()
    pattern_4_debate()
    pattern_5_hierarchical()

    print("\n" + "=" * 60)
    print("ALL PATTERNS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
