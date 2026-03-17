"""Optimization -- Token Budgets, Compile Strategies, and Prompt Caching

Fine-grained control over context window size and cost using tract's compile
strategies, compression, priority annotations, and Anthropic prompt caching.

Sections:
  1. Per-Stage Token Budgets         -- research / generation budgets
  2. Auto-Compression Triggers       -- threshold-based compression
  3. Compile Strategies              -- full vs messages vs adaptive vs ratio
  4. Progressive Compression         -- long workflows with pinned findings
  5. Prompt Caching                  -- cache_control breakpoints + priorities

Demonstrates: TokenBudgetConfig, compile(strategy=), compress(), annotate(),
              Priority, status(), transition(), to_anthropic(cache_control=True)

No LLM required -- all compression uses manual content= mode.
"""

from tract import Priority, Tract, TractConfig, TokenBudgetConfig


def _report(q: int) -> str:
    """Generate a multi-paragraph report for compile strategy demos."""
    rev = 2.1 + q * 0.8
    return (
        f"Q{q} Analysis: ${rev:.1f}M revenue ({8 + q}% QoQ growth). "
        f"Enterprise contributed {55 + q}% of revenue. {50 + q * 12}K MAU, "
        f"{max(1.2, 4.5 - q * 0.3):.1f}% churn. NPS {32 + q * 3}. "
        f"Competitor pricing changes affected {10 + q * 2}% of accounts."
    )


# ===================================================================
# 1. Per-Stage Token Budgets
# ===================================================================

def per_stage_budgets() -> None:
    """Different workflow stages get different context sizes."""
    print("=" * 60)
    print("1. Per-Stage Token Budgets")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10000))
    with Tract.open(config=config) as t:
        t.system("You are a market research analyst.")
        t.directive("research", "Gather comprehensive competitor information.")
        t.config.set(compile_strategy="full")

        for msg, data in [
            ("Competitor A", "Competitor A: $50M ARR, 30% share, enterprise focus."),
            ("Competitor B", "Competitor B: $25M ARR, 15% share, SMB freemium."),
            ("Market trends", "TAM $15B by 2027. AI tools +45% YoY. Consolidation."),
        ]:
            t.user(data, message=msg)
            t.assistant(f"Recorded: {msg}.")

        s = t.search.status()
        print(f"\n  Research: {s.token_count}/{s.token_budget_max} tokens "
              f"({s.token_count / s.token_budget_max:.0%})")

        # Pin critical finding, then compress for next stage
        for e in t.search.log():
            if e.message and "Market" in (e.message or ""):
                t.annotations.set(e.commit_hash, Priority.PINNED)
                break

        t.transition("generation")
        result = t.compression.compress(
            content="Summary: A ($50M, enterprise), B ($25M, SMB). TAM $15B."
        )
        gs = t.search.status()
        print(f"  Generation (post-compress): {gs.token_count} tokens")
    print("PASSED\n")


# ===================================================================
# 2. Auto-Compression Threshold
# ===================================================================

def auto_compression_threshold() -> None:
    """Compress when context exceeds a configured threshold."""
    print("=" * 60)
    print("2. Auto-Compression Threshold")
    print("=" * 60)

    BUDGET, THRESHOLD = 600, 450

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=BUDGET))
    with Tract.open(config=config) as t:
        t.system("You are a code reviewer.")
        t.config.set(auto_compress_threshold=THRESHOLD)
        print(f"\n  Budget: {BUDGET}, threshold: {THRESHOLD} ({THRESHOLD/BUDGET:.0%})")

        compressed = False
        for topic, review, summary in [
            ("auth", "Auth: MD5 hashing, no session expiry, SQL injection.", "P0: auth"),
            ("payments", "Payments: no backoff, PCI logging.", "P0: payments"),
            ("API", "API: CORS wildcard, stack traces leaked.", "P1: API"),
        ]:
            t.user(review, message=f"Review: {topic}")
            t.assistant(summary, message=f"Summary: {topic}")
            s = t.search.status()
            if s.token_count > THRESHOLD and not compressed:
                r = t.compression.compress(content="Auth P0, Payments P0, API P1.")
                compressed = True
                a = t.search.status()
                print(f"    [{topic}] THRESHOLD -- compressed {r.original_tokens}"
                      f" -> {r.compressed_tokens}")
            else:
                print(f"    [{topic}] {s.token_count} tokens ({s.token_count/BUDGET:.0%})")

        print(f"  Compressed: {compressed}")
    print("PASSED\n")


# ===================================================================
# 3. Compile Strategies (full / messages / adaptive / ratio)
# ===================================================================

def compile_strategies() -> None:
    """Compare full, messages, adaptive (fixed k), adaptive (ratio)."""
    print("=" * 60)
    print("3. Compile Strategy Comparison")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a data analyst.")
        for i in range(8):
            q = i + 1
            t.user(_report(q), message=f"Q{q}: ${2.1+q*0.8:.1f}M rev")
            t.assistant(f"Q{q} analysis noted.", message=f"Q{q} ack")

        ctx_full = t.compile(strategy="full")
        ctx_msg = t.compile(strategy="messages")
        ctx_k5 = t.compile(strategy="adaptive", strategy_k=5)
        ctx_r70 = t.compile(strategy="adaptive", recent_ratio=0.7)

        baseline = ctx_full.token_count
        print(f"\n  {len(t.search.log())} commits built")
        print(f"\n  {'Strategy':<30} {'Tokens':>7} {'Savings':>8}")
        print(f"  {'-' * 47}")
        for label, ctx in [
            ("full (baseline)", ctx_full),
            ("adaptive (recent_ratio=0.7)", ctx_r70),
            ("adaptive (k=5)", ctx_k5),
            ("messages only", ctx_msg),
        ]:
            sv = baseline - ctx.token_count
            pct = f"-{sv/baseline:.0%}" if sv > 0 else "---"
            print(f"  {label:<30} {ctx.token_count:>7} {pct:>8}")

        print(f"\n  Takeaway: adaptive gives a sliding scale between fidelity")
        print(f"  and cost. recent_ratio scales proportionally with length.")
    print("PASSED\n")


# ===================================================================
# 4. Progressive Compression with Pinned Findings
# ===================================================================

def progressive_compression() -> None:
    """Pin critical findings so they survive compression."""
    print("=" * 60)
    print("4. Progressive Compression with Pinned Findings")
    print("=" * 60)

    BUDGET = 600
    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=BUDGET))
    with Tract.open(config=config) as t:
        t.system("You are a supply chain risk investigator.")

        steps = [
            ("Audit", "15 suppliers: Acme 7.2, Beta 6.8, Gamma 8.1 flagged.", True),
            ("Financial", "Acme cash -40%, Beta $4.2M fines, Gamma capex frozen.", False),
            ("Risk scoring", "Acme HIGH, Beta HIGH, Gamma CRITICAL. $12M at risk.", True),
            ("Mitigation", "Ph1 dual-source Gamma, Ph2 Acme escrow, Ph3 Beta.", True),
        ]

        for i, (name, finding, pin) in enumerate(steps):
            ci = t.user(finding, message=name)
            t.assistant(f"Ack: {name}.", message=f"Ack: {name}")
            if pin:
                t.annotations.set(ci.commit_hash, Priority.PINNED)

            s = t.search.status()
            if s.token_count > BUDGET * 0.80:
                r = t.compression.compress(
                    content=f"Steps 1-{i+1}: 3 flagged suppliers. Gamma CRITICAL."
                )
                a = t.search.status()
                print(f"    [{name}] {s.token_count}->{a.token_count} tokens "
                      f"({len(r.preserved_commits)} pinned preserved)")
            else:
                print(f"    [{name}] {s.token_count} tokens")

        ctx = t.compile()
        pinned = sum(1 for p in ctx.priorities if p == "pinned")
        print(f"\n  Final: {t.search.status().token_count}/{BUDGET} tokens, "
              f"{pinned} pinned in context")
    print("PASSED\n")


# ===================================================================
# 5. Prompt Caching (Basic + Priority-Aware + Full Workflow)
# ===================================================================

def prompt_caching() -> None:
    """Anthropic cache_control breakpoints with priority-aware placement."""
    print("=" * 60)
    print("5. Prompt Caching")
    print("=" * 60)

    # --- 5a. Basic: enable with a single flag ---
    with Tract.open() as t:
        t.system("You are a research assistant.")
        t.user("What is quantum computing?")
        t.assistant("Qubits leverage superposition and entanglement.")
        t.user("How does it compare to classical?")
        t.assistant("Classical bits are 0/1; qubits enable parallelism.")

        ctx = t.compile()
        normal = ctx.to_anthropic()
        cached = ctx.to_anthropic(cache_control=True)

        assert isinstance(normal["system"], str)
        assert isinstance(cached["system"], list)
        assert cached["system"][0]["cache_control"] == {"type": "ephemeral"}

        markers = sum(
            1 for m in cached["messages"] if isinstance(m["content"], list)
            for b in m["content"] if "cache_control" in b
        )
        assert markers == 1
        print(f"\n  Basic: system={type(cached['system']).__name__}, "
              f"markers={markers + 1} (system + 1 msg)")

    # --- 5b. Priority-aware: PINNED anchors the boundary ---
    with Tract.open() as t:
        t.system("You are a coding assistant.")
        t.user("Always use type hints.", priority="pinned")
        t.assistant("Understood.")
        t.user("Write add(a, b).")
        t.assistant("def add(a: int, b: int) -> int: return a + b")

        ctx = t.compile()
        cached = ctx.to_anthropic(cache_control=True)
        for i, msg in enumerate(cached["messages"]):
            c = msg["content"]
            has = isinstance(c, list) and any("cache_control" in b for b in c)
            if has:
                print(f"  Priority: boundary at msg[{i}] (pinned content)")

    # --- 5c. Full workflow: compress + cache ---
    with Tract.open() as t:
        t.system("You are a market analyst.")
        for q, rev in [(1, "10M"), (2, "11.5M"), (3, "13M")]:
            t.user(f"Q{q} revenue: ${rev}.")
            t.assistant(f"Q{q} noted.")
        t.compression.compress(content="Revenue: Q1 $10M, Q2 $11.5M, Q3 $13M.")
        t.user("Project Q4.")
        t.assistant("Q4 projected: $14.5-15.5M.")

        params = ctx.to_anthropic_params(cache_control=True)
        assert isinstance(params["system"], list)
        print(f"  Workflow: compress + cache, {len(params['messages'])} messages")
        print(f"  Cost: 90% discount on tokens before cache boundary")

    print("PASSED\n")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    per_stage_budgets()
    auto_compression_threshold()
    compile_strategies()
    progressive_compression()
    prompt_caching()

    print("=" * 60)
    print("Summary: Optimization Patterns")
    print("=" * 60)
    print()
    print("  Pattern                       Primitives")
    print("  ----------------------------  ----------------------------------")
    print("  Per-stage budgets             TokenBudgetConfig + transition()")
    print("  Auto-compression threshold    config.set() + status() check")
    print("  Compile strategies            compile(strategy=, strategy_k=, ratio=)")
    print("  Progressive compression       compress(content=) + PINNED")
    print("  Prompt caching                to_anthropic(cache_control=True)")
    print()
    print("Done.")


test_optimization = main

if __name__ == "__main__":
    main()
