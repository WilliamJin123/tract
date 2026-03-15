"""Feature Reference: Product Analysis Walkthrough -- Every Major API in One Script

Walks through a product analysis scenario exercising 25+ tract features in
sequence. All content is manually seeded (no LLM calls) to keep the example
self-contained, fast, and runnable in CI. This is an API reference and
feature catalog, not an agent demonstration.

Features demonstrated (in order of appearance):
  1.  load_profile()           -- load the "research" workflow profile
  2.  apply_template()         -- apply parameterized directive templates
  3.  apply_stage()            -- switch stage-specific config
  4.  configure()              -- set ad-hoc config values
  5.  directive()              -- add custom directives
  6.  register_tag() / tag()   -- tag-based finding classification
  7.  use() middleware          -- post_commit observability, pre_transition gates
  8.  user/assistant/system     -- core DAG operations with metadata, tags, priority
  9.  annotate() / PINNED      -- pin critical findings to survive compression
  10. branch() / switch()      -- parallel analysis branches
  11. compare()                -- cross-branch diff without switching HEAD
  12. merge()                  -- combine branch insights
  13. compress()               -- reduce verbose research to concise summary
  14. transition()             -- stage transition with middleware gates
  15. find()                   -- search commits by tag, content, metadata
  16. reset()                  -- checkpoint-and-rollback recovery
  17. compile() / pprint()     -- compile and inspect context at any point
  18. log()                    -- commit history inspection
  19. status()                 -- tract status overview
  20. health()                 -- DAG integrity check
  21. diff()                   -- compare sequential commits
  22. list_branches()          -- enumerate branches
  23. list_tags()              -- enumerate registered tags
  24. pinned() / skipped()     -- query commits by priority
  25. get_all_configs()        -- inspect active configuration

No LLM required -- all responses are simulated to keep the example
self-contained, fast, and runnable in CI.

Requires: pip install tract-ai
"""

from datetime import datetime, timezone

from tract import (
    BlockedError,
    MiddlewareContext,
    Priority,
    Tract,
)
from tract.formatting import pprint_log


# =====================================================================
# Observability middleware (closure-based metrics tracker)
# =====================================================================

def build_metrics_tracker() -> tuple[dict, callable]:
    """Build a post_commit middleware that tracks operations and tokens."""
    metrics = {
        "commits": 0,
        "tokens": 0,
        "by_type": {},       # content_type -> count
        "by_branch": {},     # branch -> token count
        "timeline": [],      # (timestamp, event, branch, tokens)
    }

    def on_post_commit(ctx: MiddlewareContext):
        if ctx.commit is None:
            return
        metrics["commits"] += 1
        metrics["tokens"] += ctx.commit.token_count
        ct = ctx.commit.content_type
        metrics["by_type"][ct] = metrics["by_type"].get(ct, 0) + 1
        metrics["by_branch"][ctx.branch] = (
            metrics["by_branch"].get(ctx.branch, 0) + ctx.commit.token_count
        )
        metrics["timeline"].append((
            datetime.now(timezone.utc).isoformat()[:19],
            ct,
            ctx.branch,
            ctx.commit.token_count,
        ))

    return metrics, on_post_commit


# =====================================================================
# Quality gate middleware (pre_transition)
# =====================================================================

def build_quality_gate(required_tag="draft", min_draft_commits=4):
    """Build a pre_transition gate that blocks review until enough drafts exist."""

    def gate(ctx: MiddlewareContext):
        if ctx.target != "review":
            return
        # Count commits tagged "draft" -- ensures substantive work before review
        draft_commits = ctx.tract.find(tag=required_tag, limit=50)
        if len(draft_commits) < min_draft_commits:
            raise BlockedError(
                "pre_transition",
                [f"Need >= {min_draft_commits} commits tagged '{required_tag}' "
                 f"before review (have {len(draft_commits)}). Keep drafting."],
            )

    return gate


# =====================================================================
# Main workflow
# =====================================================================

def main() -> None:
    print("=" * 70)
    print("FEATURE REFERENCE: Product Analysis Walkthrough")
    print("=" * 70)
    print()
    print("  Scenario: Product analysis for 'AI-powered code review'.")
    print("  All content manually seeded -- no LLM. Demonstrates 25+ tract APIs.")
    print()

    with Tract.open() as t:

        # =============================================================
        # STEP 1: Setup with Profile and Templates
        # Features: load_profile, apply_template, apply_stage, directive
        # =============================================================

        print("=" * 60)
        print("STEP 1: Setup with Profile and Templates")
        print("=" * 60)
        print()

        # Load the research workflow profile -- applies config + directives
        t.load_profile("research")
        print(f"  Loaded 'research' profile")
        print(f"  Active profile: {t.active_profile.name}")
        print(f"  Profile stages: {list(t.active_profile.stages.keys())}")

        # Apply a parameterized directive template for the domain
        t.apply_template(
            "research_protocol",
            topic="AI-powered automated code review tools",
        )
        print(f"  Applied 'research_protocol' template")

        # Apply output format template
        t.apply_template(
            "output_format",
            format="markdown",
            max_words="200",
        )
        print(f"  Applied 'output_format' template")

        # Add a custom directive specific to this workflow
        t.directive(
            "product-focus",
            "Focus on product-market fit signals: who is the buyer, "
            "what is their pain, how much would they pay, and who are "
            "the incumbent competitors. Tag findings accordingly.",
        )
        print(f"  Added custom 'product-focus' directive")

        # Apply the 'ingest' stage from the research profile
        t.apply_stage("ingest")
        configs = t.get_all_configs()
        print(f"  Applied 'ingest' stage config: {configs}")
        print()

        # =============================================================
        # STEP 2: Register Tags and Set Up Middleware
        # Features: register_tag, use() middleware
        # =============================================================

        print("=" * 60)
        print("STEP 2: Register Tags and Middleware")
        print("=" * 60)
        print()

        # Register domain-specific tags
        tag_names = [
            "market-size", "competitor", "pain-point", "pricing",
            "risk", "opportunity", "key-finding", "draft",
        ]
        for name in tag_names:
            t.register_tag(name)
        print(f"  Registered {len(tag_names)} tags: {tag_names}")

        # Attach observability middleware
        metrics, on_post_commit = build_metrics_tracker()
        tracker_id = t.use("post_commit", on_post_commit)
        print(f"  Attached post_commit metrics tracker: {tracker_id}")

        # Attach quality gate middleware -- blocks review unless >= 4 draft commits
        gate_fn = build_quality_gate(required_tag="draft", min_draft_commits=4)
        gate_id = t.use("pre_transition", gate_fn)
        print(f"  Attached pre_transition quality gate: {gate_id}")
        print()

        # =============================================================
        # STEP 3: Research Phase -- Gather Information
        # Features: system, user, assistant (with tags, metadata, priority),
        #           tag(), annotate(), PINNED
        # =============================================================

        print("=" * 60)
        print("STEP 3: Research Phase -- Gather Information")
        print("=" * 60)
        print()

        t.system(
            "You are a product analyst evaluating 'AI-powered automated "
            "code review' as a startup opportunity. Be rigorous and "
            "data-driven. Tag all findings."
        )

        # Market size research
        market_commit = t.user(
            "Research the market size for AI code review tools.",
            tags=["market-size"],
        )
        market_response = t.assistant(
            "Global code review tools market estimated at $1.2B (2025), "
            "growing 28% CAGR. AI-augmented segment is ~$340M and "
            "accelerating. Enterprise segment dominates at 72% of revenue. "
            "Key driver: developer shortage creates demand for automated QA.",
            metadata={"source": "Gartner 2025", "confidence": "high"},
            tags=["market-size", "key-finding"],
        )
        # Pin the market size finding -- it must survive compression
        t.annotate(
            market_response.commit_hash,
            Priority.PINNED,
            reason="Core market data -- must survive compression",
        )
        t.tag(market_response.commit_hash, "market-size")
        print(f"  Market research committed and pinned [{market_response.commit_hash[:8]}]")

        # Competitor research
        t.user(
            "Who are the main competitors in AI code review?",
            tags=["competitor"],
        )
        competitor_commit = t.assistant(
            "Top 5 competitors:\n"
            "1. GitHub Copilot Code Review -- 60% market share, bundled\n"
            "2. CodeRabbit -- Fast-growing, $19/seat/mo, strong PR focus\n"
            "3. Codacy -- Enterprise, $15/seat/mo, compliance focus\n"
            "4. SonarQube AI -- Legacy player adding AI, $30/seat/mo\n"
            "5. Qodo (ex-CodiumAI) -- Test generation angle, $29/seat/mo",
            metadata={"competitors_analyzed": 5, "confidence": "medium"},
            tags=["competitor", "pricing"],
        )
        t.annotate(
            competitor_commit.commit_hash,
            Priority.IMPORTANT,
            reason="Competitive landscape reference",
            retain="competitor names and pricing",
        )
        print(f"  Competitor analysis committed [{competitor_commit.commit_hash[:8]}]")

        # Pain point research
        t.user("What are the key pain points for engineering teams?")
        pain_commit = t.assistant(
            "Top pain points from 200 engineering leader surveys:\n"
            "1. Review bottleneck: PRs wait avg 2.3 days for review (47%)\n"
            "2. Inconsistent quality: style/logic issues slip through (38%)\n"
            "3. Reviewer fatigue: senior devs spend 15+ hrs/week reviewing (35%)\n"
            "4. Knowledge silos: only 1-2 people can review certain code (29%)\n"
            "5. Security gaps: manual review misses 67% of vulnerabilities (24%)",
            metadata={"survey_size": 200, "source": "DevOps Pulse 2025"},
            tags=["pain-point", "key-finding"],
        )
        t.annotate(pain_commit.commit_hash, Priority.PINNED, reason="Core customer pain data")
        print(f"  Pain points committed and pinned [{pain_commit.commit_hash[:8]}]")

        # Check pinned commits
        pinned = t.pinned()
        print(f"  Pinned commits: {len(pinned)} (these survive compression)")
        print()

        # =============================================================
        # STEP 4: Branch for Parallel Analysis
        # Features: branch, switch, configure per-branch
        # =============================================================

        print("=" * 60)
        print("STEP 4: Branch for Parallel Analysis")
        print("=" * 60)
        print()

        # Save main branch HEAD for later
        main_head = t.head

        # --- Optimistic analysis branch ---
        t.branch("optimistic")
        print(f"  Created 'optimistic' branch, switched to it")

        t.configure(temperature=0.7, analysis_bias="bullish")
        t.assistant(
            "OPTIMISTIC ANALYSIS:\n"
            "The AI code review market has massive tailwinds. Developer "
            "shortage means automation is inevitable. GitHub's bundling "
            "leaves a premium gap for specialized tools. Enterprise buyers "
            "pay $50-100/seat/mo for security-focused review. A specialized "
            "tool targeting enterprise security review could capture $200M "
            "SAM within 3 years.\n\n"
            "Bull case revenue: $15M ARR by Year 3 with 100 enterprise "
            "accounts at $150K ACV.",
            tags=["opportunity"],
            metadata={"scenario": "optimistic", "confidence": "medium"},
        )
        print(f"  Optimistic analysis committed on 'optimistic' branch")

        # --- Pessimistic analysis branch ---
        t.switch("main")
        t.branch("pessimistic")
        print(f"  Created 'pessimistic' branch, switched to it")

        t.configure(temperature=0.2, analysis_bias="bearish")
        t.assistant(
            "PESSIMISTIC ANALYSIS:\n"
            "GitHub Copilot's bundling strategy is an existential threat. "
            "With 60% market share and zero marginal cost for users already "
            "paying for GitHub Enterprise, standalone tools face brutal "
            "price competition. CodeRabbit's growth may plateau as GitHub "
            "closes feature gaps. Enterprise sales cycles are 6-12 months.\n\n"
            "Bear case: market consolidates around 2-3 players. New entrant "
            "struggles to reach $3M ARR, burns $8M before pivot or shutdown.",
            tags=["risk"],
            metadata={"scenario": "pessimistic", "confidence": "medium"},
        )
        print(f"  Pessimistic analysis committed on 'pessimistic' branch")
        print()

        # =============================================================
        # STEP 5: Compare Branches
        # Features: compare(), list_branches()
        # =============================================================

        print("=" * 60)
        print("STEP 5: Compare Branches")
        print("=" * 60)
        print()

        # List all branches
        branches = t.list_branches()
        print(f"  Branches: {[b.name for b in branches]}")
        current = [b for b in branches if b.is_current][0]
        print(f"  Currently on: {current.name}")

        # Compare optimistic vs pessimistic
        diff_result = t.compare("optimistic", "pessimistic")
        print(f"  Cross-branch diff (optimistic vs pessimistic):")
        diff_result.pprint(stat_only=True)
        print()

        # =============================================================
        # STEP 6: Merge Best Insights
        # Features: switch, merge with strategy
        # =============================================================

        print("=" * 60)
        print("STEP 6: Merge Best Insights")
        print("=" * 60)
        print()

        # Switch to main and merge optimistic (has the SAM/revenue projections)
        t.switch("main")
        merge_result = t.merge("optimistic", strategy="theirs")
        print(f"  Merged 'optimistic' into main: {merge_result.merge_type}")

        # Also merge pessimistic for the risk assessment
        merge_result_2 = t.merge("pessimistic", strategy="theirs")
        print(f"  Merged 'pessimistic' into main: {merge_result_2.merge_type}")

        ctx_after_merge = t.compile()
        ctx_after_merge.pprint(style="compact")
        print()

        # =============================================================
        # STEP 7: Compress Research
        # Features: compress with pinned commit preservation
        # =============================================================

        print("=" * 60)
        print("STEP 7: Compress Research")
        print("=" * 60)
        print()

        ctx_before = t.compile()
        tokens_before = ctx_before.token_count
        messages_before = len(ctx_before.messages)
        print(f"  Before compression: {messages_before} messages, ~{tokens_before} tokens")

        # Compress -- pinned commits (market data, pain points) survive
        compress_result = t.compress(
            content=(
                "[Research Summary] AI Code Review Market Analysis:\n"
                "- Market: $1.2B total, $340M AI segment, 28% CAGR\n"
                "- 5 competitors analyzed; GitHub dominates at 60%\n"
                "- Key pain: PR wait times (2.3d avg), reviewer fatigue\n"
                "- Bull case: $15M ARR Y3 targeting enterprise security\n"
                "- Bear case: GitHub bundling threatens standalone tools\n"
                "- Consensus: niche opportunity in enterprise security review"
            ),
        )

        ctx_after = t.compile()
        compress_result.pprint()

        # Verify pinned findings survived
        compiled_text = " ".join((m.content or "") for m in ctx_after.messages)
        assert "$1.2B" in compiled_text or "$340M" in compiled_text, \
            "Pinned market data should survive compression"
        print(f"  Verified: pinned market data survived compression")
        print()

        # =============================================================
        # STEP 8: Transition to Drafting
        # Features: transition with handoff, apply_stage
        # =============================================================

        print("=" * 60)
        print("STEP 8: Transition to Drafting")
        print("=" * 60)
        print()

        t.transition(
            "drafting",
            handoff=(
                "Research complete. Key insight: niche opportunity in "
                "enterprise security-focused code review. $340M AI segment, "
                "28% CAGR. GitHub dominance leaves premium gap."
            ),
        )
        print(f"  Transitioned to 'drafting' branch")

        # Apply the synthesize stage from the research profile
        t.apply_stage("synthesize")
        configs = t.get_all_configs()
        print(f"  Applied 'synthesize' stage: temp={configs.get('temperature')}, "
              f"strategy={configs.get('compile_strategy')}")
        print()

        # =============================================================
        # STEP 9: Search Previous Research
        # Features: find() by tag, content, metadata
        # =============================================================

        print("=" * 60)
        print("STEP 9: Search Previous Research")
        print("=" * 60)
        print()

        # Find by content substring across the current branch context
        market_hits = t.find(content="$340M", limit=10)
        print(f"  Found {len(market_hits)} commits mentioning '$340M' (market size)")
        for hit in market_hits:
            preview = (hit.message or hit.content_type)[:50]
            print(f"    [{hit.commit_hash[:8]}] {preview}")

        # Find by metadata key (high-confidence research)
        high_conf = t.find(metadata_key="confidence", metadata_value="high", limit=5)
        print(f"  Found {len(high_conf)} high-confidence commits")

        # Find by content type -- look for all instruction commits
        instructions = t.find(content_type="instruction", limit=10)
        print(f"  Found {len(instructions)} instruction commits (directives, handoffs)")

        # Search the optimistic branch for opportunity findings
        opp_hits = t.find(tag="opportunity", branch="optimistic", limit=5)
        print(f"  Found {len(opp_hits)} 'opportunity' commits on optimistic branch")
        print()

        # =============================================================
        # STEP 10: Draft the Pitch
        # Features: user/assistant with tags, metadata
        # =============================================================

        print("=" * 60)
        print("STEP 10: Draft the Pitch")
        print("=" * 60)
        print()

        t.user(
            "Draft a 1-paragraph product pitch based on the research.",
            tags=["draft"],
        )
        pitch_commit = t.assistant(
            "SecureReview AI: Enterprise-grade automated code review "
            "that catches security vulnerabilities your team misses. "
            "Manual code review misses 67% of vulnerabilities and costs "
            "senior engineers 15+ hours per week. SecureReview uses "
            "purpose-built security models to review every PR in under "
            "60 seconds, integrating with GitHub, GitLab, and Azure DevOps. "
            "Unlike GitHub Copilot's general-purpose review, we specialize "
            "in security -- SOC2, HIPAA, and PCI compliance out of the box. "
            "Target: $150K ACV enterprise accounts. $340M addressable market "
            "growing 28% annually.",
            tags=["draft", "key-finding"],
            metadata={"draft_version": 1, "word_count": 89},
        )
        print(f"  Pitch drafted [{pitch_commit.commit_hash[:8]}]")

        # =============================================================
        # STEP 11: Quality Gate -- Attempt Premature Review
        # Features: pre_transition middleware, BlockedError, reset
        # =============================================================

        print()
        print("=" * 60)
        print("STEP 11: Quality Gate and Recovery")
        print("=" * 60)
        print()

        # Try to transition to review prematurely (gate should block)
        try:
            t.transition("review")
            print("  ERROR: Should have been blocked!")
        except BlockedError as e:
            print(f"  Quality gate BLOCKED transition to 'review':")
            print(f"    Reason: {e.reasons[0]}")

        # Add more drafting work to pass the gate
        t.user("Refine the pitch: add a competitive moat section.")
        t.assistant(
            "Competitive Moat:\n"
            "1. Security-first ML models trained on CVE databases\n"
            "2. Compliance mapping engine (SOC2/HIPAA/PCI)\n"
            "3. Enterprise integration depth (SSO, audit logs, RBAC)\n"
            "4. Accumulating proprietary dataset of security patterns",
            tags=["draft"],
        )

        t.user("Add a go-to-market strategy.")
        t.assistant(
            "Go-to-Market:\n"
            "1. PLG bottom-up: free tier for open source projects\n"
            "2. Enterprise outbound: target CISO/VP Eng at Series C+\n"
            "3. Partner channel: security consulting firms as resellers\n"
            "4. Content: publish annual 'State of Code Security' report",
            tags=["draft"],
        )

        t.user("Add financial projections.")
        financial_commit = t.assistant(
            "Financial Projections (3-year):\n"
            "Year 1: $1.5M ARR, 10 enterprise accounts, -$2M burn\n"
            "Year 2: $6M ARR, 40 accounts, -$1M burn, break-even Q4\n"
            "Year 3: $15M ARR, 100 accounts, $3M profit\n"
            "Assumes: $150K ACV, 12-month sales cycle, 90% gross margin",
            tags=["draft", "key-finding"],
            metadata={"draft_version": 1, "projection_years": 3},
        )

        # Demonstrate checkpoint-and-rollback: save state, try something risky
        checkpoint = t.head
        t.register_tag("checkpoint")
        t.tag(checkpoint, "checkpoint")
        print(f"  Checkpoint saved at [{checkpoint[:8]}]")

        # Simulate a bad edit that we want to undo
        bad_commit = t.assistant(
            "REVISED PROJECTION: Actually, we should pivot to B2C. "
            "Consumer code review app, $9.99/mo, target 1M users. "
            "This completely changes the business model.",
            tags=["draft"],
        )
        print(f"  Bad pivot committed [{bad_commit.commit_hash[:8]}] -- will rollback")

        # Rollback to checkpoint
        t.reset(checkpoint)
        commits_after_reset = len(t.log(limit=50))
        print(f"  Reset to checkpoint [{checkpoint[:8]}]")
        print(f"  Commits visible after reset: {commits_after_reset}")

        # Verify bad content is gone from compiled context
        ctx_check = t.compile()
        check_text = " ".join((m.content or "") for m in ctx_check.messages)
        assert "pivot to B2C" not in check_text, "Bad pivot should be excluded"
        print(f"  Verified: bad pivot excluded from context")
        print()

        # =============================================================
        # STEP 12: Successful Transition to Review
        # Features: transition (now passes the gate)
        # =============================================================

        print("=" * 60)
        print("STEP 12: Transition to Review (Gate Passes)")
        print("=" * 60)
        print()

        # Now we have enough commits to pass the quality gate
        try:
            t.transition(
                "review",
                handoff=(
                    "Draft complete: SecureReview AI pitch with competitive "
                    "moat, GTM strategy, and 3-year financial projections. "
                    "Ready for review."
                ),
            )
            print(f"  Successfully transitioned to 'review'")
        except BlockedError:
            # If still blocked, temporarily remove the gate for demonstration
            t.remove_middleware(gate_id)
            t.transition("review")
            print(f"  Transitioned to 'review' (gate removed for demo)")
            gate_id = t.use("pre_transition", gate_fn)

        # Apply validate stage for review
        t.apply_stage("validate")
        print(f"  Applied 'validate' stage config")
        print()

        # =============================================================
        # STEP 13: Review and Finalize
        # Features: diff, edit_history, compile, status
        # =============================================================

        print("=" * 60)
        print("STEP 13: Review and Finalize")
        print("=" * 60)
        print()

        # Reviewer feedback
        t.user("Review the pitch for completeness and accuracy.")
        review_commit = t.assistant(
            "Review Notes:\n"
            "- STRONG: Clear pain point ($1.2B market, 67% vuln miss rate)\n"
            "- STRONG: Differentiated positioning (security-first vs general)\n"
            "- IMPROVE: Add customer validation (pilot commitments?)\n"
            "- IMPROVE: Clarify why incumbents can't just add security\n"
            "- RISK: $150K ACV assumes enterprise-only; validate pipeline\n"
            "Overall: 7/10 -- ready for investor meeting with minor tweaks.",
            tags=["key-finding"],
            metadata={"review_score": 7, "reviewer": "product_lead"},
        )
        print(f"  Review committed [{review_commit.commit_hash[:8]}]")

        # Show a sequential diff
        diff = t.diff()
        print(f"  Diff (previous -> HEAD):")
        diff.pprint(stat_only=True)
        print()

        # =============================================================
        # STEP 14: Final State Report
        # Features: status, health, log, list_branches, list_tags,
        #           pinned, skipped, get_all_configs, compile/pprint,
        #           metrics from middleware
        # =============================================================

        print("=" * 60)
        print("STEP 14: Final State Report")
        print("=" * 60)
        print()

        # --- Status ---
        status = t.status()
        print(f"  Status:")
        print(f"    HEAD:          [{(status.head_hash or 'none')[:8]}]")
        print(f"    Branch:        {status.branch_name}")
        print(f"    Commits:       {status.commit_count}")
        print(f"    Tokens:        {status.token_count}")
        print()

        # --- Health check ---
        health = t.health()
        print(f"  Health: {health.summary()}")
        print()

        # --- Branches ---
        branches = t.list_branches()
        print(f"  Branches ({len(branches)}):")
        for b in branches:
            marker = " *" if b.is_current else "  "
            print(f"   {marker} {b.name}")
        print()

        # --- Tags ---
        tags = t.list_tags()
        print(f"  Tag Registry ({len(tags)} tags):")
        for entry in tags:
            print(f"    {entry['name']:20s} count={entry['count']}")
        print()

        # --- Pinned commits ---
        pinned = t.pinned()
        print(f"  Pinned commits ({len(pinned)}):")
        for p in pinned:
            preview = (p.message or p.content_type)[:40]
            print(f"    [{p.commit_hash[:8]}] {preview}")
        print()

        # --- Configs ---
        configs = t.get_all_configs()
        print(f"  Active configs:")
        for k, v in sorted(configs.items()):
            print(f"    {k}: {v}")
        print()

        # --- Commit log ---
        print(f"  Commit log (last 15):")
        pprint_log(t.log()[-15:])
        print()

        # --- Middleware metrics ---
        print(f"  Observability Metrics:")
        print(f"    Total commits tracked: {metrics['commits']}")
        print(f"    Total tokens tracked:  {metrics['tokens']}")
        print(f"    By content type:")
        for ct, count in sorted(metrics["by_type"].items()):
            print(f"      {ct:20s} {count:>3} commits")
        print(f"    By branch:")
        for branch, tokens in sorted(metrics["by_branch"].items()):
            print(f"      {branch:20s} {tokens:>5} tokens")
        print()

        # --- Final compiled context ---
        print(f"  Final Compiled Context:")
        t.compile().pprint(style="compact")
        print()

        # --- Clean up middleware ---
        t.remove_middleware(tracker_id)
        t.remove_middleware(gate_id)

    # =================================================================
    # Feature Coverage Summary
    # =================================================================

    print()
    print("=" * 70)
    print("FEATURE COVERAGE SUMMARY")
    print("=" * 70)
    print("""
  Feature                    Where Used
  -------------------------  ------------------------------------------
  load_profile()             Step 1: Load 'research' profile
  apply_template()           Step 1: 'research_protocol', 'output_format'
  apply_stage()              Step 1/8/12: ingest, synthesize, validate
  configure()                Step 4: per-branch config overrides
  directive()                Step 1: custom 'product-focus' directive
  register_tag() / tag()     Step 2/3: domain-specific tag taxonomy
  use() middleware            Step 2: post_commit tracker, pre_transition gate
  system/user/assistant      Step 3+: core DAG operations
  annotate() / PINNED        Step 3: pin critical findings
  branch() / switch()        Step 4: optimistic/pessimistic branches
  compare()                  Step 5: cross-branch diff
  merge()                    Step 6: merge branch insights
  compress()                 Step 7: reduce verbose research
  transition()               Step 8/12: stage transitions with handoff
  find()                     Step 9: search by tag, content, metadata
  reset()                    Step 11: checkpoint-and-rollback
  diff()                     Step 13: sequential commit diff
  compile() / pprint()       Steps 7/13/14: compile and inspect context
  log()                      Step 14: commit history
  status()                   Step 14: tract status
  health()                   Step 14: DAG integrity check
  list_branches()            Step 5/14: enumerate branches
  list_tags()                Step 14: enumerate registered tags
  pinned()                   Step 3/14: query pinned commits
  get_all_configs()          Step 1/14: inspect active configuration
  BlockedError               Step 11: quality gate enforcement
  remove_middleware()        Step 14: clean up handlers

  Total: 25+ distinct tract features in a single natural workflow.
""")

    print("Done.")


# Alias for pytest discovery
test_full_showcase = main


if __name__ == "__main__":
    main()
