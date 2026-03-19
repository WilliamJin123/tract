"""Adversarial Review: propose -> critique -> defend -> revise

A multi-perspective workflow where independent agents review work adversarially.
The proposer commits an implementation plan, the critic tears it apart on a
separate branch, the defender evaluates which critiques hold up, and the reviser
incorporates surviving feedback into a final version.

Uses separate t.llm.run() calls per role so each agent has its own perspective
and can't self-censor. Branching isolates each viewpoint; compare() and merge()
reconcile them. Full conversations are printed after each stage so you can
read exactly what each agent said.

Stages:
  propose  -- commit the original plan/implementation (manual seed)
  critique -- adversarial agent finds flaws on a separate branch
  defend   -- independent agent pushes back on overblown critiques
  revise   -- incorporate surviving critiques into final version

Demonstrates: multi-agent perspectives via separate t.llm.run() calls, branching
              for isolated viewpoints, compare() for cross-branch diff, merge()
              to reconcile, directives for role-specific behavior, middleware
              gates, config-per-stage (high temp for critique, low for defense),
              compile().pprint(style="chat") for full conversation inspection

Requires: LLM API key (uses Cerebras provider, gpt-oss-120b)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError, MiddlewareContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm

MODEL_ID = llm.small


def print_conversation(t: Tract, label: str) -> None:
    """Compile current branch and print the full chat transcript."""
    print(f"\n{'=' * 60}")
    print(f"  FULL CONVERSATION: {label}")
    print(f"  branch={t.current_branch}  commits={len(t.log())}")
    print(f"{'=' * 60}\n")
    t.compile().pprint(style="chat")
    print()


def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
    ) as t:

        # =============================================================
        # Tags for classifying findings
        # =============================================================

        for tag in ["flaw", "risk", "nitpick", "valid", "dismissed", "revised"]:
            t.register_tag(tag)

        # =============================================================
        # Stage 1: PROPOSE -- seed the original plan on main
        # =============================================================

        print("=== Stage 1: PROPOSE ===\n")

        t.config.set(stage="propose", temperature=0.5)

        t.system(
            "You are a senior engineer. The conversation contains a system "
            "design proposal that will be reviewed adversarially."
        )

        t.user(
            "Design proposal: Real-time notification service\n\n"
            "Architecture:\n"
            "- REST API gateway receives events from upstream services\n"
            "- Events published to Redis Pub/Sub for fan-out\n"
            "- WebSocket server subscribes to Redis and pushes to clients\n"
            "- PostgreSQL stores notification history and read status\n"
            "- Single Redis instance handles both pub/sub and caching\n\n"
            "Scaling strategy:\n"
            "- Horizontal scaling of WebSocket servers behind a load balancer\n"
            "- Redis handles cross-server message distribution\n"
            "- No message queue -- Redis Pub/Sub is fast enough\n"
            "- Client reconnection pulls missed notifications from Postgres\n\n"
            "Delivery guarantees:\n"
            "- At-most-once via Redis Pub/Sub (fire and forget)\n"
            "- Missed messages recovered on reconnect from Postgres\n"
            "- No deduplication -- clients handle idempotency",
            message="proposal: notification service design",
        )

        t.user(
            "Implementation plan:\n"
            "1. Phase 1: REST API + Postgres storage (week 1-2)\n"
            "2. Phase 2: Redis Pub/Sub + WebSocket push (week 3)\n"
            "3. Phase 3: Reconnection recovery logic (week 4)\n"
            "4. Phase 4: Load testing and horizontal scaling (week 5)\n\n"
            "Team: 2 backend engineers, no dedicated infra support\n"
            "Timeline: 5 weeks to production",
            message="proposal: implementation plan",
        )

        proposal_log = t.log()
        print(f"  Seeded {len(proposal_log)} commits on main")
        print_conversation(t, "PROPOSAL (main)")

        # =============================================================
        # Stage 2: CRITIQUE -- adversarial review on separate branch
        # =============================================================

        print("=== Stage 2: CRITIQUE ===\n")

        t.branch("critique")
        t.switch("critique")
        t.config.set(stage="critique", temperature=0.8)

        t.directive(
            "critic-role",
            "You are an adversarial reviewer. Your job is to FIND FLAWS.\n"
            "Be aggressive but specific. For each issue:\n"
            "- State the flaw clearly in one sentence\n"
            "- Explain the concrete failure scenario\n"
            "- Rate severity: critical / major / minor / nitpick\n"
            "Do NOT suggest fixes -- only identify problems.\n"
            "Do NOT be charitable. Assume the worst-case scenario.\n"
            "Commit each distinct flaw as a separate message.",
        )

        # Gate: critic must commit at least 3 findings
        findings_needed = 3

        def critique_completion_gate(ctx: MiddlewareContext):
            if ctx.target != "main":
                return
            findings = [c for c in ctx.tract.log() if c.tags and "flaw" in c.tags]
            if len(findings) < findings_needed:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= {findings_needed} flaw-tagged findings "
                     f"(have {len(findings)})"],
                )

        t.middleware.add("pre_transition", critique_completion_gate)

        print("  Running critic agent...\n")

        critic_result = t.llm.run(
            "Review the notification service proposal above. Find every flaw, "
            "risk, and questionable decision. Focus on:\n"
            "- Single points of failure\n"
            "- Scaling bottlenecks\n"
            "- Missing failure modes\n"
            "- Unrealistic assumptions\n"
            "- Security gaps\n\n"
            "Commit each flaw separately with tag 'flaw'. "
            "Rate severity in the commit message. "
            "When done, commit a summary of all findings.",
            max_steps=10,
            profile="full",
            tool_names=["commit", "tag", "get_config", "status", "log"],
        )

        critique_log = t.log()
        flaw_commits = [c for c in critique_log if c.tags and "flaw" in c.tags]
        print(f"\n  Critic: {len(critique_log)} commits, "
              f"{len(flaw_commits)} tagged as flaws")

        # Show the full critic conversation
        critic_result.pprint(style="chat")
        print_conversation(t, "CRITIQUE (critique branch)")

        # =============================================================
        # Stage 3: DEFEND -- evaluate critiques independently
        # =============================================================

        print("=== Stage 3: DEFEND ===\n")

        t.switch("main")
        t.branch("defense")
        t.switch("defense")
        t.config.set(stage="defend", temperature=0.3)

        # Compile the critique branch to get the full conversation text.
        # (compare().to_json() only has structural metadata — no message content.)
        t.switch("critique")
        critique_ctx = t.compile()
        critique_messages = "\n\n".join(
            f"[{m.get('role', '?')}]: {m.get('content', '')}"
            for m in critique_ctx.to_dicts()
            if m.get("role") != "system"
        )
        t.switch("defense")

        t.system(
            "You are an independent technical reviewer. You have the original "
            "proposal AND an adversarial critique. Your job is to evaluate each "
            "critique objectively:\n"
            "- VALID: the critique identifies a real, actionable problem\n"
            "- DISMISSED: the critique is overblown, wrong, or a nitpick "
            "that doesn't warrant changes\n\n"
            "Be rigorous. Don't dismiss real issues to be agreeable, but don't "
            "accept weak critiques either. For each critique, commit your "
            "verdict with tag 'valid' or 'dismissed'."
        )

        t.user(
            f"Here are the adversarial critiques to evaluate:\n\n{critique_messages}",
            message="defense: critique findings for review",
        )

        t.directive(
            "defender-role",
            "For each critique, respond with:\n"
            "1. The original critique (one sentence)\n"
            "2. Your verdict: VALID or DISMISSED\n"
            "3. Your reasoning (2-3 sentences)\n"
            "Commit each verdict separately. Tag with 'valid' or 'dismissed'.",
        )

        print("  Running defender agent...\n")

        defense_result = t.llm.run(
            "Evaluate each critique from the adversarial review. For each one, "
            "decide if it's VALID (real problem worth fixing) or DISMISSED "
            "(overblown, incorrect, or trivial). Commit each verdict separately "
            "with the appropriate tag.",
            max_steps=10,
            profile="full",
            tool_names=["commit", "tag", "get_config", "status", "log"],
        )

        defense_log = t.log()
        valid = [c for c in defense_log if c.tags and "valid" in c.tags]
        dismissed = [c for c in defense_log if c.tags and "dismissed" in c.tags]
        print(f"\n  Defender verdicts: {len(valid)} valid, {len(dismissed)} dismissed")

        # Show the full defender conversation
        defense_result.pprint(style="chat")
        print_conversation(t, "DEFENSE (defense branch)")

        # =============================================================
        # Stage 4: REVISE -- incorporate surviving critiques
        # =============================================================

        print("=== Stage 4: REVISE ===\n")

        t.switch("main")
        t.config.set(stage="revise", temperature=0.4)

        # Merge defense branch to bring verdicts into main
        t.merge("defense", message="merge defense verdicts into main")

        t.directive(
            "reviser-role",
            "You are revising the original proposal based on validated critiques. "
            "Only address critiques tagged 'valid' -- ignore dismissed ones.\n"
            "For each valid critique, propose a specific design change.\n"
            "Keep changes minimal and targeted. Don't redesign from scratch.\n"
            "Commit the revised proposal as a single comprehensive update.",
        )

        print("  Running reviser agent...\n")

        revise_result = t.llm.run(
            "Review the conversation history. The original proposal and the "
            "defense verdicts are both here. For every critique marked VALID, "
            "propose a concrete design change. Then commit a revised version "
            "of the proposal that addresses all valid critiques. "
            "Tag the final revision with 'revised'.",
            max_steps=8,
            profile="full",
            tool_names=["commit", "tag", "get_config", "status", "log",
                        "compile"],
        )

        # Show the full reviser conversation
        revise_result.pprint(style="chat")
        print_conversation(t, "REVISED (main branch, final)")

        # =============================================================
        # Pipeline summary
        # =============================================================

        print(f"{'=' * 60}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'=' * 60}\n")

        print(f"  Proposal commits:  {len(proposal_log)}")
        print(f"  Critique findings: {len(flaw_commits)}")
        print(f"  Defense verdicts:  {len(valid)} valid, {len(dismissed)} dismissed")
        print(f"  Revision status:   {revise_result.status}")
        print(f"  Total steps:       "
              f"{critic_result.steps + defense_result.steps + revise_result.steps}")

        print(f"\n  Branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Tag summary:")
        for entry in t.list_tags():
            if entry["count"] > 0:
                print(f"    {entry['name']:12s} count={entry['count']}")


if __name__ == "__main__":
    main()


# --- See also ---
# Error recovery:       agentic/10_error_recovery.py
# Semantic automation:  agentic/04_semantic_automation.py
