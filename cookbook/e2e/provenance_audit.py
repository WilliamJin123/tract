"""Provenance audit trail: trace every operation back to its cause.

  PART 1 -- Manual:      Build conversation, walk audit trail, time-travel
  PART 2 -- LLM / Agent: Trigger-driven ops, trace compression to its trigger

Story: A developer debugging why an agent's context looks the way it does.
The conversation had commits, edits, compression, and trigger-driven actions.
Using log(), query_by_config(), query_by_tags(), and time-travel, we
reconstruct the full history of how context evolved.

Principle 3 (Full History): every operation is recorded, every state is
reconstructable, every decision is traceable.
"""

import sys
from pathlib import Path

from tract import (
    CompressTrigger,
    Priority,
    Tract,
    TractConfig,
    TokenBudgetConfig,
)
from tract.hooks.compress import PendingCompress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: Build and Audit
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Full Audit Trail Walkthrough")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=600))
    with Tract.open(config=config) as t:
        # --- Build a realistic conversation with multiple operations ---

        # System prompt
        sys_ci = t.system("You are a financial analysis assistant.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        # Initial exchanges
        c1 = t.user("What are the key metrics for evaluating SaaS companies?")
        c2 = t.assistant(
            "Key SaaS metrics include: ARR (Annual Recurring Revenue), "
            "CAC (Customer Acquisition Cost), LTV (Lifetime Value), "
            "churn rate, NDR (Net Dollar Retention), and Rule of 40."
        )

        # Tag important content
        t.tag(c2.commit_hash, "metrics-reference")

        c3 = t.user("Explain the Rule of 40 in more detail.")
        c4 = t.assistant(
            "The Rule of 40 states that a SaaS company's combined growth "
            "rate and profit margin should exceed 40%. For example, a "
            "company growing at 30% with 15% margins scores 45 (healthy). "
            "It balances growth vs profitability trade-offs."
        )

        # Edit a commit (correct a detail)
        c4_edit = t.assistant(
            "The Rule of 40 states that a SaaS company's combined revenue "
            "growth rate and EBITDA margin should exceed 40%. A company "
            "growing at 30% with 15% EBITDA margins scores 45 (healthy). "
            "It helps investors evaluate the growth-profitability trade-off.",
            edit=c4.commit_hash,
        )

        # More conversation
        c5 = t.user("How does churn rate affect LTV calculations?")
        c6 = t.assistant(
            "LTV = ARPU / Monthly Churn Rate. A 2% monthly churn gives "
            "an average customer lifetime of 50 months. Reducing churn "
            "from 5% to 2% triples LTV, making it the highest-leverage "
            "metric for SaaS economics."
        )
        t.tag(c6.commit_hash, "metrics-reference")
        t.annotate(c6.commit_hash, Priority.PINNED, reason="key formula")

        # Manual compression of early dialogue
        t.compress(
            content="Initial discussion covered key SaaS metrics (ARR, CAC, "
            "LTV, churn, NDR, Rule of 40) and the Rule of 40 formula "
            "(growth rate + EBITDA margin > 40%).",
        )

        # --- AUDIT: Walk the full trail ---

        print("\n  --- Full Commit Log ---\n")
        log = t.log(include_edits=True)
        for i, ci in enumerate(log):
            tags_str = f" tags={ci.tags}" if ci.tags else ""
            edit_str = " [EDIT]" if ci.operation == "EDIT" else ""
            ct = ci.content_type or "?"
            msg = (ci.message or "")[:55]
            print(f"    [{i:2d}] {ci.commit_hash[:10]} {ct:>12}{edit_str}: {msg}{tags_str}")

        # --- AUDIT: Query by tags ---

        print("\n  --- Query by Tags: 'metrics-reference' ---\n")
        tagged = t.query_by_tags(["metrics-reference"])
        for ci in tagged:
            print(f"    {ci.commit_hash[:10]}: {(ci.message or '')[:60]}")

        # --- AUDIT: Time-travel ---

        print("\n  --- Time-Travel: Context at each checkpoint ---\n")

        # Before compression
        ctx_early = t.compile(at_commit=c4.commit_hash)
        print(f"    At c4 (before edit): {ctx_early.commit_count} msgs, "
              f"{ctx_early.token_count} tokens")

        # After edit
        ctx_edited = t.compile(at_commit=c4_edit.commit_hash)
        print(f"    At c4_edit (after edit): {ctx_edited.commit_count} msgs, "
              f"{ctx_edited.token_count} tokens")

        # Current state
        ctx_now = t.compile()
        print(f"    Current (after compress): {ctx_now.commit_count} msgs, "
              f"{ctx_now.token_count} tokens")

        print("\n  --- Current Compiled Context ---")
        ctx_now.pprint(style="compact")

        # --- AUDIT: Status summary ---

        status = t.status()
        print(f"\n  --- Current Status ---")
        print(f"    Commits: {status.commit_count}")
        print(f"    Tokens:  {status.token_count}")
        budget_max = status.token_budget_max or 0
        if budget_max:
            print(f"    Budget:  {status.token_count}/{budget_max} "
                  f"({status.token_count / budget_max:.0%})")


# =====================================================================
# PART 2 -- LLM / Agent: Trigger-Driven Provenance
# =====================================================================

def part2_agent():
    print("\n" + "=" * 60)
    print("PART 2 -- LLM / Agent: Trigger-Driven Provenance")
    print("=" * 60)
    print()
    print("  Triggers fire automatically. The audit trail shows which")
    print("  trigger caused each compression and what state looked like")
    print("  before and after.")

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=400))

    # Track trigger-driven compressions
    trigger_events: list[dict] = []

    def audit_compress(pending: PendingCompress):
        """Hook that records compression provenance before approving."""
        trigger_events.append({
            "original_tokens": pending.original_tokens,
            "estimated_tokens": pending.estimated_tokens,
            "summaries": len(pending.summaries),
        })
        pending.approve()

    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Set up trigger and audit hook
        t.configure_triggers([
            CompressTrigger(
                threshold=0.75,
                summary_content="Auto-compressed: investment analysis session.",
            ),
        ])
        t.on("compress", audit_compress, name="audit")

        t.system("You are a venture capital analyst.")

        # Build conversation that will trigger compression
        topics = [
            ("Evaluate Series A metrics for a fintech startup.", "ARR $2M, 3x YoY growth, 8% monthly churn."),
            ("What's the competitive landscape?", "Three major incumbents, two well-funded competitors."),
            ("Assess the founding team.", "Strong technical co-founders, first-time CEO."),
            ("What due diligence should we prioritize?", "Customer interviews, code audit, financial model review."),
            ("Give a final investment recommendation.", "Conditional pass: revisit after churn improvement."),
            ("How does this compare to our last fintech deal?", "Similar stage but better unit economics."),
            ("What follow-on strategy would you recommend?", "Reserve 2x for Series B if metrics improve."),
        ]

        for question, answer in topics:
            t.user(question)
            t.assistant(answer)

            # compile() evaluates triggers
            ctx = t.compile()
            status = t.status()
            budget_max = status.token_budget_max or 1
            usage = status.token_count / budget_max
            print(f"    Turn: {status.token_count}/{budget_max} ({usage:.0%}), "
                  f"compressions so far: {len(trigger_events)}")

        # --- AUDIT: Show trigger provenance ---

        print(f"\n  --- Trigger-Driven Compression Events ---\n")
        for i, evt in enumerate(trigger_events):
            print(f"    Event {i}: {evt['original_tokens']} -> ~{evt['estimated_tokens']} tokens "
                  f"({evt['summaries']} summaries)")

        # --- AUDIT: Full log shows compression commits ---

        print(f"\n  --- Full Log (showing compression entries) ---\n")
        for ci in t.log():
            ct = ci.content_type or "?"
            marker = " [COMPRESSED]" if ct == "summary" else ""
            msg = (ci.message or "")[:55]
            print(f"    {ci.commit_hash[:10]} {ct:>12}{marker}: {msg}")

        # --- AUDIT: Config provenance across operations ---

        if llm.api_key:
            print(f"\n  --- Config Provenance ---\n")
            # Show what model/config was used
            all_commits = t.log()
            configs_seen = set()
            for ci in all_commits:
                if ci.generation_config:
                    model = ci.generation_config.get("model", "N/A")
                    configs_seen.add(model)
            if configs_seen:
                print(f"    Models used: {configs_seen}")
            else:
                print(f"    (No generation configs recorded -- manual commits)")

        # --- AUDIT: Final state ---

        final = t.status()
        budget_max = final.token_budget_max or 1
        print(f"\n  --- Final State ---")
        print(f"    Commits: {final.commit_count}")
        print(f"    Tokens:  {final.token_count}/{budget_max} "
              f"({final.token_count / budget_max:.0%})")
        print(f"    Trigger compressions: {len(trigger_events)}")
        print(f"    Every operation is in the log, every state is reconstructable.")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()
