"""Customer Support Workflow: triage -> resolve -> escalate

Developer-driven staged workflow. The developer controls all transitions;
the agent generates content per stage following explicit instructions.
This is a template-execution pattern -- the agent fills in stage-specific
templates, it does not make workflow decisions autonomously.

NOTE: The prompts currently prescribe exactly what to commit (numbered items
with specific tags). This is scripted, not emergent. To make this a genuine
agent demo, the prompts would need to describe the *problem* and let the
agent decide what to produce. Marked for rewrite.

Stages:
  triage   -- classify severity, gather context (temperature 0.5)
  resolve  -- propose and test solutions (temperature 0.3)
  escalate -- document failure, prepare handoff (temperature 0.1)

Demonstrates: per-stage config, middleware quality gates, developer-driven
              transitions, tagging for support classification

Requires: LLM API key (uses Cerebras provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError, MiddlewareContext
from tract.formatting import pprint_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # =============================================================
        # Stage config, quality gates, and escalation threshold
        # =============================================================

        print("=== Setting Up Support Workflow ===\n")

        t.configure(
            stage="triage",
            temperature=0.5,
            compile_strategy="full",
            max_resolution_attempts=2,
        )

        # Transition gates
        def resolve_gate(ctx: MiddlewareContext):
            if ctx.target != "resolve":
                return
            count = len(ctx.tract.log())
            if count < 5:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 5 commits for resolve (have {count})"],
                )

        def escalate_gate(ctx: MiddlewareContext):
            if ctx.target != "escalate":
                return
            count = len(ctx.tract.log())
            if count < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 commits for escalate (have {count})"],
                )

        t.use("pre_transition", resolve_gate)
        t.use("pre_transition", escalate_gate)

        for tag_name in ["bug", "feature-request", "billing", "critical",
                         "solution", "workaround", "escalated"]:
            t.register_tag(tag_name)

        print(f"  Initial configs: {t.get_all_configs()}")
        print(f"  Registered 7 support tags")

        t.system(
            "You are a customer support agent. Use commit() to save every "
            "analysis, classification, or recommendation. Include tags in "
            "your commit calls to classify content."
        )

        t.user(
            "Hi, I'm having a critical issue. Our API integration stopped "
            "working after your latest update. We're getting 500 errors on "
            "all POST requests to /api/v2/orders. GET requests work fine. "
            "This is blocking our production system. We tried rolling back "
            "our client code but the errors persist. Order ID: ORD-2847."
        )

        log = StepLogger()
        _tool_names = ["commit", "tag", "get_config", "status"]

        # =============================================================
        # Stage 1: Triage
        # =============================================================
        print("\n=== Stage 1: Triage ===\n")

        result = t.run(
            "Triage this support case. Commit 2 items:\n"
            "1. Issue classification (type=bug, severity=critical, "
            "endpoint=/api/v2/orders, symptom=500 on POST). "
            "Tag=['bug','critical']\n"
            "2. Context summary (GET works, POST fails, client rollback "
            "didn't help, suggests server-side breaking change). "
            "Tag=['bug']",
            max_steps=6, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Stage 2: Resolve (developer drives transition)
        # =============================================================
        print("\n\n=== Stage 2: Resolve ===\n")

        t.transition("resolve", handoff="summary")
        t.configure(stage="resolve", temperature=0.3)

        result = t.run(
            "Propose solutions for the API 500 error. Commit 2 items:\n"
            "1. Root cause analysis (likely: v2 API schema change broke POST "
            "validation, old request body format rejected). Tag=['solution']\n"
            "2. Proposed workaround (update POST body to match new v2 schema, "
            "or use v1 endpoint as temporary fallback). Tag=['workaround']",
            max_steps=6, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Stage 3: Escalate (developer drives transition)
        # =============================================================
        print("\n\n=== Stage 3: Escalate ===\n")

        t.transition("escalate", handoff="summary")
        t.configure(stage="escalate", temperature=0.1)

        result = t.run(
            "Prepare an escalation summary. Commit 1 item:\n"
            "Document what was tried (root cause identified, workaround "
            "proposed) and recommend next action (engineering team should "
            "add backward compatibility to v2 POST endpoint). "
            "Tag=['escalated']",
            max_steps=4, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Show final state
        # =============================================================

        print(f"\n=== Final State ===\n")

        print(f"  Stage: {t.get_config('stage')}")
        print(f"  Branch: {t.current_branch}")

        print(f"\n  Branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Tags with content:")
        for entry in t.list_tags():
            if entry["count"] > 0:
                print(f"    {entry['name']:20s} count={entry['count']}")

        print(f"\n  Log (last 8 commits):")
        pprint_log(t.log()[-8:])

        print(f"\n  Stages completed: 3/3")


if __name__ == "__main__":
    main()


# --- See also ---
# Coding workflow:       workflows/01_coding_assistant.py
# Self-routing:          workflows/09_self_routing.py
# Branch patterns:       agent/06_tangent_isolation.py
