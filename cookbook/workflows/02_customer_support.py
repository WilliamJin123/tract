"""Customer Support Workflow: triage -> resolve -> escalate

Developer-driven staged workflow. The developer controls all transitions;
the agent decides what content to produce at each stage based on the
problem context. The prompts describe the goal, not the deliverables.

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

        t.config.set(
            stage="triage",
            temperature=0.5,
            compile_strategy="full",
            max_resolution_attempts=2,
        )

        # Transition gates
        def resolve_gate(ctx: MiddlewareContext):
            if ctx.target != "resolve":
                return
            count = len(ctx.tract.search.log())
            if count < 5:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 5 commits for resolve (have {count})"],
                )

        def escalate_gate(ctx: MiddlewareContext):
            if ctx.target != "escalate":
                return
            count = len(ctx.tract.search.log())
            if count < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 commits for escalate (have {count})"],
                )

        t.middleware.add("pre_transition", resolve_gate)
        t.middleware.add("pre_transition", escalate_gate)

        for tag_name in ["bug", "feature-request", "billing", "critical",
                         "solution", "workaround", "escalated"]:
            t.tags.register(tag_name)

        print(f"  Initial configs: {t.config.get_all()}")
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

        result = t.llm.run(
            "Analyze this support ticket. Classify its severity and gather "
            "relevant context. What type of issue is this, what's affected, "
            "and what has the customer already tried?",
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
        t.config.set(stage="resolve", temperature=0.3)

        result = t.llm.run(
            "Based on the triage, propose solutions. What's the likely root "
            "cause of the 500 errors on POST? What workarounds can the "
            "customer use right now while a permanent fix is developed?",
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
        t.config.set(stage="escalate", temperature=0.1)

        result = t.llm.run(
            "Prepare an escalation summary for the engineering team. "
            "Document what was investigated, what was tried, and recommend "
            "specific next actions to permanently resolve this issue.",
            max_steps=4, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Show final state
        # =============================================================

        print(f"\n=== Final State ===\n")

        print(f"  Stage: {t.config.get('stage')}")
        print(f"  Branch: {t.current_branch}")

        print(f"\n  Branches:")
        for b in t.branches.list():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Tags with content:")
        for entry in t.tags.list():
            if entry["count"] > 0:
                print(f"    {entry['name']:20s} count={entry['count']}")

        print(f"\n  Log (last 8 commits):")
        pprint_log(t.search.log()[-8:])

        print(f"\n  Stages completed: 3/3")


if __name__ == "__main__":
    main()


# --- See also ---
# Coding workflow:       workflows/01_coding_assistant.py
# Self-routing:          workflows/07_self_routing.py
# Branch patterns:       agent/01_implicit_discovery.py
