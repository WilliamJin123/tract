"""Customer Support Workflow: triage -> resolve -> escalate

An agent-driven support workflow. The agent triages a customer issue,
attempts resolution, and escalates if the quality gate fails. Branches
are used to explore alternative solutions, and middleware gates enforce
quality standards at each stage transition.

Stages:
  triage   -- classify severity, gather context (temperature 0.5)
  resolve  -- propose and test solutions (temperature 0.3)
  escalate -- document failure, prepare handoff (temperature 0.1)

Demonstrates: branching for solution exploration, middleware quality gates,
              escalation on gate failure, compile strategy per stage

Requires: LLM API key (uses Cerebras provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


def main():
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

        # Initial stage config
        t.configure(
            stage="triage",
            temperature=0.5,
            compile_strategy="full",
            max_resolution_attempts=2,
        )

        # Transition gates via middleware
        def resolve_gate(ctx):
            """Require enough triage context before resolving."""
            if ctx.target != "resolve":
                return
            count = len(ctx.tract.log())
            if count < 5:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 5 commits for resolve (have {count})"],
                )

        def escalate_gate(ctx):
            """Low bar -- agent can always escalate after minimal context."""
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

        configs = t.get_all_configs()
        print(f"  Initial configs: {configs}")

        # =============================================================
        # Register support tags
        # =============================================================

        for tag_name in ["bug", "feature-request", "billing", "critical",
                         "solution", "workaround", "escalated"]:
            t.register_tag(tag_name)

        print(f"  Registered 7 support tags")

        # =============================================================
        # System prompt: describe the support workflow
        # =============================================================

        t.system(
            "You are a customer support agent working through a structured workflow.\n\n"
            "WORKFLOW STAGES:\n"
            "1. TRIAGE -- Classify the issue (bug/feature/billing), assess severity.\n"
            "   Tag with appropriate category. Gather all needed context.\n"
            "2. RESOLVE -- Propose solutions. Use branching to explore alternatives.\n"
            "   Create a branch for each solution attempt. If a solution works,\n"
            "   tag it 'solution'. Check get_config('max_resolution_attempts').\n"
            "3. ESCALATE -- If resolution fails, transition here. Document what\n"
            "   was tried, tag as 'escalated', prepare handoff summary.\n\n"
            "Tools: commit, compile, status, log, branch, switch, tag,\n"
            "register_tag, create_metadata, get_config, transition.\n\n"
            "Use get_config to check stage and max_resolution_attempts.\n"
            "Use transition to move between stages. If you cannot resolve\n"
            "the issue after attempting solutions, transition to 'escalate'."
        )

        # =============================================================
        # Customer issue
        # =============================================================

        t.user(
            "Hi, I'm having a critical issue. Our API integration stopped "
            "working after your latest update. We're getting 500 errors on "
            "all POST requests to /api/v2/orders. GET requests work fine. "
            "This is blocking our production system. We tried rolling back "
            "our client code but the errors persist. Order ID: ORD-2847."
        )

        # =============================================================
        # Run: agent triages, attempts resolution, escalates if needed
        # =============================================================

        print("\n=== Running Agent (triage -> resolve -> escalate) ===\n")

        log = StepLogger()

        result = t.run(
            "Handle this customer support case. Follow the workflow:\n\n"
            "1. TRIAGE: Classify the issue severity and type. Tag appropriately.\n"
            "   Gather context about what's failing (POST 500s, /api/v2/orders).\n"
            "   When triage is complete, transition to 'resolve'.\n\n"
            "2. RESOLVE: Propose a fix. Create a branch 'solution/api-fix' to\n"
            "   explore the solution. Suggest the likely cause (breaking API change\n"
            "   in v2, possible schema validation issue). If the solution seems\n"
            "   viable, tag it as 'solution'. If not, and you've hit the\n"
            "   max_resolution_attempts, transition to 'escalate'.\n\n"
            "3. ESCALATE (if needed): Document what was tried, create a metadata\n"
            "   entry with the escalation summary, tag as 'escalated'.",
            max_steps=20,
            profile="full",
            tool_names=["commit", "tag", "register_tag", "branch", "switch",
                        "transition", "create_metadata", "get_config", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
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

        print(f"\n  Log (last 8 commits):")
        for ci in t.log()[-8:]:
            tags_str = f" [{', '.join(ci.tags)}]" if ci.tags else ""
            print(f"    {ci.commit_hash[:8]}  {ci.content_type:10s}{tags_str}  "
                  f"{(ci.message or '')[:45]}")


if __name__ == "__main__":
    main()


# --- See also ---
# Coding workflow:       workflows/01_coding_assistant.py
# Research pipeline:     workflows/02_research_pipeline.py
# Branch patterns:       agent/06_tangent_isolation.py
