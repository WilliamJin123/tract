"""Coding Assistant Workflow: design -> implementation -> validation

An agent-driven multi-stage workflow. Config sets stage-specific settings
(temperature, compile strategy) and middleware gates transitions. The agent
gets transition tools and decides when to move between stages -- one t.llm.run()
call drives the whole pipeline.

Stages:
  design         -- high temperature (0.9), creative exploration
  implementation -- low temperature (0.3), precise code generation
  validation     -- minimal temperature (0.1), deterministic testing

Demonstrates: t.config.set() for stage configs, t.directive() for stage
              instructions, pre_transition middleware gates, agent-driven
              stage navigation, compile strategy per stage

Requires: LLM API key (uses Cerebras provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError, MiddlewareContext
from tract.formatting import pprint_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no Claude Code credentials)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # =============================================================
        # Stage configuration via config and directives
        # =============================================================

        print("=== Setting Up Workflow ===\n")

        # Initial stage and settings
        t.config.set(
            stage="design",
            temperature=0.9,
            compile_strategy="full",
        )

        # Directive: tell the agent about the workflow structure
        t.directive(
            "workflow-stages",
            "This conversation follows a three-stage workflow:\n"
            "1. DESIGN -- Brainstorm and outline architecture\n"
            "2. IMPLEMENTATION -- Write precise, working code\n"
            "3. VALIDATION -- Write tests and verify correctness\n"
            "Use get_config to check the current stage. "
            "Use transition to move between stages.",
        )

        # Transition gates: require minimum commit count before advancing
        def impl_gate(ctx: MiddlewareContext):
            if ctx.target != "implementation":
                return
            count = len(ctx.tract.search.log())
            if count < 6:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 6 commits for implementation (have {count})"],
                )

        def validation_gate(ctx: MiddlewareContext):
            if ctx.target != "validation":
                return
            count = len(ctx.tract.search.log())
            if count < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 commits for validation (have {count})"],
                )

        t.middleware.add("pre_transition", impl_gate)
        t.middleware.add("pre_transition", validation_gate)

        configs = t.config.get_all()
        print(f"  Initial configs: {configs}")

        # =============================================================
        # System prompt: tell the agent about the workflow
        # =============================================================

        t.system(
            "You are a software engineer working through a structured workflow.\n\n"
            "WORKFLOW STAGES:\n"
            "1. DESIGN -- Brainstorm and outline the architecture. Use high creativity.\n"
            "2. IMPLEMENTATION -- Write precise, working code. Be exact.\n"
            "3. VALIDATION -- Write tests and verify correctness. Be deterministic.\n\n"
            "You have access to tools including 'transition' to move between stages.\n"
            "Use get_config to check the current stage. Use transition when you\n"
            "believe you have enough content to advance.\n\n"
            "IMPORTANT: After completing design, transition to 'implementation'.\n"
            "After implementing, transition to 'validation'. Complete all three stages."
        )

        # =============================================================
        # Run: agent drives through all stages
        # =============================================================

        print("\n=== Running Agent (design -> implementation -> validation) ===\n")

        log = StepLogger()

        result = t.llm.run(
            "Design and implement a simple stack data structure in Python with "
            "push, pop, peek, and is_empty methods. Then write 3 test cases.\n\n"
            "Start by designing the interface (what methods, what types, edge cases). "
            "When design is complete, transition to 'implementation' to write code. "
            "When implementation is complete, transition to 'validation' to write tests.",
            max_steps=15,
            profile="full",
            tool_names=["commit", "transition", "get_config", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )

        result.pprint()

        # =============================================================
        # Show final state
        # =============================================================

        print(f"\n=== Final State ===\n")

        final_configs = t.config.get_all()
        print(f"  Active configs: {final_configs}")

        print(f"\n  Branches:")
        for b in t.branches.list():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Log (last 8 commits):")
        pprint_log(t.search.log()[-8:])


if __name__ == "__main__":
    main()


# --- See also ---
# Config and directives:  getting_started/02_config_and_directives.py
# Self-routing workflow:   workflows/07_self_routing.py
# Customer support:        workflows/02_customer_support.py
