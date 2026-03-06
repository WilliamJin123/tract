"""Coding Assistant Workflow: design -> implementation -> validation

An agent-driven multi-stage workflow. Rules define stage-specific configs
(temperature, compile strategy) and transition gates. The agent gets
transition tools and decides when to move between stages -- one t.run()
call drives the whole pipeline.

Stages:
  design         -- high temperature (0.9), creative exploration
  implementation -- low temperature (0.3), precise code generation
  validation     -- minimal temperature (0.1), deterministic testing

Demonstrates: rules as stage configs, transition gates, agent-driven
              stage navigation, compile strategy per stage

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract, resolve_all_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.small


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # =============================================================
        # Stage configuration via rules
        # =============================================================

        print("=== Setting Up Workflow Rules ===\n")

        # Initial stage
        t.rule("stage", trigger="active",
               action={"type": "set_config", "key": "stage", "value": "design"})

        # Design stage: creative, full context
        t.rule("design-temp", trigger="active",
               action={"type": "set_config", "key": "temperature", "value": 0.9})
        t.rule("design-strategy", trigger="active",
               action={"type": "set_config", "key": "compile_strategy", "value": "full"})

        # Transition gates: require minimum commit count before advancing
        t.rule(
            "impl-gate",
            trigger="transition:implementation",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">=",
                    "value": 6,
                },
            },
        )
        t.rule(
            "validation-gate",
            trigger="transition:validation",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">=",
                    "value": 3,
                },
            },
        )

        configs = resolve_all_configs(t.rule_index)
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

        result = t.run(
            "Design and implement a simple stack data structure in Python with "
            "push, pop, peek, and is_empty methods. Then write 3 test cases.\n\n"
            "Start by designing the interface (what methods, what types, edge cases). "
            "When design is complete, transition to 'implementation' to write code. "
            "When implementation is complete, transition to 'validation' to write tests.",
            max_steps=15,
            on_step=lambda step, _resp: print(f"  step {step}..."),
        )

        print(f"\n=== Result ===\n")
        print(f"  Status:     {result.status}")
        print(f"  Reason:     {result.reason}")
        print(f"  Steps:      {result.steps}")
        print(f"  Tool calls: {result.tool_calls}")

        if result.final_response:
            print(f"\n  Response:\n  {result.final_response[:300]}")

        # =============================================================
        # Show final state
        # =============================================================

        print(f"\n=== Final State ===\n")

        final_configs = resolve_all_configs(t.rule_index)
        print(f"  Active configs: {final_configs}")

        print(f"\n  Branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Log (last 8 commits):")
        for ci in t.log()[-8:]:
            print(f"    {ci.commit_hash[:8]}  {ci.content_type:10s}  {ci.message[:50]}")


if __name__ == "__main__":
    main()


# --- See also ---
# Rules:                getting_started/02_rules.py
# Research workflow:    workflows/02_research_pipeline.py
# Customer support:     workflows/03_customer_support.py
