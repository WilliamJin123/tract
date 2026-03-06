"""Agent-Driven Staged Workflow

An LLM agent navigates a multi-stage task (design -> implementation ->
validation) using rules and transitions. Rules set stage-specific config
(temperature, compile_strategy) and transition gates enforce that the
agent completes enough work before advancing.

The agent decides when to transition between stages based on its own
assessment of readiness, not hardcoded triggers.

Tools exercised: create_rule, get_config, transition, commit, compile,
                 status, log, branch

Demonstrates: Rule-based stage configuration, agent-managed transitions,
              gate conditions, stage-specific LLM settings
"""

import io
import json
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.large


# Tool profile: staged workflow tools
STAGE_PROFILE = ToolProfile(
    name="stage-navigator",
    tool_configs={
        "create_rule": ToolConfig(
            enabled=True,
            description=(
                "Create a rule on the current branch. Rules fire on events "
                "(commit, compile, transition) and can set config, require/block "
                "actions, or trigger operations. Use this to set up stage-specific "
                "configuration and transition gates."
            ),
        ),
        "get_config": ToolConfig(
            enabled=True,
            description=(
                "Resolve a config value from active rules. Returns the value "
                "set by the closest rule to HEAD. Use to check current stage "
                "settings like temperature or compile_strategy."
            ),
        ),
        "transition": ToolConfig(
            enabled=True,
            description=(
                "Transition to a target branch/stage. Evaluates gate conditions "
                "(require/block rules), runs pre-transition actions, and switches "
                "to the target branch. Returns None if blocked by gates."
            ),
        ),
        "commit": ToolConfig(
            enabled=True,
            description=(
                "Record content into the tract. Use content_type='dialogue' "
                "with role='assistant' for your responses, or content_type="
                "'artifact' for structured deliverables."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description="View current compiled context to verify stage state.",
        ),
        "status": ToolConfig(
            enabled=True,
            description="Check current branch/stage, HEAD, and token count.",
        ),
        "log": ToolConfig(
            enabled=True,
            description="View recent commits on the current branch.",
        ),
        "branch": ToolConfig(
            enabled=True,
            description=(
                "Create a new branch for a stage. Each stage of the workflow "
                "lives on its own branch with stage-specific rules."
            ),
        ),
    },
)


def _log_step(step_num, response):
    """on_step callback -- print step number."""
    print(f"    [step {step_num}]")


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 70)
    print("Agent-Driven Staged Workflow: design -> implementation -> validation")
    print("=" * 70)
    print()
    print("  The agent navigates a multi-stage task using rules and transitions.")
    print("  Each stage has its own branch with stage-specific config rules.")
    print("  The agent decides when it's ready to transition to the next stage.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Register tools from the profile
        tools = t.as_tools(profile=STAGE_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a software architect working on a staged project.\n\n"
            "WORKFLOW PROTOCOL:\n"
            "1. You start on 'main'. Set up stage branches with rules.\n"
            "2. Each stage has its own branch: 'design', 'implementation', 'validation'.\n"
            "3. Create rules on each branch for stage-specific config.\n"
            "4. Use transition() to move between stages when ready.\n"
            "5. Work through all three stages to complete the task.\n\n"
            "For setup, create the branches and rules first, then transition."
        )

        # --- Phase 1: Set up the workflow ---
        print("=== Phase 1: Set up stage branches and rules ===\n")
        result = t.run(
            "Set up a three-stage workflow for building a REST API:\n\n"
            "1. Create branch 'design' and switch to it. Add a rule:\n"
            "   - name: 'design-config'\n"
            "   - trigger: 'active'\n"
            "   - action: {type: 'set_config', key: 'stage', value: 'design'}\n"
            "Then switch back to main.\n\n"
            "2. Create branch 'implementation' and switch to it. Add a rule:\n"
            "   - name: 'impl-config'\n"
            "   - trigger: 'active'\n"
            "   - action: {type: 'set_config', key: 'stage', value: 'implementation'}\n"
            "Then switch back to main.\n\n"
            "3. Create branch 'validation' and switch to it. Add a rule:\n"
            "   - name: 'validation-config'\n"
            "   - trigger: 'active'\n"
            "   - action: {type: 'set_config', key: 'stage', value: 'validation'}\n"
            "Then switch back to main.\n\n"
            "After setup, use get_config to verify the stage config on main "
            "and check status.",
            max_steps=20, on_step=_log_step,
        )
        print(f"\n  Setup result: {result.status} ({result.steps} steps)")
        if result.final_response:
            print(f"  Agent: {result.final_response[:200]}")

        # --- Phase 2: Work through the stages ---
        print("\n\n=== Phase 2: Design stage ===\n")
        # Switch to design branch for the design work
        t.switch("design")
        result = t.run(
            "You are now on the 'design' stage branch. Use get_config to "
            "verify the stage is 'design'. Then do your design work: commit "
            "an artifact with the API endpoint design for a task management "
            "REST API (use content_type='artifact', text='...'). Include "
            "at least the URL structure and HTTP methods. After your design "
            "is committed, check status.",
            max_steps=10, on_step=_log_step,
        )
        print(f"\n  Design result: {result.status} ({result.steps} steps)")
        if result.final_response:
            print(f"  Agent: {result.final_response[:200]}")

        print("\n  Design stage context:")
        t.compile().pprint(style="compact")

        # --- Phase 3: Implementation stage ---
        print("\n\n=== Phase 3: Implementation stage ===\n")
        t.switch("implementation")
        result = t.run(
            "You are now on the 'implementation' stage branch. Verify the "
            "stage with get_config. Then commit an artifact with a brief "
            "implementation plan: list 3-4 key modules/files needed and "
            "their responsibilities. Check status when done.",
            max_steps=10, on_step=_log_step,
        )
        print(f"\n  Implementation result: {result.status} ({result.steps} steps)")
        if result.final_response:
            print(f"  Agent: {result.final_response[:200]}")

        # --- Phase 4: Validation stage ---
        print("\n\n=== Phase 4: Validation stage ===\n")
        t.switch("validation")
        result = t.run(
            "You are now on the 'validation' stage branch. Verify the "
            "stage with get_config. Commit an artifact with a validation "
            "checklist: 3-4 items to verify the API design is correct "
            "(e.g., RESTful conventions, error handling, auth). Check "
            "status when done.",
            max_steps=10, on_step=_log_step,
        )
        print(f"\n  Validation result: {result.status} ({result.steps} steps)")
        if result.final_response:
            print(f"  Agent: {result.final_response[:200]}")

        # --- Final state ---
        print("\n\n=== Final State ===\n")
        branches = [b.name for b in t.list_branches()]
        print(f"  Branches: {branches}")
        print(f"  Current: {t.current_branch}")

        # Show each stage's work
        for stage in ["design", "implementation", "validation"]:
            t.switch(stage)
            ctx = t.compile()
            print(f"\n  [{stage}] {len(ctx.messages)} messages, {ctx.token_count} tokens")


if __name__ == "__main__":
    main()
