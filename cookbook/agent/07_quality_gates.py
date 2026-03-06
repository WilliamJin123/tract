"""Agent Quality Gates

An LLM agent encounters require/block gates set up via rules. When the
agent tries to transition to the next stage but doesn't meet the gate
requirements, it adapts by doing more work to satisfy the conditions.

This demonstrates rules as agent guardrails -- the agent cannot skip
stages or bypass quality checks, and must genuinely complete work before
advancing.

Tools exercised: create_rule, transition, commit, get_config, status,
                 log, compile, annotate, branch

Demonstrates: Rule-based gates (require/block), agent adapting to gate
              failures, quality enforcement without hardcoded checks,
              BlockedByRuleError handling
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


# Tool profile: quality gate tools
GATE_PROFILE = ToolProfile(
    name="gate-navigator",
    tool_configs={
        "create_rule": ToolConfig(
            enabled=True,
            description=(
                "Create a rule on the current branch. Rules fire on events "
                "and can set_config, require actions, or block actions.\n"
                "For gates, use trigger='transition:{target}' with action "
                "{type: 'require', threshold: N, message: '...'} to enforce "
                "minimum commit counts before transitioning.\n"
                "Use action {type: 'block', message: '...'} with a condition "
                "to prevent transitions under certain circumstances."
            ),
        ),
        "transition": ToolConfig(
            enabled=True,
            description=(
                "Transition to a target branch/stage. Evaluates gate conditions "
                "first. If a 'require' or 'block' rule prevents the transition, "
                "returns None. The agent must do more work to satisfy the gate "
                "and try again."
            ),
        ),
        "commit": ToolConfig(
            enabled=True,
            description=(
                "Record content into the tract. Use content_type='dialogue' for "
                "messages, 'artifact' for deliverables, 'reasoning' for analysis. "
                "Each commit counts toward gate thresholds."
            ),
        ),
        "get_config": ToolConfig(
            enabled=True,
            description=(
                "Resolve a config value from active rules. Check 'stage' to "
                "see which stage you're in, or any other rule-defined config."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check current branch, HEAD, commit count, and token count. "
                "Use this to verify your position and assess gate readiness."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description="View recent commits to understand what work has been done.",
        ),
        "compile": ToolConfig(
            enabled=True,
            description="View current compiled context.",
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Mark a commit as 'pinned' to protect it, or 'skip' to exclude. "
                "Pinned artifacts may count toward quality thresholds."
            ),
        ),
        "branch": ToolConfig(
            enabled=True,
            description="Create a new branch for a workflow stage.",
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
    print("Agent Quality Gates: require/block rules as agent guardrails")
    print("=" * 70)
    print()
    print("  The agent encounters transition gates that enforce quality.")
    print("  When blocked, it must do more work to meet the requirements.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Register tools from the profile
        tools = t.as_tools(profile=GATE_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a software engineer working through a gated workflow.\n\n"
            "GATE PROTOCOL:\n"
            "- Each stage has rules that may block transitions if you haven't "
            "done enough work.\n"
            "- When a transition returns None/blocked, read the message to "
            "understand what's missing, do the required work, then try again.\n"
            "- Use commit to record work (artifacts, analysis, deliverables).\n"
            "- Use transition to advance when ready.\n"
            "- Do NOT skip stages or bypass gates."
        )

        # --- Phase 1: Set up gated workflow ---
        print("=== Phase 1: Set up gated stages ===\n")

        # Create the research stage with a gate to implementation
        t.branch("research", switch=True)
        t.rule(
            name="research-stage",
            trigger="active",
            action={"type": "set_config", "key": "stage", "value": "research"},
        )
        # Gate: require at least a few commits before transitioning to impl
        t.rule(
            name="research-gate",
            trigger="transition:implementation",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">=",
                    "value": 3,
                },
                "message": "Need at least 3 research commits before implementation",
            },
        )
        print("  Created 'research' branch with transition gate")

        # Create the implementation stage
        t.switch("main")
        t.branch("implementation", switch=True)
        t.rule(
            name="impl-stage",
            trigger="active",
            action={"type": "set_config", "key": "stage", "value": "implementation"},
        )
        print("  Created 'implementation' branch")
        t.switch("research")

        print(f"  Starting on: {t.current_branch}")
        print(f"  Branches: {[b.name for b in t.list_branches()]}")

        # --- Phase 2: Agent tries to transition too early ---
        print("\n\n=== Phase 2: Agent attempts premature transition ===\n")
        result = t.run(
            "You are on the 'research' stage. There is a gate requiring at "
            "least 3 commits before you can transition to 'implementation'.\n\n"
            "First, try to transition to 'implementation' immediately to see "
            "the gate block you. Then, do the required research work:\n"
            "1. Commit a research artifact about API authentication options\n"
            "2. Commit a research artifact about database schema design\n"
            "3. Commit a research artifact about error handling patterns\n\n"
            "After completing the research, try the transition again.",
            max_steps=15, on_step=_log_step,
        )
        result.pprint()

        # --- Phase 3: Show final state ---
        print("\n\n=== Final State ===\n")
        print(f"  Current branch: {t.current_branch}")
        branches = [b.name for b in t.list_branches()]
        print(f"  All branches: {branches}")

        status = t.status()
        print(f"  Commits: {status.commit_count}")
        print(f"  Tokens: {status.token_count}")

        print("\n  Current context:")
        t.compile().pprint(style="compact")


if __name__ == "__main__":
    main()
