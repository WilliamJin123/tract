"""Implicit Behavior Discovery

Three scenarios where an LLM agent is given tract tools and a task, but
never told *how* to use them. The agent must discover the right behavior
on its own: branching to isolate conflicting work, navigating pre-built
workflow stages, and adapting when quality gates block premature actions.

Pattern: set up tract with tools/config, give agent a task, observe
         whether it discovers the capabilities autonomously.

Scenarios:
  1. Branch Discovery   -- isolate conflicting analyses on branches
  2. Stage Navigation   -- discover and traverse a staged workflow
  3. Quality Gate Adapt  -- satisfy hidden middleware constraints
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError, MiddlewareContext
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _header(title: str, description: str) -> None:
    print()
    print("=" * 70)
    print(f"  Scenario: {title}")
    print("=" * 70)
    print(f"  {description}")
    print()


def _branch_report(t: Tract) -> None:
    """Print compiled context for every branch."""
    branches = t.branches.list()
    print(f"\n  Branches: {[b.name for b in branches]}")
    print(f"  Current:  {t.current_branch}")
    original = t.current_branch
    for branch in branches:
        t.branches.switch(branch.name)
        print(f"\n  [{branch.name}]:")
        t.compile().pprint(style="compact")
    t.branches.switch(original)


# ---------------------------------------------------------------------------
# Scenario 1 -- Branch Discovery
# ---------------------------------------------------------------------------
# The agent must draft two independent, conflicting technical proposals.
# Branches are available but never suggested. Does the agent isolate them?

BRANCH_PROFILE = ToolProfile(
    name="researcher",
    tool_configs={
        "branch": ToolConfig(enabled=True),
        "switch": ToolConfig(enabled=True),
        "list_branches": ToolConfig(enabled=True),
        "merge": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
    },
)


def scenario_branch_discovery() -> None:
    _header(
        "Branch Discovery",
        "Two conflicting proposals -- will the agent use branches for isolation?",
    )

    with Tract.open(
        api_key=llm.api_key, base_url=llm.base_url,
        model=MODEL_ID, auto_message=llm.small,
        tool_profile=BRANCH_PROFILE,
    ) as t:
        t.system(
            "You are a senior solutions architect evaluating backend "
            "architecture options for a new product."
        )
        t.user(
            "Building a SaaS analytics platform. 10k concurrent users, "
            "50TB warehouse, sub-second queries. 12 engineers. "
            "Need to pick: microservices vs monolith."
        )
        t.assistant("I'll evaluate both options.")

        log = StepLogger()
        result = t.llm.run(
            "Write two short proposals:\n"
            "A) Why microservices is the right call\n"
            "B) Why monolith is the right call\n\n"
            "Each must stand on its own -- if one influences the other "
            "they won't be genuine independent evaluations.",
            max_steps=12, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        _branch_report(t)
        branches = t.branches.list()
        if len(branches) > 1:
            print(f"\n  Agent created {len(branches) - 1} branch(es) for isolation.")
        else:
            print("\n  Agent did not use branches.")


# ---------------------------------------------------------------------------
# Scenario 2 -- Stage Navigation
# ---------------------------------------------------------------------------
# The developer pre-creates stage branches with config metadata. The agent
# must discover the stages, produce deliverables, and navigate through them.

STAGE_PROFILE = ToolProfile(
    name="architect",
    tool_configs={
        "get_config": ToolConfig(enabled=True),
        "transition": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
        "switch": ToolConfig(enabled=True),
        "list_branches": ToolConfig(enabled=True),
    },
)


def scenario_stage_navigation() -> None:
    _header(
        "Stage Navigation",
        "Pre-built stages (design/implementation/validation) -- will the agent find and use them?",
    )

    with Tract.open(
        api_key=llm.api_key, base_url=llm.base_url,
        model=MODEL_ID, auto_message=llm.small,
        tool_profile=STAGE_PROFILE,
    ) as t:
        t.system(
            "You are a software architect. Complete the task by working "
            "through each stage of the workflow. Use get_config and "
            "list_branches to understand the available infrastructure."
        )

        # Developer pre-creates stage branches with config metadata
        for stage, temp in [("design", 0.9), ("implementation", 0.3), ("validation", 0.5)]:
            t.branches.create(stage, switch=True)
            t.config.set(stage=stage, temperature=temp)
            t.branches.switch("main")

        t.branches.switch("design")
        print(f"  Branches: {[b.name for b in t.branches.list()]}")
        print(f"  Starting on: {t.current_branch}")

        log = StepLogger()
        result = t.llm.run(
            "Design a task management REST API (title, status, assignee). "
            "Work through the available stages (design, implementation, "
            "validation) to produce a complete specification. Commit your "
            "deliverables at each stage and transition when ready.",
            max_steps=18, max_tokens=4096,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Report -- check which stages the agent visited
        print("\n\n  --- Final State ---\n")
        branches_visited = set()
        for stage in ["design", "implementation", "validation"]:
            t.branches.switch(stage)
            ctx = t.compile()
            cfg = t.config.get("stage")
            if ctx.token_count > 50:
                branches_visited.add(stage)
            print(f"  [{stage}] stage={cfg}, {len(ctx.messages)} msgs, "
                  f"{ctx.token_count} tokens")

        if len(branches_visited) >= 3:
            print(f"\n  Agent navigated all 3 stages.")
        elif branches_visited:
            print(f"\n  Agent visited {len(branches_visited)} stage(s): "
                  f"{sorted(branches_visited)}")
        else:
            print("\n  Agent did not navigate to any stages.")


# ---------------------------------------------------------------------------
# Scenario 3 -- Quality Gate Adaptation
# ---------------------------------------------------------------------------
# Middleware gates block premature transitions. The agent must produce enough
# work to pass a gate it does not know about in advance, then adapt when
# the gate blocks its first attempt.

GATE_PROFILE = ToolProfile(
    name="engineer",
    tool_configs={
        "transition": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
        "get_config": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
    },
)


def scenario_quality_gate() -> None:
    _header(
        "Quality Gate Adaptation",
        "Hidden middleware blocks transitions until 3 artifacts are committed. Can the agent adapt?",
    )

    with Tract.open(
        api_key=llm.api_key, base_url=llm.base_url,
        model=MODEL_ID, auto_message=llm.small,
        tool_profile=GATE_PROFILE,
    ) as t:
        t.system(
            "You are a software engineer working on an API project. "
            "Research topics thoroughly before moving to implementation."
        )

        # Developer sets up gated workflow infrastructure
        t.branches.create("research", switch=True)
        t.config.set(stage="research")

        # Gate: require at least 3 artifact commits before transition
        def research_gate(ctx: MiddlewareContext):
            if ctx.target == "implementation":
                entries = ctx.tract.search.log(limit=50)
                artifacts = [e for e in entries if e.content_type == "artifact"]
                if len(artifacts) < 3:
                    raise BlockedError(
                        "pre_transition",
                        f"Research incomplete: {len(artifacts)} artifact(s) "
                        f"committed, need at least 3 before moving to implementation.",
                    )

        t.middleware.add("pre_transition", research_gate)

        t.branches.switch("main")
        t.branches.create("implementation", switch=True)
        t.config.set(stage="implementation")
        t.branches.switch("research")

        print(f"  Starting on: {t.current_branch}")
        print(f"  Branches: {[b.name for b in t.branches.list()]}")

        log = StepLogger()
        result = t.llm.run(
            "Research authentication, database schema, and error handling "
            "patterns for a REST API. When you feel your research is "
            "thorough, move to implementation.",
            max_steps=15, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # If the agent never tried transitioning, force an attempt to demo the gate
        if t.current_branch != "implementation":
            print("\n  --- Agent didn't transition, forcing attempt ---")
            entries = t.search.log(limit=50)
            artifacts = [e for e in entries if e.content_type == "artifact"]
            print(f"  Artifacts committed: {len(artifacts)}")
            try:
                t.transition("implementation")
                print("  Transition succeeded!")
            except BlockedError as e:
                print(f"  Gate blocked: {e}")

        # Report
        print("\n\n  --- Final State ---\n")
        print(f"  Current branch: {t.current_branch}")
        status = t.search.status()
        print(f"  Commits: {status.commit_count}, Tokens: {status.token_count}")
        print("\n  Context:")
        t.compile().pprint(style="compact")
        print(f"\n  Reached implementation: {t.current_branch == 'implementation'}")


# ---------------------------------------------------------------------------
# Main -- run all three scenarios
# ---------------------------------------------------------------------------

def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 70)
    print("  Implicit Behavior Discovery")
    print("  Three scenarios testing whether an LLM agent discovers")
    print("  tract capabilities without explicit instructions.")
    print("=" * 70)

    scenario_branch_discovery()
    scenario_stage_navigation()
    scenario_quality_gate()

    print("\n\nDone. All three scenarios complete.")


if __name__ == "__main__":
    main()


# --- See also ---
# Branching basics (no LLM):     reference/03_branching.py
# Config per branch:              config_and_middleware/01_config_and_precedence.py
# Middleware patterns (no LLM):   config_and_middleware/02_event_automation.py
