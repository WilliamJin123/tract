"""Dynamic Tool Profile Switching

Demonstrates how an agent's available tools change as it progresses
through workflow stages. Two patterns:

  1. Stage-Based Profile Switching -- pre_transition middleware swaps
     the entire tool profile when the agent moves between stages.

  2. Progressive Capability Discovery -- agent starts with minimal
     tools and earns new capabilities by producing work. Middleware
     unlocks tools after demonstrated progress.

Demonstrates: ToolProfile, ToolConfig, switch_profile(), lock_tool(),
              unlock_tool(), pre_transition middleware, post_commit
              middleware, t.llm.run() with dynamic profiles
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, MiddlewareContext
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


def _section(num: int, title: str) -> None:
    print(f"\n{'=' * 70}\n  {num}. {title}\n{'=' * 70}\n")


def _tool_names(t: Tract, profile: ToolProfile) -> list[str]:
    """Extract tool names from a profile."""
    return sorted(td["function"]["name"] for td in t.toolkit.as_tools(profile=profile))


# =====================================================================
# Section 1: Stage-Based Profile Switching
# =====================================================================
# Three profiles with different capability sets per stage.

RESEARCH_PROFILE = ToolProfile(
    name="research",
    tool_configs={
        "commit": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
        "get_config": ToolConfig(enabled=True),
        "tag": ToolConfig(enabled=True),
    },
)

IMPLEMENTATION_PROFILE = ToolProfile(
    name="implementation",
    tool_configs={
        "commit": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
        "branch": ToolConfig(enabled=True),
        "switch": ToolConfig(enabled=True),
        "merge": ToolConfig(enabled=True),
        "compress": ToolConfig(enabled=True),
    },
)

REVIEW_PROFILE = ToolProfile(
    name="review",
    tool_configs={
        "commit": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
        "compile": ToolConfig(enabled=True),
        "get_config": ToolConfig(enabled=True),
    },
)

STAGE_PROFILES = {
    "research": RESEARCH_PROFILE,
    "implementation": IMPLEMENTATION_PROFILE,
    "review": REVIEW_PROFILE,
}


def stage_based_switching() -> None:
    _section(1, "Stage-Based Profile Switching")
    print("  research:       commit, status, log, get_config, tag")
    print("  implementation: commit, status, log, branch, switch, merge, compress")
    print("  review:         commit, status, log, compile, get_config\n")

    log = StepLogger()

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
        auto_message=llm.small,
        tool_profile=RESEARCH_PROFILE,
    ) as t:
        t.system(
            "You are a software architect working through a staged workflow. "
            "Use the tools available to you at each stage."
        )

        # Set up stage branches
        for stage in ["research", "implementation", "review"]:
            t.branch(stage, switch=True)
            t.config.set(stage=stage)
            t.switch("main")
        t.switch("research")

        # Middleware: swap profile on stage transition
        def profile_switcher(ctx: MiddlewareContext):
            target = ctx.target
            if target in STAGE_PROFILES:
                ctx.tract.toolkit.switch_profile(STAGE_PROFILES[target])
                print(f"\n  >> Profile switched to '{target}'")

        t.middleware.add("pre_transition", profile_switcher)

        # --- Research stage ---
        print(f"  --- RESEARCH ---")
        print(f"  Tools: {_tool_names(t, RESEARCH_PROFILE)}")

        result = t.llm.run(
            "Research authentication patterns for a REST API. "
            "Commit 2 findings: one on JWT tokens, one on session-based auth.",
            max_steps=6, max_tokens=1024,
            profile=RESEARCH_PROFILE,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Verify branch is NOT available in research profile
        print(f"\n  'branch' in research profile: {'branch' in _tool_names(t, RESEARCH_PROFILE)}")

        # --- Transition to implementation ---
        print(f"\n  --- IMPLEMENTATION (after transition) ---")
        t.transition("implementation")
        print(f"  Tools: {_tool_names(t, IMPLEMENTATION_PROFILE)}")

        result = t.llm.run(
            "Create a branch called 'jwt-auth' and commit a JWT implementation plan.",
            max_steps=6, max_tokens=1024,
            profile=IMPLEMENTATION_PROFILE,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # --- Transition to review ---
        print(f"\n  --- REVIEW (after transition) ---")
        t.transition("review")
        review_names = _tool_names(t, REVIEW_PROFILE)
        print(f"  Tools: {review_names}")

        result = t.llm.run(
            "Review the implementation by compiling the current context. "
            "Commit a summary of your review findings.",
            max_steps=6, max_tokens=1024,
            profile=REVIEW_PROFILE,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Verify destructive tools locked out in review
        print(f"\n  'branch' in review: {'branch' in review_names}")
        print(f"  'merge' in review:  {'merge' in review_names}")

    print("\n  PASSED")


# =====================================================================
# Section 2: Progressive Capability Discovery
# =====================================================================

MINIMAL_PROFILE = ToolProfile(
    name="minimal",
    tool_configs={
        "commit": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
    },
)


def progressive_discovery() -> None:
    _section(2, "Progressive Capability Discovery")
    print("  Start: commit, status")
    print("  After 3 commits: + branch, switch, list_branches, log")
    print("  After branch commit: + merge, compress\n")

    log = StepLogger()
    unlocked = {"branching": False, "merging": False}

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
        auto_message=llm.small,
        tool_profile=MINIMAL_PROFILE,
    ) as t:
        t.system(
            "You are a research agent. Start by committing findings. "
            "New tools will become available as you make progress."
        )

        # Middleware: unlock tools based on demonstrated progress
        def progressive_unlock(ctx: MiddlewareContext):
            entries = ctx.tract.log(limit=50)

            # After 3 commits -> unlock branching
            if len(entries) >= 3 and not unlocked["branching"]:
                for tool in ["branch", "switch", "list_branches", "log"]:
                    ctx.tract.toolkit.unlock_tool(tool)
                unlocked["branching"] = True
                print("\n  >> UNLOCKED: branch, switch, list_branches, log")

            # After creating a branch and committing there -> unlock merge
            if unlocked["branching"] and not unlocked["merging"]:
                branches = ctx.tract.list_branches()
                non_main = [b for b in branches if b.name != "main"]
                for branch in non_main:
                    ctx.tract.switch(branch.name)
                    if len(ctx.tract.log(limit=10)) >= 1:
                        for tool in ["merge", "compress"]:
                            ctx.tract.toolkit.unlock_tool(tool)
                        unlocked["merging"] = True
                        print("\n  >> UNLOCKED: merge, compress")
                        break
                ctx.tract.switch(ctx.tract.current_branch)

        t.middleware.add("post_commit", progressive_unlock)

        # --- Phase 1: Minimal tools ---
        print(f"  --- Phase 1: Minimal ---")
        print(f"  Tools: {_tool_names(t, MINIMAL_PROFILE)}")

        result = t.llm.run(
            "Research and commit 3 separate findings about database indexing: "
            "B-tree indexes, hash indexes, and composite indexes.",
            max_steps=8, max_tokens=1024,
            profile=MINIMAL_PROFILE,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        entries = t.log(limit=20)
        print(f"\n  Commits: {len(entries)}, Branching unlocked: {unlocked['branching']}")

        # --- Phase 2: Branching unlocked ---
        if unlocked["branching"]:
            phase2 = ["commit", "status", "branch", "switch", "list_branches", "log"]
            print(f"\n  --- Phase 2: Branching ---")
            print(f"  Tools: {sorted(phase2)}")

            result = t.llm.run(
                "Create a branch called 'optimization' and commit an "
                "implementation plan for database query optimization.",
                max_steps=8, max_tokens=1024,
                profile="full", tool_names=phase2,
                on_step=log.on_step, on_tool_result=log.on_tool_result,
            )
            result.pprint()

            print(f"\n  Branches: {[b.name for b in t.list_branches()]}")
            print(f"  Merging unlocked: {unlocked['merging']}")

        # --- Phase 3: Full access ---
        if unlocked["merging"]:
            phase3 = ["commit", "status", "branch", "switch",
                       "list_branches", "log", "merge", "compress"]
            print(f"\n  --- Phase 3: Full Access ---")
            print(f"  Tools: {sorted(phase3)}")

        # Summary
        print(f"\n  Progression: commit,status -> +branch,switch,log -> +merge,compress")
        print(f"  Branching unlocked: {unlocked['branching']}")
        print(f"  Merging unlocked:   {unlocked['merging']}")

    print("\n  PASSED")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    stage_based_switching()
    progressive_discovery()

    print(f"\n{'=' * 70}")
    print("  Summary")
    print(f"{'=' * 70}\n")
    print("  Pattern                         Key APIs")
    print("  ----------------------------    --------------------------------")
    print("  Stage-based profile switching    ToolProfile, switch_profile()")
    print("  Progressive capability unlock    unlock_tool(), post_commit MW")
    print()
    print("  Stage-based: swap entire profiles on transition.")
    print("  Progressive: unlock individual tools based on demonstrated work.")
    print("\nDone.")


if __name__ == "__main__":
    main()


# --- See also ---
# Implicit discovery:       agentic/01_implicit_discovery.py
# Tool compaction:           agentic/03_tool_compaction.py
# Semantic automation:       agentic/04_semantic_automation.py
