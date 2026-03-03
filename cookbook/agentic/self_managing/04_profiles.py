"""Tool Profiles -- Scoping Agent Capabilities

Three built-in profiles control what tract operations an agent can access:
- self: Full CRUD -- the agent manages its own context completely
- supervisor: Read + high-level ops -- oversight without micro-management
- observer: Read-only -- monitoring without modification

Demonstrates: as_tools(profile=), ToolExecutor profiles, capability scoping,
              Orchestrator with observer profile
"""

import sys
from pathlib import Path

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile
from tract.toolkit.profiles import SELF_PROFILE, SUPERVISOR_PROFILE, FULL_PROFILE
from tract.orchestrator import Orchestrator, OrchestratorConfig, AutonomyLevel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# Custom observer profile: read-only tools for monitoring
# =====================================================================
# The built-in profiles are "self" (agent CRUD), "supervisor" (manage
# another agent), and "full" (everything). For a read-only observer,
# we build a custom profile that only exposes inspection tools.

OBSERVER_PROFILE = ToolProfile(
    name="observer",
    tool_configs={
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check the tract's current status: branch, HEAD, token "
                "count, and budget usage. Read-only."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description=(
                "View recent commit history. Inspect what content has "
                "been recorded. Read-only."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile the current context to inspect messages and "
                "token counts. Read-only."
            ),
        ),
        "diff": ToolConfig(
            enabled=True,
            description=(
                "Compare two commits to see what changed. Read-only."
            ),
        ),
        "get_commit": ToolConfig(
            enabled=True,
            description=(
                "Get detailed information about a specific commit. "
                "Read-only."
            ),
        ),
        "list_branches": ToolConfig(
            enabled=True,
            description=(
                "List all branches with their HEAD commits. Read-only."
            ),
        ),
        "get_tags": ToolConfig(
            enabled=True,
            description="See all tags on a commit. Read-only.",
        ),
        "list_tags": ToolConfig(
            enabled=True,
            description="List all registered tags with counts. Read-only.",
        ),
        "query_by_tags": ToolConfig(
            enabled=True,
            description="Find commits by tag names. Read-only.",
        ),
    },
)


# =====================================================================
# PART 1 -- Manual: Compare profiles side-by-side
# =====================================================================

def part1_manual():
    """Show what tools each profile exposes."""
    print("=" * 60)
    print("PART 1 -- Manual: Profile Comparison")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Hello, world.")
        t.assistant("Hello! How can I help you today?")

        # Compare all profiles
        profiles = {
            "self": SELF_PROFILE,
            "supervisor": SUPERVISOR_PROFILE,
            "full": FULL_PROFILE,
            "observer": OBSERVER_PROFILE,
        }

        for name, profile in profiles.items():
            tools = t.as_tools(profile=profile, format="openai")
            tool_names = sorted(t["function"]["name"] for t in tools)

            print(f"\n  Profile '{name}': {len(tools)} tools")
            # Group by capability
            read_tools = [n for n in tool_names if n in {
                "status", "log", "compile", "diff", "get_commit",
                "list_branches", "get_tags", "list_tags", "query_by_tags",
            }]
            write_tools = [n for n in tool_names if n not in {
                "status", "log", "compile", "diff", "get_commit",
                "list_branches", "get_tags", "list_tags", "query_by_tags",
            }]

            print(f"    Read  ({len(read_tools)}): {', '.join(read_tools[:6])}"
                  f"{'...' if len(read_tools) > 6 else ''}")
            print(f"    Write ({len(write_tools)}): {', '.join(write_tools[:6])}"
                  f"{'...' if len(write_tools) > 6 else ''}")

        # Demonstrate that observer cannot write
        print(f"\n  Observer profile -- write operations blocked:\n")
        executor = ToolExecutor(t)
        executor.set_profile(OBSERVER_PROFILE)

        # These should work (read-only)
        result = executor.execute("status", {})
        print(f"    status():      success={result.success}")

        result = executor.execute("log", {"limit": 3})
        print(f"    log(limit=3):  success={result.success}")

        # These should fail (not in observer profile)
        result = executor.execute("commit", {
            "content": {"content_type": "dialogue", "role": "user", "text": "test"},
        })
        print(f"    commit():      success={result.success}  error={result.error}")

        result = executor.execute("compress", {})
        print(f"    compress():    success={result.success}  error={result.error}")

        result = executor.execute("annotate", {
            "target_hash": "abc", "priority": "pinned",
        })
        print(f"    annotate():    success={result.success}  error={result.error}")

        # Unlock a single tool despite profile
        print(f"\n  Selective unlock: add 'tag' to observer profile:\n")
        executor.unlock_tool("tag")
        print(f"    Available tools: {executor.available_tools()}")
        print(f"    'tag' in tools: {'tag' in executor.available_tools()}")
        print(f"    'commit' in tools: {'commit' in executor.available_tools()}")


# =====================================================================
# PART 3 -- Agent: Orchestrator with observer profile (read-only)
# =====================================================================

def part3_agent():
    """Orchestrator with observer profile inspects but cannot modify."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 3 -- Agent: Orchestrator with Observer Profile")
    print("=" * 60)
    print()
    print("  An observer orchestrator can inspect context but cannot modify it.")
    print("  Useful for monitoring, auditing, or analysis agents that")
    print("  should not alter the conversation state.")
    print()

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2000))

    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Build some context for the observer to inspect
        t.system("You are a software architecture advisor.")
        t.user("Should we use a monolith or microservices?")
        t.assistant(
            "For a team of 5 with a single product, start with a "
            "modular monolith. Microservices add operational overhead "
            "that small teams cannot absorb."
        )
        t.user("What about scaling concerns?")
        t.assistant(
            "A well-structured monolith scales vertically to handle "
            "significant load. Extract services only when you have "
            "clear domain boundaries and team capacity."
        )
        t.user("Good advice. Let's go with the modular monolith.")
        t.assistant(
            "Agreed. I recommend starting with clear module boundaries: "
            "auth, billing, and core product as separate modules within "
            "the monolith. This makes future extraction straightforward."
        )

        status_before = t.status()
        print(f"  Before: {status_before.commit_count} commits, "
              f"{status_before.token_count} tokens")

        # Orchestrator with observer profile: read-only inspection
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=10,
            profile=OBSERVER_PROFILE,
            task_context=(
                "You are an observer agent. Inspect this conversation using "
                "the available read-only tools. Determine: "
                "1) How many commits and tokens are used? "
                "2) What was the key decision made? "
                "3) Summarize the conversation in one sentence."
            ),
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Orchestrator completed: {result.total_tool_calls} tool calls, "
              f"state={result.state.value}")
        for step in result.steps:
            status_label = "OK" if step.success else "FAIL"
            args_short = str(step.tool_call.arguments)[:60]
            print(f"    [{status_label}] {step.tool_call.name}({args_short})")

        # Verify the observer did not modify anything
        status_after = t.status()
        print(f"\n  Verification: tract unchanged after observer inspection")
        print(f"    Commits: {status_after.commit_count}")
        print(f"    Tokens:  {status_after.token_count}")
        print(f"    Branch:  {status_after.branch_name}")


def main():
    part1_manual()
    part3_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/agentic/sidecar/03_toolkit.py            -- ToolExecutor basics and built-in profiles
# cookbook/agentic/self_managing/01_tool_hints.py    -- Custom profile with description hints
# cookbook/agentic/multi_agent/01_parent_child.py    -- Supervisor profile for managing child agents
# cookbook/developer/config/01_per_call.py           -- ToolConfig description overrides
