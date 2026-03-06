"""Multi-Agent Delegation via the Agent Loop

A parent agent delegates work to a child by branching, the child agent
runs independently on the branch, and the parent merges results back.
Unlike manual multi-agent examples, this shows delegation through the
agent loop (t.run) -- agents make their own decisions about what to
research and when they're done.

Pattern: parent branches -> child runs on branch -> parent merges

Tools exercised: branch, switch, merge, commit, compress, compile,
                 status, log, list_branches

Demonstrates: Agent-driven delegation, branch-based isolation,
              compress-then-merge for clean handoff, parent/child
              coordination through branch mechanics
"""

import io
import json
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, Session
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.large


# Tool profile for the parent: coordination tools
PARENT_PROFILE = ToolProfile(
    name="coordinator",
    tool_configs={
        "branch": ToolConfig(
            enabled=True,
            description=(
                "Create a branch for delegating work to a sub-agent. Name "
                "it descriptively (e.g. 'research-caching'). The child "
                "agent will work on this branch independently."
            ),
        ),
        "switch": ToolConfig(
            enabled=True,
            description="Switch to a branch to inspect work or return to main.",
        ),
        "merge": ToolConfig(
            enabled=True,
            description=(
                "Merge a child branch back into the current branch. Use "
                "after the child has completed and compressed their work."
            ),
        ),
        "list_branches": ToolConfig(
            enabled=True,
            description="List all branches to see available child work.",
        ),
        "compile": ToolConfig(
            enabled=True,
            description="View compiled context on the current branch.",
        ),
        "status": ToolConfig(
            enabled=True,
            description="Check current position and token count.",
        ),
        "log": ToolConfig(
            enabled=True,
            description="View commit history on current branch.",
        ),
        "commit": ToolConfig(
            enabled=True,
            description="Record coordination decisions and summaries.",
        ),
    },
)


# Tool profile for the child: research tools
CHILD_PROFILE = ToolProfile(
    name="researcher",
    tool_configs={
        "commit": ToolConfig(
            enabled=True,
            description=(
                "Record research findings. Use content_type='dialogue' with "
                "role='assistant' for analysis, or content_type='artifact' "
                "for structured deliverables."
            ),
        ),
        "compress": ToolConfig(
            enabled=True,
            description=(
                "Compress your research into a summary. Use content= to "
                "provide a concise summary of your findings. Do this before "
                "the parent merges your branch."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description="Check your branch status and token count.",
        ),
        "log": ToolConfig(
            enabled=True,
            description="View your commit history.",
        ),
        "compile": ToolConfig(
            enabled=True,
            description="View your compiled context.",
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
    print("Multi-Agent Delegation: parent branches, child researches, merge back")
    print("=" * 70)
    print()
    print("  1. Parent creates a branch for the child")
    print("  2. Child agent runs independently on the branch")
    print("  3. Child compresses findings into a summary")
    print("  4. Parent merges the branch back")
    print()

    # Use Session for multi-agent coordination
    session = Session.open()
    parent_tract = session.create_tract(display_name="coordinator")

    # Configure LLM on the parent tract
    from tract.llm.client import OpenAIClient
    from tract.models.config import LLMConfig

    parent_client = OpenAIClient(
        api_key=llm.api_key,
        base_url=llm.base_url,
        default_model=MODEL_ID,
    )
    parent_tract.configure_llm(parent_client)
    parent_tract._owns_llm_client = True
    parent_tract._default_config = LLMConfig(model=MODEL_ID)

    # Set up parent context
    parent_tract.system(
        "You are a project coordinator. You delegate research tasks to "
        "sub-agents by creating branches. Each sub-agent works independently "
        "on its branch, then you merge the results."
    )
    parent_tract.user(
        "We need to evaluate caching strategies for our microservice "
        "architecture. Research write-through, write-back, and cache-aside "
        "patterns, then give me a recommendation."
    )
    parent_tract.assistant(
        "I'll delegate this research to a specialist agent on a separate branch."
    )

    print("  Parent context (before delegation):")
    parent_tract.compile().pprint(style="compact")

    # --- Step 1: Parent creates a research branch ---
    print("\n=== Step 1: Parent creates research branch ===\n")
    parent_tract.branch("research-caching", switch=False)
    print(f"  Created branch: research-caching")
    print(f"  Parent stays on: {parent_tract.current_branch}")

    # --- Step 2: Deploy child on the branch ---
    print("\n=== Step 2: Child agent works on branch ===\n")
    child = session.deploy(
        parent_tract,
        purpose="research caching patterns",
        branch_name="research-caching",
    )

    # Configure LLM on child
    child_client = OpenAIClient(
        api_key=llm.api_key,
        base_url=llm.base_url,
        default_model=MODEL_ID,
    )
    child.configure_llm(child_client)
    child._owns_llm_client = True
    child._default_config = LLMConfig(model=MODEL_ID)

    # Set child tools
    child_tools = child.as_tools(profile=CHILD_PROFILE)
    child.set_tools(child_tools)

    # Child runs autonomously
    result = child.run(
        "Research caching patterns for microservices. Cover:\n"
        "1. Write-through caching (pros, cons, use cases)\n"
        "2. Write-back caching (pros, cons, use cases)\n"
        "3. Cache-aside pattern (pros, cons, use cases)\n\n"
        "Commit each finding as a separate artifact. When done, compress "
        "all your findings into a one-paragraph summary using the compress "
        "tool with content='<your summary>'.",
        max_steps=12, on_step=_log_step,
    )
    result.pprint()

    print(f"\n  Child commits: {len(child.log())}")
    print("\n  Child context (after research + compression):")
    child.compile().pprint(style="compact")

    # --- Step 3: Parent merges results ---
    print("\n=== Step 3: Parent merges child work ===\n")
    merge_result = parent_tract.merge("research-caching")
    merge_result.pprint()

    print("\n  Parent context after merge:")
    parent_tract.compile().pprint(style="compact")

    # --- Step 4: Parent synthesizes ---
    print("\n=== Step 4: Parent synthesizes recommendation ===\n")

    # Set parent tools for final synthesis
    parent_tools = parent_tract.as_tools(profile=PARENT_PROFILE)
    parent_tract.set_tools(parent_tools)

    result = parent_tract.run(
        "The research branch has been merged. Review the compiled context "
        "which now includes the child's research findings. Commit a final "
        "recommendation as an artifact (content_type='artifact') summarizing "
        "which caching strategy to use and why.",
        max_steps=8, on_step=_log_step,
    )
    result.pprint()

    # --- Final state ---
    print("\n\n=== Final State ===\n")
    branches = [b.name for b in parent_tract.list_branches()]
    print(f"  Branches: {branches}")
    print(f"  Parent branch: {parent_tract.current_branch}")
    print(f"  Total parent commits: {len(parent_tract.log())}")

    print("\n  Final parent context:")
    parent_tract.compile().pprint(style="compact")

    session.close()


if __name__ == "__main__":
    main()
