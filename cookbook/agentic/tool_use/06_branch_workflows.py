"""Tangent Branching -- Agent isolates off-topic questions on branches

An LLM agent working on a design task autonomously branches when the user
asks conceptual clarification questions that don't advance the project.
The agent handles the full lifecycle: branch, answer, compress, switch
back to main, and merge the one-line summary.

Key technique: custom tool descriptions via ToolProfile steer the LLM on
*when* to branch, without any hardcoded trigger logic.

Tools exercised: branch, switch, merge, compress, commit, status, log
Demonstrates: description overrides for behavioral steering, agent-managed
              branch lifecycle, compress-then-merge pattern
"""

import io
import json
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large


# =====================================================================
# Tool profile: full tangent lifecycle tools with steering descriptions
# =====================================================================

TANGENT_PROFILE = ToolProfile(
    name="tangent-manager",
    tool_configs={
        "commit": ToolConfig(
            enabled=True,
            description=(
                "Record content into the conversation. Use this to commit your "
                "answer as a dialogue message (content_type='dialogue', "
                "role='assistant') instead of returning text directly. This "
                "lets you continue making tool calls after answering."
            ),
        ),
        "branch": ToolConfig(
            enabled=True,
            description=(
                "Create a new branch. You MUST use this when the user asks a "
                "conceptual or clarification question that does not directly "
                "advance the current design or implementation discussion. "
                "Name the branch 'tangent/<topic>'. Set switch=true to work on it."
            ),
        ),
        "switch": ToolConfig(
            enabled=True,
            description=(
                "Switch to a different branch. Use this to return to 'main' "
                "after compressing a tangent branch."
            ),
        ),
        "merge": ToolConfig(
            enabled=True,
            description=(
                "Merge a branch into the current branch. After compressing a "
                "tangent branch, switch to main and merge the tangent so the "
                "one-line summary is preserved in the main context."
            ),
        ),
        "compress": ToolConfig(
            enabled=True,
            description=(
                "Compress commits into a summary. Use content= to provide a "
                "short one-sentence manual summary capturing only the key "
                "takeaway from the tangent conversation."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description="Check current branch, HEAD, and token count.",
        ),
        "log": ToolConfig(
            enabled=True,
            description="View recent commits to find hashes for compress range.",
        ),
    },
)


def run_agent_loop(t, executor, task, *, max_turns=15):
    """Agentic loop: user task -> tool calls -> final response."""
    t.user(task)

    for turn in range(max_turns):
        response = t.generate()

        if not response.tool_calls:
            text = response.text or "(empty)"
            print(f"\n  Agent: {text[:400]}")
            if len(text) > 400:
                print(f"         ...({len(text)} chars total)")
            return response

        for tc in response.tool_calls:
            result = executor.execute(tc.name, tc.arguments)
            t.tool_result(tc.id, tc.name, str(result))
            args_short = json.dumps(tc.arguments)[:80]
            ok = "OK" if result.success else "FAIL"
            output = str(result.output if result.success else result.error)[:100]
            print(f"    [{ok}] {tc.name}({args_short})")
            print(f"           {output}")

    print("  (max turns reached)")
    return None


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 70)
    print("Tangent Branching: Agent isolates off-topic questions on branches")
    print("=" * 70)
    print()
    print("  The agent handles the full tangent lifecycle via tool calls:")
    print("    1. branch('tangent/<topic>', switch=true)")
    print("    2. commit the answer as a dialogue message")
    print("    3. compress the tangent to a one-line summary")
    print("    4. switch back to main")
    print("    5. merge the tangent branch")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=TANGENT_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a senior API architect helping design a REST API for a "
            "task management app.\n\n"
            "TANGENT PROTOCOL: When the user asks a conceptual question that "
            "does not advance the design (e.g. 'what is REST?'), you must "
            "handle it entirely through tool calls:\n"
            "1. branch('tangent/<topic>', switch=true)\n"
            "2. commit your answer as a dialogue message "
            "(content_type='dialogue', role='assistant', text='...')\n"
            "3. compress(content='<one-line summary>')\n"
            "4. switch('main')\n"
            "5. merge('<branch-name>')\n"
            "6. Then respond with a short confirmation.\n\n"
            "For design questions, answer normally without branching."
        )

        # --- Phase 1: Design question (should NOT branch) ---
        print("  Initial context:")
        t.compile().pprint(style="compact")

        print("=== Phase 1: Design question (on main, no branching) ===\n")
        run_agent_loop(
            t, executor,
            "Let's design the API for a task management app. I need endpoints "
            "for creating, listing, updating, and deleting tasks. Each task has "
            "a title, description, status (todo/in_progress/done), and assignee. "
            "What's your recommended URL structure?",
        )
        print(f"\n  Branch: {t.current_branch}")

        print("\n  Context after Phase 1:")
        t.compile().pprint(style="compact")

        # --- Phase 2: Conceptual tangent (full LLM-driven lifecycle) ---
        print("\n\n=== Phase 2: Conceptual tangent (full agent lifecycle) ===\n")
        run_agent_loop(
            t, executor,
            "Wait, quick question -- what actually is REST? I keep hearing "
            "the term but I don't fully understand the principles behind it.",
        )
        print(f"\n  Branch: {t.current_branch}")
        print(f"  Branches: {[b.name for b in t.list_branches()]}")

        # --- Phase 3: Resume design ---
        if t.current_branch != "main":
            print(f"  (agent didn't return to main -- switching manually)")
            t.switch("main")

        print(f"\n\n=== Phase 3: Resume design (on {t.current_branch}) ===\n")
        run_agent_loop(
            t, executor,
            "OK, back to the API design. What status codes should each "
            "endpoint return? And should we version the API?",
        )

        # --- Final state ---
        print("\n\n=== Final context on main ===\n")
        t.compile().pprint(style="compact")

        branches = [b.name for b in t.list_branches()]
        msgs = t.compile().to_dicts()
        print(f"\n  Branch: {t.current_branch}  |  Messages: {len(msgs)}  |  "
              f"All branches: {branches}")


if __name__ == "__main__":
    main()
