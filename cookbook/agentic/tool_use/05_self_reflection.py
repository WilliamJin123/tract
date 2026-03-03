"""Self-Reflection via Tools

An LLM agent inspects its own previous output, identifies issues, and
corrects them using edit operations and diff. This is the "agent traces
its own edit history" pattern done properly — the LLM genuinely decides
what to inspect and how to fix it through tool calls.

Scenario: The agent gives an answer, then is asked to review and improve
its own work. It uses get_commit to read its previous output, diff to
compare versions, commit with edit operation to revise, and log to trace
the edit chain.

Tools exercised: commit (with edit), get_commit, diff, log, compile,
                 annotate, status

Demonstrates: LLM-driven self-correction, edit-in-place through tools,
              agent reasoning about quality of its own output,
              edit chain inspection
"""

import json
import sys
from pathlib import Path

import httpx

from tract import Tract
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


# Tool profile: self-reflection and correction tools
REFLECTION_PROFILE = ToolProfile(
    name="self-reflector",
    tool_configs={
        "commit": ToolConfig(
            enabled=True,
            description=(
                "Record content or edit a previous commit. To edit, set "
                "operation='edit' and provide edit_target with the hash of "
                "the commit to replace. The new content overwrites the old. "
                "Include a message describing why you're editing."
            ),
        ),
        "get_commit": ToolConfig(
            enabled=True,
            description=(
                "Get full details about a commit: content, type, tokens, "
                "metadata. Use this to read your own previous responses "
                "before deciding whether to edit them."
            ),
        ),
        "diff": ToolConfig(
            enabled=True,
            description=(
                "Compare two commits to see what changed. Call after an edit "
                "to verify the improvement. Shows added, removed, modified "
                "messages and token delta."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description=(
                "View recent commit history. Use op_filter='edit' to see "
                "only edit operations — this shows your correction chain."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile current context. After edits, compile shows the "
                "latest version of each message — verify your corrections "
                "appear correctly."
            ),
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Mark a commit as 'pinned' (important, keep forever) or "
                "'skip' (hide from compilation). Use to clean up failed "
                "attempts or protect good corrections."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description="Check current state: branch, HEAD, token count.",
        ),
    },
)


def run_agent_loop(t, executor, tools, task, max_turns=12):
    """Generic agentic loop: user task -> tool calls -> final response."""
    t.user(task)

    for turn in range(max_turns):
        response = t.generate()

        if not response.tool_calls:
            print(f"\n  Agent: {response.text[:200]}")
            if len(response.text) > 200:
                print(f"         ...({len(response.text)} chars total)")
            return response

        for tc in response.tool_calls:
            result = executor.execute(tc.name, tc.arguments)
            t.tool_result(tc.id, tc.name, str(result))
            args_short = json.dumps(tc.arguments)[:60]
            print(f"    -> {tc.name}({args_short})")
            output_short = str(result.output)[:80]
            print(f"       {output_short}")

    print("  (max turns reached)")
    return None


# =====================================================================
# PART 1 -- Manual: Show reflection tools, execute directly
# =====================================================================

def part1_manual():
    """Show self-reflection tools and execute them manually."""
    print("=" * 60)
    print("PART 1 -- Manual: Self-Reflection Tools")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a concise science tutor.")
        t.user("What causes rain?")
        ci = t.assistant("Rain is caused by evaporation and condensation "
                         "in the water cycle.")

        executor = ToolExecutor(t)
        executor.set_profile(REFLECTION_PROFILE)

        # Show available tools
        tools = t.as_tools(profile=REFLECTION_PROFILE, format="openai")
        print(f"\n  Reflection profile: {len(tools)} tools")
        for tool in tools:
            print(f"    - {tool['function']['name']}")

        print("\n  BEFORE edit:")
        t.compile().pprint(style="chat")

        # Inspect own response
        result = executor.execute("get_commit", {"commit_hash": ci.commit_hash})
        print(f"\n  get_commit({ci.commit_hash[:8]}):\n    {result.output[:200]}")

        # Edit it
        result = executor.execute("commit", {
            "content": {
                "content_type": "dialogue",
                "role": "assistant",
                "text": "Rain forms when water vapor in the atmosphere "
                        "condenses into droplets heavy enough to fall. "
                        "This is part of the water cycle: evaporation, "
                        "condensation, precipitation, and collection.",
            },
            "operation": "edit",
            "edit_target": ci.commit_hash,
            "message": "Expanded explanation with water cycle stages",
        })
        print(f"\n  edit commit:\n    {result.output}")

        # Log edits
        result = executor.execute("log", {"limit": 5, "op_filter": "edit"})
        print(f"\n  log(op_filter='edit'):\n    {result.output[:200]}")

        print("\n  AFTER edit:")
        t.compile().pprint(style="chat")

        print()


# =====================================================================
# PART 2 -- Agent: LLM reviews and corrects its own work
# =====================================================================

def part2_agent():
    """LLM agent inspects its own responses and improves them."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 2: SKIPPED (no API key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- Agent: LLM Reviews Its Own Work")
    print("=" * 60)
    print()
    print("  The agent will: inspect a previous response with get_commit,")
    print("  decide it needs improvement, edit it, then verify with compile.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=REFLECTION_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a meticulous assistant with self-correction tools. "
            "When asked to review your work, use get_commit to read your "
            "previous responses, then edit them if they can be improved. "
            "Use commit with operation='edit' and edit_target to replace "
            "a previous response."
        )

        # Give an initial answer (deliberately brief)
        r1 = t.chat("Explain how a compiler works in one sentence.")
        original_hash = r1.commit_info.commit_hash
        print(f"  Initial answer [{original_hash[:8]}]: {r1.text[:120]}")

        # Ask the agent to review and improve
        print("\n  --- Task: Review and improve ---")
        run_agent_loop(
            t, executor, tools,
            f"Review your previous answer about compilers (commit "
            f"{original_hash[:8]}). Use get_commit to read it, then "
            f"edit it to be more complete and accurate — mention lexing, "
            f"parsing, and code generation. Use commit with operation='edit' "
            f"and edit_target='{original_hash}'. After editing, compile to "
            f"verify the improved version appears."
        )

        print("\n  Context after agent edits:")
        t.compile().pprint(style="compact")

        # Show the edit chain
        print("\n  --- Edit chain ---")
        history = t.edit_history(original_hash)
        for i, version in enumerate(history):
            label = "ORIGINAL" if i == 0 else f"EDIT {i}"
            content = t.get_content(version)
            text = str(content)[:100]
            print(f"  v{i} ({label}) [{version.commit_hash[:8]}]: {text}...")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/history/03_edit_history.py     -- Manual edit chain tracking
# cookbook/developer/metadata/03_edit_in_place.py   -- Edit-in-place workflow
# cookbook/e2e/self_correcting_agent.py             -- Full self-correction scenario
