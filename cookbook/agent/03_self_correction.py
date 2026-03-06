"""Agent Self-Correction

An LLM agent inspects its own previous output, identifies issues, and
corrects them using edit operations. The agent genuinely decides what to
inspect and how to fix it through tool calls -- not hardcoded logic.

The agent gives an answer, then reviews and improves its own work using
get_commit to read previous output, commit with operation='edit' to
revise, and log to trace the edit chain.

Tools exercised: commit (with edit), get_commit, diff, log, compile,
                 annotate, status

Demonstrates: LLM-driven self-correction, edit-in-place through tools,
              agent reasoning about quality of its own output,
              edit chain inspection
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
                "only edit operations -- this shows your correction chain."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile current context. After edits, compile shows the "
                "latest version of each message -- verify your corrections "
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


def _log_step(step_num, response):
    """on_step callback -- print step number."""
    print(f"    [step {step_num}]")


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 60)
    print("Agent Self-Correction")
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
        # Register tools from the profile
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
        result = t.run(
            f"Review your previous answer about compilers (commit "
            f"{original_hash[:8]}). Use get_commit to read it, then "
            f"edit it to be more complete and accurate -- mention lexing, "
            f"parsing, and code generation. Use commit with operation='edit' "
            f"and edit_target='{original_hash}'. After editing, compile to "
            f"verify the improved version appears.",
            max_steps=12, on_step=_log_step,
        )
        print(f"\n  Loop result: {result.status} ({result.steps} steps, "
              f"{result.tool_calls} tool calls)")
        if result.final_response:
            print(f"  Agent: {result.final_response[:200]}")

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


if __name__ == "__main__":
    main()
