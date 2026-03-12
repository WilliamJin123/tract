"""Agent Self-Correction (Implicit)

The agent gives a deliberately brief answer, then is asked to improve it.
It has edit tools available but is never told how to use them. Will it
discover get_commit + edit to revise its own output in place?

Tools available: commit (with edit), get_commit, diff, log

Demonstrates: Can the model use inspect-then-edit-in-place to revise
              its own output when guided toward the pattern?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm
from _logging import StepLogger

MODEL_ID = llm.xlarge


PROFILE = ToolProfile(
    name="writer",
    tool_configs={
        "commit": ToolConfig(enabled=True),
        "get_commit": ToolConfig(enabled=True),
        "diff": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
    },
)


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 60)
    print("Agent Self-Correction (Implicit)")
    print("=" * 60)
    print()
    print("  The agent gives a brief answer, then is asked to improve it.")
    print("  Will it use edit operations to revise in place?")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
        tool_profile=PROFILE,
    ) as t:
        t.system(
            "You are a technical writer. Produce clear, thorough explanations.\n"
            "You have tools to manage your content: commit (with edit operation), "
            "get_commit, diff, and log. When revising earlier work, use "
            "get_commit to inspect the original, then commit with operation='edit' "
            "and edit_target=<hash> to update it in place."
        )

        # Get a deliberately brief initial answer
        r1 = t.chat("Explain how a compiler works in one sentence.")
        original_hash = r1.commit_info.commit_hash
        print(f"  Initial answer [{original_hash[:8]}]: {(r1.text or '(no response)')[:120]}")

        # Ask to improve — no mention of tools, edit operations, or hashes
        print("\n  --- Task ---")
        log = StepLogger()
        result = t.run(
            "Too brief. Expand to cover lexing, parsing, optimization, "
            "and codegen. Use get_commit to find the original, then use commit "
            "with operation='edit' and edit_target to replace it in place.",
            max_steps=8, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        print("\n  Context after improvement:")
        t.compile().pprint(style="compact")

        # Check if the agent used edit operations
        edits = [e for e in t.log(limit=30) if e.operation.value == "edit"]
        if edits:
            print(f"\n  Agent used {len(edits)} edit operation(s) to revise in place.")
            history = t.edit_history(original_hash)
            for i, version in enumerate(history):
                label = "ORIGINAL" if i == 0 else f"EDIT {i}"
                content = t.get_content(version)
                text = str(content)[:100]
                print(f"  v{i} ({label}) [{version.commit_hash[:8]}]: {text}...")
        else:
            print("\n  Agent did not use edit operations (appended instead).")


if __name__ == "__main__":
    main()
