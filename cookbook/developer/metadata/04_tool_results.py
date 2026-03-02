"""Tool Commits and Interactive Approval

Two tiers of tool interaction: manual tool commits and interactive approval
of tool calls.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides

Demonstrates: set_tools(), generate() with tool_calls, tool_result(),
              ToolCall, click.confirm() for tool approval
"""

import os
import sys
from functools import partial
from pathlib import Path

import click
from dotenv import load_dotenv

from tract import Tract
from tract.protocols import ToolCall

# Allow importing _helpers from the same directory when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import TOOLS, execute_tool as _execute_tool, call_llm  # noqa: E402
from _helpers import TRACT_OPENAI_API_KEY, TRACT_OPENAI_BASE_URL, MODEL_ID  # noqa: E402

load_dotenv()

# The directory this file lives in — tools will search here.
COOKBOOK_DIR = os.path.dirname(os.path.abspath(__file__))

# Bind execute_tool to this file's sandbox directory.
_SELF = os.path.basename(__file__)
execute_tool = partial(_execute_tool, cookbook_dir=COOKBOOK_DIR, exclude_file=_SELF)


# =============================================================================
# Part 1: Manual Tool Commits  (PART 1 — Manual)
# =============================================================================

def part1_manual_tools():
    """Manually commit tool calls and results — no LLM needed."""
    print("=" * 60)
    print("Part 1: MANUAL TOOL COMMITS  [Manual Tier]")
    print("=" * 60)
    print()
    print("  Commit tool interactions by hand using set_tools(), assistant()")
    print("  with metadata, and tool_result().  Fully deterministic.")
    print()

    with Tract.open() as t:
        t.set_tools(TOOLS)
        t.system("You are a file search agent.")
        t.user("Find all Python files in the project.")

        # Manual assistant message with a tool call
        t.assistant(
            "I'll search the directory.",
            metadata={"tool_calls": [
                {"id": "call_001", "name": "list_directory",
                 "arguments": {"path": "."}},
            ]},
        )

        # Execute the tool for real and commit the result
        listing = execute_tool("list_directory", {"path": "."})
        ci = t.tool_result("call_001", "list_directory", listing)

        file_count = len(listing.splitlines())
        t.assistant(f"Found {file_count} files in the directory.")

        print(f"  Tool result committed: {ci.commit_hash[:8]}")
        print(f"  Content: {t.get_content(ci.commit_hash)}\n")

        ctx = t.compile()
        print(f"  Compiled: {ctx.commit_count} messages, {ctx.token_count} tokens\n")
        ctx.pprint(style="compact")

    print()


# =============================================================================
# Part 2: Interactive Tool Approval  (PART 2 — Interactive)
# =============================================================================

def part2_interactive_approval():
    """After LLM requests a tool call, confirm before executing."""
    if not TRACT_OPENAI_API_KEY:
        print("=" * 60)
        print("Part 2: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print("=" * 60)
    print("Part 2: INTERACTIVE TOOL APPROVAL  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  After the LLM requests a tool call, click.confirm() before")
    print("  executing. Option to override the result manually.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.set_tools(TOOLS)
        t.system(
            "You are a file search agent. Use the provided tools to find "
            "information in the filesystem."
        )
        t.user(
            "List the files in the current directory."
        )

        # Get LLM response with tool call
        ctx = t.compile()
        raw = call_llm(ctx.to_dicts(), TOOLS)
        msg = raw["choices"][0]["message"]
        raw_tool_calls = msg.get("tool_calls")

        if raw_tool_calls:
            tool_calls = [ToolCall.from_openai(tc) for tc in raw_tool_calls]

            # Commit assistant message
            t.assistant(
                msg.get("content") or "",
                metadata={"tool_calls": [tc.to_dict() for tc in tool_calls]},
            )

            for tc in tool_calls:
                print(f"  LLM wants to call: {tc.name}({tc.arguments})")

                if click.confirm(f"    Execute tool '{tc.name}'?", default=True):
                    override = click.prompt(
                        "    Override result? (Enter to skip)", default="", show_default=False,
                    )
                    if override:
                        result_text = override
                    else:
                        result_text = execute_tool(tc.name, tc.arguments)
                    is_error = result_text.startswith("Error:")
                    t.tool_result(tc.id, tc.name, result_text, is_error=is_error)
                    print(f"    -> committed result ({len(result_text)} chars)")
                else:
                    t.tool_result(tc.id, tc.name, "(skipped by user)", is_error=False)
                    print(f"    -> skipped")
        else:
            t.assistant(msg.get("content") or "")

        print()
        t.compile().pprint(style="compact")

    print()


def main():
    part1_manual_tools()
    part2_interactive_approval()
    print("=" * 60)
    print("Done -- both tiers of tool interaction demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
