"""Tool Commits

Manual tool commits -- no LLM needed.

Demonstrates: set_tools(), assistant() with tool_calls metadata,
              tool_result(), compile(), pprint()
"""

import os
import sys
from functools import partial
from pathlib import Path

from tract import Tract

# Allow importing _helpers from the same directory when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import TOOLS, execute_tool as _execute_tool

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


if __name__ == "__main__":
    part1_manual_tools()
    print("=" * 60)
    print("Done -- manual tool commits demonstrated.")
    print("=" * 60)
