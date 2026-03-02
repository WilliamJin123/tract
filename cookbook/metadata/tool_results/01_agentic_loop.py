"""Agentic File Search + Post-Hoc Compaction

Three tiers of tool interaction: manual tool commits, interactive approval
of tool calls, and fully autonomous agentic loop with post-hoc compaction.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: set_tools(), generate() with tool_calls, tool_result(),
              ToolCall, compress_tool_calls(), agentic loop pattern,
              click.confirm() for tool approval
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

# The marker to search for — split to avoid matching THIS file in search.
_MARKER = "DIS" + "COVERY"

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


# =============================================================================
# Part 3: Agentic Loop  (PART 3 — LLM / Agent)
# =============================================================================

def part3_agentic_loop():
    if not TRACT_OPENAI_API_KEY:
        print("=" * 60)
        print("Part 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print("=" * 60)
    print("Part 3: AGENTIC LOOP + POST-HOC COMPACTION  [Agent Tier]")
    print("=" * 60)
    print()
    print("  The LLM autonomously searches files, commits tool calls and")
    print("  results, then compress_tool_calls() shortens results in-place.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # --- Register tools and start the conversation ---

        t.set_tools(TOOLS)
        t.system(
            "You are a file search agent. Use the provided tools to find "
            "information in the filesystem. Be methodical: list directories "
            "first, then search or read promising files. When you find the "
            "answer, respond with the exact text — no tool calls."
        )
        t.user(
            "One of the .py files in the 08_tool_calling/ directory "
            f"contains a comment line starting with '# {_MARKER}:'. "
            "Find it and tell me the filename, line number, exact text, "
            "and a brief explanation of what the comment means, as well as what other content exists in the file."
        )

        # --- Agentic loop: compile -> LLM -> tool calls -> execute -> repeat ---

        intermediate_hashes = []  # Track for compression later
        max_turns = 10

        for turn in range(max_turns):
            # Compile tract context and call LLM with tools
            ctx = t.compile()
            raw = call_llm(ctx.to_dicts(), TOOLS)
            msg = raw["choices"][0]["message"]

            # Parse tool calls (if any)
            raw_tool_calls = msg.get("tool_calls")
            tool_calls = (
                [ToolCall.from_openai(tc) for tc in raw_tool_calls]
                if raw_tool_calls
                else None
            )

            if not tool_calls:
                # Agent gave a final text answer — commit and done
                answer_text = msg.get("content") or ""
                answer_ci = t.assistant(answer_text)
                break

            # Commit assistant's tool-calling message (content may be null)
            assistant_text = msg.get("content") or ""
            asst_ci = t.assistant(
                assistant_text,
                metadata={"tool_calls": [tc.to_dict() for tc in tool_calls]},
            )
            intermediate_hashes.append(asst_ci.commit_hash)

            # Execute each tool call and commit results
            for tc in tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                is_error = result.startswith("Error:")
                ci = t.tool_result(tc.id, tc.name, result, is_error=is_error)
                intermediate_hashes.append(ci.commit_hash)

        # --- Full context: everything including verbose tool output ---

        print("=" * 60)
        print("FULL CONTEXT (with all tool results)")
        print("=" * 60)
        full_ctx = t.compile()
        print(f"  {len(full_ctx.messages)} messages  |  {full_ctx.token_count} tokens\n")
        full_ctx.pprint(style="compact")

        # --- Compact tool interactions ---
        # compress_tool_calls() uses EDIT commits to shorten each tool
        # result in-place.  The LLM sees the full tool-calling sequence
        # for holistic context, then produces a per-result summary that
        # is applied as an EDIT commit — preserving commit structure,
        # tool roles, and metadata (tool_call_id, name).

        print(f"\n  Compacting {len(intermediate_hashes)} intermediate messages...\n")
        compact_result = t.compress_tool_calls(
            intermediate_hashes,
            target_tokens=100,
            instructions=(
                "Summarize as a single line in the format: "
                "'line <N>: <exact comment text>' where N is the line number "
                f"and the text is the {_MARKER} comment that was found."
            ),
        )

        print(f"  ToolCompactResult:")
        print(f"    original_tokens:  {compact_result.original_tokens}")
        print(f"    compacted_tokens: {compact_result.compacted_tokens}")
        print(f"    tool_names:       {compact_result.tool_names}")
        print(f"    turn_count:       {compact_result.turn_count}")
        print(f"    edit_commits:     {len(compact_result.edit_commits)} created")
        print(f"    source_commits:   {len(compact_result.source_commits)} edited")

        # --- Compacted context: tool results shortened in-place ---

        print(f"\n{'=' * 60}")
        print("COMPACTED CONTEXT (tool results shortened in-place)")
        print("=" * 60)
        clean_ctx = t.compile()
        saved = full_ctx.token_count - clean_ctx.token_count
        print(f"  {len(clean_ctx.messages)} messages  |  {clean_ctx.token_count} tokens")
        print(f"  Saved {saved} tokens by compacting tool results\n")
        clean_ctx.pprint(style="chat")

        # --- Provenance: original tool results are still in history ---

        print("\n=== Full history (originals preserved for audit) ===\n")
        for entry in reversed(t.log()):
            print(f"  {entry}")


def main():
    part1_manual_tools()
    part2_interactive_approval()
    part3_agentic_loop()
    print("=" * 60)
    print("Done -- all 3 tiers of tool interaction demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
