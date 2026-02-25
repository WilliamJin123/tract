"""Tool Calling

Build an agentic loop where the LLM decides which tools to call, tract
commits every step, and you compress the verbose tool output afterward.

The agent's mission: find a hidden comment in THIS file by searching the
cookbook directory. Every tool call and result is committed for full
provenance. After the agent answers, compress the noisy intermediate
tool interactions into a concise summary — preserving what happened
without the raw file listings and contents.

After compression, demonstrates the tool query API for inspecting
tool history, tool_result(edit=) for surgical edits, and
configure_tool_summarization() for automatic per-tool summarization.

Demonstrates: set_tools(), generate() with tool_calls, tool_result(),
              ToolCall, compress_tool_calls() for tool result cleanup,
              find_tool_turns/results/calls for querying tool history,
              tool_result(edit=) for surgical result replacement,
              configure_tool_summarization() for automatic compression,
              agentic loop pattern, compile() before/after compression
"""

import json
import os

import httpx
from dotenv import load_dotenv

from tract import Tract
from tract.protocols import ToolCall

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"

# The directory this file lives in — tools will search here.
COOKBOOK_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------
# DISCOVERY: Tool calls are just commits with metadata — the context
# window is the source of truth.
# -----------------------------------------------------------------------


# --- Tool definitions (OpenAI function-calling format) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory. Returns one filename per line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the cookbook/fundamentals/ root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filename (not full path) inside the cookbook/fundamentals/ directory.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a text pattern across all .py files in the cookbook/fundamentals/ directory. Returns matching lines with filenames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for (case-sensitive substring match).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


# --- Tool implementations ---

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name. All paths are sandboxed to COOKBOOK_DIR."""
    if name == "list_directory":
        rel = arguments.get("path", ".")
        target = os.path.normpath(os.path.join(COOKBOOK_DIR, rel))
        if not target.startswith(COOKBOOK_DIR):
            return "Error: path outside sandbox"
        try:
            entries = sorted(os.listdir(target))
            return "\n".join(entries)
        except OSError as e:
            return f"Error: {e}"

    elif name == "read_file":
        filename = arguments["path"]
        target = os.path.normpath(os.path.join(COOKBOOK_DIR, filename))
        if not target.startswith(COOKBOOK_DIR):
            return "Error: path outside sandbox"
        try:
            with open(target) as f:
                return f.read()
        except OSError as e:
            return f"Error: {e}"

    elif name == "search_files":
        pattern = arguments["pattern"]
        matches = []
        for fname in sorted(os.listdir(COOKBOOK_DIR)):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(COOKBOOK_DIR, fname)
            try:
                with open(fpath) as f:
                    for i, line in enumerate(f, 1):
                        if pattern in line:
                            matches.append(f"{fname}:{i}: {line.rstrip()}")
            except OSError:
                continue
        return "\n".join(matches) if matches else f"No matches for '{pattern}'"

    return f"Unknown tool: {name}"


# --- LLM caller (bypasses generate() for tool-calling turns) ---

def call_llm(messages: list[dict], tools: list[dict]) -> dict:
    """Call OpenAI-compatible API with tool definitions.

    We call the API directly (via httpx) rather than using t.generate()
    because tool-calling responses often have null content, which the
    current generate() path doesn't handle. Tract manages the context;
    we manage the LLM call.
    """
    response = httpx.post(
        f"{TRACT_OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {TRACT_OPENAI_API_KEY}"},
        json={
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main():
    print("=" * 60)
    print("TOOL CALLING: Agentic File Search")
    print("=" * 60)
    print()
    print("  The agent will search the cookbook directory to find a hidden")
    print("  DISCOVERY comment in one of the .py files. Every tool call")
    print("  and result is committed to the tract for full provenance.")
    print("  Afterward, we compress the verbose results into a summary.")
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
            "One of the .py files in the cookbook/fundamentals/ directory "
            "contains a comment line starting with '# DISCOVERY:'. "
            "Find it and tell me the filename, line number, exact text, "
            "and a brief explanation of what the comment means."
        )

        # --- Agentic loop: compile → LLM → tool calls → execute → repeat ---

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
                print(f"  Agent answered after {turn + 1} turn(s).\n")
                print(f"  Answer: {answer_text}\n")
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
                print(f"  [{turn + 1}] {tc.name}({json.dumps(tc.arguments)})")
                result = execute_tool(tc.name, tc.arguments)

                # Show truncated output
                display = result[:120] + "..." if len(result) > 120 else result
                print(f"       → {display}\n")

                ci = t.tool_result(tc.id, tc.name, result)
                intermediate_hashes.append(ci.commit_hash)

        else:
            print(f"  Agent didn't finish in {max_turns} turns.\n")

        # --- Full context: everything including verbose tool output ---

        print("=" * 60)
        print("FULL CONTEXT (with all tool results)")
        print("=" * 60)
        full_ctx = t.compile()
        print(f"  {len(full_ctx.messages)} messages  |  {full_ctx.token_count} tokens\n")
        full_ctx.pprint(style="chat")

        # --- Compress tool interactions ---
        # compress_tool_calls() auto-detects the final answer and
        # preserves it verbatim while compressing the verbose tool
        # interactions into a concise summary. It uses a tool-aware
        # system prompt (TOOL_SUMMARIZE_SYSTEM) instead of the
        # general-purpose default.

        all_hashes = intermediate_hashes + [answer_ci.commit_hash]
        print(f"\n  Compressing {len(intermediate_hashes)} intermediate messages...\n")
        compress_result = t.compress_tool_calls(
            all_hashes,
            target_tokens=100,
            instructions=(
                "Summarize as a single line in the format: "
                "'line <N>: <exact comment text>' where N is the line number "
                "and the text is the DISCOVERY comment that was found."
            ),
        )

        print(f"  CompressResult:")
        print(f"    original_tokens:   {compress_result.original_tokens}")
        print(f"    compressed_tokens: {compress_result.compressed_tokens}")
        print(f"    compression_ratio: {compress_result.compression_ratio:.1%}")
        print(f"    source_commits:    {len(compress_result.source_commits)} archived")
        print(f"    summary_commits:   {len(compress_result.summary_commits)} created")

        # --- Compressed context: tool interactions summarized ---

        print(f"\n{'=' * 60}")
        print("COMPRESSED CONTEXT (tool interactions summarized)")
        print("=" * 60)
        clean_ctx = t.compile()
        saved = full_ctx.token_count - clean_ctx.token_count
        print(f"  {len(clean_ctx.messages)} messages  |  {clean_ctx.token_count} tokens")
        print(f"  Saved {saved} tokens by compressing tool output\n")
        clean_ctx.pprint(style="chat")

        # --- Provenance: original tool results are still in history ---

        print("\n=== Full history (originals preserved for audit) ===\n")
        for entry in reversed(t.log()):
            print(f"  {entry}")

    # =================================================================
    # PART 2: Tool Query API + Surgical Edits
    # =================================================================
    # These features work without an LLM — demonstrated with a fresh
    # in-memory tract using manually committed tool interactions.

    print(f"\n\n{'=' * 60}")
    print("PART 2: Tool Query API + Surgical Edits")
    print("=" * 60)

    with Tract.open() as t2:
        # Simulate a multi-tool agent session
        t2.system("You are a code analysis agent.")
        t2.user("Find the main entry point and its dependencies.")

        # Turn 1: Agent calls grep
        t2.assistant(
            "I'll search for the main function.",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "grep", "arguments": {"pattern": "def main"}},
            ]},
        )
        grep_ci = t2.tool_result("call_1", "grep", (
            "src/app.py:15: def main():\n"
            "src/app.py:16:     parser = argparse.ArgumentParser()\n"
            "src/app.py:17:     parser.add_argument('--verbose')\n"
            "src/cli.py:42: def main_cli():\n"
            "tests/test_app.py:8: def test_main():\n"
        ))

        # Turn 2: Agent calls read_file
        t2.assistant(
            "Found it in src/app.py. Let me read the full file.",
            metadata={"tool_calls": [
                {"id": "call_2", "name": "read_file", "arguments": {"path": "src/app.py"}},
            ]},
        )
        read_ci = t2.tool_result("call_2", "read_file", (
            "import argparse\nimport logging\nfrom . import db, auth, api\n\n"
            "logger = logging.getLogger(__name__)\n\n"
            "def main():\n    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--verbose')\n"
            "    # ... 200 more lines of setup code ...\n"
        ))

        t2.assistant("The main entry point is src/app.py:15, depending on db, auth, and api.")

        # --- Query API: find tool results, calls, and turns ---

        print("\n--- find_tool_results() ---")
        all_results = t2.find_tool_results()
        print(f"  Total tool results: {len(all_results)}")
        for r in all_results:
            print(f"    {r.metadata['name']}: {r.token_count} tokens")

        grep_results = t2.find_tool_results(name="grep")
        print(f"  Grep results only: {len(grep_results)}")

        print("\n--- find_tool_calls() ---")
        all_calls = t2.find_tool_calls()
        print(f"  Total tool-calling turns: {len(all_calls)}")

        print("\n--- find_tool_turns() ---")
        turns = t2.find_tool_turns()
        print(f"  Total tool turns: {len(turns)}")
        for turn in turns:
            names = ", ".join(turn.tool_names)
            print(f"    {names}: {turn.total_tokens} tokens, {len(turn.results)} result(s)")

        # --- Surgical edit: replace verbose grep output with just filenames ---

        print("\n--- tool_result(edit=) ---")
        print(f"  Before edit: grep result is {grep_ci.token_count} tokens")

        edited_ci = t2.tool_result(
            "call_1", "grep",
            "src/app.py:15, src/cli.py:42, tests/test_app.py:8",
            edit=grep_ci.commit_hash,
        )
        print(f"  After edit:  grep result is {edited_ci.token_count} tokens")
        print(f"  Original preserved at {grep_ci.commit_hash[:8]}...")

        ctx = t2.compile()
        print(f"  Compiled context: {ctx.token_count} tokens")

    # =================================================================
    # PART 3: Automatic Tool Summarization
    # =================================================================
    # configure_tool_summarization() sets up a hook that auto-summarizes
    # verbose tool results. Requires an LLM client.

    print(f"\n\n{'=' * 60}")
    print("PART 3: Automatic Tool Summarization (configure_tool_summarization)")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t3:
        # Configure per-tool summarization rules
        t3.configure_tool_summarization(
            instructions={
                "grep": "Summarize to matching filenames and line numbers only.",
                "read_file": "Keep the first 5 lines, summarize the rest.",
            },
            auto_threshold=200,  # Auto-summarize any tool result over 200 tokens
            default_instructions="Preserve key findings, omit raw output.",
        )

        t3.system("You are a code analysis agent.")
        t3.user("Analyze the project structure.")

        # This tool result will be auto-summarized because "grep" has
        # explicit instructions in the config
        t3.assistant(
            "Searching for Python files...",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "grep", "arguments": {"pattern": "*.py"}},
            ]},
        )
        ci = t3.tool_result("call_1", "grep", (
            "src/app.py:1: import argparse\n"
            "src/app.py:2: import logging\n"
            "src/db.py:1: import sqlalchemy\n"
            "src/auth.py:1: import jwt\n"
            "src/api.py:1: from flask import Flask\n"
            "tests/test_app.py:1: import pytest\n"
            "tests/test_db.py:1: import pytest\n"
        ))
        summarized_content = t3.get_content(ci)
        print(f"\n  Original grep result: 7 lines of matches")
        print(f"  Auto-summarized to: {summarized_content}")

        ctx = t3.compile()
        print(f"\n  Context after auto-summarization: {ctx.token_count} tokens")
        print(f"  (Tool results are summarized on commit — no manual step needed)")


if __name__ == "__main__":
    main()
