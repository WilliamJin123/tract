"""Agentic File Search + Post-Hoc Compaction

Scenario 1 of Tool Calling: the LLM searches cookbook files for a hidden
marker comment. Every tool call and result is committed for full provenance.
After the agent answers, compress_tool_calls() shortens results in-place.

Demonstrates: set_tools(), generate() with tool_calls, tool_result(),
              ToolCall, compress_tool_calls(), agentic loop pattern
"""

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

# The marker to search for — split to avoid matching THIS file in search.
_MARKER = "DIS" + "COVERY"


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
                        "description": "Directory path relative to the 08_tool_calling/ root.",
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
                        "description": "Filename (not full path) inside the 08_tool_calling/ directory.",
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
            "description": "Search for a text pattern across all .py files in the 08_tool_calling/ directory. Returns matching lines with filenames.",
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

_SELF = os.path.basename(__file__)  # Exclude this file from search results.


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
            if not fname.endswith(".py") or fname == _SELF:
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
    # =================================================================
    # SCENARIO 1: Agentic File Search + Post-Hoc Compaction
    # =================================================================

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


if __name__ == "__main__":
    main()
