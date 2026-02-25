"""Tool Calling

Build an agentic loop where the LLM decides which tools to call, tract
commits every step, and you clean up the verbose tool output afterward.

The agent's mission: find a hidden comment in THIS file by searching the
cookbook directory. Every tool call and result is committed for full
provenance. After the agent answers, SKIP the noisy intermediate results
to leave a clean compiled context — question in, answer out.

Demonstrates: set_tools(), generate() with tool_calls, tool_result(),
              ToolCall, annotate(SKIP) for tool result cleanup,
              agentic loop pattern, compile() before/after cleanup
"""

import json
import os

import httpx
from dotenv import load_dotenv

from tract import Priority, Tract
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
    print("  Afterward, we SKIP the verbose results for a clean context.")
    print()

    with Tract.open() as t:

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
            "Find it and tell me exactly what it says."
        )

        # --- Agentic loop: compile → LLM → tool calls → execute → repeat ---

        tool_result_hashes = []  # Track for cleanup later
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
                t.assistant(answer_text)
                print(f"  Agent answered after {turn + 1} turn(s).\n")
                print(f"  Answer: {answer_text}\n")
                break

            # Commit assistant's tool-calling message (content may be null)
            assistant_text = msg.get("content") or ""
            t.assistant(
                assistant_text,
                metadata={"tool_calls": [tc.to_dict() for tc in tool_calls]},
            )

            # Execute each tool call and commit results
            for tc in tool_calls:
                print(f"  [{turn + 1}] {tc.name}({json.dumps(tc.arguments)})")
                result = execute_tool(tc.name, tc.arguments)

                # Show truncated output
                display = result[:120] + "..." if len(result) > 120 else result
                print(f"       → {display}\n")

                ci = t.tool_result(tc.id, tc.name, result)
                tool_result_hashes.append(ci.commit_hash)

        else:
            print(f"  Agent didn't finish in {max_turns} turns.\n")

        # --- Full context: everything including verbose tool output ---

        print("=" * 60)
        print("FULL CONTEXT (with all tool results)")
        print("=" * 60)
        full_ctx = t.compile()
        print(f"  {len(full_ctx.messages)} messages  |  {full_ctx.token_count} tokens\n")
        full_ctx.pprint(style="compact")

        # --- Clean up: SKIP tool results ---
        # The agent's final answer contains the relevant info.
        # The raw tool output (file listings, file contents) is noise now.
        # SKIP hides them from compile() but preserves them in history
        # for audit — you can always un-skip or inspect via log().

        print(f"\n  Skipping {len(tool_result_hashes)} tool result(s)...\n")
        for h in tool_result_hashes:
            t.annotate(h, Priority.SKIP, reason="tool output consumed by agent")

        # --- Clean context: just the conversation, no intermediate noise ---

        print("=" * 60)
        print("CLEAN CONTEXT (tool results skipped)")
        print("=" * 60)
        clean_ctx = t.compile()
        saved = full_ctx.token_count - clean_ctx.token_count
        print(f"  {len(clean_ctx.messages)} messages  |  {clean_ctx.token_count} tokens")
        print(f"  Saved {saved} tokens by skipping tool output\n")
        clean_ctx.pprint(style="compact")

        # --- Provenance: tool results are still in history ---

        print("\n=== Full history (tool results preserved for audit) ===\n")
        for entry in reversed(t.log()):
            print(f"  {entry}")


if __name__ == "__main__":
    main()
