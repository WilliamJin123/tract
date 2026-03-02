"""Shared helpers for tool_results cookbook examples.

Provides: TOOLS, make_execute_tool(), call_llm().

These are extracted from 01_agentic_loop.py and 02_auto_summarization.py
to avoid duplication.
"""

import os

import httpx

# --- Environment variables (shared by all tool_results examples) ---

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


# --- Tool definitions (OpenAI function-calling format) ---

LIST_DIRECTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": "List files in a directory. Returns one filename per line.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the cookbook root.",
                },
            },
            "required": ["path"],
        },
    },
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file and return its contents as text.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filename (not full path) inside the cookbook directory.",
                },
            },
            "required": ["path"],
        },
    },
}

SEARCH_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search for a text pattern across all .py files in the cookbook directory. Returns matching lines with filenames.",
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
}

TOOLS = [LIST_DIRECTORY_TOOL, READ_FILE_TOOL, SEARCH_FILES_TOOL]


# --- Tool implementations ---

def execute_tool(name: str, arguments: dict, *, cookbook_dir: str, exclude_file: str = "") -> str:
    """Execute a tool by name. All paths are sandboxed to cookbook_dir.

    Parameters
    ----------
    name : str
        Tool name ("list_directory", "read_file", or "search_files").
    arguments : dict
        Tool arguments from the LLM.
    cookbook_dir : str
        Absolute path to the sandbox directory for file operations.
    exclude_file : str
        Basename of a file to exclude from search_files results (typically
        the calling script itself).
    """
    if name == "list_directory":
        rel = arguments.get("path", ".")
        target = os.path.normpath(os.path.join(cookbook_dir, rel))
        if not target.startswith(cookbook_dir):
            return "Error: path outside sandbox"
        try:
            entries = sorted(os.listdir(target))
            return "\n".join(entries)
        except OSError as e:
            return f"Error: {e}"

    elif name == "read_file":
        filename = arguments["path"]
        target = os.path.normpath(os.path.join(cookbook_dir, filename))
        if not target.startswith(cookbook_dir):
            return "Error: path outside sandbox"
        try:
            with open(target) as f:
                return f.read()
        except OSError as e:
            return f"Error: {e}"

    elif name == "search_files":
        pattern = arguments["pattern"]
        matches = []
        for fname in sorted(os.listdir(cookbook_dir)):
            if not fname.endswith(".py") or fname == exclude_file:
                continue
            fpath = os.path.join(cookbook_dir, fname)
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
