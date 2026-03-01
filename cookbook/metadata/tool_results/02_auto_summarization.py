"""Context-Aware Auto-Summarization (Noisy Tools)

Scenario 2 of Tool Calling: intentionally noisy tool results (huge directory
listing, verbose config file) are auto-summarized on commit.
include_context=True lets the summarizer see the conversation so it keeps
only what matters to the user's question.

Demonstrates: set_tools(), tool_result(), configure_tool_summarization(),
              include_context=True, get_content(), pprint()
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
COOKBOOK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    # SCENARIO 2: Context-Aware Auto-Summarization (Noisy Tools)
    # =================================================================
    # A different task with intentionally verbose tool results.
    # configure_tool_summarization(include_context=True) lets the
    # summarizer see the conversation, so it extracts ONLY what the
    # user asked about — filtering out irrelevant noise.

    print(f"\n\n{'=' * 60}")
    print("SCENARIO 2: Context-Aware Auto-Summarization")
    print("=" * 60)
    print()
    print("  The user asks a specific question. Tools return intentionally")
    print("  noisy output. The auto-summarizer sees the conversation and")
    print("  extracts only the relevant information.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t2:

        t2.set_tools(TOOLS)
        t2.system("You are a configuration auditor. Answer precisely.")
        t2.user("What is the database connection string in the project config?")

        # Enable context-aware auto-summarization — the hook fires on every
        # tool_result() commit and the LLM sees the conversation above.
        t2.configure_tool_summarization(
            auto_threshold=50,
            default_instructions="Keep ONLY facts relevant to the user's question.",
            include_context=True,
        )

        # --- Noisy tool result 1: large directory listing ---
        # The user asked about a DB connection string. A 35-entry listing
        # has one relevant file (config.yaml) buried in noise.

        fake_listing = "\n".join(
            ["README.md", "setup.py", "config.yaml", "requirements.txt",
             "Makefile", "Dockerfile", ".gitignore", ".env.example",
             "pyproject.toml", "LICENSE"]
            + [f"module_{i:02d}.py" for i in range(20)]
            + ["tests/", "docs/", "migrations/", "scripts/", "static/"]
        )
        t2.assistant(
            "Let me list the project files to find the config.",
            metadata={"tool_calls": [
                {"id": "n1", "name": "list_directory",
                 "arguments": {"path": "."}},
            ]},
        )
        ci1 = t2.tool_result("n1", "list_directory", fake_listing)

        print(f"  Tool 1: list_directory")
        print(f"    Original: {len(fake_listing)} chars, "
              f"{len(fake_listing.splitlines())} lines")
        print(f"    Summarized: {t2.get_content(ci1)}\n")

        # --- Noisy tool result 2: verbose config with one relevant line ---
        # 20 settings, only DB_CONNECTION matters for the question.

        fake_config = "\n".join([
            "# Application Configuration",
            "APP_NAME=my-service",
            "APP_VERSION=2.4.1",
            "LOG_LEVEL=INFO",
            "LOG_FORMAT=json",
            "CACHE_TTL=3600",
            "CACHE_BACKEND=redis",
            "REDIS_HOST=cache.internal",
            "REDIS_PORT=6379",
            "",
            "# Database settings",
            "DB_CONNECTION=postgresql://admin:s3cret@db.prod.internal:5432/myapp",
            "DB_POOL_SIZE=20",
            "DB_TIMEOUT=30",
            "",
            "# Feature flags",
            "ENABLE_DARK_MODE=true",
            "ENABLE_BETA=false",
            "MAX_UPLOAD_SIZE=10485760",
            "ALLOWED_ORIGINS=*.example.com",
            "SMTP_HOST=mail.example.com",
            "SMTP_PORT=587",
        ])
        t2.assistant(
            "Found config.yaml. Let me read it.",
            metadata={"tool_calls": [
                {"id": "n2", "name": "read_file",
                 "arguments": {"path": "config.yaml"}},
            ]},
        )
        ci2 = t2.tool_result("n2", "read_file", fake_config)

        print(f"  Tool 2: read_file (config.yaml)")
        print(f"    Original: {len(fake_config)} chars, "
              f"{len(fake_config.splitlines())} lines")
        print(f"    Summarized: {t2.get_content(ci2)}\n")

        # --- Noisy tool result 3: search with many irrelevant matches ---
        # Search for "connection" returns hits in logs, docs, and code.
        # Only the config line matters.

        fake_search = "\n".join([
            "module_03.py:12: # TODO: handle connection timeout",
            "module_03.py:45:     self.connection_pool = create_pool()",
            "module_07.py:8: class ConnectionManager:",
            "module_07.py:19:     def close_connection(self):",
            "module_07.py:33:     # connection retry logic",
            "module_11.py:2: import connection_utils",
            "module_15.py:88: # stale connections are cleaned up by GC",
            "config.yaml:12: DB_CONNECTION=postgresql://admin:s3cret"
            "@db.prod.internal:5432/myapp",
        ])
        t2.assistant(
            "Let me search for 'connection' across all files.",
            metadata={"tool_calls": [
                {"id": "n3", "name": "search_files",
                 "arguments": {"pattern": "connection"}},
            ]},
        )
        ci3 = t2.tool_result("n3", "search_files", fake_search)

        print(f"  Tool 3: search_files ('connection')")
        print(f"    Original: {len(fake_search)} chars, "
              f"{fake_search.count(chr(10)) + 1} matches")
        print(f"    Summarized: {t2.get_content(ci3)}\n")

        # --- Final context: see how the LLM window looks ---

        print(f"{'=' * 60}")
        print("FINAL CONTEXT (auto-summarized tool results)")
        print("=" * 60)
        ctx = t2.compile()
        print(f"  {len(ctx.messages)} messages  |  {ctx.token_count} tokens\n")
        ctx.pprint(style="chat")


if __name__ == "__main__":
    main()
