"""Tool Calling

Two scenarios requiring an LLM, plus an offline Part 2.

Scenario 1 — Agentic loop: the LLM searches cookbook files for a hidden
marker comment. Every tool call and result is committed for full provenance.
After the agent answers, compress_tool_calls() shortens results in-place.

Scenario 2 — Context-aware auto-summarization: intentionally noisy tool
results (huge directory listing, verbose config file) are auto-summarized
on commit. include_context=True lets the summarizer see the conversation
so it keeps only what matters to the user's question.

Part 2 — Offline tool management: error handling with is_error=True and
drop_failed_tool_turns(), the tool query API (find_tool_turns/results/calls),
and tool_result(edit=) for surgical edits. No LLM needed.

Demonstrates: set_tools(), generate() with tool_calls, tool_result(),
              ToolCall, compress_tool_calls(), is_error=True,
              drop_failed_tool_turns(), find_tool_turns/results/calls,
              tool_result(edit=), configure_tool_summarization(),
              include_context=True, agentic loop pattern
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
            "One of the .py files in the cookbook/fundamentals/ directory "
            f"contains a comment line starting with '# {_MARKER}:'. "
            "Find it and tell me the filename, line number, exact text, "
            "and a brief explanation of what the comment means, as well as what other content exists in the file."
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

    # =================================================================
    # PART 2: Offline Tool Management (no LLM needed)
    # =================================================================
    # Error handling, query API, surgical edits — all with pprint()
    # so you can see exactly what the context window looks like.

    print(f"\n\n{'=' * 60}")
    print("PART 2: Offline Tool Management")
    print("=" * 60)
    print()
    print("  No LLM needed. We manually commit tool interactions, then")
    print("  show error handling, the query API, and surgical edits.")
    print()

    with Tract.open() as t3:

        # --- Build a realistic multi-tool session ---

        t3.system("You are a deployment agent.")
        t3.user("Deploy the application to staging.")

        # Turn 1: Health check (success)
        t3.assistant(
            "Checking server health...",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "health_check", "arguments": {}},
            ]},
        )
        t3.tool_result("call_1", "health_check",
                        "Server is healthy. CPU: 23%, Memory: 45%")

        # Turn 2: Deploy (fails — marked as error)
        t3.assistant(
            "Deploying to staging...",
            metadata={"tool_calls": [
                {"id": "call_2", "name": "deploy", "arguments": {"env": "staging"}},
            ]},
        )
        t3.tool_result(
            "call_2", "deploy",
            "Error: Connection refused. Could not reach staging server "
            "at 10.0.1.5:8080.\n"
            "Traceback (most recent call last):\n"
            "  File '/deploy/runner.py', line 42, in deploy_to\n"
            "    conn = ssh.connect(host, port)\n"
            "ConnectionRefusedError: [Errno 111] Connection refused",
            is_error=True,
        )

        # Turn 3: Deploy retry (success)
        t3.assistant(
            "Retrying with backup server...",
            metadata={"tool_calls": [
                {"id": "call_3", "name": "deploy", "arguments": {"env": "staging-backup"}},
            ]},
        )
        deploy_ci = t3.tool_result("call_3", "deploy",
                                    "Deployed successfully to staging-backup. Build #1847.")

        # --- Full context before dropping errors ---

        print("--- Before drop_failed_tool_turns() ---\n")
        ctx_before = t3.compile()
        print(f"  {len(ctx_before.messages)} messages  |  {ctx_before.token_count} tokens\n")
        ctx_before.pprint(style="compact")

        # --- Drop error turns ---

        drop_result = t3.drop_failed_tool_turns()

        print(f"\n  drop_failed_tool_turns() -> ToolDropResult:")
        print(f"    turns_dropped:   {drop_result.turns_dropped}")
        print(f"    commits_skipped: {drop_result.commits_skipped}")
        print(f"    tokens_freed:    {drop_result.tokens_freed}")
        print(f"    tool_names:      {drop_result.tool_names}")

        # --- Context after dropping: error turn gone ---

        print(f"\n--- After drop_failed_tool_turns() ---\n")
        ctx_after = t3.compile()
        saved = ctx_before.token_count - ctx_after.token_count
        print(f"  {len(ctx_after.messages)} messages  |  {ctx_after.token_count} tokens")
        print(f"  (freed {saved} tokens)\n")
        ctx_after.pprint(style="chat")

        # --- Query API ---

        print(f"\n--- find_tool_turns() ---")
        turns = t3.find_tool_turns()
        print(f"  {len(turns)} tool turn(s) in history:")
        for turn in turns:
            names = ", ".join(turn.tool_names)
            print(f"    {names}: {turn.total_tokens} tokens, "
                  f"{len(turn.results)} result(s)")

        print(f"\n--- find_tool_results() ---")
        for r in t3.find_tool_results():
            print(f"    {r.metadata['name']}: {r.token_count} tokens")

        # --- Surgical edit: shorten the deploy result in-place ---

        print(f"\n--- tool_result(edit=) ---")
        print(f"  Before: deploy result is {deploy_ci.token_count} tokens")

        edited_ci = t3.tool_result(
            "call_3", "deploy",
            "Deployed to staging-backup. Build #1847.",
            edit=deploy_ci.commit_hash,
        )
        print(f"  After:  deploy result is {edited_ci.token_count} tokens")
        print(f"  Original preserved at {deploy_ci.commit_hash[:8]}...\n")

        print("\n--- Full history (originals preserved for audit) ---\n")
        for entry in reversed(t3.log()):
            print(f"  {entry}")


if __name__ == "__main__":
    main()
