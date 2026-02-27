"""Tool Result Hooks

Part 1 — Hook basics: t.on("tool_result", handler) intercepts every
tool_result() call. PendingToolResult fields (tool_name, content,
token_count), approve/reject flow.

Part 2 — Edit and summarize: pending.edit_result() for manual
replacement, pending.summarize() for LLM-driven summarization.
original_content preservation for provenance.

Part 3 — Declarative config: configure_tool_summarization() with
per-tool instructions, auto_threshold, and default_instructions.
The sugar layer that replaces manual hook registration.

Part 4 — Custom routing: a handler that routes different tools through
different strategies (pass-through, edit, summarize, reject).

Demonstrates: t.on("tool_result", handler), PendingToolResult,
              edit_result(), summarize(), approve(), reject(),
              original_content, configure_tool_summarization(),
              auto_threshold, per-tool instructions
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.tool_result import PendingToolResult

load_dotenv()


def _safe(text: str) -> str:
    """Sanitize text for Windows console (cp1252) — replace non-ASCII."""
    return text.encode("ascii", "replace").decode("ascii")

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# ---------------------------------------------------------------------------
# Part 1: Hook Basics (t.on("tool_result", handler))
# ---------------------------------------------------------------------------

def part1_hook_basics():
    print("=" * 60)
    print("PART 1 — Tool Result Hook Basics")
    print("=" * 60)
    print()
    print("  Every tool_result() call passes through the hook system.")
    print("  A handler can inspect, modify, or reject before commit.")

    with Tract.open() as t:
        t.system("You are a code assistant.")
        t.user("Find all Python files in the project.")

        tool_log = []

        def log_tool_results(pending: PendingToolResult):
            """Log every tool result, then approve."""
            tool_log.append({
                "tool": pending.tool_name,
                "chars": len(pending.content),
                "tokens": pending.token_count,
                "is_error": pending.is_error,
            })
            print(f"    [hook] tool={pending.tool_name}, "
                  f"{len(pending.content)} chars, "
                  f"{pending.token_count} tokens")
            pending.approve()

        t.on("tool_result", log_tool_results)

        # Simulate tool calls with fake results
        t.assistant("Let me search for Python files.", metadata={
            "tool_calls": [{"id": "tc1", "name": "find_files",
                            "arguments": {"pattern": "*.py"}}],
        })
        t.tool_result("tc1", "find_files", "src/main.py\nsrc/utils.py\ntests/test_main.py")

        t.assistant("Let me also check the config.", metadata={
            "tool_calls": [{"id": "tc2", "name": "read_file",
                            "arguments": {"path": "config.yaml"}}],
        })
        t.tool_result("tc2", "read_file", "debug: true\nlog_level: INFO\nport: 8080")

        print(f"\n  Tool log: {len(tool_log)} results intercepted")
        for entry in tool_log:
            print(f"    {entry}")

    # --- review=True: manual inspection ---
    print(f"\n  review=True returns PendingToolResult:")

    with Tract.open() as t:
        t.system("You are a code assistant.")
        t.assistant("Reading file.", metadata={
            "tool_calls": [{"id": "tc3", "name": "read_file",
                            "arguments": {"path": "secret.env"}}],
        })

        pending: PendingToolResult = t.tool_result(
            "tc3", "read_file",
            "API_KEY=sk-secret-12345\nDB_PASSWORD=hunter2",
            review=True,
        )

        print(f"    type: {type(pending).__name__}")
        print(f"    status: {pending.status}")
        print(f"    tool_name: {pending.tool_name}")
        print(f"    content: {pending.content[:40]}...")
        print(f"    token_count: {pending.token_count}")

        # Reject sensitive content
        pending.reject("Contains secrets — cannot enter context window")
        print(f"\n    After reject(): status={pending.status}")
        print(f"    reason: {pending.rejection_reason}")


# ---------------------------------------------------------------------------
# Part 2: Edit and Summarize
# ---------------------------------------------------------------------------

def part2_edit_and_summarize():
    print("\n" + "=" * 60)
    print("PART 2 — Edit and Summarize")
    print("=" * 60)

    # --- edit_result(): manual replacement ---
    print("\n  edit_result(): replace content before commit")

    with Tract.open() as t:
        t.system("You are a security auditor.")
        t.assistant("Reading config.", metadata={
            "tool_calls": [{"id": "tc4", "name": "read_config",
                            "arguments": {"path": "app.env"}}],
        })

        pending: PendingToolResult = t.tool_result(
            "tc4", "read_config",
            "APP_NAME=myapp\nSECRET_KEY=abc123\nDB_URL=postgres://user:pass@host/db",
            review=True,
        )

        print(f"    original: {pending.content[:60]}...")
        print(f"    original_content: {pending.original_content}")

        # Redact secrets
        redacted = pending.content.replace("abc123", "***").replace("user:pass", "***:***")
        pending.edit_result(redacted)

        print(f"\n    After edit_result():")
        print(f"    content: {pending.content[:60]}...")
        print(f"    original_content: {pending.original_content[:60]}...")
        print(f"    (Original preserved for provenance)")

        pending.approve()
        print(f"    Committed with redacted content")

    # --- summarize(): LLM-driven summarization ---
    print(f"\n  summarize(): LLM compresses verbose tool output")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a code auditor. Answer precisely.")
        t.user("What database is this project using?")
        t.assistant("Let me check the config files.", metadata={
            "tool_calls": [{"id": "tc5", "name": "read_file",
                            "arguments": {"path": "docker-compose.yaml"}}],
        })

        # Verbose tool output — much more than we need
        verbose_config = "\n".join([
            "version: '3.8'",
            "services:",
            "  web:",
            "    build: .",
            "    ports:",
            "      - '8080:8080'",
            "    environment:",
            "      - DEBUG=true",
            "      - LOG_LEVEL=info",
            "  db:",
            "    image: postgres:15",
            "    environment:",
            "      - POSTGRES_DB=myapp",
            "      - POSTGRES_USER=admin",
            "      - POSTGRES_PASSWORD=secret",
            "    ports:",
            "      - '5432:5432'",
            "    volumes:",
            "      - pgdata:/var/lib/postgresql/data",
            "  redis:",
            "    image: redis:7",
            "    ports:",
            "      - '6379:6379'",
            "volumes:",
            "  pgdata:",
        ])

        pending: PendingToolResult = t.tool_result(
            "tc5", "read_file", verbose_config, review=True,
        )

        print(f"    Before: {len(pending.content)} chars, {pending.token_count} tokens")

        # Summarize with instructions
        pending.summarize(
            instructions="Extract only database-related configuration.",
            include_context=True,  # LLM sees the user's question
        )

        print(f"    After summarize(): {len(pending.content)} chars")
        print(f"    Summary: {_safe(pending.content[:120])}")
        print(f"    original_content preserved: {pending.original_content is not None}")

        pending.approve()


# ---------------------------------------------------------------------------
# Part 3: Declarative Config (configure_tool_summarization)
# ---------------------------------------------------------------------------

def part3_declarative_config():
    print("\n" + "=" * 60)
    print("PART 3 — configure_tool_summarization()")
    print("=" * 60)
    print()
    print("  Sugar over t.on('tool_result', handler).")
    print("  Set per-tool instructions and auto-thresholds declaratively.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a project analyst.")
        t.user("Give me an overview of the project structure.")

        # Configure: per-tool instructions + auto-threshold for everything else
        t.configure_tool_summarization(
            instructions={
                "list_directory": "Summarize to a bullet list of top-level directories only.",
                "read_file": "Keep the first 5 lines. Summarize the rest.",
            },
            auto_threshold=50,  # Anything over 50 tokens gets summarized
            default_instructions="Keep only the most relevant information.",
            include_context=True,
        )

        # --- Tool 1: list_directory (has specific instructions) ---
        t.assistant("Let me list the project.", metadata={
            "tool_calls": [{"id": "d1", "name": "list_directory",
                            "arguments": {"path": "."}}],
        })
        big_listing = "\n".join(
            ["src/", "tests/", "docs/", "config/", "README.md", "setup.py"]
            + [f"module_{i:02d}.py" for i in range(25)]
        )
        ci1 = t.tool_result("d1", "list_directory", big_listing)
        stored1 = t.get_content(ci1)
        print(f"\n  list_directory:")
        print(f"    Original: {len(big_listing)} chars")
        print(f"    Stored:   {_safe(stored1[:100])}")

        # --- Tool 2: read_file (has specific instructions) ---
        t.assistant("Reading the main module.", metadata={
            "tool_calls": [{"id": "d2", "name": "read_file",
                            "arguments": {"path": "src/main.py"}}],
        })
        big_file = "\n".join([f"# Line {i}: some code here" for i in range(50)])
        ci2 = t.tool_result("d2", "read_file", big_file)
        stored2 = t.get_content(ci2)
        print(f"\n  read_file:")
        print(f"    Original: {len(big_file)} chars")
        print(f"    Stored:   {_safe(stored2[:100])}")

        # --- Tool 3: search_code (no specific instructions, uses default) ---
        t.assistant("Searching for imports.", metadata={
            "tool_calls": [{"id": "d3", "name": "search_code",
                            "arguments": {"query": "import"}}],
        })
        big_search = "\n".join(
            [f"module_{i:02d}.py:1: import os" for i in range(30)]
        )
        ci3 = t.tool_result("d3", "search_code", big_search)
        stored3 = t.get_content(ci3)
        print(f"\n  search_code (default instructions, over threshold):")
        print(f"    Original: {len(big_search)} chars")
        print(f"    Stored:   {_safe(stored3[:100])}")

        # --- Tool 4: small result (under threshold, passes through) ---
        t.assistant("Quick check.", metadata={
            "tool_calls": [{"id": "d4", "name": "check_version",
                            "arguments": {}}],
        })
        ci4 = t.tool_result("d4", "check_version", "v2.4.1")
        stored4 = t.get_content(ci4)
        print(f"\n  check_version (under threshold, pass-through):")
        print(f"    Original: v2.4.1")
        print(f"    Stored:   {_safe(stored4)}")


# ---------------------------------------------------------------------------
# Part 4: Custom Routing (per-tool strategy)
# ---------------------------------------------------------------------------

def part4_custom_routing():
    print("\n" + "=" * 60)
    print("PART 4 — Custom Routing Handler")
    print("=" * 60)
    print()
    print("  A single handler that routes tools through different strategies:")
    print("  pass-through, edit (redact), summarize, or reject.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a security-conscious assistant.")
        t.user("Audit the system configuration.")

        BLOCKED_TOOLS = {"run_shell", "delete_file"}
        REDACT_TOOLS = {"read_env", "get_secrets"}
        SUMMARIZE_TOOLS = {"read_file", "search_files", "list_directory"}

        def smart_router(pending: PendingToolResult):
            """Route each tool through the appropriate strategy."""
            tool = pending.tool_name

            if tool in BLOCKED_TOOLS:
                pending.reject(f"Tool '{tool}' is blocked by security policy")
                print(f"    [{tool}] REJECTED")

            elif tool in REDACT_TOOLS:
                # Redact sensitive patterns
                import re
                redacted = re.sub(
                    r'(password|secret|key|token)\s*[=:]\s*\S+',
                    r'\1=***REDACTED***',
                    pending.content,
                    flags=re.IGNORECASE,
                )
                pending.edit_result(redacted)
                pending.approve()
                print(f"    [{tool}] REDACTED and approved")

            elif tool in SUMMARIZE_TOOLS and pending.token_count > 30:
                pending.summarize(instructions="Keep only security-relevant entries.")
                pending.approve()
                print(f"    [{tool}] SUMMARIZED ({pending.token_count} tokens)")

            else:
                pending.approve()
                print(f"    [{tool}] PASS-THROUGH")

        t.on("tool_result", smart_router)

        # --- Trigger each strategy ---

        # 1. Blocked tool
        t.assistant("Running a shell command.", metadata={
            "tool_calls": [{"id": "r1", "name": "run_shell",
                            "arguments": {"cmd": "rm -rf /"}}],
        })
        result1 = t.tool_result("r1", "run_shell", "Command output here")
        if hasattr(result1, "status"):
            print(f"      Result: {type(result1).__name__}, status={result1.status}")
        else:
            print(f"      Result: {type(result1).__name__} (committed)")

        # 2. Redact tool
        t.assistant("Reading environment.", metadata={
            "tool_calls": [{"id": "r2", "name": "read_env",
                            "arguments": {}}],
        })
        t.tool_result("r2", "read_env", "APP=myapp\nSECRET_KEY=abc123\nDB_PASSWORD=hunter2")

        # 3. Summarize tool (large output)
        t.assistant("Listing files.", metadata={
            "tool_calls": [{"id": "r3", "name": "list_directory",
                            "arguments": {"path": "/"}}],
        })
        big_listing = "\n".join([f"file_{i:03d}.py" for i in range(40)])
        t.tool_result("r3", "list_directory", big_listing)

        # 4. Pass-through (small, unknown tool)
        t.assistant("Checking version.", metadata={
            "tool_calls": [{"id": "r4", "name": "get_version",
                            "arguments": {}}],
        })
        t.tool_result("r4", "get_version", "v3.1.0")

        print(f"\n  Final context:")
        ctx = t.compile()
        print(f"    {len(ctx.messages)} messages, {ctx.token_count} tokens")


# ---------------------------------------------------------------------------

def main():
    part1_hook_basics()
    part2_edit_and_summarize()
    part3_declarative_config()
    part4_custom_routing()


if __name__ == "__main__":
    main()
