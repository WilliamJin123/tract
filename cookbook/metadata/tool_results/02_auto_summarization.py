"""Context-Aware Auto-Summarization (Noisy Tools)

Three tiers of tool result management: manual surgical edit, interactive
review of pending summaries, and fully autonomous auto-summarization.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: tool_result(edit=), configure_tool_summarization(),
              include_context=True, get_content(), pprint(), click.edit()
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

# The directory this file lives in — tools will search here (one level up).
COOKBOOK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bind execute_tool to this file's sandbox directory.
_SELF = os.path.basename(__file__)
execute_tool = partial(_execute_tool, cookbook_dir=COOKBOOK_DIR, exclude_file=_SELF)


# =============================================================================
# Part 1: Manual Surgical Edit  (PART 1 — Manual)
# =============================================================================

def part1_manual_edit():
    """Edit a tool result manually — no LLM needed."""
    print("=" * 60)
    print("Part 1: MANUAL SURGICAL EDIT  [Manual Tier]")
    print("=" * 60)
    print()
    print("  Use tool_result(edit=) to surgically replace a verbose tool")
    print("  result with a trimmed version. Show token savings.")
    print()

    with Tract.open() as t:
        t.system("You are a configuration auditor.")
        t.user("What is the database connection string?")

        # Commit a verbose tool result
        t.assistant(
            "Let me read the config file.",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "read_file",
                 "arguments": {"path": "config.yaml"}},
            ]},
        )
        original_ci = t.tool_result(
            "call_1", "read_file",
            "APP_NAME=my-service\nAPP_VERSION=2.4.1\nLOG_LEVEL=INFO\n"
            "CACHE_TTL=3600\nCACHE_BACKEND=redis\nREDIS_HOST=cache.internal\n"
            "DB_CONNECTION=postgresql://admin:s3cret@db.prod.internal:5432/myapp\n"
            "DB_POOL_SIZE=20\nDB_TIMEOUT=30\nENABLE_DARK_MODE=true\n"
            "MAX_UPLOAD_SIZE=10485760\nSMTP_HOST=mail.example.com",
        )

        before_ctx = t.compile()
        print(f"  Before edit: {before_ctx.token_count} tokens")

        # Surgical edit: keep only the relevant line
        edited_ci = t.tool_result(
            "call_1", "read_file",
            "DB_CONNECTION=postgresql://admin:s3cret@db.prod.internal:5432/myapp",
            edit=original_ci.commit_hash,
        )

        after_ctx = t.compile()
        saved = before_ctx.token_count - after_ctx.token_count
        print(f"  After edit:  {after_ctx.token_count} tokens (saved {saved})")
        print(f"  Original preserved at {original_ci.commit_hash[:8]} for audit.\n")

    print()


# =============================================================================
# Part 2: Interactive Review  (PART 2 — Interactive)
# =============================================================================

def part2_interactive_review():
    """Review and edit tool results interactively before committing."""
    print("=" * 60)
    print("Part 2: INTERACTIVE REVIEW  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  After a tool result is committed, open it in $EDITOR for")
    print("  trimming, then apply the edit with confirmation.")
    print()

    with Tract.open() as t:
        t.system("You are a deployment auditor.")
        t.user("Show me the server health report.")

        # Commit a verbose tool result
        t.assistant(
            "Running health check...",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "health_check",
                 "arguments": {}},
            ]},
        )
        verbose_output = (
            "Server: prod-web-01\n"
            "CPU: 23%  Memory: 45%  Disk: 67%\n"
            "Uptime: 42 days\n"
            "Active connections: 1,247\n"
            "Request rate: 340 req/s\n"
            "Error rate: 0.02%\n"
            "Last deploy: 2026-02-28T14:30:00Z\n"
            "Docker containers: 12 running, 0 stopped\n"
            "SSL cert expires: 2026-09-15\n"
            "DNS resolution: 2.1ms avg"
        )
        ci = t.tool_result("call_1", "health_check", verbose_output)

        # Show the content and offer to edit
        content = t.get_content(ci.commit_hash)
        print(f"  Tool result ({len(verbose_output)} chars):")
        for line in verbose_output.split("\n")[:5]:
            print(f"    {line}")
        print(f"    ... ({len(verbose_output.splitlines())} lines total)\n")

        edited = click.edit(verbose_output)
        if edited and edited.strip() != verbose_output.strip():
            if click.confirm("  Approve trimmed result?"):
                t.tool_result(
                    "call_1", "health_check", edited.strip(),
                    edit=ci.commit_hash,
                )
                print(f"  Edit applied.\n")
            else:
                print(f"  Edit cancelled.\n")
        else:
            print(f"  No changes made.\n")

    print()


# =============================================================================
# Part 3: Auto-Summarization  (PART 3 — LLM / Agent)
# =============================================================================

def part3_auto_summarization():
    if not TRACT_OPENAI_API_KEY:
        print("=" * 60)
        print("Part 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print("=" * 60)
    print("Part 3: CONTEXT-AWARE AUTO-SUMMARIZATION  [Agent Tier]")
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


def main():
    part1_manual_edit()
    part2_interactive_review()
    part3_auto_summarization()
    print("=" * 60)
    print("Done -- all 3 tiers of tool result management demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
