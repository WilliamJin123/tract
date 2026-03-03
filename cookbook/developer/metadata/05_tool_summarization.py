"""Context-Aware Auto-Summarization (Noisy Tools)

Two tiers of tool result management: manual surgical edit and
LLM-powered auto-summarization.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 3 -- Auto-Summarize    configure_tool_summarization(), LLM-driven

Demonstrates: tool_result(edit=), configure_tool_summarization(),
              include_context=True, get_content(), pprint()
"""

import os
import sys
from functools import partial
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

# Allow importing _helpers from the same directory when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import TOOLS, execute_tool as _execute_tool  

# The directory this file lives in — tools will search here.
COOKBOOK_DIR = os.path.dirname(os.path.abspath(__file__))

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
# Part 3: Auto-Summarization  (PART 3 — LLM-Powered)
# =============================================================================

def part3_auto_summarization():
    if not llm.api_key:
        print("=" * 60)
        print("Part 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print("=" * 60)
    print("Part 3: CONTEXT-AWARE AUTO-SUMMARIZATION  [LLM-Powered]")
    print("=" * 60)
    print()
    print("  The user asks a specific question. Tools return intentionally")
    print("  noisy output. The auto-summarizer sees the conversation and")
    print("  extracts only the relevant information.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=llm.large,
    ) as t2:

        t2.set_tools(TOOLS)
        t2.system("You are a code auditor. Answer precisely.")
        t2.user("Which function in this directory makes HTTP calls to the LLM API?")

        # Enable context-aware auto-summarization — the hook fires on every
        # tool_result() commit and the LLM sees the conversation above.
        t2.configure_tool_summarization(
            auto_threshold=50,
            default_instructions="Keep ONLY facts relevant to the user's question.",
            include_context=True,
        )

        # --- Noisy tool result 1: directory listing ---
        # The user asked about HTTP/LLM calls. A full directory listing
        # has many files; the LLM should focus on _helpers.py.

        real_listing = execute_tool("list_directory", {"path": "."})
        t2.assistant(
            "Let me list the files to find the relevant module.",
            metadata={"tool_calls": [
                {"id": "n1", "name": "list_directory",
                 "arguments": {"path": "."}},
            ]},
        )
        ci1 = t2.tool_result("n1", "list_directory", real_listing)

        print(f"  Tool 1: list_directory")
        print(f"    Original: {len(real_listing)} chars, "
              f"{len(real_listing.splitlines())} entries")
        print(f"    Summarized: {t2.get_content(ci1)}\n")

        # --- Noisy tool result 2: full file read ---
        # _helpers.py has ~160 lines; only call_llm() is relevant.

        real_file = execute_tool("read_file", {"path": "_helpers.py"})
        t2.assistant(
            "Found _helpers.py. Let me read it.",
            metadata={"tool_calls": [
                {"id": "n2", "name": "read_file",
                 "arguments": {"path": "_helpers.py"}},
            ]},
        )
        ci2 = t2.tool_result("n2", "read_file", real_file)

        print(f"  Tool 2: read_file (_helpers.py)")
        print(f"    Original: {len(real_file)} chars, "
              f"{len(real_file.splitlines())} lines")
        print(f"    Summarized: {t2.get_content(ci2)}\n")

        # --- Noisy tool result 3: search across all files ---
        # Search for "httpx" finds matches in multiple files; only
        # the call_llm() function is relevant to the question.

        real_search = execute_tool("search_files", {"pattern": "httpx"})
        t2.assistant(
            "Let me search for 'httpx' across all files.",
            metadata={"tool_calls": [
                {"id": "n3", "name": "search_files",
                 "arguments": {"pattern": "httpx"}},
            ]},
        )
        ci3 = t2.tool_result("n3", "search_files", real_search)

        print(f"  Tool 3: search_files ('httpx')")
        print(f"    Original: {len(real_search)} chars, "
              f"{real_search.count(chr(10)) + 1} matches")
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
    part3_auto_summarization()
    print("=" * 60)
    print("Done -- both parts of tool result management demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
