"""Tool Query and Audit

Find, filter, and inspect tool-related commits across a conversation.
Use the query API to answer questions like "which tools were called?",
"how many tokens did grep results consume?", and "which tool turns
are worth compressing?"  Then act on the answers — bulk-edit verbose
results, selectively compress by tool name, or export an audit log.

No LLM required for the query and edit operations (Parts 1-2).
Part 3 uses an LLM for selective compression by tool name.

Part 1: Query API — find_tool_results(), find_tool_calls(), find_tool_turns()
Part 2: Surgical edits — tool_result(edit=) to trim verbose results
Part 3: Selective compression — compress_tool_calls(name=) to compress
        only specific tool types while leaving others untouched

Each part uses its own Tract instance for clarity.

Demonstrates: find_tool_results(name=, after=), find_tool_calls(name=),
              find_tool_turns(name=), ToolTurn (all_hashes, result_hashes,
              total_tokens, tool_names), tool_result(edit=) for surgical
              replacement, compress_tool_calls(name=) for selective
              compression, token accounting before/after edits
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


def _build_agent_session(t):
    """Build a realistic multi-tool agent session for auditing.

    Simulates a coding agent that uses grep, read_file, and bash tools
    to investigate a bug. Returns commit hashes for key tool results.
    """
    t.system("You are a code debugging agent.")
    t.user("Find and fix the authentication bug in the login module.")

    # Turn 1: grep for auth-related files (verbose)
    t.assistant(
        "Let me search for authentication code.",
        metadata={"tool_calls": [
            {"id": "c1", "name": "grep", "arguments": {"pattern": "authenticate"}},
        ]},
    )
    grep1_ci = t.tool_result("c1", "grep", (
        "src/auth/login.py:15: def authenticate(username, password):\n"
        "src/auth/login.py:22:     if not authenticate_ldap(username, password):\n"
        "src/auth/login.py:31:     logger.info(f'authenticate() called for {username}')\n"
        "src/auth/session.py:8: from auth.login import authenticate\n"
        "src/auth/session.py:45: result = authenticate(user, pw)\n"
        "src/auth/middleware.py:12: from auth.login import authenticate\n"
        "src/auth/middleware.py:33: auth_result = authenticate(req.user, req.pass)\n"
        "tests/test_auth.py:10: from auth.login import authenticate\n"
        "tests/test_auth.py:25: assert authenticate('admin', 'pass123')\n"
        "tests/test_auth.py:30: assert not authenticate('admin', 'wrong')\n"
        "docs/api.md:88: The `authenticate()` function validates credentials.\n"
        "docs/api.md:92: See `authenticate()` for LDAP integration details."
    ))

    # Turn 2: read the main auth file (verbose)
    t.assistant(
        "Let me read the login module.",
        metadata={"tool_calls": [
            {"id": "c2", "name": "read_file", "arguments": {"path": "src/auth/login.py"}},
        ]},
    )
    read1_ci = t.tool_result("c2", "read_file", (
        "import hashlib\n"
        "import logging\n"
        "from datetime import datetime, timedelta\n"
        "\n"
        "from ldap3 import Connection, Server\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "MAX_ATTEMPTS = 5\n"
        "LOCKOUT_DURATION = timedelta(minutes=15)\n"
        "_failed_attempts = {}  # BUG: not thread-safe\n"
        "\n"
        "\n"
        "def authenticate(username, password):\n"
        "    if _is_locked_out(username):\n"
        "        logger.warning(f'{username} is locked out')\n"
        "        return False\n"
        "    \n"
        "    if not authenticate_ldap(username, password):\n"
        "        _record_failure(username)\n"
        "        return False\n"
        "    \n"
        "    _failed_attempts.pop(username, None)  # Clear on success\n"
        "    logger.info(f'authenticate() called for {username}')\n"
        "    return True\n"
        "\n"
        "\n"
        "def _is_locked_out(username):\n"
        "    if username not in _failed_attempts:\n"
        "        return False\n"
        "    count, last_time = _failed_attempts[username]\n"
        "    if datetime.now() - last_time > LOCKOUT_DURATION:\n"
        "        del _failed_attempts[username]\n"
        "        return False\n"
        "    return count >= MAX_ATTEMPTS\n"
        "\n"
        "\n"
        "def _record_failure(username):\n"
        "    count, _ = _failed_attempts.get(username, (0, None))\n"
        "    _failed_attempts[username] = (count + 1, datetime.now())\n"
    ))

    # Turn 3: another grep (verbose)
    t.assistant(
        "Let me check the test file for the failing test.",
        metadata={"tool_calls": [
            {"id": "c3", "name": "grep", "arguments": {"pattern": "thread"}},
        ]},
    )
    grep2_ci = t.tool_result("c3", "grep", (
        "src/auth/login.py:11: _failed_attempts = {}  # BUG: not thread-safe\n"
        "src/server.py:5: import threading\n"
        "src/server.py:18: thread_pool = ThreadPoolExecutor(max_workers=10)\n"
        "tests/test_concurrent.py:8: import threading\n"
        "tests/test_concurrent.py:22: threads = [threading.Thread(target=auth_worker)]\n"
        "tests/test_concurrent.py:35: # This test intermittently fails due to race"
    ))

    # Turn 4: read the test file
    t.assistant(
        "Found it. Let me read the concurrent test.",
        metadata={"tool_calls": [
            {"id": "c4", "name": "read_file",
             "arguments": {"path": "tests/test_concurrent.py"}},
        ]},
    )
    read2_ci = t.tool_result("c4", "read_file", (
        "import threading\n"
        "import pytest\n"
        "from auth.login import authenticate\n"
        "\n"
        "def auth_worker(results, idx):\n"
        "    results[idx] = authenticate('admin', 'wrong_pass')\n"
        "\n"
        "def test_concurrent_auth():\n"
        "    results = [None] * 10\n"
        "    threads = [threading.Thread(target=auth_worker, args=(results, i))\n"
        "              for i in range(10)]\n"
        "    for t in threads: t.start()\n"
        "    for t in threads: t.join()\n"
        "    # This test intermittently fails due to race\n"
        "    assert all(r is False for r in results)\n"
    ))

    # Turn 5: bash to run the failing test
    t.assistant(
        "Let me run the test to confirm the race condition.",
        metadata={"tool_calls": [
            {"id": "c5", "name": "bash",
             "arguments": {"command": "python -m pytest tests/test_concurrent.py -v"}},
        ]},
    )
    bash_ci = t.tool_result("c5", "bash", (
        "===== test session starts =====\n"
        "collected 1 item\n"
        "tests/test_concurrent.py::test_concurrent_auth FAILED\n"
        "FAILED tests/test_concurrent.py::test_concurrent_auth\n"
        "AssertionError: assert all(r is False for r in results)\n"
        "1 failed in 0.12s"
    ))

    # Agent's conclusion
    t.assistant(
        "The bug is a race condition in `_failed_attempts` — it's a plain dict "
        "shared across threads without synchronization. The fix is to use "
        "`threading.Lock` to guard access to `_failed_attempts`."
    )

    return {
        "grep1": grep1_ci,
        "read1": read1_ci,
        "grep2": grep2_ci,
        "read2": read2_ci,
        "bash": bash_ci,
    }


# =============================================================================
# Part 1: Query API
# =============================================================================
# find_tool_results(), find_tool_calls(), find_tool_turns() let you
# inspect the tool history without parsing commits manually. ToolTurn
# pairs each tool-calling assistant message with its result(s).

def part1_query_api():
    print("=" * 60)
    print("Part 1: QUERY API")
    print("=" * 60)
    print()

    t = Tract.open()
    _build_agent_session(t)

    # --- find_tool_turns(): the primary audit entry point ---

    print("  find_tool_turns() — pairs tool calls with their results:\n")
    all_turns = t.find_tool_turns()
    print(f"  {len(all_turns)} tool turn(s) total\n")

    total_tokens = 0
    for i, turn in enumerate(all_turns):
        names = ", ".join(turn.tool_names)
        print(f"    Turn {i+1}: {names}")
        print(f"      results:      {len(turn.results)}")
        print(f"      total_tokens: {turn.total_tokens}")
        print(f"      all_hashes:   {len(turn.all_hashes)} commits")
        total_tokens += turn.total_tokens

    print(f"\n  Total tool tokens: {total_tokens}")

    # --- Filter by tool name ---

    print(f"\n  find_tool_turns(name='grep') — filter by tool:\n")
    grep_turns = t.find_tool_turns(name="grep")
    print(f"  {len(grep_turns)} grep turn(s)")
    for turn in grep_turns:
        print(f"    {turn.total_tokens} tokens, {len(turn.results)} result(s)")

    # --- find_tool_results(): just the results ---

    print(f"\n  find_tool_results() — all results:\n")
    results = t.find_tool_results()
    for r in results:
        name = r.metadata.get("name", "?")
        print(f"    {name:12s}  {r.token_count:4d} tokens  {r.commit_hash[:8]}")

    # --- find_tool_results(name=) for specific tools ---

    print(f"\n  find_tool_results(name='read_file'):\n")
    read_results = t.find_tool_results(name="read_file")
    for r in read_results:
        print(f"    {r.token_count} tokens  {r.commit_hash[:8]}")

    # --- find_tool_calls(): assistant-side commit with tool_calls metadata ---

    print(f"\n  find_tool_calls() — assistant messages that requested tools:\n")
    calls = t.find_tool_calls()
    for c in calls:
        tc_meta = c.metadata.get("tool_calls", [])
        names = [tc["name"] for tc in tc_meta]
        print(f"    {', '.join(names):20s}  {c.commit_hash[:8]}")

    # --- Token budget analysis ---

    ctx = t.compile()
    tool_pct = (total_tokens / ctx.token_count * 100) if ctx.token_count else 0
    print(f"\n  Token budget analysis:")
    print(f"    Total context:  {ctx.token_count} tokens")
    print(f"    Tool content:   {total_tokens} tokens ({tool_pct:.0f}%)")
    print(f"    Non-tool:       {ctx.token_count - total_tokens} tokens")

    t.close()


# =============================================================================
# Part 2: Surgical Edits
# =============================================================================
# tool_result(edit=hash) replaces a tool result in-place. The original
# is preserved in history (visible via log()). Use this to trim verbose
# results after the fact without rerunning the agent.

def part2_surgical_edits():
    print(f"\n{'=' * 60}")
    print("Part 2: SURGICAL EDITS (tool_result(edit=))")
    print("=" * 60)
    print()

    t = Tract.open()
    refs = _build_agent_session(t)

    ctx_before = t.compile()
    print(f"  BEFORE edits: {ctx_before.token_count} tokens, "
          f"{len(ctx_before.messages)} messages\n")

    # --- Trim verbose grep results ---
    # Keep only the lines that matter for the bug (thread-related)

    print("  Editing grep results to keep only relevant lines...\n")

    edited_grep1 = t.tool_result(
        "c1", "grep",
        "src/auth/login.py:15: def authenticate(username, password):\n"
        "src/auth/login.py:22:     if not authenticate_ldap(username, password):\n"
        "src/auth/session.py:45: result = authenticate(user, pw)",
        edit=refs["grep1"].commit_hash,
    )
    print(f"    grep1: {refs['grep1'].token_count} -> {edited_grep1.token_count} tokens")

    edited_grep2 = t.tool_result(
        "c3", "grep",
        "src/auth/login.py:11: _failed_attempts = {}  # BUG: not thread-safe\n"
        "tests/test_concurrent.py:35: # This test intermittently fails due to race",
        edit=refs["grep2"].commit_hash,
    )
    print(f"    grep2: {refs['grep2'].token_count} -> {edited_grep2.token_count} tokens")

    # --- Trim verbose file reads ---
    # Keep only the key function, not the full file

    print("\n  Editing read_file results to keep only key sections...\n")

    edited_read1 = t.tool_result(
        "c2", "read_file",
        "_failed_attempts = {}  # BUG: not thread-safe\n"
        "\n"
        "def authenticate(username, password):\n"
        "    # ... validates against LDAP, records failures\n"
        "    # Race condition: _failed_attempts is unprotected dict",
        edit=refs["read1"].commit_hash,
    )
    print(f"    read1: {refs['read1'].token_count} -> {edited_read1.token_count} tokens")

    # --- Token accounting ---

    ctx_after = t.compile()
    saved = ctx_before.token_count - ctx_after.token_count
    print(f"\n  AFTER edits: {ctx_after.token_count} tokens, "
          f"{len(ctx_after.messages)} messages")
    print(f"  Saved {saved} tokens ({saved/ctx_before.token_count*100:.0f}% reduction)\n")

    # --- Originals preserved in history ---

    print("  Originals are preserved — log(include_edits=True) shows both:\n")
    log = t.log(include_edits=True)
    edit_count = sum(1 for e in log if e.operation.value == "edit")
    print(f"    {len(log)} total entries, {edit_count} edits")

    t.close()


# =============================================================================
# Part 3: Selective Compression
# =============================================================================
# compress_tool_calls(name=) compresses only specific tool types,
# leaving others untouched. Useful when grep results are verbose but
# bash output is already concise.

def part3_selective_compression():
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("Part 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("Part 3: SELECTIVE COMPRESSION (compress_tool_calls(name=))")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        refs = _build_agent_session(t)

        ctx_before = t.compile()
        print(f"  BEFORE: {ctx_before.token_count} tokens\n")

        # --- Compress only grep results ---
        # bash and read_file results stay untouched

        print("  compress_tool_calls(name='grep') — only grep turns:\n")
        grep_result = t.compress_tool_calls(
            name="grep",
            instructions="One line per file: 'filename: relevant finding'",
        )
        print(f"    original_tokens:  {grep_result.original_tokens}")
        print(f"    compacted_tokens: {grep_result.compacted_tokens}")
        print(f"    turn_count:       {grep_result.turn_count}")
        print(f"    tool_names:       {grep_result.tool_names}")

        ctx_after_grep = t.compile()
        print(f"\n  After grep compression: {ctx_after_grep.token_count} tokens")

        # --- Compress read_file results too ---

        print(f"\n  compress_tool_calls(name='read_file'):\n")
        read_result = t.compress_tool_calls(
            name="read_file",
            instructions="Summarize the file's purpose and key findings in 1-2 lines.",
        )
        print(f"    original_tokens:  {read_result.original_tokens}")
        print(f"    compacted_tokens: {read_result.compacted_tokens}")
        print(f"    turn_count:       {read_result.turn_count}")

        ctx_after_both = t.compile()
        total_saved = ctx_before.token_count - ctx_after_both.token_count
        print(f"\n  AFTER all selective compressions: {ctx_after_both.token_count} tokens")
        print(f"  Total saved: {total_saved} tokens\n")

        # --- bash results were untouched ---

        print("  bash results were never compressed (already concise):")
        bash_results = t.find_tool_results(name="bash")
        for r in bash_results:
            content = t.get_content(r.commit_hash)
            preview = content[:60] if isinstance(content, str) else str(content)[:60]
            print(f"    {r.token_count} tokens: {preview}...")

        # --- Final context ---

        print(f"\n  Final context:\n")
        ctx_after_both.pprint(style="compact")


def main():
    part1_query_api()
    part2_surgical_edits()
    part3_selective_compression()


if __name__ == "__main__":
    main()
