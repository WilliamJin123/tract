"""Surgical Edits

tool_result(edit=hash) replaces a tool result in-place. The original
is preserved in history (visible via log()). Use this to trim verbose
results after the fact without rerunning the agent.

No LLM required — all edit operations work offline.

Demonstrates: tool_result(edit=) for surgical replacement,
              token accounting before/after edits,
              log(include_edits=True), originals preserved in history
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


if __name__ == "__main__":
    part2_surgical_edits()
