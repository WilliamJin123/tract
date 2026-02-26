"""Query API

Find, filter, and inspect tool-related commits across a conversation.
Use find_tool_turns(), find_tool_results(), and find_tool_calls() to
answer questions like "which tools were called?" and "how many tokens
did grep results consume?"

No LLM required — all query operations work offline.

Demonstrates: find_tool_results(name=, after=), find_tool_calls(name=),
              find_tool_turns(name=), ToolTurn (all_hashes, result_hashes,
              total_tokens, tool_names), token budget analysis
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


def _build_agent_session(t: Tract):
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


if __name__ == "__main__":
    part1_query_api()
