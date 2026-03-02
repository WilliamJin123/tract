"""Shared helpers for queries cookbook examples.

Provides: build_agent_session().

Extracted from 02_surgical_edits.py and 03_selective_compression.py
to avoid duplication.
"""

from tract import Tract


def build_agent_session(t: Tract) -> dict:
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
