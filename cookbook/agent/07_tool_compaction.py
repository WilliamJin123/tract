"""Tool Compaction in Agent Loops

Agent loops with heavy tool use accumulate verbose tool results that
eat the context window. This example shows three compaction approaches:

  1. Manual edit: trim a tool result with tool_result(edit=hash)
  2. Drop failures: skip error tool turns with tools.drop_failed_turns()
  3. Batch compaction: compress_tool_calls() summarizes all results

Then demonstrates a post_tool_execute middleware that auto-triggers
compaction when tool results exceed a token threshold.

Demonstrates: compress_tool_calls(), tools.find_turns(),
              tools.drop_failed_turns(), tool_result(edit=),
              post_tool_execute middleware for auto-compaction

Requires: LLM API key for compress_tool_calls() (uses Cerebras)
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, TractConfig, TokenBudgetConfig, MiddlewareContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


def main() -> None:

    # =================================================================
    # 1. Manual edit: trim a verbose tool result in-place
    # =================================================================
    # No LLM needed. The edit= parameter creates an EDIT commit that
    # replaces the original in compiled context.

    print("=== 1. Manual Tool Result Edit ===\n")

    with Tract.open() as t:
        t.system("You are a code search agent.")
        t.user("Find all Python files.")

        t.assistant("Searching...", metadata={"tool_calls": [
            {"id": "c1", "name": "find_files", "arguments": {"pattern": "*.py"}},
        ]})

        # Verbose result (imagine 500 lines of file listing)
        verbose_result = (
            "src/main.py\nsrc/utils.py\nsrc/config.py\nsrc/models.py\n"
            "src/routes.py\nsrc/auth.py\nsrc/cache.py\nsrc/logging.py\n"
            "tests/test_main.py\ntests/test_utils.py\ntests/conftest.py"
        )
        orig = t.tool_result("c1", "find_files", verbose_result)

        before = t.compile().token_count
        print(f"  Original tool result ({orig.token_count} tokens):")
        print(f"    {verbose_result[:70]}...")

        # Edit: keep only the summary
        summary = "Found 11 Python files (8 in src/, 3 in tests/)"
        edited = t.tool_result("c1", "find_files", summary,
            edit=orig.commit_hash)

        after = t.compile().token_count
        print(f"  Edited to:")
        print(f'    "{summary}"')
        print(f"  Tokens: {before} -> {after} ({before - after} saved)")

        # Show what compilation looks like after the edit
        print(f"\n  Compiled context:")
        t.compile().pprint(style="compact")

    # =================================================================
    # 2. Drop failed tool turns
    # =================================================================
    # Error tool results waste context. tools.drop_failed_turns()
    # annotates them as SKIP so they fall out of compilation.

    print("\n=== 2. Drop Failed Tool Turns ===\n")

    with Tract.open() as t:
        t.system("You are a deployment agent.")
        t.user("Deploy to production.")

        # Failed attempt
        t.assistant("Deploying...", metadata={"tool_calls": [
            {"id": "c1", "name": "deploy", "arguments": {"env": "prod"}},
        ]})
        t.tool_result("c1", "deploy", "Error: Connection refused", is_error=True)
        print('  deploy(env=prod) -> "Error: Connection refused" [FAIL]')

        # Another failure
        t.assistant("Retrying...", metadata={"tool_calls": [
            {"id": "c2", "name": "deploy", "arguments": {"env": "prod"}},
        ]})
        t.tool_result("c2", "deploy", "Error: Timeout after 30s", is_error=True)
        print('  deploy(env=prod) -> "Error: Timeout after 30s" [FAIL]')

        # Success
        t.assistant("Third attempt...", metadata={"tool_calls": [
            {"id": "c3", "name": "deploy", "arguments": {"env": "prod"}},
        ]})
        t.tool_result("c3", "deploy", "Deployed build #1847 to prod.")
        t.assistant("Deployment complete.")
        print('  deploy(env=prod) -> "Deployed build #1847 to prod." [OK]')

        before = t.compile().token_count
        print(f"\n  Before dropping failed turns: {before} tokens")
        print(f"  Context includes {len(t.tools.find_turns())} tool turns (2 failed, 1 success)")

        drop = t.tools.drop_failed_turns()
        after = t.compile().token_count

        print(f"  After drop: {after} tokens ({before - after} saved)")
        print(f"  Dropped {drop.turns_dropped} failed turn(s) ({drop.commits_skipped} commits)")

        print(f"\n  Compiled context (failed turns removed):")
        t.compile().pprint(style="compact")

    # =================================================================
    # 3. Batch compaction with compress_tool_calls()
    # =================================================================
    # Uses an LLM to summarize all tool results at once while
    # preserving the tool turn structure (EDIT commits, not collapse).

    print("\n=== 3. Batch Tool Compaction (LLM) ===\n")

    if not llm.api_key:
        print("  SKIPPED (no API key -- set CEREBRAS_API_KEY)")
    else:
        with Tract.open(
            api_key=llm.api_key,
            base_url=llm.base_url,
            model=MODEL_ID,
        ) as t:
            t.system("You are a code review agent.")
            t.user("Review the authentication module.")

            # Simulate multiple tool calls with verbose results
            files = {
                "auth.py": (
                    "import jwt\nimport bcrypt\nfrom datetime import datetime, timedelta\n\n"
                    "class AuthService:\n    def __init__(self, secret_key, algorithm='HS256'):\n"
                    "        self.secret_key = secret_key\n        self.algorithm = algorithm\n\n"
                    "    def create_token(self, user_id, expires_in=3600):\n"
                    "        payload = {'user_id': user_id, 'exp': datetime.utcnow() + "
                    "timedelta(seconds=expires_in)}\n"
                    "        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)\n\n"
                    "    def verify_token(self, token):\n        try:\n"
                    "            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])\n"
                    "        except jwt.ExpiredSignatureError:\n            return None\n\n"
                    "    def hash_password(self, password):\n"
                    "        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())\n\n"
                    "    def check_password(self, password, hashed):\n"
                    "        return bcrypt.checkpw(password.encode(), hashed)\n"
                ),
                "middleware.py": (
                    "from functools import wraps\nfrom flask import request, jsonify, g\n\n"
                    "def require_auth(f):\n    @wraps(f)\n"
                    "    def decorated(*args, **kwargs):\n"
                    "        token = request.headers.get('Authorization', '').replace('Bearer ', '')\n"
                    "        if not token:\n            return jsonify({'error': 'Missing token'}), 401\n"
                    "        payload = g.auth_service.verify_token(token)\n"
                    "        if not payload:\n            return jsonify({'error': 'Invalid token'}), 401\n"
                    "        g.current_user = payload['user_id']\n"
                    "        return f(*args, **kwargs)\n    return decorated\n"
                ),
                "test_auth.py": (
                    "import pytest\nfrom auth import AuthService\n\n"
                    "def test_create_and_verify():\n    svc = AuthService('secret')\n"
                    "    token = svc.create_token(42)\n    payload = svc.verify_token(token)\n"
                    "    assert payload['user_id'] == 42\n\n"
                    "def test_expired_token():\n    svc = AuthService('secret')\n"
                    "    token = svc.create_token(42, expires_in=-1)\n"
                    "    assert svc.verify_token(token) is None\n\n"
                    "def test_password_hash():\n    svc = AuthService('secret')\n"
                    "    hashed = svc.hash_password('p@ssw0rd')\n"
                    "    assert svc.check_password('p@ssw0rd', hashed)\n"
                    "    assert not svc.check_password('wrong', hashed)\n"
                ),
            }

            for fname, content in files.items():
                t.assistant(f"Reading {fname}...", metadata={"tool_calls": [
                    {"id": f"read_{fname}", "name": "read_file",
                     "arguments": {"path": fname}},
                ]})
                t.tool_result(f"read_{fname}", "read_file", content)
                lines = content.count("\n") + 1
                print(f"  read_file({fname}) -> {lines} lines of code")

            before = t.compile().token_count
            turns_before = len(t.tools.find_turns())
            print(f"\n  Before compaction: {turns_before} tool turns, {before} tokens")

            # Compact all tool results at once
            print("  Calling compress_tool_calls()...")
            result = t.compression.compress_tool_calls(
                instructions="Summarize each file's purpose, key classes/functions, "
                             "and any security concerns. Keep it concise.",
            )

            after = t.compile().token_count
            print(f"  After compaction:  {after} tokens ({result.compression_ratio:.1%} ratio)")
            print(f"  EDIT commits created: {len(result.edit_commits)}")

            # Show what the compacted context looks like
            print(f"\n  Compiled context after compaction:")
            t.compile().pprint(style="compact")

    # =================================================================
    # 4. Auto-compact middleware pattern
    # =================================================================
    # A post_tool_execute handler that tracks tool result tokens and
    # triggers compaction when they exceed a budget.

    print("\n=== 4. Auto-Compact Middleware Pattern ===\n")

    with Tract.open() as t:

        compact_state = {"result_tokens": 0, "compactions": 0}

        def auto_compact_tools(ctx: MiddlewareContext):
            """Track tool result tokens, trigger compaction at threshold."""
            if not ctx.commit or "tool_result" not in (ctx.commit.tags or []):
                return
            compact_state["result_tokens"] += ctx.commit.token_count

            threshold = 300  # tokens
            if compact_state["result_tokens"] > threshold:
                # In production with configured LLM:
                # ctx.tract.compression.compress_tool_calls(
                #     instructions="Keep key findings only.",
                # )
                compact_state["compactions"] += 1
                compact_state["result_tokens"] = 0  # reset after compaction
                print(f"    >> Auto-compact triggered ({compact_state['compactions']}x)")

        t.middleware.add("post_commit", auto_compact_tools)

        t.system("You are a search agent.")
        # Simulate tool-heavy loop
        for i in range(8):
            t.assistant(f"Searching {i}...", metadata={"tool_calls": [
                {"id": f"s{i}", "name": "search", "arguments": {"q": f"topic {i}"}},
            ]})
            result_text = (f"Found {i*3+1} results for topic {i}. "
                f"Key finding: {'detailed analysis ' * 5}conclusion #{i}.")
            t.tool_result(f"s{i}", "search", result_text)
            print(f"  search(topic {i}) -> {result_text[:55]}...")

        print(f"\n  Compaction triggers: {compact_state['compactions']}")
        print(f"  (In production, compress_tool_calls() runs at each trigger)")

    # =================================================================
    # Putting it together
    # =================================================================
    # In a real agent loop, combine these patterns:
    #
    #   t.middleware.add("post_commit", auto_compact_tools)   # background compaction
    #   result = t.llm.run("Do research...",            # agent loop
    #       max_steps=20,
    #       tool_profile="self",
    #   )
    #   t.tools.drop_failed_turns()                  # clean up errors
    #   t.compression.compress_tool_calls()                     # final compaction
    #
    # The middleware handles compaction DURING the loop (zero agent
    # steps wasted). The post-loop calls handle final cleanup.

    print("\n=== Summary ===\n")
    print("  Manual edit:           tool_result(edit=hash) -- no LLM")
    print("  Drop failures:         tools.drop_failed_turns() -- no LLM")
    print("  Batch compaction:      compress_tool_calls() -- LLM")
    print("  Auto-compact:          post_commit middleware + compress_tool_calls()")
    print()
    print("  Key insight: middleware compacts IN the loop without burning")
    print("  agent steps. The agent never knows it happened.")


if __name__ == "__main__":
    main()


# --- See also ---
# Tool tracking reference:  reference/06_tool_tracking.py
# Tangent isolation:        agent/01_implicit_discovery.py
# Autonomous behaviors:     config_and_middleware/06_autonomous_behaviors.py
# Budget management:        optimization/01_budget_management.py
