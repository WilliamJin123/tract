"""Surgical Edits

tool_result(edit=hash) replaces a tool result in-place. The original
is preserved in history (visible via log()). Use this to trim verbose
results after the fact without rerunning the agent.

No LLM required — all edit operations work offline.

Demonstrates: tool_result(edit=) for surgical replacement,
              token accounting before/after edits,
              log(include_edits=True), originals preserved in history
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

# Allow importing _helpers from the same directory when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import build_agent_session  

MODEL_ID = llm.large


def part2_surgical_edits():
    print(f"\n{'=' * 60}")
    print("PART 2 -- Manual: SURGICAL EDITS (tool_result(edit=))")
    print("=" * 60)
    print()

    t = Tract.open()
    refs = build_agent_session(t)

    ctx_before = t.compile()
    ctx_before.pprint(style="compact")
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
    ctx_after.pprint(style="compact")
    saved = ctx_before.token_count - ctx_after.token_count
    print(f"\n  AFTER edits: {ctx_after.token_count} tokens, "
          f"{len(ctx_after.messages)} messages")
    print(f"  Saved {saved} tokens ({saved/ctx_before.token_count*100:.0f}% reduction)\n")

    # --- Originals preserved in history ---

    log = t.log()
    edit_count = sum(1 for e in log if e.operation.value == "edit")
    print(f"    {len(log)} total entries, {edit_count} edits")

    t.close()


def main():
    part2_surgical_edits()


if __name__ == "__main__":
    main()
