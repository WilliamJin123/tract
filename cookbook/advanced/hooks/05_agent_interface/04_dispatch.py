"""apply_decision() and execute_tool(): dispatch an LLM's structured
decision to the correct method. Whitelist guards prevent calling private
methods. Full end-to-end: Pending -> to_tools() -> LLM picks action ->
apply_decision() executes it.
"""

import json
import os

from dotenv import load_dotenv

from typing import Any

from tract import Tract
from tract.hooks.gc import PendingGC
from tract.hooks.tool_result import PendingToolResult
from tract.models.compression import GCResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def dispatch_demo() -> None:
    print("\n" + "=" * 60)
    print("PART 4 — apply_decision() / execute_tool()")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a project management assistant helping track sprints and deliverables.")
        t.chat("We have 5 stories in this sprint. How should we prioritize when two are blocked?")

        t.branch("work")
        for i in range(3):
            t.user(f"Work Q{i}")
            t.assistant(f"Work A{i}")
        t.switch("main")
        t.delete_branch("work", force=True)

        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)

        # Show starting state
        pending.pprint()

        # --- execute_tool(): call by name ---
        print(f"\n  execute_tool('exclude', {{'commit_hash': ...}}):")
        first_hash: str = pending.commits_to_remove[0]
        pending.execute_tool("exclude", {"commit_hash": first_hash})
        print(f"    Excluded {first_hash[:12]}")
        print(f"    Remaining: {len(pending.commits_to_remove)} commits")

        # --- apply_decision(): structured LLM output ---
        print(f"\n  apply_decision() — simulating LLM response:")

        # Simulate what an LLM would return after seeing to_tools()
        llm_decision: dict[str, Any] = {
            "action": "approve",
            "args": {},
        }
        print(f"    LLM decision: {json.dumps(llm_decision)}")

        result: GCResult = pending.apply_decision(llm_decision)
        print(f"    Dispatched! status={pending.status}")

    # --- Whitelist security ---
    print(f"\n  Whitelist security:")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a project management assistant helping track sprints and deliverables.")
        t.assistant("Tool.", metadata={
            "tool_calls": [{"id": "z1", "name": "test",
                            "arguments": {}}],
        })
        pending2: PendingToolResult = t.tool_result(
            "z1", "test", "output", review=True,
        )

        # Try to call a private method — blocked
        try:
            pending2.execute_tool("_execute_fn")
        except ValueError as e:
            print(f"    execute_tool('_execute_fn') -> ValueError: {e}")

        # Try to call a non-whitelisted method — blocked
        try:
            pending2.execute_tool("pprint")
        except ValueError as e:
            print(f"    execute_tool('pprint') -> ValueError: {e}")

        # Only whitelisted actions work
        print(f"\n    Allowed actions: {sorted(pending2._public_actions)}")
        pending2.approve()

    # --- Full pipeline summary ---
    print(f"\n  Full agent pipeline:")
    print(f"    1. Operation returns Pending (review=True or via hook)")
    print(f"    2. Send to_dict() as context + to_tools() as tools to LLM")
    print(f"    3. LLM returns {{action, args}} in tool_call response")
    print(f"    4. pending.apply_decision(llm_response) dispatches safely")
    print(f"    5. Whitelist blocks private/unauthorized methods")


if __name__ == "__main__":
    dispatch_demo()
