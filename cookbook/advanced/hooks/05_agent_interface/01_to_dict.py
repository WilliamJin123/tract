"""to_dict(): serialize any Pending to a structured dict for LLM consumption.
Shows the JSON output with operation, status, fields, and available_actions.
Works identically on every Pending subclass.
"""

import json
import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.gc import PendingGC
from tract.hooks.tool_result import PendingToolResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def to_dict_demo() -> None:
    print("=" * 60)
    print("PART 1 — to_dict(): Structured Serialization")
    print("=" * 60)
    print()
    print("  Every Pending subclass serializes to a consistent dict format.")
    print("  Send this to an LLM to let it inspect state and pick an action.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # --- PendingGC ---
        # GC removes orphaned commits. Create orphans by branching then deleting.
        t.system("You are a data analysis assistant helping with pandas and visualization.")
        t.chat("How do I merge two DataFrames on a composite key in pandas?", max_tokens=500)

        t.branch("temp")
        for i in range(3):
            t.user(f"Temp Q{i}")
            t.assistant(f"Temp A{i}")
        t.switch("main")
        t.delete_branch("temp", force=True)

        pending_gc: PendingGC = t.gc(orphan_retention_days=0, review=True)

        # Show the pending state via pprint before serializing
        print("\n  PendingGC state:")
        pending_gc.pprint()

        # The lesson: to_dict() output is what you send to an LLM
        gc_dict: dict = pending_gc.to_dict()
        print(f"\n  PendingGC.to_dict():")
        print(json.dumps(gc_dict, indent=4))

        print(f"\n  Key structure:")
        print(f"    operation:         {gc_dict['operation']}")
        print(f"    pending_id:        {gc_dict['pending_id'][:12]}...")
        print(f"    status:            {gc_dict['status']}")
        print(f"    fields:            {list(gc_dict['fields'].keys())}")
        print(f"    available_actions: {gc_dict['available_actions']}")

        pending_gc.approve()  # Clean up

    # --- PendingToolResult ---
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a code analysis assistant.")
        t.assistant("Calling tool.", metadata={
            "tool_calls": [{"id": "x1", "name": "grep",
                            "arguments": {"pattern": "TODO"}}],
        })

        pending_tr: PendingToolResult = t.tool_result(
            "x1", "grep", "main.py:10: # TODO fix this", review=True,
        )
        tr_dict: dict = pending_tr.to_dict()
        print(f"\n  PendingToolResult.to_dict():")
        print(json.dumps(tr_dict, indent=4))

        print(f"\n  Same structure, different fields:")
        print(f"    operation:         {tr_dict['operation']}")
        print(f"    fields:            {list(tr_dict['fields'].keys())}")
        print(f"    available_actions: {tr_dict['available_actions']}")

        pending_tr.approve()


if __name__ == "__main__":
    to_dict_demo()
