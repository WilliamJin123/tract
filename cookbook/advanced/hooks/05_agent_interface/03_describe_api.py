"""describe_api(): human/LLM-readable markdown documentation auto-generated
from type hints and docstrings. Useful for system prompts or documentation.
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.gc import PendingGC
from tract.hooks.tool_result import PendingToolResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def describe_api_demo():
    print("\n" + "=" * 60)
    print("PART 3 — describe_api(): Markdown Documentation")
    print("=" * 60)
    print()
    print("  Auto-generated markdown from type hints and docstrings.")
    print("  Useful for system prompts or documentation.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a technical writing assistant helping with documentation.")
        t.chat("How should I structure API reference docs for a Python library?")
        t.branch("temp")
        t.user("Temp Q")
        t.assistant("Temp A")
        t.switch("main")
        t.delete_branch("temp", force=True)

        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)
        api_doc = pending.describe_api()

        print(f"\n  PendingGC.describe_api():\n")
        print(api_doc)

        pending.approve()

    # --- PendingToolResult API ---
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a technical writing assistant helping with documentation.")
        t.assistant("Calling.", metadata={
            "tool_calls": [{"id": "y1", "name": "test",
                            "arguments": {}}],
        })

        pending_tr: PendingToolResult = t.tool_result(
            "y1", "test", "test output", review=True,
        )
        api_doc_tr = pending_tr.describe_api()
        print(f"\n  PendingToolResult.describe_api():\n")
        print(api_doc_tr)

        pending_tr.approve()


if __name__ == "__main__":
    describe_api_demo()
