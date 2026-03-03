"""describe_api(): human/LLM-readable markdown documentation auto-generated
from type hints and docstrings. Useful for system prompts or documentation.
"""

import sys
from pathlib import Path

from tract import Tract
from tract.hooks.gc import PendingGC
from tract.hooks.tool_result import PendingToolResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm  

MODEL_ID = llm.large


def describe_api_demo() -> None:
    print("\n" + "=" * 60)
    print("PART 3 — describe_api(): Markdown Documentation")
    print("=" * 60)
    print()
    print("  Auto-generated markdown from type hints and docstrings.")
    print("  Useful for system prompts or documentation.")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a technical writing assistant helping with documentation.")
        t.chat("How should I structure API reference docs for a Python library?", max_tokens=500)
        t.branch("temp")
        t.user("Temp Q")
        t.assistant("Temp A")
        t.switch("main")
        t.delete_branch("temp", force=True)

        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)

        # Show the pending state before generating docs
        print(f"\n  PendingGC state:")
        pending.pprint()

        api_doc: str = pending.describe_api()

        print(f"\n  PendingGC.describe_api():\n")
        print(api_doc)

        pending.approve()

    # --- PendingToolResult API ---
    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
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

        # Show the pending state before generating docs
        print(f"\n  PendingToolResult state:")
        pending_tr.pprint()

        api_doc_tr: str = pending_tr.describe_api()
        print(f"\n  PendingToolResult.describe_api():\n")
        print(api_doc_tr)

        pending_tr.approve()


if __name__ == "__main__":
    describe_api_demo()
