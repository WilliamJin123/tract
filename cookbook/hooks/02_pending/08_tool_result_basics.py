"""Tool result hook basics: t.on("tool_result", handler) intercepts every
tool_result() call. PendingToolResult fields (tool_name, content,
token_count), approve/reject flow, and review=True for manual inspection.
"""

import os
import sys
from pathlib import Path

from tract import Tract
from tract.hooks.event import HookEvent
from tract.hooks.tool_result import PendingToolResult
from tract.models.commit import CommitInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm  

MODEL_ID = llm.large


def hook_basics() -> None:
    print("=" * 60)
    print("PART 1 -- Tool Result Hook Basics")
    print("=" * 60)
    print()
    print("  Every tool_result() call passes through the hook system.")
    print("  A handler can inspect, modify, or reject before commit.")

    # --- Without hooks (baseline) ---
    # Without t.on("tool_result", ...), tool_result() auto-commits
    # immediately (tier 3). With a handler, you intercept every result.
    # With review=True, the caller gets a PendingToolResult (tier 1).

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a code analysis assistant with access to development tools.")
        t.user("Find all Python files in the project.")

        tool_log: list[dict[str, object]] = []

        def log_tool_results(pending: PendingToolResult) -> None:
            """Log every tool result, then approve."""
            tool_log.append({
                "tool": pending.tool_name,
                "chars": len(pending.content),
                "tokens": pending.token_count,
                "is_error": pending.is_error,
            })
            pending.approve()

        t.on("tool_result", log_tool_results, name="logger")

        # Real tool results from the local filesystem
        _dir = os.path.dirname(os.path.abspath(__file__))
        py_files = "\n".join(
            f for f in sorted(os.listdir(_dir)) if f.endswith(".py")
        )
        t.assistant("Let me search for Python files.", metadata={
            "tool_calls": [{"id": "tc1", "name": "find_files",
                            "arguments": {"pattern": "*.py"}}],
        })
        t.tool_result("tc1", "find_files", py_files)

        # Read the first 8 lines of this script as a realistic file snippet
        with open(os.path.abspath(__file__)) as _f:
            snippet = "\n".join(_f.read().splitlines()[:8])
        t.assistant("Let me also check a file.", metadata={
            "tool_calls": [{"id": "tc2", "name": "read_file",
                            "arguments": {"path": os.path.basename(__file__)}}],
        })
        t.tool_result("tc2", "read_file", snippet)

        print(f"\n  Context after tool results committed:")
        t.compile().pprint(style="compact")

        print(f"\n  Tool log: {len(tool_log)} results intercepted")
        for entry in tool_log:
            print(f"    {entry}")

        # Show hook activity via hook_log
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")

        t.print_hooks()

    # --- review=True: manual inspection ---
    print(f"\n  review=True returns PendingToolResult for manual inspection:")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a code analysis assistant with access to development tools.")
        t.assistant("Reading file.", metadata={
            "tool_calls": [{"id": "tc3", "name": "read_file",
                            "arguments": {"path": "secret.env"}}],
        })

        pending: PendingToolResult = t.tool_result(
            "tc3", "read_file",
            "API_KEY=sk-secret-12345\nDB_PASSWORD=hunter2",
            review=True,
        )

        # pprint shows all fields, status, and available actions
        pending.pprint()

        # Reject sensitive content
        pending.reject("Contains secrets -- cannot enter context window")
        print(f"\n    After reject():")
        pending.pprint()

        print(f"\n    Context (rejected tool result not included):")
        t.compile().pprint(style="chat")


if __name__ == "__main__":
    hook_basics()
