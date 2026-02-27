"""Recursion guard: nested hookable operations auto-approve.

When a hook handler triggers another hookable operation, Tract detects
the re-entry and auto-approves the inner operation -- no infinite loop,
no second hook call.  This demo uses compress as the outer hook (rich
output to inspect) and triggers a tool_result inside it.
"""

import os

from dotenv import load_dotenv

from typing import Any

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.event import HookEvent
from tract.hooks.pending import Pending
from tract.hooks.tool_result import PendingToolResult
from tract.models.commit import CommitInfo
from tract.models.compression import CompressResult
from tract.protocols import CompiledContext

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def recursion_guard() -> None:
    """Compress hook triggers tool_result inside itself -- inner auto-approves."""
    print("=" * 60)
    print("Recursion Guard -- Nested Operations Auto-Approve")
    print("=" * 60)
    print()
    print("  A compress hook fires and calls t.tool_result() inside itself.")
    print("  Tract detects the re-entry and auto-approves the inner operation.")
    print("  The outer hook continues normally -- no recursion, no double-fire.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Seed a conversation worth compressing
        sys_ci = t.system("You are a DevOps assistant helping engineers with CI/CD pipelines and infrastructure.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("How do I set up GitHub Actions for a Python project with pytest and linting?")
        t.chat("What about adding Docker build and push steps to the pipeline?")
        t.chat("How should I handle secrets and environment variables in CI?")
        t.chat("What's the best strategy for running tests in parallel?")

        # Set up a tool call assistant message so tool_result has context
        t.assistant("Let me check the codebase.", metadata={
            "tool_calls": [{"id": "call_rc", "name": "lint", "arguments": {"file": "main.py"}}],
        })

        ctx_before: CompiledContext = t.compile()
        print(f"\n  Before: {len(ctx_before.messages)} messages, {ctx_before.token_count} tokens")

        inner_results: list[Any] = []

        def compress_triggers_tool_result(pending: PendingCompress) -> None:
            """Compress hook that triggers a tool_result inside itself."""
            print(f"\n  [hook] PendingCompress received:")
            pending.pprint(verbose=True)

            # --- Trigger a nested hookable operation ---
            print(f"\n  [hook] Calling t.tool_result() inside the compress hook...")
            inner: CommitInfo = t.tool_result(
                "call_rc", "lint",
                "main.py:1: C0114 Missing module docstring\nmain.py:5: W0611 Unused import os",
            )
            inner_results.append(inner)

            # The inner tool_result auto-approved (recursion guard)
            print(f"  [hook] Inner tool_result returned: {type(inner).__name__}")
            print(f"  [hook] (Recursion guard: inner op auto-approved, hook NOT re-entered)")

            # Now approve the outer compress
            print(f"\n  [hook] Approving the outer compress.")
            result: CompressResult = pending.approve()
            print(f"  [hook] Compress result: {type(result).__name__}, ratio={result.compression_ratio:.1%}")

        t.on("compress", compress_triggers_tool_result, name="compress_triggers_tool_result")

        # Also register a tool_result hook to prove it does NOT fire for the inner call
        tool_hook_fired: list[bool] = []

        def tool_result_hook(pending: PendingToolResult) -> None:
            tool_hook_fired.append(True)
            pending.approve()

        t.on("tool_result", tool_result_hook, name="tool_result_hook")

        t.print_hooks()

        # Trigger compress -- our hook fires, which triggers tool_result inside
        print("\n  Triggering compress...")
        result: CompressResult = t.compress(target_tokens=200)
        assert isinstance(result, CompressResult), f"Expected CompressResult, got {type(result).__name__}"

        # The tool_result hook should NOT have fired (recursion guard bypassed it)
        print(f"\n  tool_result_hook fired? {bool(tool_hook_fired)} (should be False -- recursion guard)")

        ctx_after: CompiledContext = t.compile()
        print(f"  After: {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")

        # hook_log shows both entries: outer compress handled by hook,
        # inner tool_result auto-approved by recursion guard
        print("\n  Hook log (showing recursion guard):")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation:15s} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    recursion_guard()
