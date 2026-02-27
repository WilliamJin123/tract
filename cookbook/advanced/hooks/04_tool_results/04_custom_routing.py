"""Custom routing: a handler that routes different tools through different
strategies (pass-through, edit, summarize, reject) based on tool name.
"""

import os
import re

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.event import HookEvent
from tract.hooks.tool_result import PendingToolResult
from tract.models.commit import CommitInfo
from tract.protocols import CompiledContext

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def custom_routing() -> None:
    print("\n" + "=" * 60)
    print("PART 4 -- Custom Routing Handler")
    print("=" * 60)
    print()
    print("  A single handler that routes tools through different strategies:")
    print("  pass-through, edit (redact), summarize, or reject.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a security-conscious assistant.")
        t.user("Audit the system configuration.")

        BLOCKED_TOOLS: set[str] = {"run_shell", "delete_file"}
        REDACT_TOOLS: set[str] = {"read_env", "get_secrets"}
        SUMMARIZE_TOOLS: set[str] = {"read_file", "search_files", "list_directory"}

        def smart_router(pending: PendingToolResult) -> None:
            """Route each tool through the appropriate strategy."""
            tool = pending.tool_name

            if tool in BLOCKED_TOOLS:
                pending.reject(f"Tool '{tool}' is blocked by security policy")

            elif tool in REDACT_TOOLS:
                redacted = re.sub(
                    r'(password|secret|key|token)\s*[=:]\s*\S+',
                    r'\1=***REDACTED***',
                    pending.content,
                    flags=re.IGNORECASE,
                )
                pending.edit_result(redacted)
                pending.approve()

            elif tool in SUMMARIZE_TOOLS and pending.token_count > 30:
                pending.summarize(instructions="Keep only security-relevant entries.")
                pending.approve()

            else:
                pending.approve()

        t.on("tool_result", smart_router, name="smart_router")

        # --- Trigger each strategy ---

        # 1. Blocked tool
        t.assistant("Running a shell command.", metadata={
            "tool_calls": [{"id": "r1", "name": "run_shell",
                            "arguments": {"cmd": "rm -rf /"}}],
        })
        result1: CommitInfo | PendingToolResult = t.tool_result("r1", "run_shell", "Command output here")
        if isinstance(result1, PendingToolResult):
            print("  Blocked tool returned PendingToolResult:")
            result1.pprint()
        else:
            print(f"  Result: {type(result1).__name__} (committed)")

        # 2. Redact tool
        t.assistant("Reading environment.", metadata={
            "tool_calls": [{"id": "r2", "name": "read_env",
                            "arguments": {}}],
        })
        t.tool_result("r2", "read_env", "APP=myapp\nSECRET_KEY=abc123\nDB_PASSWORD=hunter2")

        # 3. Summarize tool (large output)
        t.assistant("Listing files.", metadata={
            "tool_calls": [{"id": "r3", "name": "list_directory",
                            "arguments": {"path": "/"}}],
        })
        big_listing = "\n".join([f"file_{i:03d}.py" for i in range(40)])
        t.tool_result("r3", "list_directory", big_listing)

        # 4. Pass-through (small, unknown tool)
        t.assistant("Checking version.", metadata={
            "tool_calls": [{"id": "r4", "name": "get_version",
                            "arguments": {}}],
        })
        t.tool_result("r4", "get_version", "v3.1.0")

        # hook_log shows the routing decisions
        print(f"\n  Hook log ({len(t.hook_log)} events):")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")

        print(f"\n  Final context:")
        ctx: CompiledContext = t.compile()
        print(f"    {len(ctx.messages)} messages, {ctx.token_count} tokens")

        t.print_hooks()


if __name__ == "__main__":
    custom_routing()
