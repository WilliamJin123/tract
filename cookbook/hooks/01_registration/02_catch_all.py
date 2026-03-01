"""Catch-all "*" handler catching multiple operation types.

The wildcard "*" registration intercepts every hookable operation --
tool_result, compress, gc, anything.  A single handler can build a
universal audit logger.  When a specific handler is also registered,
it takes priority for its operation while "*" handles the rest.
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.event import HookEvent
from tract.hooks.pending import Pending
from tract.models.compression import CompressResult, GCResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def catch_all() -> None:
    """A single '*' handler intercepts tool_result, compress, and gc."""
    print("=" * 60)
    print("Catch-All '*' Handler -- Multiple Operation Types")
    print("=" * 60)
    print()
    print("  t.on('*', handler) catches every hookable operation.")
    print("  We trigger tool_result, compress, and gc -- all three hit the same handler.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        audit_log: list[dict[str, str]] = []

        def universal_logger(pending: Pending) -> None:
            """Log the operation type and auto-approve everything."""
            entry: dict[str, str] = {
                "operation": pending.operation,
                "type": type(pending).__name__,
                "id": pending.pending_id[:8],
            }
            audit_log.append(entry)
            print(f"  [*] Caught: {pending.operation} ({type(pending).__name__})")
            pending.approve()

        t.on("*", universal_logger, name="universal_logger")

        # -- 1. Trigger tool_result -----------------------------------
        print("\n--- Step 1: tool_result ---")
        sys_ci = t.system("You are a code assistant with access to development tools.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.assistant("I'll search the codebase for that pattern.", metadata={
            "tool_calls": [{"id": "call_w1", "name": "grep", "arguments": {"pattern": "FIXME"}}],
        })
        t.tool_result(
            "call_w1", "grep",
            "src/app.py:88: # FIXME handle timeout\nlib/cache.py:12: # FIXME eviction policy",
        )

        # -- 2. Trigger compress --------------------------------------
        print("\n--- Step 2: compress ---")

        # Seed enough conversation for compress to work with
        t.chat("How do I set up GitHub Actions for a Python project with pytest and linting?", max_tokens=500)
        t.chat("What about adding Docker build and push steps to the pipeline?", max_tokens=500)
        t.chat("How should I handle secrets and environment variables in CI?", max_tokens=500)
        t.chat("What's the best strategy for running tests in parallel?", max_tokens=500)

        t.compress(target_tokens=200)

        # -- 3. Trigger gc --------------------------------------------
        print("\n--- Step 3: gc ---")

        # Create orphans: branch, add commits, delete branch
        t.branch("catch-all-temp")
        t.user("Experiment on catch-all-temp: throwaway question")
        t.assistant("Throwaway response for catch-all-temp experiment")
        t.switch("main")
        t.delete_branch("catch-all-temp", force=True)

        t.gc(orphan_retention_days=0)

        # -- Show the audit log has all three operations ---------------
        print("\n--- Audit log ---")
        for i, entry in enumerate(audit_log):
            print(f"  [{i}] {entry['operation']:15s}  {entry['type']}")

        operations_seen: set[str] = {e["operation"] for e in audit_log}
        print(f"\n  Operations caught: {sorted(operations_seen)}")
        assert "tool_result" in operations_seen, "tool_result should be in audit log"
        assert "compress" in operations_seen, "compress should be in audit log"
        assert "gc" in operations_seen, "gc should be in audit log"

        # -- Specific handler takes priority over "*" -----------------
        print()
        print("=" * 60)
        print("Specific handler overrides '*' for compress")
        print("=" * 60)
        print()
        print("  Register t.on('compress', specific) -- it takes priority.")
        print("  '*' still handles gc and tool_result.")

        specific_log: list[str] = []

        def specific_compress_hook(pending: PendingCompress) -> None:
            """Specific handler for compress only."""
            specific_log.append(pending.operation)
            print(f"  [compress-specific] Handling compress (not '*')")
            pending.approve()

        t.on("compress", specific_compress_hook, name="specific_compress")

        print(f"\n  hook_names: {t.hook_names}")

        # Reset audit_log to track only new events
        audit_log.clear()

        # Trigger another tool_result -- still hits "*"
        print("\n--- tool_result (should hit '*') ---")
        t.assistant("Let me check another file.", metadata={
            "tool_calls": [{"id": "call_w2", "name": "read_file", "arguments": {"path": "README.md"}}],
        })
        t.tool_result("call_w2", "read_file", "# My Project\nA sample readme file.")

        # Trigger another compress -- should hit specific handler, NOT "*"
        print("\n--- compress (should hit specific handler) ---")
        t.chat("How do I configure Dependabot for automatic dependency updates?", max_tokens=500)
        t.chat("What about setting up CodeQL for security scanning?", max_tokens=500)
        t.compress(target_tokens=200)

        # Trigger another gc -- still hits "*"
        print("\n--- gc (should hit '*') ---")
        t.branch("catch-all-temp2")
        t.user("Another throwaway question for priority test")
        t.assistant("Another throwaway response for priority test")
        t.switch("main")
        t.delete_branch("catch-all-temp2", force=True)
        t.gc(orphan_retention_days=0)

        print(f"\n  '*' audit log: {[e['operation'] for e in audit_log]}")
        print(f"  specific log:  {specific_log}")
        print(f"  ('*' handled tool_result + gc, specific handler handled compress)")

        t.print_hooks()

        # Show full hook_log
        print("\n--- Full hook log ---")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {evt.operation:15s} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    catch_all()
