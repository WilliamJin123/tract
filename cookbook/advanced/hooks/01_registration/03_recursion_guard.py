"""Recursion guard: nested hookable operations auto-approve to prevent loops.

When a hook handler triggers another hookable operation, Tract detects the
re-entry and auto-approves the inner operation without calling the hook again.
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.gc import PendingGC
from tract.models.compression import GCResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def recursion_guard():
    """Nested hookable operations inside a hook auto-approve (no infinite loop)."""
    print("=" * 60)
    print("PART 3 — Recursion Guard")
    print("=" * 60)
    print()
    print("  If a hook handler triggers another hookable operation,")
    print("  Tract auto-approves the inner one (no recursion).")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Pre-create two sets of orphans so both GC calls find work
        t.system("You are a DevOps assistant helping engineers with CI/CD pipelines and infrastructure.")
        t.chat("Help me write a Dockerfile for a FastAPI app with multi-stage builds.")

        # Orphan set 1 (for outer gc)
        t.branch("orphan1")
        t.user("Throwaway experiment orphan1: question 0")
        t.assistant("Response to orphan1 experiment 0")
        t.switch("main")
        t.delete_branch("orphan1", force=True)

        # Orphan set 2 (for inner gc to find during recursion)
        t.branch("orphan2")
        t.user("Throwaway experiment orphan2: question 0")
        t.assistant("Response to orphan2 experiment 0")
        t.switch("main")
        t.delete_branch("orphan2", force=True)

        def gc_then_gc(pending: PendingGC):
            """GC hook that tries to trigger ANOTHER gc inside itself."""
            pending.pprint()

            # Try to gc again inside the hook -- the inner gc will
            # auto-approve (recursion guard), our hook won't re-enter.
            inner = t.gc(orphan_retention_days=0)
            print(f"    [inner] gc returned: {type(inner).__name__}")
            if isinstance(inner, GCResult):
                print(f"    (Recursion guard: inner gc auto-approved, hook NOT re-entered)")

            pending.approve()

        t.on("gc", gc_then_gc, name="gc_then_gc")
        t.gc(orphan_retention_days=0)

        # hook_log shows the "skipped" entry from the recursion guard
        print("\n  Hook log (showing recursion guard 'skipped' entry):")
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    recursion_guard()
