"""Catch-all "*" handler that intercepts every hookable operation.

Shows how a single registration can build a universal audit logger, and how
a specific handler takes priority over the catch-all when both are registered.
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.gc import PendingGC

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def catch_all():
    """A single '*' handler intercepts every hookable operation."""
    print("=" * 60)
    print("PART 2 — Catch-All '*' Handler")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        audit_log = []

        def universal_logger(pending):
            """Log every pending operation, then auto-approve."""
            entry = f"{pending.operation} (id={pending.pending_id[:8]})"
            audit_log.append(entry)
            pending.approve()

        t.on("*", universal_logger, name="universal_logger")

        # Seed content
        t.system("You are a DevOps assistant helping engineers with CI/CD pipelines and infrastructure.")
        t.chat("What's the best way to monitor container health in a Kubernetes cluster?")

        # Create orphans, then GC -- fires through "*" (no specific handler)
        t.branch("star-temp")
        t.user("Throwaway experiment star-temp: question 0")
        t.assistant("Response to star-temp experiment 0")
        t.switch("main")
        t.delete_branch("star-temp", force=True)
        t.gc(orphan_retention_days=0)

        # A specific handler takes priority over "*"
        print("\n  Specific handler overrides '*':")

        specific_log = []

        def specific_gc_hook(pending: PendingGC):
            specific_log.append("specific")
            pending.approve()

        t.on("gc", specific_gc_hook, name="specific_gc")

        # hook_names shows both registrations and their priority
        print(f"  hook_names: {t.hook_names}")

        t.branch("star-temp2")
        t.user("Throwaway experiment star-temp2: question 0")
        t.assistant("Response to star-temp2 experiment 0")
        t.switch("main")
        t.delete_branch("star-temp2", force=True)
        t.gc(orphan_retention_days=0)

        print(f"\n  Audit log: {audit_log}")
        print(f"  Specific log: {specific_log}")
        print(f"  ('*' fired for gc once, then specific handler took over)")

        t.print_hooks()


if __name__ == "__main__":
    catch_all()
