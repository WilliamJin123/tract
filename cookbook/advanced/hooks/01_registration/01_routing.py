"""Three-tier hook routing: review=True > registered hook > auto-approve.

Register a handler with t.on(), see how the three routing tiers determine
which code fires when a hookable operation runs, and remove hooks with t.off().
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.gc import PendingGC

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def registration_and_routing():
    """t.on() registers, t.off() removes, three tiers decide what fires."""
    print("=" * 60)
    print("PART 1 — Registration and Three-Tier Routing")
    print("=" * 60)

    # Helper: create orphaned commits by branching then deleting
    def _make_orphans(t: Tract, branch_name="temp", count=3):
        """Branch, add commits, delete branch -> orphaned commits."""
        t.branch(branch_name)
        for i in range(count):
            t.user(f"Throwaway experiment {branch_name}: question {i}")
            t.assistant(f"Response to {branch_name} experiment {i}")
        t.switch("main")
        t.delete_branch(branch_name, force=True)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a DevOps assistant helping engineers with CI/CD pipelines and infrastructure.")
        t.chat("How do I set up GitHub Actions for a Python project with pytest and linting?")

        # --- Tier 3 (default): No hook registered -> auto-approve ----------
        print("\n  Tier 3 (no hook): gc() auto-approves silently")
        _make_orphans(t, "tier3-temp")
        result = t.gc(orphan_retention_days=0)
        print(f"    gc returned: {type(result).__name__}")

        # hook_log captures the auto-approve even with no handler
        last_event = t.hook_log[-1]
        print(f"    hook_log: {last_event.handler_name} -> {last_event.result}")

        # --- Tier 2: Registered hook fires ---------------------------------
        print("\n  Tier 2 (hook registered): gc() fires the handler")

        def my_gc_hook(pending: PendingGC):
            """Handler that inspects the pending and approves it."""
            pending.pprint()
            pending.approve()

        t.on("gc", my_gc_hook, name="my_gc_hook")
        t.print_hooks()

        _make_orphans(t, "tier2-temp")
        t.gc(orphan_retention_days=0)

        # hook_log shows the handler that fired
        last_event = t.hook_log[-1]
        print(f"    hook_log: {last_event.handler_name} -> {last_event.result}")

        # --- Tier 1: review=True bypasses hooks entirely -------------------
        print("\n  Tier 1 (review=True): returns pending to caller")

        _make_orphans(t, "tier1-temp")
        pending = t.gc(orphan_retention_days=0, review=True)
        print(f"    gc(review=True) returned PendingGC (hook NOT called)")
        pending.pprint()

        # Approve manually
        pending.approve()
        print(f"    After approve(): status={pending.status}")

        # --- t.off() removes the hook --------------------------------------
        print("\n  t.off('gc') removes the handler:")
        t.off("gc")
        t.print_hooks()


if __name__ == "__main__":
    registration_and_routing()
