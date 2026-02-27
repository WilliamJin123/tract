"""Hook Registration and Routing

Part 1 — Register a handler with t.on(), see the three-tier routing model
(review=True > registered hook > auto-approve), and remove hooks with t.off().

Part 2 — The catch-all "*" handler intercepts every hookable operation with
a single registration. Shows how to build a universal audit logger.

Part 3 — The recursion guard: what happens when a hook handler triggers
another hookable operation. Tract auto-approves nested operations to
prevent infinite loops.

Demonstrates: t.on(), t.off(), t.hooks, review=True, catch-all "*",
              _fire_hook three-tier routing, recursion guard, PendingCompress,
              PendingGC, pending.approve(), pending.reject(), pending.pprint()
"""

from tract import Tract, Priority


# ---------------------------------------------------------------------------
# Part 1: Register, Fire, Remove
# ---------------------------------------------------------------------------

def part1_registration_and_routing():
    """t.on() registers, t.off() removes, three tiers decide what fires."""
    print("=" * 60)
    print("PART 1 — Registration and Three-Tier Routing")
    print("=" * 60)

    # Helper: create orphaned commits by branching then deleting
    def _make_orphans(t: Tract, branch_name="temp", count=3):
        """Branch, add commits, delete branch → orphaned commits."""
        t.branch(branch_name)
        for i in range(count):
            t.user(f"{branch_name} Q{i}")
            t.assistant(f"{branch_name} A{i}")
        t.switch("main")
        t.delete_branch(branch_name, force=True)

    with Tract.open() as t:
        t.system("You are a concise assistant.")
        t.user("Main question.")
        t.assistant("Main answer.")

        # --- Tier 3 (default): No hook registered → auto-approve ----------
        print("\n  Tier 3 (no hook): gc() auto-approves silently")
        _make_orphans(t, "tier3-temp")
        result = t.gc(orphan_retention_days=0)
        print(f"    gc returned: {type(result).__name__}")
        print(f"    (Auto-approved — no hook, no review)")

        # --- Tier 2: Registered hook fires ---------------------------------
        print("\n  Tier 2 (hook registered): gc() fires the handler")

        gc_log = []

        def my_gc_hook(pending):
            """Handler that logs the pending and approves it."""
            gc_log.append(pending.operation)
            print(f"    Hook fired! operation={pending.operation}")
            print(f"    Commits to remove: {len(pending.commits_to_remove)}")
            pending.approve()

        t.on("gc", my_gc_hook)
        print(f"    Registered hooks: {list(t.hooks.keys())}")

        _make_orphans(t, "tier2-temp")
        t.gc(orphan_retention_days=0)
        print(f"    Hook was called: {gc_log}")

        # --- Tier 1: review=True bypasses hooks entirely -------------------
        print("\n  Tier 1 (review=True): returns pending to caller")

        _make_orphans(t, "tier1-temp")
        pending = t.gc(orphan_retention_days=0, review=True)
        print(f"    gc(review=True) returned: {type(pending).__name__}")
        print(f"    status: {pending.status}")
        print(f"    Hook was NOT called (review takes priority)")

        # Approve manually
        pending.approve()
        print(f"    After approve(): status={pending.status}")

        # --- t.off() removes the hook --------------------------------------
        print("\n  t.off('gc') removes the handler:")
        t.off("gc")
        print(f"    Registered hooks: {list(t.hooks.keys())}")
        print(f"    (Back to tier 3 — auto-approve)")


# ---------------------------------------------------------------------------
# Part 2: Catch-All "*" Handler
# ---------------------------------------------------------------------------

def part2_catch_all():
    """A single '*' handler intercepts every hookable operation."""
    print("\n" + "=" * 60)
    print("PART 2 — Catch-All '*' Handler")
    print("=" * 60)

    with Tract.open() as t:
        audit_log = []

        def universal_logger(pending):
            """Log every pending operation, then auto-approve."""
            entry = f"{pending.operation} (id={pending.pending_id[:8]})"
            audit_log.append(entry)
            print(f"    [audit] {entry}")
            pending.approve()

        t.on("*", universal_logger)

        # Seed content
        t.system("You are a concise assistant.")
        t.user("Main question.")
        t.assistant("Main answer.")

        # Create orphans, then GC — fires through "*" (no specific handler)
        t.branch("star-temp")
        t.user("Orphan Q")
        t.assistant("Orphan A")
        t.switch("main")
        t.delete_branch("star-temp", force=True)
        t.gc(orphan_retention_days=0)

        # A specific handler takes priority over "*"
        print("\n  Specific handler overrides '*':")

        specific_log = []

        def specific_gc_hook(pending):
            specific_log.append("specific")
            print(f"    [specific] gc hook fired")
            pending.approve()

        t.on("gc", specific_gc_hook)

        t.branch("star-temp2")
        t.user("Orphan Q2")
        t.assistant("Orphan A2")
        t.switch("main")
        t.delete_branch("star-temp2", force=True)
        t.gc(orphan_retention_days=0)

        print(f"\n  Audit log: {audit_log}")
        print(f"  Specific log: {specific_log}")
        print(f"  ('*' fired for gc once, then specific handler took over)")


# ---------------------------------------------------------------------------
# Part 3: Recursion Guard
# ---------------------------------------------------------------------------

def part3_recursion_guard():
    """Nested hookable operations inside a hook auto-approve (no infinite loop)."""
    print("\n" + "=" * 60)
    print("PART 3 — Recursion Guard")
    print("=" * 60)
    print()
    print("  If a hook handler triggers another hookable operation,")
    print("  Tract auto-approves the inner one (no recursion).")

    with Tract.open() as t:
        # Track what fires
        fire_log = []

        # Pre-create two sets of orphans so both GC calls find work
        t.system("Assistant.")
        t.user("Main.")
        t.assistant("Reply.")

        # Orphan set 1 (for outer gc)
        t.branch("orphan1")
        t.user("O1-Q")
        t.assistant("O1-A")
        t.switch("main")
        t.delete_branch("orphan1", force=True)

        # Orphan set 2 (for inner gc to find during recursion)
        t.branch("orphan2")
        t.user("O2-Q")
        t.assistant("O2-A")
        t.switch("main")
        t.delete_branch("orphan2", force=True)

        def gc_then_gc(pending):
            """GC hook that tries to trigger ANOTHER gc inside itself."""
            fire_log.append(f"outer-gc (status={pending.status})")
            print(f"    [outer] gc hook fired, {len(pending.commits_to_remove)} commits")

            # Try to gc again inside the hook — the inner gc will
            # auto-approve (recursion guard), our hook won't re-enter.
            inner = t.gc(orphan_retention_days=0)
            fire_log.append(f"inner-gc (type={type(inner).__name__})")
            print(f"    [inner] gc returned: {type(inner).__name__}")
            print(f"    (Recursion guard: inner gc auto-approved, hook NOT re-entered)")

            pending.approve()

        t.on("gc", gc_then_gc)
        t.gc(orphan_retention_days=0)

        print(f"\n  Fire log: {fire_log}")
        print(f"  (outer fired once, inner auto-approved — no infinite loop)")


# ---------------------------------------------------------------------------

def main():
    part1_registration_and_routing()
    part2_catch_all()
    part3_recursion_guard()


if __name__ == "__main__":
    main()
