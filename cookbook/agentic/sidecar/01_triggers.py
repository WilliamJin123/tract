"""Autonomous triggers: threshold-based automation for context management.

Triggers are threshold-based rules that fire automatically when conditions are
met during normal tract operations.  They eliminate manual housekeeping by
watching token counts, branch divergence, content types, and other metrics,
then proposing (or executing) maintenance actions on your behalf.

When triggers fire
------------------
Each trigger declares a ``fires_on`` event:

* **"compile"** -- evaluated on every ``compile()`` call.  Good for token-
  budget concerns (compression, GC) that only matter when you materialise
  the context window.
* **"commit"** -- evaluated on every ``commit()`` (and the shorthand methods
  ``system()``, ``user()``, ``assistant()``).  Good for per-message
  reactions like auto-pinning instructions.

Autonomy spectrum
-----------------
Every ``TriggerAction`` carries an ``autonomy`` field that controls routing:

* **autonomous** -- execute immediately, no approval needed.  The action is
  applied inside the same call that evaluated the trigger.
  Example: ``PinTrigger`` silently annotates instruction commits.
* **collaborative** -- routed through the hook system for approval.  If no
  hook is registered the trigger's ``default_handler`` decides (usually
  auto-approve).  Example: ``CompressTrigger`` fires a ``PendingTrigger``
  hook so callers can inspect / modify / reject.
* **supervised** -- require explicit human review before execution.  The
  action is surfaced but never auto-approved.

Hook interception lifecycle
---------------------------
For collaborative triggers the flow is::

    evaluate(tract) -> TriggerAction -> PendingTrigger -> hook handler -> approve / reject

Register a handler with ``t.on("trigger", handler)`` to intercept.  Inside
the handler you can inspect the pending action, ``modify_params()``, then
call ``approve()`` or ``reject(reason)``.

Built-in triggers
-----------------
==============================  ==========  ================================
Trigger                         fires_on    Default behaviour
==============================  ==========  ================================
``CompressTrigger``             compile     collaborative -- compress range
``PinTrigger``                  commit      autonomous -- annotate PINNED
``RebaseTrigger``               compile     collaborative -- rebase branch
``GCTrigger``                   compile     collaborative -- garbage-collect
``MergeTrigger``                compile     collaborative -- merge branch
``BranchTrigger``               commit      collaborative -- create branch
``ArchiveTrigger``              compile     collaborative -- archive branch
==============================  ==========  ================================

This cookbook demonstrates all seven triggers, hook interception, and the
autonomy spectrum end-to-end.
"""

from tract import (
    ArchiveTrigger,
    BranchTrigger,
    CompressTrigger,
    GCTrigger,
    MergeTrigger,
    PinTrigger,
    RebaseTrigger,
    TokenBudgetConfig,
    Tract,
    TractConfig,
)
from tract.hooks.trigger import PendingTrigger


def trigger_basics() -> None:
    """Part 1: Create triggers, configure them, and watch them fire."""
    print("=" * 60)
    print("PART 1 -- Trigger Basics")
    print("=" * 60)

    compress_trigger = CompressTrigger(threshold=0.5, summary_content="Condensed.")
    pin_trigger = PinTrigger(pin_types={"instruction"})

    print(f"\n  {compress_trigger}")
    print(f"  {pin_trigger}")

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=200))
    with Tract.open(config=config) as t:
        t.configure_triggers([compress_trigger, pin_trigger])

        # PinTrigger fires on "commit" for instruction content (autonomous)
        ci = t.system("You are a helpful assistant.")
        annotations = t.get_annotations(ci.commit_hash)
        print(f"\n  Committed instruction: {ci.commit_hash[:12]}")
        print(f"  Auto-pinned: {annotations[0].priority.name if annotations else 'NO'}")

        # Add tokens to trip CompressTrigger (fires on "compile")
        for i in range(8):
            t.user(f"Message {i} with padding to increase tokens.")
        print(f"  Commits before compile: {len(t.log(limit=100))}")

        # compile() fires the trigger; default_handler auto-approves compression
        ctx = t.compile()
        print(f"\n  After compile: {ctx.token_count} tokens, {len(ctx.messages)} msgs")
        ctx.pprint(style="compact")

    print("\n  Flow: evaluate() -> TriggerAction -> PendingTrigger -> hook -> approve/reject")


def new_triggers() -> None:
    """Part 2: Demonstrate RebaseTrigger, GCTrigger, MergeTrigger, BranchTrigger, ArchiveTrigger."""
    print("\n" + "=" * 60)
    print("PART 2 -- More Triggers: Rebase, GC, Merge, Branch, Archive")
    print("=" * 60)

    # -- RebaseTrigger: fires when branch diverges from target --------
    print("\n  RebaseTrigger(divergence_commits=5)")
    rebase = RebaseTrigger(target_branch="main", divergence_commits=5)
    with Tract.open() as t:
        for i in range(3):
            t.user(f"Main msg {i}")
        t.branch("feature")
        t.switch("feature")
        for i in range(3):
            t.user(f"Feature msg {i}")
        t.switch("main")
        for i in range(6):
            t.user(f"More main work {i}")
        t.switch("feature")
        action = rebase.evaluate(t)
        if action:
            print(f"  {action}")

    # -- GCTrigger: fires when dead commits exceed threshold ----------
    print(f"\n  GCTrigger(max_dead_commits=3)")
    gc_trigger = GCTrigger(max_dead_commits=3)
    with Tract.open() as t:
        t.system("Be helpful.")
        for i in range(5):
            t.user(f"Turn {i}")
        t.compress(content="Summary.")
        action = gc_trigger.evaluate(t)
        if action:
            print(f"  {action}")

    # -- MergeTrigger: fires when branch is complete ------------------
    print(f"\n  MergeTrigger(completion_commits=3, idle_seconds=0)")
    merge = MergeTrigger(target_branch="main", completion_commits=3, idle_seconds=0)
    with Tract.open() as t:
        t.user("Initial on main")
        t.branch("feature-x")
        t.switch("feature-x")
        for i in range(4):
            t.user(f"Feature-x item {i}")
        action = merge.evaluate(t)
        if action:
            print(f"  {action}")

    # -- BranchTrigger: fires when content types switch rapidly ----
    print(f"\n  BranchTrigger(content_type_window=5, switch_threshold=2)")
    branch_trigger = BranchTrigger(content_type_window=5, switch_threshold=2)
    with Tract.open() as t:
        # Mix content types to trigger tangent detection
        t.system("You are a helpful assistant.")          # instruction
        t.user("What is Python?")                         # dialogue
        t.assistant("Python is a programming language.")   # dialogue
        t.tool_result("t1", "code_gen", "def hello(): pass")  # tool_io
        t.user("Now explain decorators.")                  # dialogue
        action = branch_trigger.evaluate(t)
        if action:
            print(f"  {action}")
        else:
            print(f"  Not fired (transitions below threshold)")

    # -- ArchiveTrigger: fires when branch is stale ----------------
    print(f"\n  ArchiveTrigger(stale_days=0, min_commits=5)")
    archive = ArchiveTrigger(stale_days=0, min_commits=5)
    with Tract.open() as t:
        t.user("Start on main")
        t.branch("experiment")
        t.switch("experiment")
        t.user("Quick note on experiment branch.")
        # Branch has 1 commit and stale_days=0 -> should fire
        action = archive.evaluate(t)
        if action:
            print(f"  {action}")
        else:
            print(f"  Not fired (branch not stale enough or too many commits)")


def hook_interception() -> None:
    """Part 3: Intercept trigger actions via t.on('trigger', handler)."""
    print("\n" + "=" * 60)
    print("PART 3 -- Hook Interception")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=150))
    with Tract.open(config=config) as t:
        trigger = CompressTrigger(threshold=0.3, summary_content="Summarized.")
        intercepted: list[PendingTrigger] = []

        def intercept(pending: PendingTrigger) -> None:
            intercepted.append(pending)
            print(f"\n  [hook] trigger={pending.trigger_name}, action={pending.action_type}")
            print(f"  [hook] reason: {pending.reason}")
            print(f"  [hook] params: {pending.action_params}")
            pending.modify_params({"target_tokens": 50})
            print(f"  [hook] modified: {pending.action_params}")
            pending.approve()

        t.on("trigger", intercept, name="intercept")
        t.configure_triggers([trigger])
        for i in range(8):
            t.user(f"Padding message {i} to push tokens over threshold.")

        ctx = t.compile()
        print(f"\n  Hook fired {len(intercepted)} time(s)")
        print(f"  Context: {ctx.token_count} tokens, {len(ctx.messages)} msgs")

        # Demonstrate rejection
        print("\n  -- Rejection demo --")
        t.off("trigger")
        rejected: list[str] = []

        def reject(pending: PendingTrigger) -> None:
            pending.reject("Not now.")
            rejected.append(pending.rejection_reason or "")

        t.on("trigger", reject, name="reject")
        for i in range(5):
            t.user(f"More content {i}.")
        t.compile()
        if rejected:
            print(f"  Rejected: {rejected[0]!r}")


def autonomy_spectrum() -> None:
    """Part 4: Autonomous vs collaborative triggers and hook overrides."""
    print("\n" + "=" * 60)
    print("PART 4 -- Autonomy Spectrum")
    print("=" * 60)

    # Autonomous: PinTrigger executes immediately, no hook routing
    print("\n  Autonomous (PinTrigger) -- executes without asking")
    with Tract.open() as t:
        t.configure_triggers([PinTrigger(pin_types={"instruction"})])
        ci = t.system("You are a code reviewer.")
        ann = t.get_annotations(ci.commit_hash)
        print(f"    instruction -> {ann[0].priority.name if ann else 'not pinned'}")
        ci2 = t.user("Review this PR.")
        ann2 = t.get_annotations(ci2.commit_hash)
        print(f"    dialogue    -> {'pinned' if ann2 else 'not pinned (correct)'}")

    # Collaborative: CompressTrigger routes through hook system
    print("\n  Collaborative (CompressTrigger) -- fires hook for review")
    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=150))
    with Tract.open(config=config) as t:
        t.configure_triggers([CompressTrigger(threshold=0.3, summary_content="Summary.")])
        for i in range(6):
            t.user(f"Content block {i} with enough words to grow tokens.")
        ctx = t.compile()
        print(f"    No hook -> default_handler auto-approved ({ctx.token_count} tokens)")

    # Hook override: t.on("trigger") intercepts collaborative triggers
    print("\n  Hook override -- t.on('trigger') takes priority")
    with Tract.open(config=config) as t:
        overridden: list[str] = []

        def override(pending: PendingTrigger) -> None:
            overridden.append(pending.action_type)
            print(f"    [hook] intercepted {pending.action_type}: {pending.reason}")
            pending.approve()

        t.on("trigger", override, name="override")
        t.configure_triggers([CompressTrigger(threshold=0.3, summary_content="Overridden.")])
        for i in range(8):
            t.user(f"Fill {i}.")
        t.compile()
        print(f"    Overrode {len(overridden)} action(s): {overridden}")


def main() -> None:
    trigger_basics()
    new_triggers()
    hook_interception()
    autonomy_spectrum()


if __name__ == "__main__":
    main()
