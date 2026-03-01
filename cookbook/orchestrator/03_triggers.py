"""Autonomous triggers: threshold-based automation for context management.

Triggers evaluate on every compile() or commit() and fire actions automatically
or collaboratively.  This cookbook covers all built-in triggers and hook interception.
"""

from tract import (
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

    print(f"\n  CompressTrigger: fires_on={compress_trigger.fires_on}, priority={compress_trigger.priority}")
    print(f"  PinTrigger:      fires_on={pin_trigger.fires_on}, priority={pin_trigger.priority}")

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
        print(f"  After compile: {ctx.token_count} tokens, {len(ctx.messages)} msgs")

    print("\n  Flow: evaluate() -> TriggerAction -> PendingTrigger -> hook -> approve/reject")


def new_triggers() -> None:
    """Part 2: Demonstrate RebaseTrigger, GCTrigger, and MergeTrigger."""
    print("\n" + "=" * 60)
    print("PART 2 -- New Triggers: Rebase, GC, Merge")
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
            print(f"  Fired: {action.action_type}, autonomy={action.autonomy}")
            print(f"  Reason: {action.reason}")

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
            print(f"  Fired: {action.reason}")

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
            print(f"  Fired: params={action.params}")
            print(f"  Reason: {action.reason}")


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
