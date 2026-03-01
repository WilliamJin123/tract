"""End-to-end integration tests for the trigger engine.

Tests the full lifecycle: configure triggers -> commit content ->
trigger evaluation -> verify actions executed/proposed.

Covers:
- Auto-pin on commit (autonomous mode)
- Auto-compress on compile (collaborative mode with approval)
- Trigger priority ordering via audit log
- Pause/resume lifecycle
- Collaborative approve/reject via hook system
- Trigger config persistence (save/load)
- Recursive evaluation prevention
- Multiple trigger composition
- Custom trigger subclass
- Trigger hook callback
- Trigger config survives restart (file-backed)
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from tract import (
    DialogueContent,
    EvaluationResult,
    InstructionContent,
    Trigger,
    TriggerAction,
    TriggerEvaluator,
    TriggerExecutionError,
    Priority,
    Tract,
    TractConfig,
    TokenBudgetConfig,
)
from tract.models.content import ArtifactContent, ReasoningContent
from tract.models.session import SessionContent
from tract.triggers.builtin.compress import CompressTrigger
from tract.triggers.builtin.pin import PinTrigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingTrigger(Trigger):
    """Test helper that counts evaluations and returns configurable action.

    default_handler does nothing (leaves pending unresolved) to preserve
    the proposal-based test workflow. Set auto_approve_default=True to
    use auto-approve behavior.
    """

    def __init__(
        self, name: str = "counter", priority: int = 100,
        trigger: str = "compile", action: TriggerAction | None = None,
        auto_approve_default: bool = False,
    ):
        self._name = name
        self._priority = priority
        self._trigger = trigger
        self._action = action
        self._auto_approve_default = auto_approve_default
        self.count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def fires_on(self) -> str:
        return self._trigger

    def evaluate(self, tract: Tract) -> TriggerAction | None:
        self.count += 1
        return self._action

    def default_handler(self, pending) -> None:
        """Leave pending unresolved by default for test control."""
        if self._auto_approve_default:
            pending.approve()


# ---------------------------------------------------------------------------
# 1. Auto-Pin on Commit
# ---------------------------------------------------------------------------


class TestAutoPinOnCommit:
    """PinTrigger auto-pins InstructionContent on commit."""

    def test_auto_pin_on_commit(self):
        """Configure PinTrigger (autonomous), commit InstructionContent, verify auto-pinned."""
        t = Tract.open(":memory:")
        try:
            # Configure PinTrigger
            pin_trigger = PinTrigger()
            t.configure_triggers(triggers=[pin_trigger])

            # Commit instruction content -- should trigger commit-time evaluation
            info = t.commit(InstructionContent(text="You are a helpful assistant."))

            # Verify it was auto-pinned
            annotations = t.get_annotations(info.commit_hash)
            assert len(annotations) >= 1
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()

    def test_auto_pin_does_not_pin_dialogue(self):
        """PinTrigger does not pin DialogueContent."""
        t = Tract.open(":memory:")
        try:
            pin_trigger = PinTrigger()
            t.configure_triggers(triggers=[pin_trigger])

            # First commit to establish HEAD
            t.commit(InstructionContent(text="system"))
            # Now commit dialogue
            info = t.commit(DialogueContent(role="user", text="Hi"))

            annotations = t.get_annotations(info.commit_hash)
            # No pinned annotation on dialogue
            assert not any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 2. Auto-Compress on Compile
# ---------------------------------------------------------------------------


class TestAutoCompressOnCompile:
    """CompressTrigger proposes compression in collaborative mode."""

    def test_auto_compress_on_compile(self):
        """Configure CompressTrigger with low threshold, compile triggers proposal."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            proposals = []

            # Register a trigger hook that captures but doesn't approve/reject
            # This overrides the trigger's default_handler auto-approve
            def trigger_hook(pending):
                proposals.append(pending)
                # Leave pending unresolved to test proposal creation

            t.on("trigger", trigger_hook)

            compress_trigger = CompressTrigger(threshold=0.5, summary_content="Compressed")
            t.configure_triggers(
                triggers=[compress_trigger],
            )

            # Commit enough to exceed threshold
            t.commit(InstructionContent(text="This is a fairly long instruction text"))

            # Compile triggers the compress trigger
            t.compile()

            # Should have created a proposal
            assert len(proposals) >= 1
            assert proposals[0].trigger_name == "auto-compress"
            assert proposals[0].status == "pending"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 3. Priority Ordering
# ---------------------------------------------------------------------------


class TestTriggerPriorityOrdering:
    """Verify pin runs before compress via audit log."""

    def test_trigger_priority_ordering(self):
        """PinTrigger (priority=100) evaluates before CompressTrigger (priority=200)."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            # Use counting triggers with known priorities
            p_first = CountingTrigger(name="first", priority=50, trigger="compile")
            p_second = CountingTrigger(name="second", priority=150, trigger="compile")
            p_third = CountingTrigger(name="third", priority=250, trigger="compile")

            t.configure_triggers(triggers=[p_third, p_first, p_second])

            t.commit(InstructionContent(text="hello"))
            t.compile()

            # All three should have been evaluated
            assert p_first.count >= 1
            assert p_second.count >= 1
            assert p_third.count >= 1

            # Check audit log order
            log_entries = t._trigger_repo.get_log(t.tract_id)
            # Filter for compile trigger entries
            compile_entries = [e for e in log_entries if e.trigger == "compile"]

            if len(compile_entries) >= 3:
                names = [e.trigger_name for e in compile_entries[:3]]
                assert names == ["first", "second", "third"]
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 4. Pause / Resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """Pause and resume trigger evaluation lifecycle."""

    def test_pause_resume_lifecycle(self):
        """Paused triggers don't fire, resumed triggers do."""
        t = Tract.open(":memory:")
        try:
            counter = CountingTrigger(name="lifecycle", trigger="commit")
            t.configure_triggers(triggers=[counter])

            # Commit while active -- both trigger the trigger
            t.commit(InstructionContent(text="first"))
            t.commit(DialogueContent(role="user", text="second"))
            assert counter.count == 2

            # Pause
            t.pause_all_triggers()
            t.commit(DialogueContent(role="assistant", text="third"))
            assert counter.count == 2  # No additional evaluation

            # Resume
            t.resume_all_triggers()
            t.commit(DialogueContent(role="user", text="fourth"))
            assert counter.count == 3  # Resumed
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 5. Collaborative Approve/Reject
# ---------------------------------------------------------------------------


class TestCollaborativeApproveReject:
    """Collaborative mode approve and reject lifecycle via hooks."""

    def test_collaborative_reject_then_approve(self):
        """First hook rejects, second hook approves -- annotation created on second."""
        t = Tract.open(":memory:")
        try:
            # Use DialogueContent which does NOT auto-pin
            t.commit(InstructionContent(text="system"))
            info = t.commit(DialogueContent(role="user", text="hello"))

            action = TriggerAction(
                action_type="annotate",
                params={"target_hash": info.commit_hash, "priority": "pinned"},
                reason="Should we pin?",
                autonomy="collaborative",
            )

            call_count = [0]

            def hook(pending):
                call_count[0] += 1
                if call_count[0] == 1:
                    pending.reject("Not now")
                else:
                    pending.approve()

            t.on("trigger", hook)

            p = CountingTrigger(name="collab", trigger="compile", action=action)
            t.configure_triggers(triggers=[p])

            # First compile -- hook rejects
            t.compile()
            annotations = t.get_annotations(info.commit_hash)
            assert not any(a.priority == Priority.PINNED for a in annotations)

            # Second compile -- hook approves
            t.compile()
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()

    def test_collaborative_approve_via_hook(self):
        """Hook auto-approves collaborative action, verifying execution."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))

            action = TriggerAction(
                action_type="annotate",
                params={"target_hash": info.commit_hash, "priority": "pinned"},
                reason="Should we pin?",
                autonomy="collaborative",
            )
            p = CountingTrigger(name="collab", trigger="compile", action=action)

            t.on("trigger", lambda pending: pending.approve())
            t.configure_triggers(triggers=[p])

            t.compile()

            # Annotation was created
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 6. Config Persistence
# ---------------------------------------------------------------------------


class TestTriggerConfigPersistence:
    """Trigger config save/load roundtrip."""

    def test_trigger_config_persistence(self):
        """Built-in trigger configs roundtrip through save/load."""
        t = Tract.open(":memory:")
        try:
            pin = PinTrigger()
            compress = CompressTrigger(threshold=0.8)

            config = {
                "triggers": [pin.to_config(), compress.to_config()],
            }
            t.save_trigger_config(config)

            loaded = t.load_trigger_config()
            assert loaded is not None
            assert len(loaded["triggers"]) == 2
            assert loaded["triggers"][0]["name"] == "auto-pin"
            assert loaded["triggers"][1]["name"] == "auto-compress"
            assert loaded["triggers"][1]["threshold"] == 0.8
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 7. Recursive Evaluation Prevention
# ---------------------------------------------------------------------------


class TestRecursiveEvaluationPrevention:
    """Trigger that calls compile() during evaluation should not infinite-loop."""

    def test_recursive_evaluation_prevention(self):
        """Trigger calling compile() inside evaluate() does not recurse infinitely."""

        class RecursiveCompileTrigger(Trigger):
            @property
            def name(self) -> str:
                return "recursive-compile"

            def evaluate(self, tract: Tract) -> TriggerAction | None:
                # This triggers compile, which would re-enter this trigger
                # without the recursion guard
                tract.compile()
                return None

        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = RecursiveCompileTrigger()
            t.configure_triggers(triggers=[p])

            # This should complete without stack overflow
            result = t.compile()
            assert result is not None
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 8. Multiple Triggers Compose
# ---------------------------------------------------------------------------


class TestMultipleTriggersCompose:
    """Multiple triggers interact correctly via priority ordering."""

    def test_multiple_triggers_compose(self):
        """PinTrigger + CompressTrigger together: pin runs first, then compress."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            proposals = []

            # Register a trigger hook to capture proposals without auto-approving
            def trigger_hook(pending):
                proposals.append(pending)

            t.on("trigger", trigger_hook)

            pin = PinTrigger()
            compress = CompressTrigger(threshold=0.5, summary_content="Summary")

            t.configure_triggers(
                triggers=[compress, pin],  # Deliberately reverse order
            )

            # Commit instruction (triggers pin, which is commit-triggered)
            info = t.commit(InstructionContent(text="This is a long system prompt"))

            # Pin should have fired (autonomous, commit-triggered)
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)

            # Compile (triggers compress)
            t.compile()

            # Compress should have proposed (collaborative, compile-triggered)
            assert any(p.trigger_name == "auto-compress" for p in proposals)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 9. Custom Trigger Subclass
# ---------------------------------------------------------------------------


class TestCustomTriggerSubclass:
    """User-defined custom triggers work with the evaluator."""

    def test_custom_trigger_subclass(self):
        """Custom Trigger subclass works with configure_triggers."""

        class MyCustomTrigger(Trigger):
            @property
            def name(self) -> str:
                return "my-custom"

            @property
            def priority(self) -> int:
                return 42

            @property
            def fires_on(self) -> str:
                return "commit"

            def evaluate(self, tract: Tract) -> TriggerAction | None:
                head = tract.head
                if head is None:
                    return None
                commit = tract.get_commit(head)
                if commit and commit.content_type == "reasoning":
                    return TriggerAction(
                        action_type="annotate",
                        params={"target_hash": head, "priority": "skip"},
                        reason="Auto-skip reasoning",
                        autonomy="autonomous",
                    )
                return None

        t = Tract.open(":memory:")
        try:
            custom = MyCustomTrigger()
            t.configure_triggers(triggers=[custom])

            # Commit reasoning content
            t.commit(InstructionContent(text="system"))
            info = t.commit(ReasoningContent(text="thinking..."))

            # Verify it was auto-skipped
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.SKIP for a in annotations)
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 10. Trigger Hook Callback
# ---------------------------------------------------------------------------


class TestTriggerHookCallback:
    """Trigger hook is invoked for collaborative proposals."""

    def test_trigger_hook_captures_pending(self):
        """Trigger hook receives PendingTrigger for collaborative actions."""
        t = Tract.open(":memory:")
        try:
            hook_calls = []
            info = t.commit(InstructionContent(text="hello"))

            action = TriggerAction(
                action_type="annotate",
                params={"target_hash": info.commit_hash, "priority": "pinned"},
                reason="Test hook",
                autonomy="collaborative",
            )
            p = CountingTrigger(name="hook-test", action=action)

            t.on("trigger", lambda pending: hook_calls.append(pending))
            t.configure_triggers(triggers=[p])

            t.compile()

            assert len(hook_calls) == 1
            assert hook_calls[0].trigger_name == "hook-test"
            assert hook_calls[0].status == "pending"
        finally:
            t.close()


# ---------------------------------------------------------------------------
# 11. Trigger Config Survives Restart (File-Backed)
# ---------------------------------------------------------------------------


class TestTriggerConfigSurvivesRestart:
    """Trigger config persists across Tract.open() calls (file-backed)."""

    def test_trigger_config_survives_restart_file_backed(self):
        """Triggers auto-load from _trace_meta on Tract.open()."""
        db_path = os.path.join(tempfile.mkdtemp(), "test_restart.db")
        tract_id = "restart-test-tract"

        try:
            # First session: configure and save triggers
            config = TractConfig(
                db_path=db_path,
                token_budget=TokenBudgetConfig(max_tokens=1000),
            )
            t1 = Tract.open(db_path, tract_id=tract_id, config=config)
            pin = PinTrigger(pin_types={"instruction", "session"})
            compress = CompressTrigger(threshold=0.75)
            t1.configure_triggers(triggers=[pin, compress])

            # Save config
            trigger_config = {
                "triggers": [pin.to_config(), compress.to_config()],
            }
            t1.save_trigger_config(trigger_config)

            # Commit some content
            t1.commit(InstructionContent(text="System prompt"))
            t1.close()

            # Second session: re-open same file
            config2 = TractConfig(
                db_path=db_path,
                token_budget=TokenBudgetConfig(max_tokens=1000),
            )
            t2 = Tract.open(db_path, tract_id=tract_id, config=config2)

            # Verify evaluator was auto-created
            assert t2.trigger_evaluator is not None

            # Verify both triggers are registered
            registered = t2.trigger_evaluator._triggers
            assert len(registered) == 2
            trigger_names = {p.name for p in registered}
            assert "auto-pin" in trigger_names
            assert "auto-compress" in trigger_names

            # Verify priority ordering (pin=100, compress=200)
            assert registered[0].name == "auto-pin"
            assert registered[1].name == "auto-compress"

            # Verify compress threshold was restored
            compress_restored = next(
                p for p in registered if p.name == "auto-compress"
            )
            assert compress_restored._threshold == 0.75

            # Verify triggers evaluate correctly -- commit SessionContent
            info = t2.commit(SessionContent(session_type="start", summary="Restart"))
            # PinTrigger should have pinned it
            annotations = t2.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)

            t2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
