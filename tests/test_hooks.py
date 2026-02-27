"""Comprehensive tests for the hook system (Phase 1).

Tests the Pending base class, hook registration on Tract, three-tier routing
for compress(), recursion guard, unresolved handler warning, PendingCompress
lifecycle, _public_actions whitelist, and ValidationResult/HookRejection models.

All tests are standalone and can run in any order.
"""

from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone

import pytest

from tract import CompressResult, Tract


# ---------------------------------------------------------------------------
# Hook-system imports (new in Phase 1)
# ---------------------------------------------------------------------------

from tract.hooks.pending import Pending
from tract.hooks.compress import PendingCompress
from tract.hooks.validation import ValidationResult, HookRejection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compressible_tract() -> Tract:
    """Create an in-memory Tract with 3 commits (system + user + assistant).

    This is the minimum required to exercise compress().
    """
    t = Tract.open(":memory:")
    t.system("You are a helpful assistant.")
    t.user("Hello, how are you?")
    t.assistant("I'm doing well, thank you!")
    return t


def _make_larger_tract(n_user: int = 5) -> Tract:
    """Create an in-memory Tract with a system prompt and n_user user+assistant pairs."""
    t = Tract.open(":memory:")
    t.system("System prompt for compression tests.")
    for i in range(n_user):
        t.user(f"User message {i + 1}")
        t.assistant(f"Assistant reply {i + 1}")
    return t


# ===========================================================================
# 1. Pending Base Class
# ===========================================================================


class TestPendingBaseClass:
    """Tests for the Pending base class identity and lifecycle."""

    def test_pending_has_auto_generated_id(self):
        """Pending instances get a unique hex pending_id on creation."""
        # PendingCompress is a concrete subclass we can instantiate through
        # the compress pipeline, but for unit testing the base we need a
        # minimal concrete subclass.  Since Pending is abstract-ish (used
        # as base), we test via PendingCompress created through compress.
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert isinstance(pending, PendingCompress)
        assert hasattr(pending, "pending_id")
        assert isinstance(pending.pending_id, str)
        assert len(pending.pending_id) == 16  # uuid hex truncated to 16 chars
        # Validate it's valid hex
        int(pending.pending_id, 16)
        t.close()

    def test_pending_has_created_at(self):
        """Pending has a created_at datetime set on creation."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert hasattr(pending, "created_at")
        assert isinstance(pending.created_at, datetime)
        t.close()

    def test_pending_initial_status_is_pending(self):
        """Pending starts with status='pending'."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert pending.status == "pending"
        t.close()

    def test_approve_changes_status_to_approved(self):
        """approve() changes status from 'pending' to 'approved'."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert pending.status == "pending"
        pending.approve()
        assert pending.status == "approved"
        t.close()

    def test_reject_changes_status_and_stores_reason(self):
        """reject() changes status to 'rejected' and stores rejection_reason."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.reject("Not a good summary")
        assert pending.status == "rejected"
        assert pending.rejection_reason == "Not a good summary"
        t.close()

    def test_cannot_approve_after_reject(self):
        """Calling approve() after reject() raises an error."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.reject("bad")
        with pytest.raises(Exception):  # Could be ValueError or custom
            pending.approve()
        t.close()

    def test_cannot_reject_after_approve(self):
        """Calling reject() after approve() raises an error."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.approve()
        with pytest.raises(Exception):  # Could be ValueError or custom
            pending.reject("too late")
        t.close()

    def test_cannot_approve_twice(self):
        """Calling approve() a second time raises an error."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.approve()
        with pytest.raises(Exception):
            pending.approve()
        t.close()

    def test_cannot_reject_twice(self):
        """Calling reject() a second time raises an error."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.reject("first")
        with pytest.raises(Exception):
            pending.reject("second")
        t.close()

    def test_apply_decision_approve(self):
        """apply_decision({'action': 'approve'}) calls approve()."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.apply_decision({"action": "approve"})
        assert pending.status == "approved"
        t.close()

    def test_apply_decision_blocks_internal_methods(self):
        """apply_decision({'action': '_execute_fn'}) raises ValueError (whitelist)."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        with pytest.raises(ValueError):
            pending.apply_decision({"action": "_execute_fn"})
        t.close()

    def test_execute_tool_reject(self):
        """execute_tool('reject', {'reason': 'bad'}) works."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.execute_tool("reject", {"reason": "bad"})
        assert pending.status == "rejected"
        assert pending.rejection_reason == "bad"
        t.close()

    def test_execute_tool_blocks_internal_methods(self):
        """execute_tool('_execute_fn', {}) raises ValueError (whitelist)."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        with pytest.raises(ValueError):
            pending.execute_tool("_execute_fn", {})
        t.close()

    def test_unique_pending_ids(self):
        """Each Pending instance gets a unique pending_id."""
        t = _make_larger_tract(3)
        p1 = t.compress(content="summary 1", review=True)
        p1.reject("skip")  # resolve so we can re-compress if needed

        # Create another tract for a second pending
        t2 = _make_compressible_tract()
        p2 = t2.compress(content="summary 2", review=True)

        assert p1.pending_id != p2.pending_id
        t.close()
        t2.close()


# ===========================================================================
# 2. Hook Registration on Tract
# ===========================================================================


class TestHookRegistration:
    """Tests for t.on(), t.off(), t.hooks."""

    def test_register_compress_hook(self):
        """t.on('compress', handler) registers a hook."""
        t = Tract.open(":memory:")
        handler = lambda p: p.approve()
        t.on("compress", handler)
        assert "compress" in t.hooks
        t.close()

    def test_hooks_returns_dict(self):
        """t.hooks returns a dict of registered hooks."""
        t = Tract.open(":memory:")
        assert isinstance(t.hooks, dict)
        assert len(t.hooks) == 0

        handler = lambda p: p.approve()
        t.on("compress", handler)
        assert len(t.hooks) == 1
        assert "compress" in t.hooks
        t.close()

    def test_off_removes_hook(self):
        """t.off('compress') removes the registered hook."""
        t = Tract.open(":memory:")
        handler = lambda p: p.approve()
        t.on("compress", handler)
        assert "compress" in t.hooks

        t.off("compress")
        assert "compress" not in t.hooks
        t.close()

    def test_on_commit_raises_valueerror(self):
        """t.on('commit', handler) raises ValueError -- commit is not hookable."""
        t = Tract.open(":memory:")
        with pytest.raises(ValueError, match="not a hookable"):
            t.on("commit", lambda p: None)
        t.close()

    def test_on_user_raises_valueerror(self):
        """t.on('user', handler) raises ValueError -- user is not hookable."""
        t = Tract.open(":memory:")
        with pytest.raises(ValueError, match="not a hookable"):
            t.on("user", lambda p: None)
        t.close()

    def test_on_compile_raises_valueerror(self):
        """t.on('compile', handler) raises ValueError -- compile is not hookable."""
        t = Tract.open(":memory:")
        with pytest.raises(ValueError, match="not a hookable"):
            t.on("compile", lambda p: None)
        t.close()

    def test_on_catchall_works(self):
        """t.on('*', handler) works as catch-all hook."""
        t = Tract.open(":memory:")
        handler = lambda p: p.approve()
        t.on("*", handler)
        assert "*" in t.hooks
        t.close()

    def test_second_on_stacks(self):
        """Registering a second hook for the same operation stacks handlers."""
        t = Tract.open(":memory:")
        calls = []

        handler1 = lambda p: calls.append("handler1") or p.approve()
        handler2 = lambda p: calls.append("handler2") or p.approve()

        t.on("compress", handler1)
        t.on("compress", handler2)

        # One key for "compress", but with two handlers
        assert len([k for k in t.hooks if k == "compress"]) == 1
        assert len(t.hooks["compress"]) == 2

        # The first handler resolves -> second never fires
        t.system("sys")
        t.user("hello")
        t.assistant("hi")
        t.compress(content="summary")

        assert "handler1" in calls
        assert "handler2" not in calls
        t.close()

    def test_non_hookable_operations_raise_valueerror(self):
        """All non-hookable operations raise ValueError on t.on()."""
        t = Tract.open(":memory:")
        non_hookable = [
            "commit", "user", "assistant", "system", "compile",
            "log", "status", "diff", "annotate", "edit",
            "branch", "switch", "checkout", "reset",
        ]
        for op in non_hookable:
            with pytest.raises(ValueError, match="not a hookable"):
                t.on(op, lambda p: None)
        t.close()


# ===========================================================================
# 3. Three-Tier Routing for compress()
# ===========================================================================


class TestThreeTierRouting:
    """Tests for the three-tier routing: review=True > hook > auto-approve."""

    def test_no_hook_no_review_auto_approves(self):
        """Without hook or review=True, compress() auto-approves and returns CompressResult."""
        t = _make_compressible_tract()
        result = t.compress(content="summary")
        assert isinstance(result, CompressResult)
        t.close()

    def test_review_true_returns_pending(self):
        """review=True returns PendingCompress with status='pending'."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert isinstance(pending, PendingCompress)
        assert pending.status == "pending"
        t.close()

    def test_review_pending_can_edit_and_approve(self):
        """PendingCompress from review=True supports edit_summary and approve."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary text", review=True)
        assert isinstance(pending, PendingCompress)

        pending.edit_summary(0, "Revised summary")
        result = pending.approve()
        assert isinstance(result, CompressResult)
        t.close()

    def test_hook_fires_on_compress(self):
        """When a hook is registered, it fires and receives a PendingCompress."""
        t = _make_compressible_tract()
        received = []

        def my_hook(pending):
            received.append(pending)
            pending.approve()

        t.on("compress", my_hook)
        result = t.compress(content="summary")

        assert isinstance(result, CompressResult)
        assert len(received) == 1
        assert isinstance(received[0], PendingCompress)
        t.close()

    def test_hook_reject_prevents_compression(self):
        """A hook that rejects prevents the compression from executing."""
        t = _make_compressible_tract()
        old_head = t.head

        def rejecting_hook(pending):
            pending.reject("I don't like this summary")

        t.on("compress", rejecting_hook)
        # When hook rejects, compress should either return None or raise
        # or return a result indicating rejection -- behavior depends on impl.
        # The key check is that head doesn't change (no commits applied).
        result = t.compress(content="summary")
        # Head should be unchanged since compression was rejected
        assert t.head == old_head
        t.close()

    def test_review_true_overrides_hook(self):
        """review=True takes precedence over registered hook; hook should NOT fire."""
        t = _make_compressible_tract()
        hook_fired = []

        def my_hook(pending):
            hook_fired.append(True)
            pending.approve()

        t.on("compress", my_hook)
        pending = t.compress(content="summary", review=True)

        # Should return pending, not auto-approve through hook
        assert isinstance(pending, PendingCompress)
        assert pending.status == "pending"
        assert len(hook_fired) == 0
        t.close()

    def test_catchall_hook_fires_for_compress(self):
        """t.on('*', handler) fires for compress when no specific compress hook exists."""
        t = _make_compressible_tract()
        received = []

        def catchall(pending):
            received.append(pending)
            pending.approve()

        t.on("*", catchall)
        result = t.compress(content="summary")

        assert isinstance(result, CompressResult)
        assert len(received) == 1
        t.close()

    def test_specific_hook_overrides_catchall(self):
        """A specific compress hook takes precedence over catch-all."""
        t = _make_compressible_tract()
        calls = []

        def specific(pending):
            calls.append("specific")
            pending.approve()

        def catchall(pending):
            calls.append("catchall")
            pending.approve()

        t.on("*", catchall)
        t.on("compress", specific)
        result = t.compress(content="summary")

        assert isinstance(result, CompressResult)
        assert "specific" in calls
        assert "catchall" not in calls
        t.close()


# ===========================================================================
# 4. Recursion Guard
# ===========================================================================


class TestRecursionGuard:
    """Tests for the _in_hook recursion guard."""

    def test_recursive_compress_in_hook_auto_approves(self):
        """compress() called inside a hook handler auto-approves (no recursion).

        The inner compress auto-approves because _in_hook is True,
        skipping the hook handler. This verifies no infinite recursion.
        Note: the inner compress changes HEAD, so we verify the recursion
        guard works by checking the inner compress returns a CompressResult
        (i.e., it auto-approved and committed successfully). The outer
        pending cannot be approved after HEAD changed, which is correct
        TOCTOU behavior. The test verifies the recursion guard fires
        and the inner compress succeeds.
        """
        t = _make_larger_tract(4)
        inner_results = []
        hook_entered = []

        def recursive_hook(pending):
            hook_entered.append(True)
            # This nested compress should auto-approve (recursion guard)
            inner_result = t.compress(content="inner summary")
            inner_results.append(inner_result)
            # Don't approve outer pending -- HEAD changed due to inner compress

        t.on("compress", recursive_hook)
        result = t.compress(content="outer summary")

        # Hook was entered (not infinite loop)
        assert len(hook_entered) == 1
        # Inner call succeeded (auto-approved, recursion guard skipped hook)
        assert len(inner_results) == 1
        assert isinstance(inner_results[0], CompressResult)
        # Outer pending is returned (still pending since handler did not approve)
        assert isinstance(result, PendingCompress)
        t.close()

    def test_in_hook_flag_resets_after_handler(self):
        """After a hook handler completes, _in_hook resets so hooks fire again."""
        t = _make_larger_tract(4)
        call_count = []

        def counting_hook(pending):
            call_count.append(1)
            pending.approve()

        t.on("compress", counting_hook)

        # First compress -- hook should fire
        t.compress(content="first summary")
        assert len(call_count) == 1

        # Add more commits so we have something to compress again
        t.user("Another message")
        t.assistant("Another reply")

        # Second compress -- hook should fire again (flag reset)
        t.compress(content="second summary")
        assert len(call_count) == 2
        t.close()


# ===========================================================================
# 5. Unresolved Handler Warning
# ===========================================================================


class TestUnresolvedHandlerWarning:
    """Tests for warnings when a hook handler doesn't resolve the pending."""

    def test_unresolved_handler_emits_warning(self):
        """Handler that doesn't approve or reject emits a warning."""
        t = _make_compressible_tract()

        def forgetful_hook(pending):
            pass  # Forgot to approve or reject

        t.on("compress", forgetful_hook)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.compress(content="summary")
            # Should have a warning about unresolved pending
            hook_warnings = [
                x for x in w
                if "approve" in str(x.message).lower()
                or "reject" in str(x.message).lower()
                or "unresolved" in str(x.message).lower()
                or "pending" in str(x.message).lower()
            ]
            assert len(hook_warnings) >= 1, (
                f"Expected warning about unresolved handler, got: {[str(x.message) for x in w]}"
            )
        t.close()


# ===========================================================================
# 6. PendingCompress Lifecycle
# ===========================================================================


class TestPendingCompressLifecycle:
    """Tests for PendingCompress fields and lifecycle."""

    def test_summaries_contain_summary_text(self):
        """pending.summaries contains the summary text."""
        t = _make_compressible_tract()
        pending = t.compress(content="My excellent summary", review=True)
        assert isinstance(pending.summaries, list)
        assert len(pending.summaries) >= 1
        assert "My excellent summary" in pending.summaries[0]
        t.close()

    def test_source_commits_lists_compressed_hashes(self):
        """pending.source_commits lists the commit hashes being compressed."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert isinstance(pending.source_commits, list)
        assert len(pending.source_commits) >= 1
        # Each element should be a string hash
        for h in pending.source_commits:
            assert isinstance(h, str)
            assert len(h) > 0
        t.close()

    def test_original_tokens_positive(self):
        """pending.original_tokens is a positive integer."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert pending.original_tokens > 0
        t.close()

    def test_edit_summary_updates_text(self):
        """edit_summary(0, 'new') updates summaries[0]."""
        t = _make_compressible_tract()
        pending = t.compress(content="original", review=True)
        assert pending.summaries[0] == "original"
        pending.edit_summary(0, "revised text")
        assert pending.summaries[0] == "revised text"
        t.close()

    def test_edit_summary_invalid_index_raises(self):
        """edit_summary with out-of-range index raises IndexError."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        with pytest.raises(IndexError):
            pending.edit_summary(999, "bad index")
        t.close()

    def test_approve_returns_compress_result(self):
        """approve() returns CompressResult with correct data."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        result = pending.approve()

        assert isinstance(result, CompressResult)
        assert result.compression_id
        assert len(result.summary_commits) >= 1
        assert len(result.source_commits) >= 1
        assert result.new_head
        t.close()

    def test_approve_sets_status(self):
        """After approve, status is 'approved'."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.approve()
        assert pending.status == "approved"
        t.close()

    def test_approve_updates_tract_head(self):
        """approve() causes the tract's HEAD to update."""
        t = _make_compressible_tract()
        old_head = t.head
        pending = t.compress(content="summary", review=True)
        result = pending.approve()
        assert t.head != old_head
        assert t.head == result.new_head
        t.close()

    def test_reject_leaves_tract_unchanged(self):
        """reject() does not modify the tract state."""
        t = _make_compressible_tract()
        old_head = t.head
        pending = t.compress(content="summary", review=True)
        pending.reject("not good")
        assert t.head == old_head
        t.close()


# ===========================================================================
# 7. _public_actions Whitelist
# ===========================================================================


class TestPublicActionsWhitelist:
    """Tests for the _public_actions whitelist on PendingCompress."""

    def _get_actions(self):
        """Get _public_actions from a PendingCompress instance."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        actions = pending._public_actions
        t.close()
        return actions

    def test_approve_in_public_actions(self):
        """'approve' is in PendingCompress._public_actions."""
        assert "approve" in self._get_actions()

    def test_reject_in_public_actions(self):
        """'reject' is in PendingCompress._public_actions."""
        assert "reject" in self._get_actions()

    def test_edit_summary_in_public_actions(self):
        """'edit_summary' is in PendingCompress._public_actions."""
        assert "edit_summary" in self._get_actions()

    def test_internal_methods_not_in_public_actions(self):
        """Internal methods (starting with _) are NOT in _public_actions."""
        for action in self._get_actions():
            assert not action.startswith("_"), (
                f"Internal method '{action}' should not be in _public_actions"
            )

    def test_execute_fn_not_in_public_actions(self):
        """'_execute_fn' is NOT in _public_actions."""
        assert "_execute_fn" not in self._get_actions()


# ===========================================================================
# 8. ValidationResult and HookRejection
# ===========================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_passed(self):
        """ValidationResult with passed=True."""
        vr = ValidationResult(passed=True)
        assert vr.passed is True
        assert vr.diagnosis is None

    def test_validation_failed_with_diagnosis(self):
        """ValidationResult with passed=False and diagnosis."""
        vr = ValidationResult(passed=False, diagnosis="missing key info", index=0)
        assert not vr.passed
        assert vr.diagnosis == "missing key info"
        assert vr.index == 0

    def test_validation_failed_no_index(self):
        """ValidationResult failed without index."""
        vr = ValidationResult(passed=False, diagnosis="general problem")
        assert not vr.passed
        assert vr.diagnosis == "general problem"


class TestHookRejection:
    """Tests for HookRejection dataclass."""

    def test_hook_rejection_creation(self):
        """HookRejection stores reason, pending, and rejection_source."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        hr = HookRejection(
            reason="Bad summary quality",
            pending=pending,
            rejection_source="hook",
        )
        assert hr.reason == "Bad summary quality"
        assert hr.pending is pending
        assert hr.rejection_source == "hook"
        t.close()

    def test_hook_rejection_with_source(self):
        """HookRejection can include rejection_source 'validation'."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        hr = HookRejection(
            reason="low quality",
            pending=pending,
            rejection_source="validation",
        )
        assert hr.reason == "low quality"
        assert hr.rejection_source == "validation"
        t.close()


# ===========================================================================
# 9. ValueError on Non-Hookable Operations (comprehensive)
# ===========================================================================


class TestNonHookableOperationsComprehensive:
    """Exhaustive test that all non-hookable operations reject hook registration."""

    @pytest.mark.parametrize("op", [
        "commit", "user", "assistant", "system", "compile",
        "log", "status", "diff", "annotate", "edit",
        "branch", "switch", "checkout", "reset",
    ])
    def test_on_non_hookable_raises_valueerror(self, op):
        """t.on('{op}', handler) raises ValueError."""
        t = Tract.open(":memory:")
        with pytest.raises(ValueError, match="not a hookable"):
            t.on(op, lambda p: None)
        t.close()


# ===========================================================================
# 10. Hook with Edit Before Approve
# ===========================================================================


class TestHookEditBeforeApprove:
    """Tests for hooks that modify pending before approving."""

    def test_hook_edits_summary_before_approve(self):
        """Hook can edit_summary then approve, and the result uses edited text."""
        t = _make_compressible_tract()

        def editing_hook(pending):
            pending.edit_summary(0, "Hook-edited summary")
            pending.approve()

        t.on("compress", editing_hook)
        result = t.compress(content="original summary")

        assert isinstance(result, CompressResult)

        # Verify compiled output uses the edited summary
        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "Hook-edited" in messages_text
        t.close()

    def test_review_edit_then_approve(self):
        """Caller can edit a pending from review=True and then approve."""
        t = _make_compressible_tract()
        pending = t.compress(content="initial draft", review=True)
        pending.edit_summary(0, "Polished summary")
        result = pending.approve()

        assert isinstance(result, CompressResult)

        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "Polished summary" in messages_text
        t.close()


# ===========================================================================
# 11. Backward Compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Tests that existing compress behavior is preserved."""

    def test_compress_with_content_no_hooks(self):
        """t.compress(content='summary') works with no hooks (backward compat)."""
        t = _make_compressible_tract()
        result = t.compress(content="summary text")
        assert isinstance(result, CompressResult)
        assert len(result.summary_commits) >= 1
        assert len(result.source_commits) >= 1
        t.close()

    def test_compress_content_updates_head(self):
        """compress with manual content updates HEAD."""
        t = _make_compressible_tract()
        old_head = t.head
        result = t.compress(content="summary")
        assert t.head != old_head
        assert t.head == result.new_head
        t.close()

    def test_compress_content_in_compiled_output(self):
        """Manual summary text appears in compiled output after compress."""
        t = _make_compressible_tract()
        t.compress(content="A concise summary of the conversation.")
        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "concise summary" in messages_text
        t.close()

    def test_compress_review_true_returns_pending_compress(self):
        """review=True returns PendingCompress."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert isinstance(pending, PendingCompress)
        assert type(pending).__name__ == "PendingCompress"
        t.close()

    def test_compress_result_fields(self):
        """CompressResult from hook-aware compress has all expected fields."""
        t = _make_compressible_tract()
        result = t.compress(content="summary")

        assert isinstance(result, CompressResult)
        assert isinstance(result.compression_id, str)
        assert isinstance(result.original_tokens, int)
        assert isinstance(result.compressed_tokens, int)
        assert isinstance(result.source_commits, tuple)
        assert isinstance(result.summary_commits, tuple)
        assert isinstance(result.preserved_commits, tuple)
        assert isinstance(result.compression_ratio, float)
        assert isinstance(result.new_head, str)
        t.close()


# ===========================================================================
# 12. Edge Cases and Integration
# ===========================================================================


class TestEdgeCases:
    """Edge cases and integration scenarios for the hook system."""

    def test_off_nonexistent_hook_is_noop(self):
        """t.off() for a non-registered operation should not raise."""
        t = Tract.open(":memory:")
        # Should not raise
        t.off("compress")
        t.close()

    def test_hooks_dict_is_copy(self):
        """t.hooks returns a copy -- mutating it doesn't affect registrations."""
        t = Tract.open(":memory:")
        handler = lambda p: p.approve()
        t.on("compress", handler)

        hooks = t.hooks
        hooks.pop("compress", None)  # Mutate the returned dict

        # Original registration should be unaffected
        assert "compress" in t.hooks
        t.close()

    def test_hook_receives_pending_with_correct_operation(self):
        """Hook handler receives PendingCompress (correct subclass for compress)."""
        t = _make_compressible_tract()
        received_types = []

        def type_checker(pending):
            received_types.append(type(pending).__name__)
            pending.approve()

        t.on("compress", type_checker)
        t.compress(content="summary")

        assert received_types == ["PendingCompress"]
        t.close()

    def test_multiple_compress_calls_with_hook(self):
        """Multiple compress calls each fire the hook independently."""
        t = _make_larger_tract(4)
        call_count = []

        def counting_hook(pending):
            call_count.append(1)
            pending.approve()

        t.on("compress", counting_hook)

        # First compress
        t.compress(content="first summary")

        # Add more content and compress again
        t.user("Follow-up question")
        t.assistant("Follow-up answer")
        t.compress(content="second summary")

        assert len(call_count) == 2
        t.close()

    def test_hook_handler_exception_propagates(self):
        """If a hook handler raises an exception, it propagates to the caller."""
        t = _make_compressible_tract()

        def broken_hook(pending):
            raise RuntimeError("Handler crashed!")

        t.on("compress", broken_hook)

        with pytest.raises(RuntimeError, match="Handler crashed"):
            t.compress(content="summary")
        t.close()

    def test_apply_decision_reject(self):
        """apply_decision({'action': 'reject', 'reason': 'bad'}) calls reject()."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.apply_decision({"action": "reject", "args": {"reason": "not acceptable"}})
        assert pending.status == "rejected"
        assert pending.rejection_reason == "not acceptable"
        t.close()

    def test_execute_tool_approve(self):
        """execute_tool('approve', {}) calls approve()."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        pending.execute_tool("approve", {})
        assert pending.status == "approved"
        t.close()

    def test_execute_tool_edit_summary(self):
        """execute_tool('edit_summary', {'index': 0, 'new_text': '...'}) works."""
        t = _make_compressible_tract()
        pending = t.compress(content="original", review=True)
        pending.execute_tool("edit_summary", {"index": 0, "new_text": "tool-edited"})
        assert pending.summaries[0] == "tool-edited"
        t.close()

    def test_hook_with_tract_context_manager(self):
        """Hooks work correctly when Tract is used as context manager."""
        with Tract.open(":memory:") as t:
            t.system("sys")
            t.user("hello")
            t.assistant("hi")

            fired = []

            def hook(pending):
                fired.append(True)
                pending.approve()

            t.on("compress", hook)
            result = t.compress(content="summary")

            assert isinstance(result, CompressResult)
            assert len(fired) == 1


# ===========================================================================
# 13. PendingCompress Isinstance Checks
# ===========================================================================


class TestPendingCompressInheritance:
    """Verify PendingCompress is a proper subclass of Pending."""

    def test_pending_compress_is_subclass_of_pending(self):
        """PendingCompress inherits from Pending."""
        assert issubclass(PendingCompress, Pending)

    def test_pending_compress_instance_is_pending(self):
        """A PendingCompress instance passes isinstance check for Pending."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        assert isinstance(pending, Pending)
        assert isinstance(pending, PendingCompress)
        t.close()
