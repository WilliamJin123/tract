"""Tests for hook pass_through() middleware and handler ordering.

Tests cover:
- pass_through() allows next handler to fire
- pass_through() auto-approves when last handler
- pass_through + approve/reject combinations
- pass_through logged in hook_log
- _check_resolved guard with pass_through
- Handler ordering: before, after, at parameters
- Named handlers and duplicate detection
- off() by name
- hook_names property
"""

from __future__ import annotations

import warnings

import pytest

from tract import Tract
from tract.hooks.event import HookEvent
from tract.hooks.pending import PendingStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compressible_tract() -> Tract:
    """Create an in-memory Tract with 3 commits (system + user + assistant)."""
    t = Tract.open(":memory:")
    t.system("You are a helpful assistant.")
    t.user("Hello, how are you?")
    t.assistant("I'm doing well, thank you!")
    return t


def _make_gc_tract() -> Tract:
    """Create a tract with orphaned commits suitable for gc()."""
    t = Tract.open(":memory:")
    t.system("System prompt")
    t.user("User message 1")
    t.assistant("Assistant reply 1")
    t.compress(content="Summary of conversation")
    return t


# ===========================================================================
# pass_through tests
# ===========================================================================


class TestPassThrough:
    def test_pass_through_next_handler_fires(self):
        """Handler calls pass_through(), next handler fires."""
        t = _make_compressible_tract()
        call_order = []

        def handler_a(p):
            call_order.append("a")
            p.pass_through()

        def handler_b(p):
            call_order.append("b")
            p.approve()

        t.on("compress", handler_a)
        t.on("compress", handler_b)
        t.compress(content="summary")

        assert call_order == ["a", "b"]
        t.close()

    def test_last_handler_pass_through_auto_approves(self):
        """Last handler calls pass_through(), pending is auto-approved."""
        t = _make_compressible_tract()
        call_order = []

        def handler_a(p):
            call_order.append("a")
            p.pass_through()

        t.on("compress", handler_a)
        t.compress(content="summary")

        assert call_order == ["a"]
        # The compress should have succeeded (auto-approved)
        ctx = t.compile()
        # After compress, messages should be reduced
        assert len(ctx.messages) > 0
        t.close()

    def test_first_pass_through_second_approves(self):
        """First handler passes through, second approves."""
        t = _make_compressible_tract()
        call_order = []

        def handler_a(p):
            call_order.append("a")
            p.pass_through()

        def handler_b(p):
            call_order.append("b")
            p.approve()

        t.on("compress", handler_a)
        t.on("compress", handler_b)
        t.compress(content="summary")

        assert call_order == ["a", "b"]
        t.close()

    def test_first_pass_through_second_rejects(self):
        """First handler passes through, second rejects."""
        t = _make_compressible_tract()
        call_order = []

        def handler_a(p):
            call_order.append("a")
            p.pass_through()

        def handler_b(p):
            call_order.append("b")
            p.reject("not today")

        t.on("compress", handler_a)
        t.on("compress", handler_b)
        result = t.compress(content="summary")

        assert call_order == ["a", "b"]
        # The result should be the pending (rejected)
        assert result is not None
        t.close()

    def test_pass_through_logged_in_hook_log(self):
        """pass_through is logged in hook_log with result='passed_through'."""
        t = _make_compressible_tract()

        def handler_a(p):
            p.pass_through()

        def handler_b(p):
            p.approve()

        t.on("compress", handler_a)
        t.on("compress", handler_b)
        t.compress(content="summary")

        log = t.hook_log
        compress_events = [e for e in log if e.operation == "compress"]
        # Should have at least two events: passed_through + approved
        assert len(compress_events) >= 2
        assert compress_events[-2].result == "passed_through"
        assert compress_events[-2].handler_name == "handler_a"
        assert compress_events[-1].result == "approved"
        assert compress_events[-1].handler_name == "handler_b"
        t.close()

    def test_cannot_pass_through_after_approve(self):
        """Cannot call pass_through after approve (guard)."""
        t = _make_compressible_tract()

        def handler(p):
            p.approve()
            # This should raise because status is already approved
            with pytest.raises(RuntimeError, match="Cannot modify"):
                p.pass_through()

        t.on("compress", handler)
        t.compress(content="summary")
        t.close()

    def test_approve_stops_iteration_no_pass_through(self):
        """First handler approve() stops iteration; second never fires."""
        t = _make_compressible_tract()
        call_order = []

        def handler_a(p):
            call_order.append("a")
            p.approve()

        def handler_b(p):
            call_order.append("b")
            p.pass_through()

        t.on("compress", handler_a)
        t.on("compress", handler_b)
        t.compress(content="summary")

        assert call_order == ["a"]
        t.close()


# ===========================================================================
# Handler ordering tests
# ===========================================================================


class TestHandlerOrdering:
    def test_default_appends_in_order(self):
        """Default: handlers append in registration order."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)

        assert t.hook_names["compress"] == ["h1", "h2"]
        t.close()

    def test_before_true_prepends(self):
        """before=True: prepends handler."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2, before=True)

        assert t.hook_names["compress"] == ["h2", "h1"]
        t.close()

    def test_before_name_inserts_before(self):
        """before='name': inserts before named handler."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        def h3(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)
        t.on("compress", h3, before="h2")

        assert t.hook_names["compress"] == ["h1", "h3", "h2"]
        t.close()

    def test_after_name_inserts_after(self):
        """after='name': inserts after named handler."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        def h3(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)
        t.on("compress", h3, after="h1")

        assert t.hook_names["compress"] == ["h1", "h3", "h2"]
        t.close()

    def test_at_zero_inserts_at_beginning(self):
        """at=0: inserts at beginning."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2, at=0)

        assert t.hook_names["compress"] == ["h2", "h1"]
        t.close()

    def test_at_n_inserts_at_specific_index(self):
        """at=N: inserts at specific index."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        def h3(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)
        t.on("compress", h3, at=1)

        assert t.hook_names["compress"] == ["h1", "h3", "h2"]
        t.close()

    def test_name_parameter_used(self):
        """name= is used; falls back to __name__."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        t.on("compress", h1, name="custom_name")
        t.on("compress", lambda p: p.approve())  # falls back to __name__ -> <lambda>

        names = t.hook_names["compress"]
        assert names[0] == "custom_name"
        assert names[1] == "<lambda>"
        t.close()

    def test_duplicate_name_raises(self):
        """Duplicate name raises ValueError."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1, name="my_handler")
        with pytest.raises(ValueError, match="already registered"):
            t.on("compress", h2, name="my_handler")
        t.close()

    def test_before_nonexistent_raises(self):
        """before='nonexistent' raises ValueError."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        t.on("compress", h1)
        with pytest.raises(ValueError, match="No handler named"):
            t.on("compress", lambda p: None, name="h2", before="nonexistent")
        t.close()

    def test_after_nonexistent_raises(self):
        """after='nonexistent' raises ValueError."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        t.on("compress", h1)
        with pytest.raises(ValueError, match="No handler named"):
            t.on("compress", lambda p: None, name="h2", after="nonexistent")
        t.close()

    def test_at_out_of_range_raises(self):
        """at=999 raises IndexError."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        t.on("compress", h1)
        with pytest.raises(IndexError, match="out of range"):
            t.on("compress", lambda p: None, name="h2", at=999)
        t.close()

    def test_multiple_positioning_args_raises(self):
        """Multiple positioning args raises ValueError."""
        t = Tract.open(":memory:")

        with pytest.raises(ValueError, match="Only one of"):
            t.on("compress", lambda p: None, name="h1", before=True, at=0)
        t.close()


# ===========================================================================
# off() by name + hook_names property
# ===========================================================================


class TestOffByNameAndHookNames:
    def test_off_by_name_string(self):
        """off('compress', 'name_string') removes by name."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)

        assert t.hook_names["compress"] == ["h1", "h2"]

        t.off("compress", "h1")
        assert t.hook_names["compress"] == ["h2"]
        t.close()

    def test_hook_names_property(self):
        """hook_names property returns correct names."""
        t = Tract.open(":memory:")

        def handler_a(p):
            p.approve()

        def handler_b(p):
            p.approve()

        t.on("compress", handler_a, name="alpha")
        t.on("compress", handler_b, name="beta")
        t.on("gc", handler_a, name="gamma")

        names = t.hook_names
        assert names["compress"] == ["alpha", "beta"]
        assert names["gc"] == ["gamma"]
        t.close()

    def test_off_by_name_clears_key_when_empty(self):
        """off() by name removes key when list becomes empty."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        t.on("compress", h1)
        t.off("compress", "h1")
        assert "compress" not in t.hooks
        t.close()

    def test_off_by_callable_still_works(self):
        """off() by callable reference still works with _HookEntry."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)
        t.off("compress", h1)

        assert t.hook_names["compress"] == ["h2"]
        t.close()

    def test_before_false_is_ignored(self):
        """before=False is treated as None (appends)."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2, before=False)

        assert t.hook_names["compress"] == ["h1", "h2"]
        t.close()
