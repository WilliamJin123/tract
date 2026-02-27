"""Tests for hook stacking and observability.

Tests cover:
- Stacked handler registration and firing order
- Selective handler removal with off(op, handler)
- Catch-all stacking and priority
- HookEvent log recording
- print_hooks() method
- Recursion guard with stacked handlers
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
# 1. Single handler fires normally
# ===========================================================================


class TestSingleHandler:
    def test_single_handler_fires(self):
        """A single registered handler fires and resolves the pending."""
        t = _make_compressible_tract()
        calls = []

        def my_handler(p):
            calls.append("fired")
            p.approve()

        t.on("compress", my_handler)
        t.compress(content="summary")

        assert calls == ["fired"]
        t.close()


# ===========================================================================
# 2. Multiple handlers fire in registration order
# ===========================================================================


class TestMultipleHandlers:
    def test_fire_in_registration_order(self):
        """Multiple handlers fire in the order they were registered."""
        t = _make_compressible_tract()
        order = []

        def h1(p):
            order.append("h1")
            # Don't resolve -- let next handler run

        def h2(p):
            order.append("h2")
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)

        # Suppress unresolved warning for h1
        t.compress(content="summary")

        assert order == ["h1", "h2"]
        t.close()


# ===========================================================================
# 3. First handler that approves stops iteration
# ===========================================================================


class TestApproveStopsIteration:
    def test_first_approve_stops(self):
        """The first handler that calls approve() stops the chain."""
        t = _make_compressible_tract()
        calls = []

        def h1(p):
            calls.append("h1")
            p.approve()

        def h2(p):
            calls.append("h2")
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)
        t.compress(content="summary")

        assert calls == ["h1"]
        t.close()


# ===========================================================================
# 4. First handler that rejects stops iteration
# ===========================================================================


class TestRejectStopsIteration:
    def test_first_reject_stops(self):
        """The first handler that calls reject() stops the chain."""
        t = _make_compressible_tract()
        calls = []

        def h1(p):
            calls.append("h1")
            p.reject("nope")

        def h2(p):
            calls.append("h2")
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)

        result = t.compress(content="summary")
        # When rejected, compress returns the PendingCompress (not a CompressResult)
        assert result is not None
        assert calls == ["h1"]
        t.close()


# ===========================================================================
# 5. off(op, handler) removes specific handler
# ===========================================================================


class TestOffSpecificHandler:
    def test_remove_specific_handler(self):
        """off(op, handler) removes only that handler; others remain."""
        t = _make_compressible_tract()
        calls = []

        def h1(p):
            calls.append("h1")
            p.approve()

        def h2(p):
            calls.append("h2")
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)

        # Remove h1 specifically
        t.off("compress", h1)

        assert "compress" in t.hooks
        assert len(t.hooks["compress"]) == 1

        t.compress(content="summary")
        assert calls == ["h2"]
        t.close()

    def test_remove_last_handler_cleans_key(self):
        """off(op, handler) on the last handler removes the key entirely."""
        t = Tract.open(":memory:")

        handler = lambda p: p.approve()
        t.on("compress", handler)
        t.off("compress", handler)

        assert "compress" not in t.hooks
        t.close()

    def test_remove_nonexistent_handler_is_noop(self):
        """off(op, handler) for an unregistered handler is a no-op."""
        t = Tract.open(":memory:")

        def h1(p):
            p.approve()

        def h2(p):
            p.approve()

        t.on("compress", h1)
        t.off("compress", h2)  # h2 was never registered

        assert "compress" in t.hooks
        assert len(t.hooks["compress"]) == 1
        t.close()


# ===========================================================================
# 6. off(op) without handler clears all
# ===========================================================================


class TestOffClearsAll:
    def test_off_clears_all_handlers(self):
        """off(op) without a handler clears all handlers for that operation."""
        t = Tract.open(":memory:")

        t.on("compress", lambda p: p.approve())
        t.on("compress", lambda p: p.approve())
        t.on("compress", lambda p: p.approve())

        assert len(t.hooks["compress"]) == 3

        t.off("compress")
        assert "compress" not in t.hooks
        t.close()


# ===========================================================================
# 7. "*" catch-all works with stacking
# ===========================================================================


class TestCatchAllStacking:
    def test_catchall_fires_when_no_specific(self):
        """Catch-all handlers fire when no operation-specific handler exists."""
        t = _make_compressible_tract()
        calls = []

        def catch1(p):
            calls.append("catch1")

        def catch2(p):
            calls.append("catch2")
            p.approve()

        t.on("*", catch1)
        t.on("*", catch2)
        t.compress(content="summary")

        assert calls == ["catch1", "catch2"]
        t.close()


# ===========================================================================
# 8. Specific handlers take priority over "*"
# ===========================================================================


class TestSpecificOverCatchAll:
    def test_specific_takes_priority(self):
        """Specific handlers fire instead of catch-all handlers."""
        t = _make_compressible_tract()
        calls = []

        def specific(p):
            calls.append("specific")
            p.approve()

        def catchall(p):
            calls.append("catchall")
            p.approve()

        t.on("compress", specific)
        t.on("*", catchall)
        t.compress(content="summary")

        assert calls == ["specific"]
        t.close()


# ===========================================================================
# 9. hooks property returns dict[str, list[Callable]]
# ===========================================================================


class TestHooksProperty:
    def test_hooks_returns_list_values(self):
        """hooks property returns dict with list values."""
        t = Tract.open(":memory:")

        h1 = lambda p: p.approve()
        h2 = lambda p: p.approve()

        t.on("compress", h1)
        t.on("compress", h2)
        t.on("gc", h1)

        hooks = t.hooks
        assert isinstance(hooks, dict)
        assert isinstance(hooks["compress"], list)
        assert len(hooks["compress"]) == 2
        assert isinstance(hooks["gc"], list)
        assert len(hooks["gc"]) == 1
        t.close()

    def test_hooks_is_deep_copy(self):
        """Mutating the returned hooks dict/list doesn't affect internals."""
        t = Tract.open(":memory:")

        handler = lambda p: p.approve()
        t.on("compress", handler)

        hooks = t.hooks
        hooks["compress"].clear()  # Mutate the list
        hooks.pop("compress", None)  # Mutate the dict

        # Internal state unaffected
        assert "compress" in t.hooks
        assert len(t.hooks["compress"]) == 1
        t.close()


# ===========================================================================
# 10. hook_log records events for each firing
# ===========================================================================


class TestHookLogRecords:
    def test_hook_log_records_approved(self):
        """hook_log records an event when a handler approves."""
        t = _make_compressible_tract()

        def my_handler(p):
            p.approve()

        t.on("compress", my_handler)
        t.compress(content="summary")

        log = t.hook_log
        assert len(log) >= 1
        evt = log[-1]
        assert isinstance(evt, HookEvent)
        assert evt.operation == "compress"
        assert evt.handler_name == "my_handler"
        assert evt.resolved is True
        assert evt.result == "approved"
        t.close()

    def test_hook_log_records_rejected(self):
        """hook_log records an event when a handler rejects."""
        t = _make_compressible_tract()

        def my_rejecter(p):
            p.reject("no")

        t.on("compress", my_rejecter)
        t.compress(content="summary")

        log = t.hook_log
        assert len(log) >= 1
        evt = log[-1]
        assert evt.result == "rejected"
        assert evt.handler_name == "my_rejecter"
        t.close()

    def test_hook_log_records_unresolved(self):
        """hook_log records an unresolved event when handler doesn't resolve."""
        t = _make_compressible_tract()

        def lazy_handler(p):
            pass  # Does nothing

        t.on("compress", lazy_handler)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t.compress(content="summary")

        log = t.hook_log
        assert len(log) >= 1
        evt = log[-1]
        assert evt.resolved is False
        assert evt.result == "unresolved"
        assert evt.handler_name == "lazy_handler"
        t.close()


# ===========================================================================
# 11. hook_log records auto-approved when no handler
# ===========================================================================


class TestHookLogAutoApproved:
    def test_auto_approved_logged(self):
        """hook_log records auto-approved when no handler is registered."""
        t = _make_compressible_tract()

        # No hooks registered; compress auto-approves
        t.compress(content="summary")

        log = t.hook_log
        assert len(log) >= 1
        evt = log[-1]
        assert evt.result == "auto-approved"
        assert evt.handler_name == "(none)"
        assert evt.resolved is True
        t.close()


# ===========================================================================
# 12. hook_log records recursion guard skips
# ===========================================================================


class TestHookLogRecursionGuard:
    def test_recursion_guard_logged(self):
        """hook_log records a 'skipped' event from the recursion guard."""
        t = Tract.open(":memory:")
        t.system("System prompt")
        t.user("User message 1")
        t.assistant("Reply 1")
        t.user("User message 2")
        t.assistant("Reply 2")
        t.user("User message 3")
        t.assistant("Reply 3")

        def recursive_handler(p):
            # This handler triggers gc() inside a compress hook,
            # which triggers the recursion guard on the gc path.
            t.gc(orphan_retention_days=0)
            p.approve()

        t.on("compress", recursive_handler)
        t.compress(content="outer summary")

        # Should have at least two events: gc skipped + compress approved
        log = t.hook_log
        skipped_events = [e for e in log if e.result == "skipped"]
        assert len(skipped_events) >= 1
        assert skipped_events[0].handler_name == "(recursion guard)"
        t.close()


# ===========================================================================
# 13. print_hooks() runs without error
# ===========================================================================


class TestPrintHooks:
    def test_print_hooks_no_crash(self, capsys):
        """print_hooks() runs without error and produces output."""
        t = _make_compressible_tract()

        def my_handler(p):
            p.approve()

        t.on("compress", my_handler)
        t.compress(content="summary")

        t.print_hooks()
        captured = capsys.readouterr()
        assert "Registered Hooks" in captured.out
        assert "Hook Log" in captured.out
        assert "compress" in captured.out
        t.close()

    def test_print_hooks_empty(self, capsys):
        """print_hooks() with no hooks shows '(none)'."""
        t = Tract.open(":memory:")
        t.print_hooks()
        captured = capsys.readouterr()
        assert "(none)" in captured.out
        t.close()


# ===========================================================================
# 14. Recursion guard still works with stacked handlers
# ===========================================================================


class TestRecursionGuardWithStacking:
    def test_recursion_guard_applies_to_stack(self):
        """Recursion guard auto-approves even when multiple handlers stacked."""
        t = Tract.open(":memory:")
        t.system("System prompt")
        t.user("User message 1")
        t.assistant("Reply 1")
        t.user("User message 2")
        t.assistant("Reply 2")

        outer_calls = []

        def h1(p):
            outer_calls.append("h1")
            # This inner gc triggers the recursion guard
            t.gc(orphan_retention_days=0)
            p.approve()

        def h2(p):
            outer_calls.append("h2")
            # Should never fire because h1 already approved
            p.approve()

        t.on("compress", h1)
        t.on("compress", h2)
        t.compress(content="outer")

        # Only h1 should fire for the outer call
        assert outer_calls == ["h1"]

        # The gc inside h1 should have triggered the recursion guard
        log = t.hook_log
        skipped = [e for e in log if e.result == "skipped"]
        assert len(skipped) >= 1
        t.close()
