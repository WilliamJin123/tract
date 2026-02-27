"""Tests for Phase D: pprint() enhancements and __repr__ across all Pending subclasses.

Tests __repr__ format for each subclass, pprint() runs without crash,
verbose mode, compress token ratio display, merge conflict display,
and guidance panel rendering.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from tract import Tract
from tract.hooks.pending import Pending
from tract.hooks.compress import PendingCompress
from tract.hooks.merge import PendingMerge
from tract.hooks.gc import PendingGC
from tract.hooks.rebase import PendingRebase
from tract.hooks.tool_result import PendingToolResult
from tract.hooks.policy import PendingPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tract() -> Tract:
    """Create an in-memory Tract with a few commits."""
    t = Tract.open(":memory:")
    t.system("You are a helpful assistant.")
    t.user("Hello!")
    t.assistant("Hi there!")
    return t


def _capture_pprint(pending, *, verbose: bool = False) -> str:
    """Capture pprint output by monkeypatching Console creation.

    Since pprint() creates its own Console, we monkeypatch the Rich Console
    class to redirect output to a StringIO buffer.
    """
    import tract.hooks.pending as pending_mod

    buf = StringIO()
    original_console = Console

    class CapturingConsole(Console):
        def __init__(self, **kwargs):
            kwargs["file"] = buf
            kwargs["force_terminal"] = False
            kwargs["no_color"] = True
            super().__init__(**kwargs)

    pending_mod.Console = CapturingConsole  # type: ignore[attr-defined]
    try:
        # We need to also patch it in the module's pprint import scope
        # Since pprint imports Console inside the method, we patch the
        # rich.console module
        import rich.console
        original_rc = rich.console.Console
        rich.console.Console = CapturingConsole  # type: ignore[misc]
        try:
            pending.pprint(verbose=verbose)
        finally:
            rich.console.Console = original_rc
    finally:
        pending_mod.Console = original_console  # type: ignore[attr-defined]

    return buf.getvalue()


# ===========================================================================
# 1. __repr__ Tests
# ===========================================================================


class TestPendingRepr:
    """Tests for __repr__() on the base Pending class."""

    def test_repr_format(self):
        t = _make_tract()
        p = Pending(operation="test_op", tract=t)
        r = repr(p)
        assert "<Pending: test_op, pending, id=" in r
        assert len(p.pending_id[:8]) == 8
        t.close()

    def test_repr_with_different_status(self):
        t = _make_tract()
        p = Pending(operation="test_op", tract=t, status="approved")
        r = repr(p)
        assert "approved" in r
        t.close()


class TestPendingCompressRepr:
    """Tests for __repr__() on PendingCompress."""

    def test_repr_with_reduction(self):
        t = _make_tract()
        p = PendingCompress(
            operation="compress", tract=t,
            summaries=["summary one", "summary two"],
            original_tokens=1000, estimated_tokens=300,
        )
        r = repr(p)
        assert "<PendingCompress:" in r
        assert "2 summaries" in r
        assert "1000->300 tokens" in r
        assert "70% reduction" in r
        assert "pending>" in r
        t.close()

    def test_repr_zero_original_tokens(self):
        t = _make_tract()
        p = PendingCompress(
            operation="compress", tract=t,
            summaries=[], original_tokens=0, estimated_tokens=0,
        )
        r = repr(p)
        assert "<PendingCompress:" in r
        assert "0 summaries" in r
        # No reduction percentage when original_tokens is 0
        assert "% reduction" not in r
        t.close()


class TestPendingMergeRepr:
    """Tests for __repr__() on PendingMerge."""

    def test_repr_format(self):
        t = _make_tract()
        p = PendingMerge(
            operation="merge", tract=t,
            source_branch="feature", target_branch="main",
            conflicts=["c1", "c2", "c3"],
            resolutions={"c1": "resolved"},
        )
        r = repr(p)
        assert "<PendingMerge:" in r
        assert "feature->main" in r
        assert "1/3 resolved" in r
        assert "pending>" in r
        t.close()


class TestPendingGCRepr:
    """Tests for __repr__() on PendingGC."""

    def test_repr_format(self):
        t = _make_tract()
        p = PendingGC(
            operation="gc", tract=t,
            commits_to_remove=["abc123", "def456"],
            tokens_to_free=500,
        )
        r = repr(p)
        assert "<PendingGC:" in r
        assert "2 commits" in r
        assert "~500 tokens" in r
        assert "pending>" in r
        t.close()


class TestPendingRebaseRepr:
    """Tests for __repr__() on PendingRebase."""

    def test_repr_format(self):
        t = _make_tract()
        p = PendingRebase(
            operation="rebase", tract=t,
            replay_plan=["abc123"],
            target_base="def456789abcdef0",
        )
        r = repr(p)
        assert "<PendingRebase:" in r
        assert "1 commits onto def45678..." in r
        assert "pending>" in r
        t.close()

    def test_repr_empty_target_base(self):
        t = _make_tract()
        p = PendingRebase(
            operation="rebase", tract=t,
            replay_plan=[], target_base="",
        )
        r = repr(p)
        assert "???" in r
        t.close()


class TestPendingToolResultRepr:
    """Tests for __repr__() on PendingToolResult."""

    def test_repr_format(self):
        t = _make_tract()
        p = PendingToolResult(
            operation="tool_result", tract=t,
            tool_name="search", token_count=100,
        )
        r = repr(p)
        assert "<PendingToolResult:" in r
        assert "search" in r
        assert "100 tokens" in r
        assert "pending>" in r
        t.close()


class TestPendingPolicyRepr:
    """Tests for __repr__() on PendingPolicy."""

    def test_repr_format(self):
        t = _make_tract()
        p = PendingPolicy(
            operation="policy", tract=t,
            policy_name="auto_compress", action_type="compress",
        )
        r = repr(p)
        assert "<PendingPolicy:" in r
        assert "auto_compress" in r
        assert "compress" in r
        assert "pending>" in r
        t.close()


# ===========================================================================
# 2. pprint() Tests -- no-crash and output verification
# ===========================================================================


class TestPprintNoCrash:
    """Verify pprint() runs without exceptions on every subclass."""

    def test_pending_pprint(self):
        t = _make_tract()
        p = Pending(operation="test_op", tract=t)
        output = _capture_pprint(p)
        assert "test_op" in output
        t.close()

    def test_compress_pprint(self):
        t = _make_tract()
        p = PendingCompress(
            operation="compress", tract=t,
            summaries=["summary text"],
            original_tokens=1000, estimated_tokens=300,
        )
        output = _capture_pprint(p)
        assert "compress" in output
        t.close()

    def test_merge_pprint(self):
        t = _make_tract()
        p = PendingMerge(
            operation="merge", tract=t,
            source_branch="feature", target_branch="main",
            conflicts=["c1"], resolutions={},
        )
        output = _capture_pprint(p)
        assert "merge" in output
        t.close()

    def test_gc_pprint(self):
        t = _make_tract()
        p = PendingGC(
            operation="gc", tract=t,
            commits_to_remove=["abc123"], tokens_to_free=500,
        )
        output = _capture_pprint(p)
        assert "gc" in output
        t.close()

    def test_rebase_pprint(self):
        t = _make_tract()
        p = PendingRebase(
            operation="rebase", tract=t,
            replay_plan=["abc123"], target_base="def456",
        )
        output = _capture_pprint(p)
        assert "rebase" in output
        t.close()

    def test_tool_result_pprint(self):
        t = _make_tract()
        p = PendingToolResult(
            operation="tool_result", tract=t,
            tool_name="search", token_count=100,
        )
        output = _capture_pprint(p)
        assert "tool_result" in output
        t.close()

    def test_policy_pprint(self):
        t = _make_tract()
        p = PendingPolicy(
            operation="policy", tract=t,
            policy_name="auto_compress", action_type="compress",
        )
        output = _capture_pprint(p)
        assert "policy" in output
        t.close()


class TestPprintVerbose:
    """Verify pprint(verbose=True) runs without exceptions."""

    def test_pending_verbose(self):
        t = _make_tract()
        p = Pending(operation="test_op", tract=t)
        output = _capture_pprint(p, verbose=True)
        assert "test_op" in output
        t.close()

    def test_compress_verbose(self):
        t = _make_tract()
        p = PendingCompress(
            operation="compress", tract=t,
            summaries=["This is a summary that is long enough to test truncation " * 3],
            original_tokens=1000, estimated_tokens=300,
        )
        output = _capture_pprint(p, verbose=True)
        assert "Summary previews" in output
        t.close()


# ===========================================================================
# 3. Compress pprint details
# ===========================================================================


class TestCompressPprintDetails:
    """Verify PendingCompress._pprint_details() shows token ratio and summaries."""

    def test_token_ratio_shown(self):
        t = _make_tract()
        p = PendingCompress(
            operation="compress", tract=t,
            summaries=["summary"],
            original_tokens=1000, estimated_tokens=300,
        )
        output = _capture_pprint(p)
        assert "1000" in output
        assert "300" in output
        assert "70%" in output
        t.close()

    def test_summary_previews_in_verbose(self):
        t = _make_tract()
        short_summary = "A short summary."
        long_summary = "X" * 200
        p = PendingCompress(
            operation="compress", tract=t,
            summaries=[short_summary, long_summary],
            original_tokens=500, estimated_tokens=200,
        )
        output = _capture_pprint(p, verbose=True)
        assert "A short summary." in output
        # Long summary should be truncated with ...
        assert "..." in output
        t.close()

    def test_guidance_panel_shown(self):
        t = _make_tract()
        p = PendingCompress(
            operation="compress", tract=t,
            summaries=["summary"],
            original_tokens=1000, estimated_tokens=300,
            guidance="Focus on key decisions and outcomes.",
        )
        output = _capture_pprint(p)
        assert "Guidance" in output
        assert "Focus on key decisions" in output
        t.close()


# ===========================================================================
# 4. Merge pprint details
# ===========================================================================


class TestMergePprintDetails:
    """Verify PendingMerge._pprint_details() shows branch info and conflicts."""

    def test_branch_info_shown(self):
        t = _make_tract()
        p = PendingMerge(
            operation="merge", tract=t,
            source_branch="feature", target_branch="main",
            conflicts=[], resolutions={},
        )
        output = _capture_pprint(p)
        assert "feature" in output
        assert "main" in output
        t.close()

    def test_conflict_count_shown(self):
        t = _make_tract()
        p = PendingMerge(
            operation="merge", tract=t,
            source_branch="feature", target_branch="main",
            conflicts=["hash1", "hash2", "hash3"],
            resolutions={"hash1": "resolved content"},
        )
        output = _capture_pprint(p)
        # Should show 3 conflicts total
        assert "3" in output
        t.close()

    def test_guidance_panel_on_merge(self):
        t = _make_tract()
        p = PendingMerge(
            operation="merge", tract=t,
            source_branch="feature", target_branch="main",
            conflicts=[], resolutions={},
            guidance="Prefer the incoming version for all conflicts.",
        )
        output = _capture_pprint(p)
        assert "Guidance" in output
        assert "Prefer the incoming version" in output
        t.close()
