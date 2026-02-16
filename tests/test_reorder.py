"""Tests for compile-time commit reordering and safety checks.

Tests that compile(order=...) reorders messages without mutating the DAG,
that structural safety checks detect EDIT-before-target and broken
response chains, and that reordered compiles bypass the cache.
"""

from __future__ import annotations

import pytest

from tract import (
    CommitNotFoundError,
    CommitOperation,
    DialogueContent,
    ReorderWarning,
    Tract,
)
from tract.operations.compression import check_reorder_safety


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from tests.conftest import make_tract_with_commits


# ===========================================================================
# 1. Basic reordering tests
# ===========================================================================


class TestBasicReordering:
    """Tests for compile(order=...) reordering of compiled messages."""

    def test_compile_default_order(self):
        """compile() without order returns commits in chain order."""
        t, hashes = make_tract_with_commits(3)

        result = t.compile()

        assert len(result.messages) == 3
        assert result.commit_hashes == hashes

    def test_compile_reversed_order(self):
        """compile(order=[h3, h2, h1]) returns messages in reversed order."""
        t, hashes = make_tract_with_commits(3)

        reversed_order = list(reversed(hashes))
        result, warnings = t.compile(order=reversed_order)

        assert len(result.messages) == 3
        # Messages should be in reversed order
        assert result.messages[0].content == "Message 3"
        assert result.messages[1].content == "Message 2"
        assert result.messages[2].content == "Message 1"
        assert result.commit_hashes == reversed_order

    def test_compile_partial_order(self):
        """compile(order=[h3, h1]) puts h3 first, h1 second, h2 at end.

        Partial ordering: commits not in `order` are appended at their
        original relative positions after the ordered ones.
        """
        t, hashes = make_tract_with_commits(3)

        partial_order = [hashes[2], hashes[0]]
        result, warnings = t.compile(order=partial_order)

        assert len(result.messages) == 3
        # h3 first, h1 second, h2 (not in order) appended last
        assert result.messages[0].content == "Message 3"
        assert result.messages[1].content == "Message 1"
        assert result.messages[2].content == "Message 2"

    def test_compile_order_preserves_content(self):
        """Reordered messages have same content, just different positions."""
        t, hashes = make_tract_with_commits(3)

        original = t.compile()
        reordered, _ = t.compile(order=list(reversed(hashes)))

        # Same set of contents
        original_contents = {m.content for m in original.messages}
        reordered_contents = {m.content for m in reordered.messages}
        assert original_contents == reordered_contents

    def test_compile_order_recounts_tokens(self):
        """Token count is recalculated for the new message order."""
        t, hashes = make_tract_with_commits(3)

        original = t.compile()
        reordered, _ = t.compile(order=list(reversed(hashes)))

        # Token counts should be the same since same messages, just reordered
        assert reordered.token_count == original.token_count


# ===========================================================================
# 2. Safety checks tests (direct function + compile integration)
# ===========================================================================


class TestSafetyChecks:
    """Tests for reorder safety check warnings.

    Safety checks operate on the commit DB, so they can detect issues
    with any commit hash list -- including EDIT commits. The compile()
    integration tests use compiled-output hashes (which exclude EDITs
    since the compiler resolves them). Direct function tests can test
    EDIT scenarios.
    """

    def test_safety_edit_before_target_direct(self):
        """Direct: EDIT before its target produces edit_before_target warning."""
        t = Tract.open()
        c1 = t.commit(DialogueContent(role="user", text="Original"))
        c2 = t.commit(
            DialogueContent(role="user", text="Edited"),
            operation=CommitOperation.EDIT,
            response_to=c1.commit_hash,
        )

        # Call check_reorder_safety directly with EDIT before target
        warnings = check_reorder_safety(
            [c2.commit_hash, c1.commit_hash],
            t._commit_repo,
            t._blob_repo,
        )

        edit_warnings = [w for w in warnings if w.warning_type == "edit_before_target"]
        assert len(edit_warnings) == 1
        assert edit_warnings[0].commit_hash == c2.commit_hash
        assert edit_warnings[0].severity == "structural"

    def test_safety_response_chain_break_direct(self):
        """Direct: response_to target not in order produces chain_break warning."""
        t = Tract.open()
        c1 = t.commit(DialogueContent(role="user", text="Original"))
        c2 = t.commit(
            DialogueContent(role="user", text="Edited"),
            operation=CommitOperation.EDIT,
            response_to=c1.commit_hash,
        )

        # Only include c2 (edit), not c1 (target) -- chain break
        warnings = check_reorder_safety(
            [c2.commit_hash],
            t._commit_repo,
            t._blob_repo,
        )

        chain_breaks = [w for w in warnings if w.warning_type == "response_chain_break"]
        assert len(chain_breaks) == 1
        assert chain_breaks[0].commit_hash == c2.commit_hash

    def test_safety_no_warnings_normal_order(self):
        """Normal order of APPEND-only commits produces no warnings."""
        t, hashes = make_tract_with_commits(3)

        result, warnings = t.compile(order=hashes)

        assert warnings == []

    def test_safety_check_disabled(self):
        """check_safety=False suppresses all warnings even for problematic order."""
        t, hashes = make_tract_with_commits(3)

        # Reversed order with check_safety=False
        result, warnings = t.compile(
            order=list(reversed(hashes)),
            check_safety=False,
        )

        assert warnings == []

    def test_warnings_accessible(self):
        """compile(order=...) returns (CompiledContext, list[ReorderWarning]) tuple."""
        t, hashes = make_tract_with_commits(3)

        result_tuple = t.compile(order=hashes)

        assert isinstance(result_tuple, tuple)
        assert len(result_tuple) == 2
        result, warnings = result_tuple
        assert hasattr(result, "messages")
        assert isinstance(warnings, list)


# ===========================================================================
# 3. Edge cases
# ===========================================================================


class TestReorderEdgeCases:
    """Edge case tests for compile(order=...)."""

    def test_compile_order_invalid_hash(self):
        """compile(order=["nonexistent"]) raises CommitNotFoundError."""
        t, hashes = make_tract_with_commits(3)

        with pytest.raises(CommitNotFoundError):
            t.compile(order=["nonexistent_hash_abcdef"])

    def test_compile_order_empty_list(self):
        """compile(order=[]) returns all messages in original order."""
        t, hashes = make_tract_with_commits(3)

        result, warnings = t.compile(order=[])

        # Empty order = no reordering effect, all messages appended as remaining
        assert len(result.messages) == 3
        assert result.messages[0].content == "Message 1"
        assert result.messages[1].content == "Message 2"
        assert result.messages[2].content == "Message 3"
        assert warnings == []

    def test_compile_order_bypasses_cache(self):
        """compile(order=...) bypasses the compile cache."""
        t, hashes = make_tract_with_commits(3)

        # First compile to populate cache
        cached_result = t.compile()
        assert cached_result.commit_count == 3

        # compile(order=...) should bypass cache and still work
        result, warnings = t.compile(order=list(reversed(hashes)))

        # Messages should be reversed (not from cache)
        assert result.messages[0].content == "Message 3"
        assert result.messages[1].content == "Message 2"
        assert result.messages[2].content == "Message 1"
