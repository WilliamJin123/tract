"""Tests for retry() and validate() on PendingCompress and PendingMerge,
plus auto_retry() integration with both types.

Covers Phase A of the hook system deferred features:
- PendingCompress.validate(): passes, fails empty, fails too short, fails token ratio
- PendingCompress.retry(): replaces summary, recalculates tokens, rejects bad index
- PendingMerge.validate(): passes, fails missing resolution, fails empty resolution
- PendingMerge.retry(): re-resolves via mocked LLM
- auto_retry(): happy path (approve), exhausted retries (reject+HookRejection), merge path
- Edge cases: retry on already-approved pending, validate with no groups
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tract import Tract, DialogueContent, InstructionContent
from tract.hooks.compress import PendingCompress
from tract.hooks.merge import PendingMerge
from tract.hooks.retry import auto_retry
from tract.hooks.validation import HookRejection, ValidationResult
from tract.models.commit import CommitOperation


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


def _make_conflict_tract() -> tuple[Tract, str]:
    """Create a tract with two branches that produce a conflict.

    Creates a both_edit conflict: main and feature both EDIT the
    same base commit. Returns (tract, source_branch_name).
    """
    t = Tract.open(":memory:")

    # Base commit on main
    base = t.commit(InstructionContent(text="original"))
    base_hash = base.commit_hash

    # Create feature branch (at current HEAD = base)
    t.branch("feature")

    # Feature branch: EDIT the base commit
    t.switch("feature")
    t.commit(
        DialogueContent(role="assistant", text="feature edit"),
        operation=CommitOperation.EDIT,
        edit_target=base_hash,
    )

    # Main branch: EDIT the same base commit
    t.switch("main")
    t.commit(
        DialogueContent(role="assistant", text="main edit"),
        operation=CommitOperation.EDIT,
        edit_target=base_hash,
    )

    return t, "feature"


# ===========================================================================
# 1. PendingCompress.validate()
# ===========================================================================


class TestPendingCompressValidate:
    """Tests for PendingCompress.validate()."""

    def test_validate_passes_good_summaries(self):
        """validate() returns passed=True for good summaries."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="A well-formed summary of the conversation.", review=True)
            assert isinstance(pending, PendingCompress)
            result = pending.validate()
            assert result.passed is True
            assert result.diagnosis is None
            assert result.index is None
        finally:
            t.close()

    def test_validate_fails_empty_summary(self):
        """validate() fails on an empty summary."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="placeholder", review=True)
            assert isinstance(pending, PendingCompress)
            # Replace with empty
            pending.summaries[0] = ""
            result = pending.validate()
            assert result.passed is False
            assert result.index == 0
            assert "empty" in result.diagnosis.lower()
        finally:
            t.close()

    def test_validate_fails_too_short_summary(self):
        """validate() fails on a suspiciously short summary (< 10 chars)."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="placeholder", review=True)
            assert isinstance(pending, PendingCompress)
            # Replace with trivially short text
            pending.summaries[0] = "hi"
            result = pending.validate()
            assert result.passed is False
            assert result.index == 0
            assert "short" in result.diagnosis.lower()
        finally:
            t.close()

    def test_validate_fails_token_ratio(self):
        """validate() fails if summary exceeds target_tokens * 1.5."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="placeholder", review=True)
            assert isinstance(pending, PendingCompress)
            # Set a very low target so any real text exceeds it
            pending._target_tokens = 2
            pending.summaries[0] = "This is a summary that definitely has more than three tokens in it."
            result = pending.validate()
            assert result.passed is False
            assert result.index == 0
            assert "token" in result.diagnosis.lower()
        finally:
            t.close()

    def test_validate_passes_within_token_ratio(self):
        """validate() passes if summary is within target_tokens * 1.5."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="placeholder", review=True)
            assert isinstance(pending, PendingCompress)
            # Set a generous target
            pending._target_tokens = 10000
            pending.summaries[0] = "A reasonable summary that fits within the budget."
            result = pending.validate()
            assert result.passed is True
        finally:
            t.close()

    def test_validate_no_summaries_passes(self):
        """validate() with empty summaries list returns passed=True."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="placeholder", review=True)
            assert isinstance(pending, PendingCompress)
            pending.summaries = []
            result = pending.validate()
            assert result.passed is True
        finally:
            t.close()


# ===========================================================================
# 2. PendingCompress.retry()
# ===========================================================================


class TestPendingCompressRetry:
    """Tests for PendingCompress.retry()."""

    def test_retry_replaces_summary_at_index(self):
        """retry() replaces the summary at the given index with LLM output."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="old summary text", review=True)
            assert isinstance(pending, PendingCompress)

            original_summary = pending.summaries[0]

            # Mock the LLM call and compression internals
            # retry() does a deferred import from tract.operations.compression
            with patch(
                "tract.operations.compression._summarize_group",
                return_value="improved summary text",
            ), patch(
                "tract.operations.compression._build_messages_text",
                return_value="mock messages text",
            ):
                # Need _groups to be set for retry
                pending._groups = [["fake_commit"]]
                # Also need an LLM client mock
                mock_client = MagicMock()
                with patch.object(t, "_resolve_llm_client", return_value=mock_client):
                    pending.retry(0, guidance="Make it better")

            assert pending.summaries[0] == "improved summary text"
            assert pending.summaries[0] != original_summary
        finally:
            t.close()

    def test_retry_recalculates_estimated_tokens(self):
        """retry() recalculates estimated_tokens after replacing summary."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="old summary", review=True)
            assert isinstance(pending, PendingCompress)

            with patch(
                "tract.operations.compression._summarize_group",
                return_value="a much longer improved summary with more tokens inside",
            ), patch(
                "tract.operations.compression._build_messages_text",
                return_value="mock messages text",
            ):
                pending._groups = [["fake_commit"]]
                mock_client = MagicMock()
                with patch.object(t, "_resolve_llm_client", return_value=mock_client):
                    pending.retry(0)

            # Tokens should have been recalculated
            new_tokens = pending.estimated_tokens
            expected = t._token_counter.count_text("a much longer improved summary with more tokens inside")
            assert new_tokens == expected
        finally:
            t.close()

    def test_retry_bad_index_raises_index_error(self):
        """retry() raises IndexError for an out-of-range index."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="summary", review=True)
            assert isinstance(pending, PendingCompress)
            pending._groups = [["fake_commit"]]

            with pytest.raises(IndexError, match="out of range"):
                pending.retry(99)
        finally:
            t.close()

    def test_retry_negative_index_raises_index_error(self):
        """retry() raises IndexError for a negative index."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="summary", review=True)
            assert isinstance(pending, PendingCompress)
            pending._groups = [["fake_commit"]]

            with pytest.raises(IndexError, match="out of range"):
                pending.retry(-1)
        finally:
            t.close()

    def test_retry_no_groups_raises_runtime_error(self):
        """retry() raises RuntimeError if no groups are available."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="summary", review=True)
            assert isinstance(pending, PendingCompress)
            # _groups should be None for manual content compressions
            pending._groups = None

            with pytest.raises(RuntimeError, match="no compression groups"):
                pending.retry(0)
        finally:
            t.close()

    def test_retry_on_approved_raises(self):
        """retry() after approve() raises RuntimeError."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="summary", review=True)
            assert isinstance(pending, PendingCompress)
            pending.approve()

            with pytest.raises(RuntimeError):
                pending.retry(0)
        finally:
            t.close()


# ===========================================================================
# 3. PendingMerge.validate()
# ===========================================================================


class TestPendingMergeValidate:
    """Tests for PendingMerge.validate()."""

    def test_validate_passes_with_all_resolutions(self):
        """validate() passes when all conflicts have non-empty resolutions."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Merged content",
                    reasoning="Test",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            assert isinstance(pending, PendingMerge)
            result = pending.validate()
            assert result.passed is True
        finally:
            t.close()

    def test_validate_fails_missing_resolution(self):
        """validate() fails when a conflict has no resolution."""
        t, source = _make_conflict_tract()
        try:
            pending = t.merge(source, review=True)
            assert isinstance(pending, PendingMerge)
            assert len(pending.conflicts) > 0
            assert len(pending.resolutions) == 0

            result = pending.validate()
            assert result.passed is False
            assert "no resolution" in result.diagnosis.lower()
        finally:
            t.close()

    def test_validate_fails_empty_resolution(self):
        """validate() fails when a resolution is empty."""
        t, source = _make_conflict_tract()
        try:
            pending = t.merge(source, review=True)
            assert isinstance(pending, PendingMerge)

            # Set empty resolution for all conflicts
            for conflict in pending.conflicts:
                if conflict.target_hash:
                    pending.set_resolution(conflict.target_hash, "")

            result = pending.validate()
            assert result.passed is False
            assert "empty" in result.diagnosis.lower()
        finally:
            t.close()


# ===========================================================================
# 4. PendingMerge.retry()
# ===========================================================================


class TestPendingMergeRetry:
    """Tests for PendingMerge.retry()."""

    def test_retry_resolves_via_mocked_llm(self):
        """retry() re-resolves conflicts using a mocked LLM client."""
        t, source = _make_conflict_tract()
        try:
            pending = t.merge(source, review=True)
            assert isinstance(pending, PendingMerge)
            assert len(pending.resolutions) == 0

            # Mock the LLM client that _resolve_llm_client returns
            mock_client = MagicMock()
            mock_client.chat.return_value = {
                "choices": [{"message": {"content": "LLM resolved content"}}],
                "model": "test-model",
            }

            with patch.object(
                t, "_resolve_llm_client", return_value=mock_client
            ):
                pending.retry()

            # All conflicts should now have resolutions
            for conflict in pending.conflicts:
                if conflict.target_hash:
                    assert conflict.target_hash in pending.resolutions
                    assert pending.resolutions[conflict.target_hash] == "LLM resolved content"
        finally:
            t.close()

    def test_retry_updates_guidance(self):
        """retry() with guidance updates self.guidance and guidance_source."""
        t, source = _make_conflict_tract()
        try:
            pending = t.merge(source, review=True)
            assert isinstance(pending, PendingMerge)

            mock_client = MagicMock()
            mock_client.chat.return_value = {
                "choices": [{"message": {"content": "resolved"}}],
                "model": "test-model",
            }

            with patch.object(
                t, "_resolve_llm_client", return_value=mock_client
            ):
                pending.retry(guidance="Prefer feature branch content")

            assert pending.guidance == "Prefer feature branch content"
            assert pending.guidance_source == "user"
        finally:
            t.close()

    def test_retry_on_approved_raises(self):
        """retry() after approve raises RuntimeError."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Content",
                    reasoning="Test",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            assert isinstance(pending, PendingMerge)
            pending.approve()

            with pytest.raises(RuntimeError):
                pending.retry()
        finally:
            t.close()


# ===========================================================================
# 5. auto_retry() integration
# ===========================================================================


class TestAutoRetry:
    """Tests for auto_retry() with both PendingCompress and PendingMerge."""

    def test_auto_retry_compress_happy_path(self):
        """auto_retry() approves PendingCompress when validation passes."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(
                content="A thorough summary of the conversation that is long enough.",
                review=True,
            )
            assert isinstance(pending, PendingCompress)

            result = auto_retry(pending)
            # Should have been approved (validation passes for a good summary)
            assert pending.status == "approved"
            # Result should be a CompressResult, not a HookRejection
            assert not isinstance(result, HookRejection)
        finally:
            t.close()

    def test_auto_retry_compress_exhausted_retries(self):
        """auto_retry() rejects PendingCompress after exhausting retries."""
        t = _make_compressible_tract()
        try:
            pending = t.compress(content="placeholder", review=True)
            assert isinstance(pending, PendingCompress)

            # Set an always-failing summary (too short)
            pending.summaries[0] = "hi"

            # Mock retry to keep returning short text
            with patch.object(
                pending, "retry",
                side_effect=lambda index, guidance="", **kw: None,
            ):
                result = auto_retry(pending, max_retries=2)

            assert isinstance(result, HookRejection)
            assert pending.status == "rejected"
            assert "short" in result.reason.lower()
            assert result.metadata["max_retries"] == 2
        finally:
            t.close()

    def test_auto_retry_merge_happy_path(self):
        """auto_retry() approves PendingMerge when validation passes."""
        t, source = _make_conflict_tract()
        try:
            def resolver(conflict):
                from tract.llm.protocols import Resolution
                return Resolution(
                    action="resolved",
                    content_text="Good merged content that is valid",
                    reasoning="Test",
                )

            pending = t.merge(source, resolver=resolver, review=True)
            assert isinstance(pending, PendingMerge)

            result = auto_retry(pending)
            assert pending.status == "approved"
            assert not isinstance(result, HookRejection)
        finally:
            t.close()

    def test_auto_retry_merge_retries_then_succeeds(self):
        """auto_retry() retries merge and approves on second try."""
        t, source = _make_conflict_tract()
        try:
            pending = t.merge(source, review=True)
            assert isinstance(pending, PendingMerge)
            assert len(pending.resolutions) == 0

            # Mock retry to add resolutions on first call
            def mock_retry(*, guidance="", **kwargs):
                for conflict in pending.conflicts:
                    th = getattr(conflict, "target_hash", None)
                    if th:
                        pending.resolutions[th] = "Resolved by retry"

            with patch.object(pending, "retry", side_effect=mock_retry):
                result = auto_retry(pending, max_retries=3)

            assert pending.status == "approved"
            assert not isinstance(result, HookRejection)
        finally:
            t.close()

    def test_auto_retry_merge_exhausted(self):
        """auto_retry() rejects PendingMerge after exhausting retries."""
        t, source = _make_conflict_tract()
        try:
            pending = t.merge(source, review=True)
            assert isinstance(pending, PendingMerge)

            # Mock retry that never adds resolutions
            with patch.object(
                pending, "retry",
                side_effect=lambda guidance="", **kw: None,
            ):
                result = auto_retry(pending, max_retries=2)

            assert isinstance(result, HookRejection)
            assert pending.status == "rejected"
            assert result.metadata["max_retries"] == 2
        finally:
            t.close()

    def test_auto_retry_unsupported_type(self):
        """auto_retry() raises TypeError for unsupported types."""
        mock_pending = MagicMock(spec=[])  # No validate attribute
        del mock_pending.validate  # Ensure hasattr returns False

        with pytest.raises(TypeError, match="does not support validate"):
            auto_retry(mock_pending)
