"""Tests for sliding window compression strategy.

Tests the ``strategy="sliding_window"`` parameter of :meth:`Tract.compress`,
which keeps the most recent ``window_size`` commits in full detail and
compresses everything older into summaries.
"""

from __future__ import annotations

import json

import pytest

from tract import (
    CompressResult,
    CompressionError,
    DialogueContent,
    Priority,
    Tract,
)
from tests.conftest import make_tract_with_commits


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM client for testing compression without real API calls."""

    def __init__(self, responses=None):
        self.responses = responses or [
            "Previously in this conversation: summary text."
        ]
        self._call_count = 0
        self.last_messages = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        text = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_commit_texts(t: Tract) -> list[str]:
    """Return commit texts in chain order (oldest first) from current HEAD."""
    compiled = t.compile()
    messages = compiled.to_dicts()
    return [m.get("text", m.get("content", "")) for m in messages]


def _count_commits(t: Tract) -> int:
    """Count total commits on the current branch."""
    head = t.head
    if head is None:
        return 0
    ancestors = list(t._commit_repo.get_ancestors(head))
    return len(ancestors)


# ===========================================================================
# 1. Basic sliding window tests
# ===========================================================================


class TestSlidingWindowBasic:
    """Basic sliding window functionality."""

    def test_sliding_window_keeps_last_n(self):
        """Sliding window keeps the last N commits in full detail."""
        t, hashes = make_tract_with_commits(8)
        result = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of older messages",
        )

        assert isinstance(result, CompressResult)
        # Should have summary commit(s) + 3 window commits
        total = _count_commits(t)
        # At minimum: 1 summary + 3 window = 4
        assert total >= 4
        # The last 3 messages should still be present
        texts = _get_commit_texts(t)
        assert "Message 6" in texts[-3]
        assert "Message 7" in texts[-2]
        assert "Message 8" in texts[-1]

    def test_sliding_window_compresses_older(self):
        """Everything before the window gets compressed."""
        t, hashes = make_tract_with_commits(10)
        result = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of first 7 messages",
        )

        assert isinstance(result, CompressResult)
        # Source commits should be the 7 pre-window commits
        assert len(result.source_commits) == 7
        assert result.compression_id

    def test_window_larger_than_total_raises(self):
        """Window larger than total commits means nothing to compress."""
        t, hashes = make_tract_with_commits(3)
        with pytest.raises(CompressionError, match="Nothing to compress"):
            t.compress(
                strategy="sliding_window",
                window_size=10,
                content="Should not be used",
            )

    def test_window_equal_to_total_raises(self):
        """Window equal to total commits means nothing to compress."""
        t, hashes = make_tract_with_commits(5)
        with pytest.raises(CompressionError, match="Nothing to compress"):
            t.compress(
                strategy="sliding_window",
                window_size=5,
                content="Should not be used",
            )

    def test_window_of_zero_compresses_everything(self):
        """Window of 0 compresses all commits (except PINNED)."""
        t, hashes = make_tract_with_commits(5)
        result = t.compress(
            strategy="sliding_window",
            window_size=0,
            content="Everything compressed",
        )

        assert isinstance(result, CompressResult)
        assert len(result.source_commits) == 5

    def test_single_commit_window_one_raises(self):
        """Single commit with window=1 means nothing to compress."""
        t, hashes = make_tract_with_commits(1)
        with pytest.raises(CompressionError, match="Nothing to compress"):
            t.compress(
                strategy="sliding_window",
                window_size=1,
                content="Should not be used",
            )


# ===========================================================================
# 2. PINNED commit handling
# ===========================================================================


class TestSlidingWindowPinned:
    """PINNED commits survive outside the window."""

    def test_pinned_outside_window_survives(self):
        """PINNED commits outside the window are preserved verbatim."""
        t, hashes = make_tract_with_commits(8)
        # Pin commit 2 (index 1) -- it's outside window_size=3
        t.annotate(hashes[1], Priority.PINNED, reason="important")

        result = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of non-pinned older messages",
        )

        assert isinstance(result, CompressResult)
        assert len(result.preserved_commits) == 1
        # The pinned commit's text should still be present
        texts = _get_commit_texts(t)
        assert any("Message 2" in text for text in texts)

    def test_all_pre_window_pinned_raises(self):
        """If all pre-window commits are PINNED, nothing to compress."""
        t, hashes = make_tract_with_commits(5)
        # Pin the first two commits (everything outside window_size=3)
        t.annotate(hashes[0], Priority.PINNED, reason="pin1")
        t.annotate(hashes[1], Priority.PINNED, reason="pin2")

        with pytest.raises(CompressionError, match="Nothing to compress"):
            t.compress(
                strategy="sliding_window",
                window_size=3,
                content="Should not be used",
            )

    def test_pinned_inside_window_unaffected(self):
        """PINNED commits inside the window are replayed normally."""
        t, hashes = make_tract_with_commits(8)
        # Pin a commit inside the window (last 3)
        t.annotate(hashes[7], Priority.PINNED, reason="pinned in window")

        result = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of older messages",
        )

        assert isinstance(result, CompressResult)
        texts = _get_commit_texts(t)
        # Window commit should still be present
        assert any("Message 8" in text for text in texts)


# ===========================================================================
# 3. Manual content mode
# ===========================================================================


class TestSlidingWindowManual:
    """Manual content mode with sliding window."""

    def test_manual_content_works(self):
        """Manual content parameter works with sliding window."""
        t, hashes = make_tract_with_commits(10)
        result = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Custom summary of early conversation",
        )

        assert isinstance(result, CompressResult)
        texts = _get_commit_texts(t)
        assert any("Custom summary" in text for text in texts)

    def test_manual_content_window_zero(self):
        """Manual content with window=0 compresses everything."""
        t, hashes = make_tract_with_commits(5)
        result = t.compress(
            strategy="sliding_window",
            window_size=0,
            content="Full conversation summary",
        )

        texts = _get_commit_texts(t)
        assert any("Full conversation summary" in text for text in texts)
        # No window commits replayed
        total = _count_commits(t)
        assert total == 1  # Just the summary commit


# ===========================================================================
# 4. LLM mode
# ===========================================================================


class TestSlidingWindowLLM:
    """LLM-based summarization with sliding window."""

    def test_llm_mode_works(self):
        """LLM summarization works with sliding window strategy."""
        t, hashes = make_tract_with_commits(8)
        mock = MockLLMClient(responses=["LLM summary of older context."])
        t.config.configure_llm(mock)

        result = t.compress(
            strategy="sliding_window",
            window_size=3,
        )

        assert isinstance(result, CompressResult)
        assert mock._call_count >= 1
        texts = _get_commit_texts(t)
        assert any("LLM summary" in text for text in texts)
        # Window commits preserved
        assert any("Message 6" in text for text in texts)
        assert any("Message 7" in text for text in texts)
        assert any("Message 8" in text for text in texts)

    def test_llm_mode_no_client_no_content_raises(self):
        """No LLM client and no content raises CompressionError."""
        t, hashes = make_tract_with_commits(8)
        with pytest.raises(CompressionError):
            t.compress(
                strategy="sliding_window",
                window_size=3,
            )


# ===========================================================================
# 5. Multiple compress calls (window slides)
# ===========================================================================


class TestSlidingWindowMultiple:
    """Multiple compress calls with sliding window."""

    def test_window_slides_on_second_compress(self):
        """After adding more commits, the window slides forward."""
        t, hashes = make_tract_with_commits(8)

        # First compress: window=3 keeps last 3
        result1 = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of messages 1-5",
        )
        assert isinstance(result1, CompressResult)

        # Add more commits
        for i in range(9, 12):
            t.commit(DialogueContent(role="user", text=f"Message {i}"))

        # Second compress: window=3 keeps last 3 (new ones)
        result2 = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of everything before last 3",
        )
        assert isinstance(result2, CompressResult)

        texts = _get_commit_texts(t)
        # The newest 3 messages should be present
        assert any("Message 9" in text for text in texts)
        assert any("Message 10" in text for text in texts)
        assert any("Message 11" in text for text in texts)

    def test_already_compressed_not_double_compressed(self):
        """Summary commits from prior compression can be re-compressed."""
        t, hashes = make_tract_with_commits(10)

        # First compress
        t.compress(
            strategy="sliding_window",
            window_size=5,
            content="First summary",
        )

        # Add more commits
        for i in range(11, 16):
            t.commit(DialogueContent(role="user", text=f"Message {i}"))

        # Second compress -- the first summary is now outside the window
        result = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Second summary covering first summary and more",
        )

        assert isinstance(result, CompressResult)
        texts = _get_commit_texts(t)
        # Latest 3 should be present
        assert any("Message 13" in text for text in texts)
        assert any("Message 14" in text for text in texts)
        assert any("Message 15" in text for text in texts)


# ===========================================================================
# 6. Integration tests
# ===========================================================================


class TestSlidingWindowIntegration:
    """Integration tests: sliding window with other operations."""

    def test_commit_10_window_3_verify_structure(self):
        """Commit 10 items, window=3, verify 7 compressed + 3 full."""
        t, hashes = make_tract_with_commits(10)

        result = t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of messages 1 through 7",
        )

        assert isinstance(result, CompressResult)
        assert len(result.source_commits) == 7

        texts = _get_commit_texts(t)
        # Should have summary + 3 window commits = 4 messages
        assert len(texts) == 4
        assert "Summary of messages 1 through 7" in texts[0]
        assert "Message 8" in texts[1]
        assert "Message 9" in texts[2]
        assert "Message 10" in texts[3]

    def test_sliding_window_then_commit(self):
        """Can commit new messages after sliding window compression."""
        t, hashes = make_tract_with_commits(8)
        t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of older messages",
        )

        # Add a new commit after compression
        info = t.commit(DialogueContent(role="user", text="New message after compress"))
        assert info.commit_hash is not None

        texts = _get_commit_texts(t)
        assert any("New message after compress" in text for text in texts)

    def test_sliding_window_then_branch(self):
        """Can branch after sliding window compression."""
        t, hashes = make_tract_with_commits(8)
        t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of older messages",
        )

        # Branch should work
        t.branch("feature")
        t.checkout("feature")
        info = t.commit(DialogueContent(role="user", text="Feature commit"))
        assert info.commit_hash is not None

    def test_sliding_window_compile_produces_valid_output(self):
        """Compiled output after sliding window is valid."""
        t, hashes = make_tract_with_commits(8)
        t.compress(
            strategy="sliding_window",
            window_size=3,
            content="Summary of older messages",
        )

        compiled = t.compile()
        messages = compiled.to_dicts()
        assert len(messages) >= 4  # summary + 3 window
        # All messages should have role and text
        for m in messages:
            assert "role" in m
            assert "text" in m or "content" in m

    def test_default_strategy_unchanged(self):
        """Default strategy still works exactly as before."""
        t, hashes = make_tract_with_commits(5)
        result = t.compress(
            content="Default compression summary",
        )

        assert isinstance(result, CompressResult)
        assert len(result.source_commits) == 5

    def test_explicit_default_strategy(self):
        """Passing strategy='default' explicitly works same as omitting it."""
        t, hashes = make_tract_with_commits(5)
        result = t.compress(
            strategy="default",
            content="Default compression summary",
        )

        assert isinstance(result, CompressResult)
        assert len(result.source_commits) == 5
