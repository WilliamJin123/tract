"""Comprehensive tests for compression operations.

Tests compression engine with all three autonomy modes (autonomous LLM,
collaborative review, manual content), PINNED/SKIP handling, provenance
tracking, edge cases, and multi-pinned interleaving.
"""

from __future__ import annotations

import pytest

from tract import (
    CommitInfo,
    CompressResult,
    CompressionError,
    DialogueContent,
    InstructionContent,
    PendingCompress,
    PendingCompression,
    Priority,
    Tract,
    TraceError,
)


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

from tests.conftest import make_tract_with_commits


# ===========================================================================
# 1. Autonomous mode tests
# ===========================================================================


class TestAutonomousMode:
    """Tests for LLM-based autonomous compression (default, no review)."""

    def test_compress_all_normal_commits(self):
        """Compress 5 normal commits into a summary."""
        t, hashes = make_tract_with_commits(5)
        old_head = t.head

        mock = MockLLMClient()
        t.configure_llm(mock)

        result = t.compress()

        assert isinstance(result, CompressResult)
        assert len(result.summary_commits) >= 1
        assert len(result.source_commits) == 5
        assert 0 <= result.compression_ratio <= 1.0
        assert result.new_head != old_head
        assert result.compression_id

    def test_compress_preserves_pinned(self):
        """PINNED commits survive compression verbatim."""
        t, hashes = make_tract_with_commits(5)
        # Pin commit 3 (index 2)
        t.annotate(hashes[2], Priority.PINNED, reason="important")

        mock = MockLLMClient(responses=[
            "Summary of first group",
            "Summary of second group",
        ])
        t.configure_llm(mock)

        result = t.compress()

        assert isinstance(result, CompressResult)
        assert len(result.preserved_commits) == 1
        assert hashes[2] in result.preserved_commits

        # Verify compiled output includes pinned content
        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "Message 3" in messages_text

    def test_compress_ignores_skip(self):
        """SKIP commits are excluded from compression entirely."""
        t, hashes = make_tract_with_commits(5)
        # Skip commit 2 (index 1)
        t.annotate(hashes[1], Priority.SKIP, reason="not needed")

        mock = MockLLMClient()
        t.configure_llm(mock)

        result = t.compress()

        assert isinstance(result, CompressResult)
        # Source commits should not include the skipped one
        assert hashes[1] not in result.source_commits
        # But should include the other 4
        assert len(result.source_commits) == 4

    def test_compress_with_range(self):
        """Compress a specific range using from_commit/to_commit."""
        t, hashes = make_tract_with_commits(8)

        mock = MockLLMClient()
        t.configure_llm(mock)

        result = t.compress(from_commit=hashes[2], to_commit=hashes[5])

        assert isinstance(result, CompressResult)
        # Should compress commits 3-6 (indices 2-5)
        assert len(result.source_commits) == 4
        for i in [2, 3, 4, 5]:
            assert hashes[i] in result.source_commits

    def test_compress_with_commit_list(self):
        """Compress specific commits by hash list."""
        t, hashes = make_tract_with_commits(5)

        mock = MockLLMClient()
        t.configure_llm(mock)

        target = [hashes[0], hashes[2], hashes[4]]
        result = t.compress(commits=target)

        assert isinstance(result, CompressResult)
        assert len(result.source_commits) == 3
        for h in target:
            assert h in result.source_commits

    def test_compress_with_target_tokens(self):
        """target_tokens is passed through to the LLM prompt."""
        t, hashes = make_tract_with_commits(3)

        mock = MockLLMClient()
        t.configure_llm(mock)

        result = t.compress(target_tokens=100)

        assert isinstance(result, CompressResult)
        # Verify the LLM was called with target tokens in prompt
        assert mock.last_messages is not None
        user_msg = mock.last_messages[-1]["content"]
        assert "100 tokens" in user_msg

    def test_compress_with_instructions(self):
        """Custom instructions are included in the LLM prompt."""
        t, hashes = make_tract_with_commits(3)

        mock = MockLLMClient()
        t.configure_llm(mock)

        result = t.compress(instructions="focus on code decisions")

        assert isinstance(result, CompressResult)
        assert mock.last_messages is not None
        user_msg = mock.last_messages[-1]["content"]
        assert "focus on code decisions" in user_msg


# ===========================================================================
# 2. Collaborative mode tests
# ===========================================================================


class TestCollaborativeMode:
    """Tests for LLM compression with review before commit."""

    def test_compress_review_true(self):
        """review=True returns PendingCompress with summaries."""
        t, hashes = make_tract_with_commits(5)

        mock = MockLLMClient()
        t.configure_llm(mock)

        result = t.compress(review=True)

        assert isinstance(result, PendingCompress)
        assert len(result.summaries) >= 1
        assert len(result.source_commits) == 5
        assert result.original_tokens > 0
        assert result.estimated_tokens > 0

    def test_pending_edit_summary(self):
        """Edit a summary before approving."""
        t, hashes = make_tract_with_commits(3)

        mock = MockLLMClient(responses=["Original summary"])
        t.configure_llm(mock)

        pending = t.compress(review=True)
        assert isinstance(pending, PendingCompress)

        # Edit the summary
        pending.edit_summary(0, "Edited summary text")
        assert pending.summaries[0] == "Edited summary text"

        # Approve
        result = pending.approve()
        assert isinstance(result, CompressResult)

        # Verify edited text appears in compiled output
        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "Edited summary text" in messages_text

    def test_pending_approve(self):
        """PendingCompression.approve() creates commits and returns CompressResult."""
        t, hashes = make_tract_with_commits(5)
        old_head = t.head

        mock = MockLLMClient()
        t.configure_llm(mock)

        pending = t.compress(review=True)
        assert isinstance(pending, PendingCompress)

        result = pending.approve()

        assert isinstance(result, CompressResult)
        assert result.new_head != old_head
        assert len(result.summary_commits) >= 1
        assert result.compression_id

    def test_approve_compression_method(self):
        """Tract.approve_compression() works as alternative to pending.approve()."""
        t, hashes = make_tract_with_commits(5)

        mock = MockLLMClient()
        t.configure_llm(mock)

        pending = t.compress(review=True)
        assert isinstance(pending, PendingCompress)

        result = t.approve_compression(pending)

        assert isinstance(result, CompressResult)
        assert len(result.summary_commits) >= 1


# ===========================================================================
# 3. Manual mode tests
# ===========================================================================


class TestManualMode:
    """Tests for manual content compression (no LLM needed)."""

    def test_compress_manual_content(self):
        """Manual summary text is used directly."""
        t, hashes = make_tract_with_commits(5)

        result = t.compress(content="My manual summary of the conversation")

        assert isinstance(result, CompressResult)
        assert len(result.summary_commits) >= 1

        # Verify manual text in compiled output
        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "My manual summary" in messages_text

    def test_compress_manual_no_llm_required(self):
        """Manual mode works without configure_llm()."""
        t, hashes = make_tract_with_commits(3)
        # Do NOT call configure_llm()

        result = t.compress(content="Manual summary without LLM")

        assert isinstance(result, CompressResult)
        assert len(result.summary_commits) >= 1

    def test_compress_manual_preserves_pinned(self):
        """Manual mode preserves PINNED commits (at boundary, no interleaving)."""
        t, hashes = make_tract_with_commits(5)
        # Pin the LAST commit -- creates only 1 group of NORMAL (first 4),
        # so manual mode is valid (no interleaving).
        t.annotate(hashes[4], Priority.PINNED, reason="keep this")

        result = t.compress(content="Manual summary")

        assert isinstance(result, CompressResult)
        assert len(result.preserved_commits) == 1
        assert hashes[4] in result.preserved_commits

        # Pinned content should be in compiled output
        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "Message 5" in messages_text


# ===========================================================================
# 4. Provenance tests
# ===========================================================================


class TestProvenance:
    """Tests for compression provenance recording."""

    def test_compression_record_created(self):
        """CompressionRecord exists in DB with correct data."""
        t, hashes = make_tract_with_commits(5)

        result = t.compress(content="Summary for provenance test")

        assert isinstance(result, CompressResult)
        assert result.compression_id

        # Query the compression record
        record = t._event_repo.get_event(result.compression_id)
        assert record is not None
        assert record.tract_id == t.tract_id
        assert record.original_tokens > 0

    def test_original_commits_unreachable(self):
        """Original commits remain in DB but are not reachable from HEAD."""
        t, hashes = make_tract_with_commits(5)
        old_hashes = set(hashes)

        result = t.compress(content="Summary")

        # Original commits still exist in DB
        for h in old_hashes:
            row = t._commit_repo.get(h)
            assert row is not None, f"Original commit {h} should still exist"

        # But they are not in the current chain
        log = t.log(limit=100)
        current_hashes = {c.commit_hash for c in log}
        assert not old_hashes.intersection(current_hashes), \
            "Original commits should not be in current chain"

    def test_provenance_query_sources(self):
        """Query source commits from compression record."""
        t, hashes = make_tract_with_commits(5)

        result = t.compress(content="Summary")

        sources = t._event_repo.get_commits(result.compression_id, "source")
        source_hashes = {s.commit_hash for s in sources}

        for h in hashes:
            assert h in source_hashes

    def test_provenance_query_results(self):
        """Query result commits from compression record."""
        t, hashes = make_tract_with_commits(5)

        result = t.compress(content="Summary")

        results = t._event_repo.get_commits(result.compression_id, "result")
        result_hashes = {r.commit_hash for r in results}

        for h in result.summary_commits:
            assert h in result_hashes


# ===========================================================================
# 5. Edge cases and errors
# ===========================================================================


class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_compress_no_commits_raises(self):
        """compress() on empty tract raises error."""
        t = Tract.open()

        with pytest.raises((CompressionError, TraceError)):
            t.compress(content="Summary")

    def test_compress_no_llm_no_content_raises(self):
        """compress() without LLM or content raises CompressionError."""
        t, hashes = make_tract_with_commits(3)

        with pytest.raises(CompressionError, match="No LLM client"):
            t.compress()

    def test_compress_all_pinned_raises(self):
        """All commits pinned raises CompressionError."""
        t = Tract.open()
        # InstructionContent defaults to PINNED
        h1 = t.commit(InstructionContent(text="System prompt 1"))
        h2 = t.commit(InstructionContent(text="System prompt 2"))

        with pytest.raises(CompressionError, match="Nothing to compress"):
            t.compress(content="Summary")

    def test_compress_clears_cache(self):
        """Compile cache is cleared after compression."""
        t, hashes = make_tract_with_commits(5)

        # Compile to populate cache
        before = t.compile()
        assert before.commit_count == 5

        # Compress
        result = t.compress(content="Summary of all messages")
        assert isinstance(result, CompressResult)

        # Compile again -- should be different (from the new chain)
        after = t.compile()
        assert after.commit_count < before.commit_count

    def test_compress_then_compile_coherent(self):
        """After compression, compile() returns coherent messages."""
        t, hashes = make_tract_with_commits(5)

        t.compress(content="This is a summary of messages 1-5")

        compiled = t.compile()
        assert len(compiled.messages) >= 1
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "summary of messages" in messages_text


# ===========================================================================
# 6. Multi-pinned interleaving tests
# ===========================================================================


class TestMultiPinnedInterleaving:
    """Tests for correct ordering with multiple PINNED commits."""

    def test_pinned_interleaving_order(self):
        """Compressed output preserves correct interleaving order.

        Input: [c1, c2(pinned), c3, c4(pinned), c5]
        Expected output: [summary_1, pinned_2, summary_2, pinned_4, summary_3]
        """
        t, hashes = make_tract_with_commits(5)
        t.annotate(hashes[1], Priority.PINNED, reason="pin2")
        t.annotate(hashes[3], Priority.PINNED, reason="pin4")

        mock = MockLLMClient(responses=[
            "Summary of group 1",
            "Summary of group 2",
            "Summary of group 3",
        ])
        t.configure_llm(mock)

        result = t.compress()

        assert isinstance(result, CompressResult)
        assert len(result.preserved_commits) == 2
        assert hashes[1] in result.preserved_commits
        assert hashes[3] in result.preserved_commits

        # Verify compiled output order
        compiled = t.compile()
        texts = [m.content for m in compiled.messages]

        # We expect 5 messages: 3 summaries + 2 pinned
        assert len(texts) == 5

        # Check that pinned content appears at correct positions
        # Position 0: summary of c1
        assert "Summary of group 1" in texts[0]
        # Position 1: pinned c2 (Message 2)
        assert "Message 2" in texts[1]
        # Position 2: summary of c3
        assert "Summary of group 2" in texts[2]
        # Position 3: pinned c4 (Message 4)
        assert "Message 4" in texts[3]
        # Position 4: summary of c5
        assert "Summary of group 3" in texts[4]

    def test_pinned_at_boundaries(self):
        """Pin first and last commit in range."""
        t, hashes = make_tract_with_commits(5)
        t.annotate(hashes[0], Priority.PINNED, reason="first")
        t.annotate(hashes[4], Priority.PINNED, reason="last")

        mock = MockLLMClient(responses=["Summary of middle 3"])
        t.configure_llm(mock)

        result = t.compress()

        assert isinstance(result, CompressResult)
        assert len(result.preserved_commits) == 2
        assert hashes[0] in result.preserved_commits
        assert hashes[4] in result.preserved_commits

        # 3 normal commits should be compressed
        assert len(result.source_commits) == 3

        # Compiled output: pinned_first, summary, pinned_last
        compiled = t.compile()
        texts = [m.content for m in compiled.messages]
        assert len(texts) == 3
        assert "Message 1" in texts[0]
        assert "Summary of middle 3" in texts[1]
        assert "Message 5" in texts[2]


# ===========================================================================
# 7. LLM error path tests
# ===========================================================================


class TestLLMErrorPaths:
    """Tests for LLM response error handling in compression."""

    def test_malformed_response_raises(self):
        """Malformed LLM response (missing expected keys) raises CompressionError."""
        t, hashes = make_tract_with_commits(5)

        class MalformedLLM:
            def chat(self, messages, **kwargs):
                return {"bad": "structure"}

            def close(self):
                pass

        t.configure_llm(MalformedLLM())

        with pytest.raises(CompressionError, match="Invalid LLM response structure"):
            t.compress()

    def test_empty_response_raises(self):
        """LLM returning empty content raises CompressionError."""
        t, hashes = make_tract_with_commits(5)

        class EmptyLLM:
            def chat(self, messages, **kwargs):
                return {"choices": [{"message": {"content": ""}}]}

            def close(self):
                pass

        t.configure_llm(EmptyLLM())

        with pytest.raises(CompressionError, match="empty summary"):
            t.compress()

    def test_missing_choices_key_raises(self):
        """LLM response without 'choices' key raises CompressionError."""
        t, hashes = make_tract_with_commits(5)

        class NoChoicesLLM:
            def chat(self, messages, **kwargs):
                return {"result": "text"}

            def close(self):
                pass

        t.configure_llm(NoChoicesLLM())

        with pytest.raises(CompressionError):
            t.compress()


# ===========================================================================
# 8. Manual mode with PINNED interleaving tests
# ===========================================================================


class TestManualModePinnedError:
    """Tests for manual mode behavior with and without PINNED interleaving."""

    def test_manual_with_pinned_interleaving_raises(self):
        """Manual mode with PINNED interleaving raises CompressionError."""
        t, hashes = make_tract_with_commits(5)
        t.annotate(hashes[2], Priority.PINNED, reason="important")

        with pytest.raises(CompressionError, match="Manual mode.*separate groups"):
            t.compress(content="my summary")

    def test_manual_without_interleaving_works(self):
        """Manual mode without PINNED interleaving succeeds."""
        t, hashes = make_tract_with_commits(5)
        # No pinned commits -- single group, manual mode should work

        result = t.compress(content="my summary")

        assert isinstance(result, CompressResult)
        assert len(result.summary_commits) >= 1
        assert len(result.source_commits) == 5


# ===========================================================================
# 9. Stacked and re-compression tests
# ===========================================================================


class TestStackedCompression:
    """Tests for compress -> commit -> compress again (iterative compression)."""

    def test_compress_commit_compress_again(self):
        """Stacked compression: compress, add more commits, compress again."""
        t, hashes = make_tract_with_commits(5)

        # First compression
        result1 = t.compress(content="Summary of first 5 messages")
        assert isinstance(result1, CompressResult)
        head_after_first = t.head

        # Add more commits after compression
        h6 = t.commit(DialogueContent(role="user", text="New message 6"))
        h7 = t.commit(DialogueContent(role="assistant", text="New message 7"))
        h8 = t.commit(DialogueContent(role="user", text="New message 8"))

        # Second compression -- compresses the summary + new messages
        result2 = t.compress(content="Summary of everything so far")
        assert isinstance(result2, CompressResult)
        assert result2.new_head != head_after_first

        # Final compiled output should have the latest summary
        compiled = t.compile()
        messages_text = " ".join(m.content for m in compiled.messages)
        assert "Summary of everything" in messages_text

    def test_re_compress_summary_commits(self):
        """Re-compression: compress a range that includes a previous summary commit."""
        t, hashes = make_tract_with_commits(5)

        # First compression
        result1 = t.compress(content="Summary of first batch")
        assert isinstance(result1, CompressResult)

        # The current chain now has the summary commit(s)
        compiled_mid = t.compile()
        assert compiled_mid.commit_count >= 1

        # Add a couple more commits
        t.commit(DialogueContent(role="user", text="After first compress"))
        t.commit(DialogueContent(role="assistant", text="Response after compress"))

        # Re-compress everything (including the previous summary)
        result2 = t.compress(content="Re-compressed summary of all")
        assert isinstance(result2, CompressResult)

        compiled_final = t.compile()
        messages_text = " ".join(m.content for m in compiled_final.messages)
        assert "Re-compressed summary" in messages_text

    def test_toctou_guard_blocks_stale_approve(self):
        """Approving a PendingCompression after HEAD changed raises error."""
        t, hashes = make_tract_with_commits(5)

        mock = MockLLMClient()
        t.configure_llm(mock)

        # Plan compression (collaborative mode)
        pending = t.compress(review=True)
        assert isinstance(pending, PendingCompress)

        # Add a new commit, changing HEAD
        t.commit(DialogueContent(role="user", text="Sneaky new commit"))

        # Approve should fail because HEAD changed
        with pytest.raises(CompressionError, match="HEAD changed"):
            pending.approve()
