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
    PendingToolResult,
    Priority,
    Tract,
    ToolDropResult,
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


# ===========================================================================
# 7. Prompt variants tests
# ===========================================================================


class TestPromptVariants:
    """Tests for summarization system prompt variants."""

    def test_default_prompt_is_neutral(self):
        """DEFAULT_SUMMARIZE_SYSTEM should not mention 'conversation history'."""
        from tract.prompts.summarize import DEFAULT_SUMMARIZE_SYSTEM

        assert isinstance(DEFAULT_SUMMARIZE_SYSTEM, str)
        assert len(DEFAULT_SUMMARIZE_SYSTEM) > 50
        assert "Previously in this conversation" not in DEFAULT_SUMMARIZE_SYSTEM

    def test_conversation_prompt_has_prefix(self):
        """CONVERSATION_SUMMARIZE_SYSTEM should use 'Previously in this conversation:'."""
        from tract.prompts.summarize import CONVERSATION_SUMMARIZE_SYSTEM

        assert "Previously in this conversation:" in CONVERSATION_SUMMARIZE_SYSTEM
        assert "conversation history" in CONVERSATION_SUMMARIZE_SYSTEM

    def test_tool_prompt_mentions_tool_calls(self):
        """TOOL_SUMMARIZE_SYSTEM should be tuned for tool-call compression."""
        from tract.prompts.summarize import TOOL_SUMMARIZE_SYSTEM

        assert "tool" in TOOL_SUMMARIZE_SYSTEM.lower()
        assert "outcomes" in TOOL_SUMMARIZE_SYSTEM.lower()

    def test_all_prompts_exportable(self):
        """All prompt variants should be importable from tract."""
        from tract import (
            DEFAULT_SUMMARIZE_SYSTEM,
            CONVERSATION_SUMMARIZE_SYSTEM,
            TOOL_SUMMARIZE_SYSTEM,
        )

        assert all(isinstance(p, str) for p in [
            DEFAULT_SUMMARIZE_SYSTEM,
            CONVERSATION_SUMMARIZE_SYSTEM,
            TOOL_SUMMARIZE_SYSTEM,
        ])


# ===========================================================================
# 8. compress_tool_calls() tests
# ===========================================================================


class TestCompressToolCalls:
    """Tests for the compress_tool_calls() EDIT-based compaction method."""

    def _build_tool_call_tract(self):
        """Build a tract with a realistic tool-call sequence.

        Returns (tract, tool_result_hashes, answer_hash).
        """
        t = Tract.open()
        t.system("You are a search agent.")
        t.user("Find the hidden comment.")

        # Turn 1: assistant calls a tool
        asst1 = t.assistant(
            "",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "search",
                 "arguments": {"q": "DISCOVERY"}},
            ]},
        )

        # Turn 1: tool result
        tr1 = t.tool_result("call_1", "search", "file.py:36: # DISCOVERY: some text")

        # Turn 2: final answer (no tool_calls)
        answer = t.assistant(
            "Found it! The comment is at file.py line 36: '# DISCOVERY: some text'. "
            "It means that tool calls are just commits."
        )

        return t, [tr1], answer.commit_hash

    def test_compress_tool_calls_basic(self):
        """compress_tool_calls() edits tool results and preserves structure."""
        import json
        from tract.models.compression import ToolCompactResult

        t, tool_results, answer_hash = self._build_tool_call_tract()

        mock = MockLLMClient(responses=[
            json.dumps(["Found DISCOVERY comment at file.py:36."])
        ])
        t.configure_llm(mock)

        result = t.compress_tool_calls(target_tokens=50)

        assert isinstance(result, ToolCompactResult)
        assert len(result.edit_commits) == 1
        assert len(result.source_commits) == 1
        assert result.turn_count == 1
        assert "search" in result.tool_names

        # The final answer should survive in compiled context
        ctx = t.compile()
        messages = ctx.to_dicts()
        assert "Found it!" in messages[-1]["content"]

        # Tool result should be the compacted version
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "file.py:36" in tool_msgs[0]["content"]

        # tool_call_id and name metadata should be preserved
        assert tool_msgs[0].get("tool_call_id") == "call_1"

    def test_compress_tool_calls_uses_compact_prompt(self):
        """compress_tool_calls() should use TOOL_COMPACT_SYSTEM by default."""
        import json
        from tract.prompts.summarize import TOOL_COMPACT_SYSTEM

        t, _, _ = self._build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Summary."])])
        t.configure_llm(mock)

        t.compress_tool_calls()

        assert mock.last_messages is not None
        system_msg = mock.last_messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == TOOL_COMPACT_SYSTEM

    def test_compress_tool_calls_custom_system_prompt(self):
        """system_prompt= overrides the default compact prompt."""
        import json

        t, _, _ = self._build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Custom."]) ])
        t.configure_llm(mock)

        custom = "You are a custom compactor."
        t.compress_tool_calls(system_prompt=custom)

        system_msg = mock.last_messages[0]
        assert system_msg["content"] == custom

    def test_compress_tool_calls_preserves_metadata(self):
        """After compaction, tool_call_id and name survive on edited commits."""
        import json

        t, tool_results, _ = self._build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted result."])])
        t.configure_llm(mock)

        result = t.compress_tool_calls()

        # Check the edit commit preserves metadata
        edited_ci = t.get_commit(result.edit_commits[0])
        assert edited_ci is not None
        meta = edited_ci.metadata or {}
        assert meta.get("tool_call_id") == "call_1"
        assert meta.get("name") == "search"
        assert edited_ci.operation.value == "edit"

    def test_compress_tool_calls_find_tool_turns_still_works(self):
        """After compaction, find_tool_turns() still finds the turns."""
        import json

        t, _, _ = self._build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted."])])
        t.configure_llm(mock)

        t.compress_tool_calls()

        # Tool turns should still be discoverable
        turns = t.find_tool_turns()
        assert len(turns) >= 1

    def test_compress_tool_calls_only_tool_results(self):
        """Compaction works when all commits are tool-related (no final answer)."""
        import json

        t = Tract.open()
        t.system("Agent.")
        t.user("Do something.")

        t.assistant(
            "",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "fn", "arguments": {}},
            ]},
        )
        t.tool_result("call_1", "fn", "result data")

        mock = MockLLMClient(responses=[json.dumps(["Compacted result."])])
        t.configure_llm(mock)

        from tract.models.compression import ToolCompactResult
        result = t.compress_tool_calls()
        assert isinstance(result, ToolCompactResult)
        assert result.turn_count == 1


# ---------------------------------------------------------------------------
# tool_result(edit=) parameter
# ---------------------------------------------------------------------------


class TestToolResultEdit:
    """Tests for tool_result(edit=) parameter."""

    def test_tool_result_edit_basic(self):
        """Edit replaces content, original in history."""
        with Tract.open() as t:
            original = t.tool_result("call_1", "grep", "200 lines of verbose output")
            edited = t.tool_result(
                "call_1", "grep", "config.py:42: DATABASE_URL=...",
                edit=original.commit_hash,
            )
            assert edited.commit_hash != original.commit_hash
            # Compiled context shows the edited version
            ctx = t.compile()
            tool_msgs = [m for m in ctx.messages if m.role == "tool"]
            assert len(tool_msgs) == 1
            assert "DATABASE_URL" in tool_msgs[0].content

    def test_tool_result_edit_preserves_metadata(self):
        """tool_call_id and name survive in the edit."""
        with Tract.open() as t:
            original = t.tool_result("call_1", "grep", "verbose output")
            edited = t.tool_result(
                "call_1", "grep", "concise",
                edit=original.commit_hash,
            )
            ci = t.get_commit(edited.commit_hash)
            assert ci.metadata["tool_call_id"] == "call_1"
            assert ci.metadata["name"] == "grep"

    def test_tool_result_edit_compiled_shows_new(self):
        """compile() serves the edited version."""
        with Tract.open() as t:
            t.system("test")
            original = t.tool_result("call_1", "read", "this is old content")
            t.tool_result(
                "call_1", "read", "this is new content",
                edit=original.commit_hash,
            )
            ctx = t.compile()
            tool_msgs = [m for m in ctx.messages if m.role == "tool"]
            assert len(tool_msgs) == 1
            assert tool_msgs[0].content == "this is new content"

    def test_tool_result_edit_log_shows_both(self):
        """log() shows both the original and edit commits."""
        with Tract.open() as t:
            original = t.tool_result("call_1", "grep", "verbose")
            edited = t.tool_result(
                "call_1", "grep", "concise",
                edit=original.commit_hash,
            )
            log_entries = t.log(limit=10)
            hashes = [e.commit_hash for e in log_entries]
            assert edited.commit_hash in hashes
            # Original should be accessible via get_commit even if not in the main chain
            orig_ci = t.get_commit(original.commit_hash)
            assert orig_ci is not None


class TestPendingToolResult:
    """Tests for PendingToolResult hook system."""

    def test_hook_fires_on_tool_result(self):
        """Handler receives PendingToolResult."""
        received = []

        def handler(pending):
            received.append(pending)
            pending.approve()

        with Tract.open() as t:
            t.on("tool_result", handler)
            t.tool_result("c1", "grep", "verbose output")
            assert len(received) == 1
            assert received[0].tool_name == "grep"
            assert received[0].content == "verbose output"

    def test_hook_edit_result(self):
        """Handler edits content, commit has new content."""

        def handler(pending):
            pending.edit_result("concise output")
            pending.approve()

        with Tract.open() as t:
            t.on("tool_result", handler)
            ci = t.tool_result("c1", "grep", "verbose output")
            content = t.get_content(ci)
            assert content == "concise output"

    def test_hook_reject(self):
        """Handler rejects, returns PendingToolResult not CommitInfo."""

        def handler(pending):
            pending.reject("too verbose")

        with Tract.open() as t:
            t.on("tool_result", handler)
            result = t.tool_result("c1", "grep", "verbose")
            assert isinstance(result, PendingToolResult)
            assert result.status == "rejected"

    def test_hook_no_fire_on_edit(self):
        """edit= bypasses the hook."""
        fired = []

        def handler(pending):
            fired.append(True)
            pending.approve()

        with Tract.open() as t:
            t.on("tool_result", handler)
            orig = t.tool_result("c1", "grep", "verbose")
            fired.clear()  # Reset after first hook fire
            t.tool_result("c1", "grep", "edited", edit=orig.commit_hash)
            assert len(fired) == 0  # Hook did NOT fire for edit

    def test_hook_passthrough(self):
        """Handler approves without changes."""

        def handler(pending):
            pending.approve()

        with Tract.open() as t:
            t.on("tool_result", handler)
            ci = t.tool_result("c1", "grep", "original content")
            content = t.get_content(ci)
            assert content == "original content"

    def test_hook_carries_token_count(self):
        """PendingToolResult has token_count set."""
        received = []

        def handler(pending):
            received.append(pending)
            pending.approve()

        with Tract.open() as t:
            t.on("tool_result", handler)
            t.tool_result("c1", "grep", "some text with a few tokens")
            assert received[0].token_count > 0

    def test_no_hook_commits_directly(self):
        """Without handler, commits directly as before."""
        with Tract.open() as t:
            ci = t.tool_result("c1", "grep", "direct output")
            assert ci.commit_hash  # Got a CommitInfo back
            content = t.get_content(ci)
            assert content == "direct output"

    def test_hook_summarize(self):
        """Handler calls summarize(), LLM runs, original preserved."""
        class MockLLM:
            def chat(self, messages, **kwargs):
                return {"choices": [{"message": {"content": "LLM-summarized"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

        def handler(pending):
            pending.summarize(instructions="keep filenames only")
            assert pending.original_content == "verbose grep output with many lines"
            assert pending.content == "LLM-summarized"
            pending.approve()

        with Tract.open() as t:
            t._llm_client = MockLLM()
            t.on("tool_result", handler)
            ci = t.tool_result("c1", "grep", "verbose grep output with many lines")
            content = t.get_content(ci)
            assert content == "LLM-summarized"

    def test_review_returns_pending(self):
        """review=True returns PendingToolResult without committing."""
        with Tract.open() as t:
            pending = t.tool_result("c1", "grep", "verbose", review=True)
            assert isinstance(pending, PendingToolResult)
            assert pending.status == "pending"
            # Can approve manually
            ci = pending.approve()
            assert ci.commit_hash


class TestToolSummarizationConfig:
    """Tests for configure_tool_summarization()."""

    def test_config_per_tool_instructions(self):
        """Tool with instructions gets summarized."""
        class MockLLM:
            def chat(self, messages, **kwargs):
                return {"choices": [{"message": {"content": "summarized grep output"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

        with Tract.open() as t:
            t._llm_client = MockLLM()
            t.configure_tool_summarization(
                instructions={"grep": "summarize to filenames only"},
            )
            ci = t.tool_result("c1", "grep", "a]" * 500)  # verbose
            content = t.get_content(ci)
            assert content == "summarized grep output"

    def test_config_auto_threshold(self):
        """Results over threshold get summarized."""
        class MockLLM:
            def chat(self, messages, **kwargs):
                return {"choices": [{"message": {"content": "auto-summarized"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

        with Tract.open() as t:
            t._llm_client = MockLLM()
            t.configure_tool_summarization(auto_threshold=10)
            # Short content should pass through
            ci_short = t.tool_result("c1", "grep", "hi")
            content_short = t.get_content(ci_short)
            assert content_short == "hi"
            # Long content should get summarized
            ci_long = t.tool_result("c2", "grep", "very verbose " * 100)
            content_long = t.get_content(ci_long)
            assert content_long == "auto-summarized"

    def test_config_under_threshold_passthrough(self):
        """Small results pass through unchanged."""
        with Tract.open() as t:
            t.configure_tool_summarization(auto_threshold=1000)
            ci = t.tool_result("c1", "grep", "small result")
            content = t.get_content(ci)
            assert content == "small result"

    def test_config_default_instructions(self):
        """Unlisted tools use default_instructions when over threshold."""
        received_instructions = []
        class MockLLM:
            def chat(self, messages, **kwargs):
                # Capture the user prompt to check instructions
                received_instructions.append(messages[-1]["content"])
                return {"choices": [{"message": {"content": "default-summarized"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

        with Tract.open() as t:
            t._llm_client = MockLLM()
            t.configure_tool_summarization(
                instructions={"grep": "filenames only"},
                auto_threshold=10,
                default_instructions="be concise",
            )
            ci = t.tool_result("c1", "unknown_tool", "verbose " * 100)
            content = t.get_content(ci)
            assert content == "default-summarized"
            assert "be concise" in received_instructions[0]

    def test_config_override_with_custom_handler(self):
        """User can replace the auto handler with a custom one."""
        with Tract.open() as t:
            t.configure_tool_summarization(
                instructions={"grep": "filenames only"},
            )
            # Override with custom handler
            def custom(pending):
                pending.edit_result("custom edited")
                pending.approve()
            t.on("tool_result", custom)
            ci = t.tool_result("c1", "grep", "verbose")
            content = t.get_content(ci)
            assert content == "custom edited"


class TestCompressToolCallsAutoDetect:
    """Tests for compress_tool_calls() with auto-detect (no explicit commits)."""

    def test_auto_detect_all_turns(self):
        """compress_tool_calls() without commits auto-detects all tool turns."""
        import json
        from tract.models.compression import ToolCompactResult

        with Tract.open() as t:
            t._llm_client = MockLLMClient([json.dumps(["Tool summary."])])
            t.system("test")
            # Simulate a tool-calling turn
            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            tr = t.tool_result("c1", "grep", "verbose grep output " * 20)
            t.assistant("The answer is 42.")
            # Auto-detect: no commits arg
            result = t.compress_tool_calls()
            assert isinstance(result, ToolCompactResult)
            assert result.source_commits == (tr.commit_hash,)
            assert len(result.edit_commits) == 1
            assert result.turn_count == 1

    def test_auto_detect_by_name(self):
        """name= filter only selects matching tool turns for compaction."""
        import json
        from tract.models.compression import ToolCompactResult

        with Tract.open() as t:
            t._llm_client = MockLLMClient([json.dumps(["Grep summary."])])
            t.system("test")
            # grep turn
            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            grep_result = t.tool_result("c1", "grep", "verbose grep output " * 20)
            # read_file turn
            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c2", "name": "read_file", "arguments": {}}]},
            )
            read_result = t.tool_result("c2", "read_file", "file content " * 20)
            # Only compact grep turns
            result = t.compress_tool_calls(name="grep")
            assert isinstance(result, ToolCompactResult)
            # source_commits should only contain the grep tool result hash
            assert result.source_commits == (grep_result.commit_hash,)
            assert "grep" in result.tool_names
            # read_file hash should NOT be in source commits
            assert read_result.commit_hash not in result.source_commits

    def test_explicit_commits_still_works(self):
        """Passing explicit commits= scopes compaction to matching turns."""
        import json
        from tract.models.compression import ToolCompactResult

        with Tract.open() as t:
            t._llm_client = MockLLMClient([json.dumps(["Compacted."])])
            t.system("test")
            asst = t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            tr = t.tool_result("c1", "grep", "verbose " * 20)
            answer = t.assistant("answer")
            result = t.compress_tool_calls(
                [asst.commit_hash, tr.commit_hash, answer.commit_hash],
            )
            assert isinstance(result, ToolCompactResult)
            assert result.source_commits == (tr.commit_hash,)
            assert len(result.edit_commits) == 1


# ---------------------------------------------------------------------------
# is_error + drop_failed_tool_turns tests
# ---------------------------------------------------------------------------


class TestToolResultIsError:
    """Tests for is_error field on tool_result()."""

    def test_tool_result_is_error_metadata(self):
        """is_error=True stores is_error in commit metadata."""
        with Tract.open() as t:
            t.system("test")
            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            ci = t.tool_result("c1", "grep", "Error: file not found", is_error=True)
            assert ci.metadata["is_error"] is True

    def test_tool_result_is_error_false_default(self):
        """No is_error key in metadata by default."""
        with Tract.open() as t:
            t.system("test")
            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            ci = t.tool_result("c1", "grep", "some result")
            assert "is_error" not in (ci.metadata or {})

    def test_tool_result_is_error_with_edit(self):
        """is_error=True works with edit= path too."""
        with Tract.open() as t:
            t.system("test")
            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            original = t.tool_result("c1", "grep", "original result")
            edited = t.tool_result(
                "c1", "grep", "Error: permission denied",
                edit=original.commit_hash,
                is_error=True,
            )
            assert edited.metadata["is_error"] is True


class TestDropFailedToolTurns:
    """Tests for drop_failed_tool_turns()."""

    def _make_tool_turn(self, t, call_id, name, content, *, is_error=False):
        """Helper: commit a tool call + result pair."""
        t.assistant(
            "",
            metadata={"tool_calls": [{"id": call_id, "name": name, "arguments": {}}]},
        )
        return t.tool_result(call_id, name, content, is_error=is_error)

    def test_drop_failed_tool_turns_basic(self):
        """Drops error turns and keeps clean ones."""
        with Tract.open() as t:
            t.system("test")
            t.user("find files")

            self._make_tool_turn(t, "c1", "grep", "found matches")
            self._make_tool_turn(t, "c2", "bash", "Error: command failed", is_error=True)
            self._make_tool_turn(t, "c3", "read_file", "file contents")

            result = t.drop_failed_tool_turns()
            assert result.turns_dropped == 1
            assert "bash" in result.tool_names

            # Compile should exclude the error turn
            ctx = t.compile()
            contents = [m.content for m in ctx.messages]
            full = " ".join(contents)
            assert "Error: command failed" not in full
            assert "found matches" in full
            assert "file contents" in full

    def test_drop_failed_tool_turns_no_errors(self):
        """Returns zero result, not an exception, when no errors found."""
        with Tract.open() as t:
            t.system("test")
            self._make_tool_turn(t, "c1", "grep", "ok result")

            result = t.drop_failed_tool_turns()
            assert result.turns_dropped == 0
            assert result.commits_skipped == 0
            assert result.tokens_freed == 0
            assert result.tool_names == ()

    def test_drop_failed_tool_turns_name_filter(self):
        """name= scoping limits which turns are considered."""
        with Tract.open() as t:
            t.system("test")
            self._make_tool_turn(t, "c1", "grep", "Error: grep failed", is_error=True)
            self._make_tool_turn(t, "c2", "bash", "Error: bash failed", is_error=True)

            result = t.drop_failed_tool_turns(name="grep")
            assert result.turns_dropped == 1
            assert result.tool_names == ("grep",)

            # bash error turn still compiles (was not filtered)
            ctx = t.compile()
            contents = [m.content for m in ctx.messages]
            full = " ".join(contents)
            assert "Error: bash failed" in full
            assert "Error: grep failed" not in full

    def test_drop_failed_tool_turns_skips_entire_turn(self):
        """Both call + results are gone from compile() for error turns."""
        with Tract.open() as t:
            t.system("test")
            t.user("do stuff")

            # Commit a tool call with assistant text
            t.assistant(
                "I will search for files.",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            t.tool_result("c1", "grep", "Error: no results", is_error=True)

            result = t.drop_failed_tool_turns()
            assert result.commits_skipped == 2  # call + result

            ctx = t.compile()
            contents = [m.content for m in ctx.messages]
            full = " ".join(contents)
            # Neither the assistant tool-calling message nor the result should appear
            assert "I will search for files." not in full
            assert "Error: no results" not in full

    def test_drop_failed_tool_turns_result_type(self):
        """ToolDropResult is a proper frozen dataclass with correct field types."""
        with Tract.open() as t:
            t.system("test")
            self._make_tool_turn(t, "c1", "grep", "Error", is_error=True)

            result = t.drop_failed_tool_turns()
            assert isinstance(result, ToolDropResult)
            assert isinstance(result.turns_dropped, int)
            assert isinstance(result.commits_skipped, int)
            assert isinstance(result.tokens_freed, int)
            assert isinstance(result.tool_names, tuple)

    def test_drop_failed_tool_turns_tokens_freed(self):
        """tokens_freed sums call + result token counts."""
        with Tract.open() as t:
            t.system("test")
            self._make_tool_turn(t, "c1", "grep", "x " * 100, is_error=True)

            result = t.drop_failed_tool_turns()
            assert result.tokens_freed > 0


class TestSummarizeIncludeContext:
    """Tests for context-aware summarize()."""

    def test_summarize_include_context(self):
        """include_context=True passes conversation context to the LLM."""
        with Tract.open() as t:
            t._llm_client = MockLLMClient(["Relevant summary."])
            t.system("You are a code helper.")
            t.user("Find the main function.")

            t.assistant(
                "Searching...",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )

            pending = t.tool_result("c1", "grep", "verbose output " * 20, review=True)
            pending.summarize(include_context=True)

            # Verify context was included in the prompt
            llm = t._llm_client
            user_msg = llm.last_messages[1]["content"]
            assert "conversation so far" in user_msg
            assert "You are a code helper" in user_msg
            assert "Find the main function" in user_msg

            # Verify context-aware system prompt was used
            sys_msg = llm.last_messages[0]["content"]
            from tract.prompts.summarize import TOOL_CONTEXT_SUMMARIZE_SYSTEM
            assert sys_msg == TOOL_CONTEXT_SUMMARIZE_SYSTEM

    def test_summarize_custom_system_prompt(self):
        """system_prompt= overrides the default system prompt."""
        with Tract.open() as t:
            t._llm_client = MockLLMClient(["Custom summary."])
            t.system("test")

            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )

            pending = t.tool_result("c1", "grep", "output", review=True)
            pending.summarize(system_prompt="You are a custom summarizer.")

            llm = t._llm_client
            sys_msg = llm.last_messages[0]["content"]
            assert sys_msg == "You are a custom summarizer."

    def test_summarize_custom_system_prompt_with_context(self):
        """Explicit system_prompt takes priority over TOOL_CONTEXT_SUMMARIZE_SYSTEM."""
        with Tract.open() as t:
            t._llm_client = MockLLMClient(["Summary."])
            t.system("test")

            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )

            pending = t.tool_result("c1", "grep", "output", review=True)
            pending.summarize(include_context=True, system_prompt="Custom prompt.")

            llm = t._llm_client
            sys_msg = llm.last_messages[0]["content"]
            assert sys_msg == "Custom prompt."
            # Context should still be in the user message
            user_msg = llm.last_messages[1]["content"]
            assert "conversation so far" in user_msg

    def test_configure_tool_summarization_include_context(self):
        """include_context threads through configure_tool_summarization hook."""
        with Tract.open() as t:
            t._llm_client = MockLLMClient(["Auto summary."])
            t.system("test context")
            t.user("do something")

            t.configure_tool_summarization(
                instructions={"grep": "summarize grep results"},
                include_context=True,
            )

            t.assistant(
                "",
                metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
            )
            ci = t.tool_result("c1", "grep", "verbose " * 30)

            # Verify the LLM was called with context
            llm = t._llm_client
            user_msg = llm.last_messages[1]["content"]
            assert "conversation so far" in user_msg
            assert "test context" in user_msg
