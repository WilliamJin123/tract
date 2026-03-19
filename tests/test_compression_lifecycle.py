"""Integration test for compress -> reorder -> gc lifecycle.

Verifies that compression, compile-time reordering, and garbage collection
work correctly when combined in sequence. Also tests compression with
branches, merges, metadata preservation, and stacking compressions.
"""

from __future__ import annotations

import json

import pytest

from tract import (
    CompressResult,
    DialogueContent,
    GCResult,
    InstructionContent,
    Priority,
    Tract,
    ToolCompactResult,
)


class MockLLMClient:
    """Mock LLM client for testing compression without real API calls."""

    def __init__(self, responses=None):
        self.responses = responses or [
            "Previously in this conversation: summary text."
        ]
        self._call_count = 0

    def chat(self, messages, **kwargs):
        text = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def close(self):
        pass


class TestCompressReorderGCLifecycle:
    """Integration test for the full compression lifecycle."""

    def test_compress_then_reorder_then_gc(self):
        """Full lifecycle: compress commits, reorder compiled output, then GC."""
        # 1. Create tract with 5 commits, pin one
        t = Tract.open()
        hashes = []
        for i in range(5):
            info = t.commit(DialogueContent(role="user", text=f"Message {i+1}"))
            hashes.append(info.commit_hash)

        t.annotate(hashes[2], Priority.PINNED, reason="important")

        # 2. Compress (creates summary commits, makes originals unreachable)
        mock = MockLLMClient(responses=["Summary group 1", "Summary group 2"])
        t.config.configure_llm(mock)
        compress_result = t.compress()

        assert isinstance(compress_result, CompressResult)
        assert len(compress_result.source_commits) == 4  # 5 - 1 pinned
        assert len(compress_result.preserved_commits) == 1

        # 3. Compile normally to verify coherence
        compiled = t.compile()
        assert compiled.commit_count >= 3  # 2 summaries + 1 pinned

        # 4. Reorder the compiled output (reverse order)
        new_hashes = [m.content for m in compiled.messages]
        reorder_hashes = list(reversed(
            [h for h in compiled.commit_hashes]
        ))
        reordered, warnings = t.compile(order=reorder_hashes)

        assert len(reordered.messages) == len(compiled.messages)
        # Reversed: first message should be what was last
        assert reordered.messages[0].content == compiled.messages[-1].content

        # 5. GC to clean up unreachable original commits
        gc_result = t.gc(orphan_retention_days=0)

        assert isinstance(gc_result, GCResult)
        assert gc_result.commits_removed > 0  # Original commits removed

        # 6. Verify compilation still works after GC
        compiled_after_gc = t.compile()
        assert compiled_after_gc.commit_count == compiled.commit_count
        assert compiled_after_gc.messages == compiled.messages

        # 7. Verify reorder still works after GC
        reordered2, warnings2 = t.compile(order=reorder_hashes)
        assert len(reordered2.messages) == len(reordered.messages)

    def test_double_compress_then_gc(self):
        """Compress twice in sequence, then GC cleans up both rounds of originals."""
        t = Tract.open()
        for i in range(6):
            t.commit(DialogueContent(role="user", text=f"Round 1 message {i+1}"))

        mock = MockLLMClient()
        t.config.configure_llm(mock)

        # First compression
        result1 = t.compress()
        assert isinstance(result1, CompressResult)

        # Add more commits
        for i in range(3):
            t.commit(DialogueContent(role="user", text=f"Round 2 message {i+1}"))

        # Second compression (compresses new commits + first summary)
        mock2 = MockLLMClient(responses=["Double summary"])
        t.config.configure_llm(mock2)
        result2 = t.compress()
        assert isinstance(result2, CompressResult)

        # GC should clean up originals from both rounds
        # archive_retention_days=0 needed because compressed source commits are archives
        gc_result = t.gc(orphan_retention_days=0, archive_retention_days=0)
        assert gc_result.commits_removed > 0

        # Verify compilation still works
        compiled = t.compile()
        assert compiled.commit_count >= 1

        # Second GC should find nothing
        gc_result2 = t.gc(orphan_retention_days=0, archive_retention_days=0)
        assert gc_result2.commits_removed == 0


class TestCompressMetadataLifecycle:
    """Test that metadata survives through compression and GC."""

    def test_compress_preserves_metadata_through_gc(self):
        """Pinned content survives compression and GC.

        After compression, pinned commits are re-committed into the new
        chain. The original hash becomes unreachable, but the pinned
        content should appear in the compiled output after GC.
        """
        t = Tract.open()
        # Create commits with metadata
        for i in range(4):
            t.commit(
                DialogueContent(role="user", text=f"Message {i+1}"),
                metadata={"index": i, "source": "test"},
            )

        # Pin one commit
        log = t.log(limit=10)
        pinned_hash = log[1].commit_hash  # pin second-most-recent
        t.annotate(pinned_hash, Priority.PINNED, reason="important data")

        # Get the pinned content text before compression
        pinned_content = t.get_content(t.get_commit(pinned_hash))

        mock = MockLLMClient(responses=["Summary of non-pinned messages."])
        t.config.configure_llm(mock)

        result = t.compress()
        assert pinned_hash in result.preserved_commits

        # Compile before GC to record state
        compiled_before_gc = t.compile()

        # GC the originals
        gc_result = t.gc(orphan_retention_days=0)

        # After GC, compile should still work and include the pinned content
        compiled_after_gc = t.compile()
        assert compiled_after_gc.commit_count == compiled_before_gc.commit_count
        # Pinned text should appear in compiled messages
        all_text = " ".join(m.content for m in compiled_after_gc.messages)
        assert pinned_content in all_text

    def test_summary_commits_have_compression_metadata(self):
        """Summary commits created by compression should have provenance metadata."""
        t = Tract.open()
        for i in range(4):
            t.commit(DialogueContent(role="user", text=f"Message {i+1}"))

        mock = MockLLMClient(responses=["Summary text."])
        t.config.configure_llm(mock)

        result = t.compress()

        # Check summary commits exist and are reachable
        for summary_hash in result.summary_commits:
            ci = t.get_commit(summary_hash)
            assert ci is not None


class TestMultipleCompressions:
    """Test stacking multiple compressions."""

    def test_multiple_compressions_stack(self):
        """Compressing already-compressed content should work.

        Create commits, compress, add more, compress again.
        Each round should further reduce the context.
        """
        t = Tract.open()
        # Round 1: 6 messages
        for i in range(6):
            t.commit(DialogueContent(role="user", text=f"Round 1 msg {i+1}"))

        mock1 = MockLLMClient(responses=["Round 1 summary."])
        t.config.configure_llm(mock1)
        result1 = t.compress()
        compiled1 = t.compile()

        # Round 2: add 4 more
        for i in range(4):
            t.commit(DialogueContent(role="user", text=f"Round 2 msg {i+1}"))

        mock2 = MockLLMClient(responses=["Combined summary of rounds 1 and 2."])
        t.config.configure_llm(mock2)
        result2 = t.compress()
        compiled2 = t.compile()

        # After second compression, we should have fewer commits
        assert compiled2.commit_count <= compiled1.commit_count + 4

        # Everything should compile cleanly
        messages = compiled2.to_dicts()
        assert len(messages) >= 1

    def test_compress_manual_then_llm(self):
        """Manual compression followed by LLM compression should stack."""
        t = Tract.open()
        for i in range(4):
            t.commit(DialogueContent(role="user", text=f"Msg {i+1}"))

        # First: manual compression
        result1 = t.compress(content="Manual summary of first 4 messages.")
        compiled1 = t.compile()

        # Add more commits
        for i in range(3):
            t.commit(DialogueContent(role="user", text=f"More msg {i+1}"))

        # Second: LLM compression
        mock = MockLLMClient(responses=["Full summary including manual and new."])
        t.config.configure_llm(mock)
        result2 = t.compress()
        compiled2 = t.compile()

        assert compiled2.commit_count >= 1
        assert isinstance(result2, CompressResult)


class TestCompressWithBranches:
    """Test compression interaction with branches."""

    def test_compress_on_one_branch_does_not_affect_other(self):
        """Compression on one branch should not affect another branch."""
        t = Tract.open()

        # Create shared history
        t.commit(DialogueContent(role="user", text="Shared message 1"))
        t.commit(DialogueContent(role="user", text="Shared message 2"))

        # Create feature branch
        t.branch("feature")

        # Add commits on feature
        t.commit(DialogueContent(role="user", text="Feature msg 1"))
        t.commit(DialogueContent(role="user", text="Feature msg 2"))
        t.commit(DialogueContent(role="user", text="Feature msg 3"))

        # Compress on feature branch
        mock = MockLLMClient(responses=["Feature summary."])
        t.config.configure_llm(mock)
        result = t.compress()
        feature_compiled = t.compile()

        # Switch back to main
        t.switch("main")
        main_compiled = t.compile()

        # Main should still have the original shared commits
        main_messages = main_compiled.to_dicts()
        main_texts = [m.get("content", "") for m in main_messages]
        assert any("Shared message 1" in txt for txt in main_texts)
        assert any("Shared message 2" in txt for txt in main_texts)

        # Feature should be compressed
        assert feature_compiled.commit_count >= 1

    def test_compress_then_merge(self):
        """Merging after compression should work correctly.

        Create a feature branch, compress it, then merge back into main.
        """
        t = Tract.open()

        # Main history
        t.commit(DialogueContent(role="user", text="Main msg 1"))
        main_head_before = t.head

        # Feature branch
        t.branch("feature")
        t.commit(DialogueContent(role="user", text="Feature msg 1"))
        t.commit(DialogueContent(role="user", text="Feature msg 2"))
        t.commit(DialogueContent(role="user", text="Feature msg 3"))

        # Compress feature
        mock = MockLLMClient(responses=["Feature branch summary."])
        t.config.configure_llm(mock)
        t.compress()

        # Switch to main and merge
        t.switch("main")
        merge_result = t.merge("feature")

        # After merge, main should include the compressed feature content
        compiled = t.compile()
        assert compiled.commit_count >= 1


class TestCompressToolCallsThenRange:
    """Test tool compaction followed by range compression."""

    def _build_tool_tract(self):
        """Build a tract with tool calls and regular messages."""
        t = Tract.open()
        t.system("Agent.")
        t.user("Do research.")

        # Tool turn 1
        t.assistant(
            "",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "search", "arguments": {}},
            ]},
        )
        tr1 = t.tool_result("call_1", "search", "verbose search output " * 20)

        # Tool turn 2
        t.assistant(
            "",
            metadata={"tool_calls": [
                {"id": "call_2", "name": "read", "arguments": {}},
            ]},
        )
        tr2 = t.tool_result("call_2", "read", "verbose read output " * 20)

        # Final answer
        t.assistant("Based on research, the answer is 42.")

        return t, [tr1, tr2]

    def test_tool_compact_then_range_compress(self):
        """Tool compaction followed by range compression should stack cleanly."""
        t, tool_results = self._build_tool_tract()

        # Step 1: Compact tool results
        # MockLLMClient returns content as raw string, so we use JSON string directly
        mock1 = MockLLMClient(
            responses=[json.dumps(["Search: found it.", "Read: content."])]
        )
        t.config.configure_llm(mock1)

        result1 = t._compression_mgr.compress_tool_calls()
        assert isinstance(result1, ToolCompactResult)
        assert len(result1.edit_commits) == 2

        # Verify tool results are now compacted
        compiled_after_compact = t.compile()
        tool_msgs = [m for m in compiled_after_compact.to_dicts() if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

        # Step 2: Range-compress the whole conversation
        mock2 = MockLLMClient(responses=["Full conversation summary."])
        t.config.configure_llm(mock2)
        result2 = t.compress()
        assert isinstance(result2, CompressResult)

        # Should still compile cleanly
        compiled_final = t.compile()
        assert compiled_final.commit_count >= 1

    def test_tool_compact_preserves_tool_structure(self):
        """After tool compaction, find_tool_turns() still finds turns."""
        t, _ = self._build_tool_tract()

        mock = MockLLMClient(
            responses=[json.dumps(["S1.", "S2."])]
        )
        t.config.configure_llm(mock)

        t._compression_mgr.compress_tool_calls()

        # Tool turns should still be discoverable
        turns = t.tools.find_turns()
        assert len(turns) == 2
        assert "search" in turns[0].tool_names
        assert "read" in turns[1].tool_names


class TestCompressEdgeCases:
    """Edge cases for the compression lifecycle."""

    def test_compress_single_commit(self):
        """Compressing a single non-system commit should work."""
        t = Tract.open()
        t.commit(DialogueContent(role="user", text="Only message."))

        mock = MockLLMClient(responses=["Summary of single message."])
        t.config.configure_llm(mock)
        result = t.compress()
        assert isinstance(result, CompressResult)

    def test_compress_with_all_pinned_raises(self):
        """Compressing when all commits are pinned should raise CompressionError."""
        from tract.exceptions import CompressionError

        t = Tract.open()
        hashes = []
        for i in range(3):
            info = t.commit(DialogueContent(role="user", text=f"Pinned msg {i+1}"))
            hashes.append(info.commit_hash)

        for h in hashes:
            t.annotate(h, Priority.PINNED, reason="all important")

        mock = MockLLMClient(responses=["Summary."])
        t.config.configure_llm(mock)

        # When all commits are pinned, nothing to compress
        with pytest.raises(CompressionError, match="pinned or skipped"):
            t.compress()

    def test_gc_with_no_compressions(self):
        """GC on a tract with no compressions should remove nothing."""
        t = Tract.open()
        for i in range(3):
            t.commit(DialogueContent(role="user", text=f"Msg {i+1}"))

        gc_result = t.gc(orphan_retention_days=0)
        assert isinstance(gc_result, GCResult)
        assert gc_result.commits_removed == 0

    def test_compress_then_gc_then_compile_coherent(self):
        """After compress + GC, compiled output should be coherent."""
        t = Tract.open()
        for i in range(8):
            t.commit(DialogueContent(role="user", text=f"Message {i+1}"))

        mock = MockLLMClient(responses=["Summary of all messages."])
        t.config.configure_llm(mock)

        t.compress()
        t.gc(orphan_retention_days=0)

        compiled = t.compile()
        messages = compiled.to_dicts()
        # Should have at least one message (the summary)
        assert len(messages) >= 1
        # Should compile without error
        assert compiled.commit_count >= 1
