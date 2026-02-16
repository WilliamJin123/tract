"""Integration test for compress -> reorder -> gc lifecycle.

Verifies that compression, compile-time reordering, and garbage collection
work correctly when combined in sequence.
"""

from __future__ import annotations

import pytest

from tract import (
    CompressResult,
    DialogueContent,
    GCResult,
    Priority,
    Tract,
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
        t.configure_llm(mock)
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
        t.configure_llm(mock)

        # First compression
        result1 = t.compress()
        assert isinstance(result1, CompressResult)

        # Add more commits
        for i in range(3):
            t.commit(DialogueContent(role="user", text=f"Round 2 message {i+1}"))

        # Second compression (compresses new commits + first summary)
        mock2 = MockLLMClient(responses=["Double summary"])
        t.configure_llm(mock2)
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
