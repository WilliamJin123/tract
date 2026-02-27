"""Tests for improve=True on shorthand methods (system/user/assistant).

The improve feature commits the original text first, then uses an LLM to
produce an improved version and applies it as an EDIT commit targeting the
original.  This preserves the original (recoverable via log/restore) while
giving the user polished content by default.
"""

from __future__ import annotations

import warnings

import pytest

from tract import Tract
from tract.models.annotations import Priority
from tract.models.commit import CommitOperation


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class _MockLLMClient:
    """Minimal mock conforming to the LLMClient protocol."""

    def __init__(self, response_text: str = "improved text"):
        self._response_text = response_text
        self.call_count = 0
        self.last_messages: list | None = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.call_count += 1
        return {
            "choices": [{"message": {"content": self._response_text}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model": "mock-model",
        }


class _FailingLLMClient:
    """Mock that always raises on chat()."""

    def chat(self, messages, **kwargs):
        raise RuntimeError("LLM service unavailable")


class _EmptyLLMClient:
    """Mock that returns empty content."""

    def chat(self, messages, **kwargs):
        return {
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            "model": "mock-model",
        }


class _IdenticalLLMClient:
    """Mock that returns the same text it received (no improvement)."""

    def __init__(self, original_text: str):
        self._original = original_text

    def chat(self, messages, **kwargs):
        return {
            "choices": [{"message": {"content": self._original}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            "model": "mock-model",
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUserImprove:
    """user(text, improve=True) creates an EDIT commit."""

    def test_user_improve_creates_edit(self):
        """user(improve=True) returns the EDIT commit info."""
        t = Tract.open(":memory:")
        t._llm_client = _MockLLMClient("polished user message")

        info = t.user("rough user message", improve=True)

        # The returned info should be the EDIT commit
        assert info.operation == CommitOperation.EDIT
        assert info.message == "improve: user message"

        # Log should show 2 commits: EDIT (newest) + APPEND (original)
        history = t.log()
        assert len(history) == 2
        assert history[0].operation == CommitOperation.EDIT
        assert history[1].operation == CommitOperation.APPEND


class TestAssistantImprove:
    """assistant(text, improve=True) creates an EDIT commit."""

    def test_assistant_improve_creates_edit(self):
        """assistant(improve=True) returns the EDIT commit info."""
        t = Tract.open(":memory:")
        t._llm_client = _MockLLMClient("polished assistant reply")

        info = t.assistant("rough assistant reply", improve=True)

        assert info.operation == CommitOperation.EDIT
        assert info.message == "improve: assistant message"

        history = t.log()
        assert len(history) == 2
        assert history[0].operation == CommitOperation.EDIT
        assert history[1].operation == CommitOperation.APPEND


class TestSystemImprove:
    """system(text, improve=True) creates an EDIT commit."""

    def test_system_improve_creates_edit(self):
        """system(improve=True) returns the EDIT commit info."""
        t = Tract.open(":memory:")
        t._llm_client = _MockLLMClient("polished system instruction")

        info = t.system("rough system instruction", improve=True)

        assert info.operation == CommitOperation.EDIT
        assert info.message == "improve: system message"

        history = t.log()
        assert len(history) == 2
        assert history[0].operation == CommitOperation.EDIT
        assert history[1].operation == CommitOperation.APPEND


class TestOriginalPreserved:
    """The original commit is preserved and the EDIT targets it."""

    def test_edit_targets_original(self):
        """EDIT commit's edit_target matches the original commit hash."""
        t = Tract.open(":memory:")
        t._llm_client = _MockLLMClient("better text")

        # Commit with improve
        info = t.user("original text", improve=True)

        # info is the EDIT; get the original from log
        history = t.log()
        original = history[1]  # oldest
        edit = history[0]  # newest

        assert original.operation == CommitOperation.APPEND
        assert edit.operation == CommitOperation.EDIT
        # The EDIT commit's edit_target should be the original's hash
        assert edit.edit_target == original.commit_hash


class TestImproveWithoutLLMRaises:
    """improve=True without an LLM client raises ValueError."""

    def test_user_improve_no_llm_raises(self):
        t = Tract.open(":memory:")
        # No _llm_client set
        with pytest.raises(ValueError, match="improve=True requires an LLM client"):
            t.user("text", improve=True)

    def test_system_improve_no_llm_raises(self):
        t = Tract.open(":memory:")
        with pytest.raises(ValueError, match="improve=True requires an LLM client"):
            t.system("text", improve=True)

    def test_assistant_improve_no_llm_raises(self):
        t = Tract.open(":memory:")
        with pytest.raises(ValueError, match="improve=True requires an LLM client"):
            t.assistant("text", improve=True)


class TestLLMFailureKeepsOriginal:
    """LLM failure returns original CommitInfo with a warning."""

    def test_llm_failure_returns_original_with_warning(self):
        t = Tract.open(":memory:")
        t._llm_client = _FailingLLMClient()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = t.user("my message", improve=True)

        # Should return the original APPEND commit
        assert info.operation == CommitOperation.APPEND

        # Only 1 commit in history (no EDIT)
        history = t.log()
        assert len(history) == 1

        # Warning was issued
        assert len(w) == 1
        assert "LLM improvement failed" in str(w[0].message)
        assert "user" in str(w[0].message)


class TestLLMReturnsEmpty:
    """LLM returns empty text -> keeps original (no useless EDIT)."""

    def test_empty_response_keeps_original(self):
        t = Tract.open(":memory:")
        t._llm_client = _EmptyLLMClient()

        info = t.user("my message", improve=True)

        assert info.operation == CommitOperation.APPEND
        history = t.log()
        assert len(history) == 1  # No EDIT created


class TestLLMReturnsIdentical:
    """LLM returns identical text -> keeps original (no useless EDIT)."""

    def test_identical_response_keeps_original(self):
        t = Tract.open(":memory:")
        t._llm_client = _IdenticalLLMClient("exact same text")

        info = t.user("exact same text", improve=True)

        assert info.operation == CommitOperation.APPEND
        history = t.log()
        assert len(history) == 1  # No EDIT created


class TestImproveDefaultFalse:
    """improve=False (default) is a noop -- no EDIT commit created."""

    def test_default_no_improve(self):
        t = Tract.open(":memory:")
        # Even with an LLM client, improve=False should not call it
        mock = _MockLLMClient("should not be called")
        t._llm_client = mock

        info = t.user("hello")

        assert info.operation == CommitOperation.APPEND
        assert mock.call_count == 0
        history = t.log()
        assert len(history) == 1

    def test_explicit_false_no_improve(self):
        t = Tract.open(":memory:")
        mock = _MockLLMClient("should not be called")
        t._llm_client = mock

        info = t.assistant("hello", improve=False)

        assert info.operation == CommitOperation.APPEND
        assert mock.call_count == 0


class TestImproveWithPriority:
    """improve=True with priority=PINNED -> annotation on original, EDIT follows."""

    def test_priority_pinned_with_improve(self):
        t = Tract.open(":memory:")
        t._llm_client = _MockLLMClient("improved pinned text")

        info = t.user(
            "important message",
            priority=Priority.PINNED,
            improve=True,
        )

        # The returned info is the EDIT commit
        assert info.operation == CommitOperation.EDIT

        # Log: EDIT (newest) + APPEND (original)
        history = t.log()
        assert len(history) == 2

        # The annotation was applied to the original (APPEND) commit
        original = history[1]
        assert original.operation == CommitOperation.APPEND

        # Verify the annotation exists on the original commit
        annotations = t.get_annotations(original.commit_hash)
        assert len(annotations) >= 1
        assert annotations[-1].priority == Priority.PINNED
