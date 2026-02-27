"""Tests for two_stage=True on compress() and regenerate_guidance().

Phase B of the hook system deferred features:
- compress(two_stage=True) generates guidance before summaries
- Guidance stored on PendingCompress
- Guidance threaded into summarize prompt
- regenerate_guidance() works after initial generation
- edit_guidance() after two_stage
- Error handling for two_stage without LLM
- Default two_stage=False skips guidance
- review=True + two_stage=True returns pending with guidance
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from tract import Tract
from tract.hooks.compress import PendingCompress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm_client(responses: list[str] | None = None):
    """Create a mock LLM client that returns sequential responses.

    Args:
        responses: List of response strings. Each .chat() call returns the next one.
            If None, returns "mock guidance" then "mock summary" repeatedly.
    """
    if responses is None:
        responses = ["mock guidance", "mock summary"]

    mock = MagicMock()
    side_effects = [
        {"choices": [{"message": {"content": r}}]}
        for r in responses
    ]
    mock.chat.side_effect = side_effects
    return mock


def _make_compressible_tract_with_llm(
    responses: list[str] | None = None,
) -> tuple[Tract, MagicMock]:
    """Create an in-memory Tract with 3 commits and a mock LLM client.

    Returns:
        Tuple of (tract, mock_client).
    """
    t = Tract.open(":memory:")
    t.system("You are a helpful assistant.")
    t.user("Hello, how are you?")
    t.assistant("I'm doing well, thank you!")

    mock_client = _make_mock_llm_client(responses)
    t._llm_client = mock_client
    return t, mock_client


def _make_compressible_tract() -> Tract:
    """Create an in-memory Tract with 3 commits (system + user + assistant)."""
    t = Tract.open(":memory:")
    t.system("You are a helpful assistant.")
    t.user("Hello, how are you?")
    t.assistant("I'm doing well, thank you!")
    return t


# ===========================================================================
# 1. two_stage=True generates guidance before summaries
# ===========================================================================


class TestTwoStageGuidanceGeneration:
    """compress(two_stage=True) generates guidance via LLM before summaries."""

    def test_two_stage_makes_guidance_call_then_summary_call(self):
        """With two_stage=True, LLM is called at least twice: once for
        guidance, then once per group for summarization."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["focus on key decisions", "summarized conversation"]
        )
        pending = t.compress(review=True, two_stage=True)

        assert isinstance(pending, PendingCompress)
        # At least 2 calls: 1 guidance + 1 summary (for 1 group)
        assert mock_client.chat.call_count >= 2

        # First call should be the guidance call (system prompt contains "guidance")
        first_call_args = mock_client.chat.call_args_list[0]
        first_messages = first_call_args[0][0]
        assert any("guidance" in msg["content"].lower() for msg in first_messages
                    if msg["role"] == "system")
        t.close()

    def test_two_stage_guidance_precedes_summary(self):
        """The guidance call happens before the summary call."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["focus on important topics", "here is the summary"]
        )
        pending = t.compress(review=True, two_stage=True)

        calls = mock_client.chat.call_args_list
        assert len(calls) >= 2

        # First call: guidance (system prompt about "analyzing a conversation")
        guidance_system = calls[0][0][0][0]  # first positional arg, first message
        assert guidance_system["role"] == "system"
        assert "analyzing" in guidance_system["content"].lower()

        # Second call: summary (system prompt about summarization)
        summary_system = calls[1][0][0][0]
        assert summary_system["role"] == "system"
        t.close()


# ===========================================================================
# 2. Guidance stored on PendingCompress
# ===========================================================================


class TestGuidanceStoredOnPending:
    """Guidance text and source are stored on the PendingCompress."""

    def test_guidance_field_populated(self):
        """PendingCompress.guidance contains the LLM-generated guidance."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["focus on architecture decisions", "compact summary"]
        )
        pending = t.compress(review=True, two_stage=True)

        assert pending.guidance == "focus on architecture decisions"
        t.close()

    def test_guidance_source_is_llm(self):
        """PendingCompress.guidance_source is 'llm' for two_stage."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["guidance text here", "summary text here"]
        )
        pending = t.compress(review=True, two_stage=True)

        assert pending.guidance_source == "llm"
        t.close()


# ===========================================================================
# 3. Guidance threaded into summarize prompt
# ===========================================================================


class TestGuidanceThreadedIntoPrompt:
    """The guidance text is included in the instructions for summarization."""

    def test_summary_call_includes_guidance_in_prompt(self):
        """The second LLM call (summarization) includes guidance text."""
        guidance_content = "Focus on API design decisions and ignore greetings"
        t, mock_client = _make_compressible_tract_with_llm(
            responses=[guidance_content, "summary incorporating guidance"]
        )
        pending = t.compress(review=True, two_stage=True)

        # The summary call is the second one
        calls = mock_client.chat.call_args_list
        assert len(calls) >= 2

        summary_call_messages = calls[1][0][0]
        # The user message in the summary call should contain the guidance
        user_messages = [m for m in summary_call_messages if m["role"] == "user"]
        assert len(user_messages) > 0
        user_text = user_messages[0]["content"]
        assert guidance_content in user_text
        t.close()


# ===========================================================================
# 4. regenerate_guidance() works after initial generation
# ===========================================================================


class TestRegenerateGuidance:
    """regenerate_guidance() re-generates guidance using LLM."""

    def test_regenerate_guidance_returns_new_text(self):
        """regenerate_guidance() returns newly generated guidance."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=[
                "initial guidance",
                "summary text",
                "regenerated guidance",  # for regenerate_guidance()
            ]
        )
        pending = t.compress(review=True, two_stage=True)
        assert pending.guidance == "initial guidance"

        new_guidance = pending.regenerate_guidance()
        assert new_guidance == "regenerated guidance"
        assert pending.guidance == "regenerated guidance"
        assert pending.guidance_source == "llm"
        t.close()

    def test_regenerate_guidance_after_edit_sets_user_plus_llm(self):
        """regenerate_guidance() after edit_guidance() sets source to 'user+llm'."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=[
                "initial guidance",
                "summary text",
                "regenerated after edit",
            ]
        )
        pending = t.compress(review=True, two_stage=True)
        pending.edit_guidance("user edited guidance")
        assert pending.guidance_source == "user"

        new_guidance = pending.regenerate_guidance()
        assert new_guidance == "regenerated after edit"
        assert pending.guidance_source == "user+llm"
        t.close()


# ===========================================================================
# 5. edit_guidance() after two_stage
# ===========================================================================


class TestEditGuidanceAfterTwoStage:
    """edit_guidance() works on two_stage PendingCompress."""

    def test_edit_guidance_changes_source_from_llm_to_user(self):
        """edit_guidance() on LLM-generated guidance sets source to 'user'."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["llm generated guidance", "summary"]
        )
        pending = t.compress(review=True, two_stage=True)
        assert pending.guidance_source == "llm"

        pending.edit_guidance("manually edited guidance")
        assert pending.guidance == "manually edited guidance"
        assert pending.guidance_source == "user"
        t.close()


# ===========================================================================
# 6. two_stage=True without LLM raises error
# ===========================================================================


class TestTwoStageWithoutLLM:
    """two_stage=True without an LLM client raises CompressionError."""

    def test_two_stage_without_llm_raises(self):
        """compress(two_stage=True) without LLM raises CompressionError."""
        from tract.exceptions import CompressionError

        t = _make_compressible_tract()
        # No LLM client configured
        with pytest.raises(CompressionError, match="two_stage=True requires an LLM"):
            t.compress(two_stage=True)
        t.close()


# ===========================================================================
# 7. two_stage=False (default) skips guidance
# ===========================================================================


class TestTwoStageFalseDefault:
    """two_stage=False (default) skips guidance generation."""

    def test_default_no_guidance_call(self):
        """Without two_stage, only summary LLM calls are made (1 per group)."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["summary only"]
        )
        pending = t.compress(review=True, two_stage=False)

        # Only 1 call: the summary call (no guidance call)
        assert mock_client.chat.call_count == 1
        assert pending.guidance is None
        assert pending.guidance_source is None
        t.close()

    def test_default_none_no_guidance_call(self):
        """With two_stage=None (default), no guidance call is made."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["summary only"]
        )
        pending = t.compress(review=True)

        assert mock_client.chat.call_count == 1
        assert pending.guidance is None
        t.close()


# ===========================================================================
# 8. review=True + two_stage=True returns pending with guidance
# ===========================================================================


class TestReviewPlusTwoStage:
    """review=True + two_stage=True returns PendingCompress with guidance populated."""

    def test_review_two_stage_returns_pending_with_guidance(self):
        """compress(review=True, two_stage=True) returns PendingCompress with
        guidance and summaries both populated."""
        t, mock_client = _make_compressible_tract_with_llm(
            responses=["detailed guidance about focus areas", "compressed summary"]
        )
        pending = t.compress(review=True, two_stage=True)

        assert isinstance(pending, PendingCompress)
        assert pending.status == "pending"
        assert pending.guidance == "detailed guidance about focus areas"
        assert pending.guidance_source == "llm"
        assert len(pending.summaries) >= 1
        assert pending.summaries[0] == "compressed summary"
        assert pending._two_stage is True
        t.close()
