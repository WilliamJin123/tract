"""Tests for configurable token tolerance in compression.

Verifies that the token_tolerance parameter controls how many tokens
above target_tokens a summary can be before the validator rejects it.
"""

from __future__ import annotations

import pytest

from tract import (
    CompressResult,
    DialogueContent,
    PendingCompress,
    RetryExhaustedError,
    Tract,
)
from tests.conftest import make_tract_with_commits


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM client that returns a fixed summary string."""

    def __init__(self, response: str = "Summary text."):
        self.response = response
        self._call_count = 0

    def chat(self, messages, **kwargs):
        self._call_count += 1
        return {
            "choices": [{"message": {"content": self.response}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    def close(self):
        pass


class TestDefaultTolerance:
    """Default tolerance (500) is applied when no token_tolerance specified."""

    def test_default_tolerance_accepts_within_500(self):
        """Summary within target + 500 passes with default tolerance."""
        # Generate a summary that is ~600 tokens (target=150, default tol=500 -> limit=650)
        # Using a short word repeated to control token count approximately
        summary = " ".join(["word"] * 500)  # ~500 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # target_tokens=150, default tolerance=500 -> limit=650
        # ~500 tokens is under 650, should pass
        result = t.compress(target_tokens=150)
        assert isinstance(result, CompressResult)

    def test_default_tolerance_rejects_over_limit(self):
        """Summary exceeding target + 500 fails with default tolerance."""
        # Generate a summary that is ~700+ tokens (target=100, default tol=500 -> limit=600)
        summary = " ".join(["word"] * 700)  # ~700 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # target_tokens=100, default tolerance=500 -> limit=600
        # ~700 tokens exceeds 600, should fail validation and exhaust retries
        with pytest.raises(RetryExhaustedError):
            t.compress(target_tokens=100, max_retries=1)


class TestCustomTolerance:
    """Custom token_tolerance is respected."""

    def test_custom_tolerance_accepts_within_range(self):
        """Summary within target + custom tolerance passes."""
        summary = " ".join(["word"] * 200)  # ~200 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # target_tokens=100, token_tolerance=200 -> limit=300
        # ~200 tokens is under 300, should pass
        result = t.compress(target_tokens=100, token_tolerance=200)
        assert isinstance(result, CompressResult)

    def test_custom_tolerance_rejects_over_range(self):
        """Summary exceeding target + custom tolerance fails."""
        summary = " ".join(["word"] * 400)  # ~400 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # target_tokens=100, token_tolerance=50 -> limit=150
        # ~400 tokens exceeds 150, should fail and exhaust retries
        with pytest.raises(RetryExhaustedError):
            t.compress(target_tokens=100, token_tolerance=50, max_retries=1)


class TestToleranceEdgeCases:
    """Edge cases for token_tolerance behavior."""

    def test_tolerance_zero_strict_mode(self):
        """token_tolerance=0 means only target_tokens allowed (strict)."""
        # A summary with ~200 tokens should fail with target=50, tolerance=0 (limit=50)
        summary = " ".join(["word"] * 200)  # ~200 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        with pytest.raises(RetryExhaustedError):
            t.compress(target_tokens=50, token_tolerance=0, max_retries=1)

    def test_tolerance_zero_passes_when_under_target(self):
        """token_tolerance=0 passes when summary is at or below target_tokens."""
        summary = "Short."  # ~2 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # target_tokens=1000, tolerance=0 -> limit=1000
        # ~2 tokens is well under 1000, should pass
        result = t.compress(target_tokens=1000, token_tolerance=0)
        assert isinstance(result, CompressResult)

    def test_no_target_tokens_ignores_tolerance(self):
        """When target_tokens is None, token_tolerance has no effect."""
        summary = " ".join(["word"] * 1000)  # ~1000 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # No target_tokens -- token validation should not trigger at all
        result = t.compress(token_tolerance=50)
        assert isinstance(result, CompressResult)

    def test_large_tolerance_allows_verbose_summaries(self):
        """A large tolerance allows significantly longer summaries."""
        summary = " ".join(["word"] * 800)  # ~800 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # target_tokens=100, token_tolerance=1000 -> limit=1100
        # ~800 tokens is under 1100, should pass
        result = t.compress(target_tokens=100, token_tolerance=1000)
        assert isinstance(result, CompressResult)

    def test_review_mode_with_tolerance(self):
        """token_tolerance works correctly with review=True."""
        summary = " ".join(["word"] * 200)  # ~200 tokens
        mock = MockLLMClient(response=summary)
        t, hashes = make_tract_with_commits(3)
        t.configure_llm(mock)

        # review=True returns PendingCompress, but validation still ran during generation
        pending = t.compress(
            target_tokens=100, token_tolerance=200, review=True,
        )
        assert isinstance(pending, PendingCompress)
        # The summary was generated successfully (within tolerance)
        assert len(pending.summaries) >= 1
