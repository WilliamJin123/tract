"""Tests for tract.llm.errors -- LLM-specific error hierarchy.

Covers:
- All five custom exception classes
- Inheritance chain (each -> LLMClientError -> TraceError -> Exception)
- LLMRateLimitError message formatting and retry_after attribute
- raise / catch semantics at every level of the hierarchy
"""

from __future__ import annotations

import pytest

from tract.exceptions import TraceError
from tract.llm.errors import (
    LLMAuthError,
    LLMClientError,
    LLMConfigError,
    LLMRateLimitError,
    LLMResponseError,
    LLMToolUseError,
)


# ---------------------------------------------------------------------------
# Inheritance hierarchy
# ---------------------------------------------------------------------------

class TestInheritanceHierarchy:
    """Every LLM error should be catchable as LLMClientError and TraceError."""

    @pytest.mark.parametrize(
        "cls",
        [LLMClientError, LLMConfigError, LLMRateLimitError, LLMAuthError, LLMResponseError, LLMToolUseError],
    )
    def test_all_are_trace_errors(self, cls):
        assert issubclass(cls, TraceError)

    @pytest.mark.parametrize(
        "cls",
        [LLMConfigError, LLMRateLimitError, LLMAuthError, LLMResponseError, LLMToolUseError],
    )
    def test_leaf_classes_are_llm_client_errors(self, cls):
        assert issubclass(cls, LLMClientError)

    def test_llm_client_error_is_base(self):
        assert issubclass(LLMClientError, TraceError)
        assert issubclass(LLMClientError, Exception)

    def test_mro_does_not_skip_llm_client_error(self):
        """Leaf classes must go through LLMClientError, not directly to TraceError."""
        for cls in (LLMConfigError, LLMAuthError, LLMResponseError, LLMToolUseError):
            mro = cls.__mro__
            assert mro.index(LLMClientError) < mro.index(TraceError)


# ---------------------------------------------------------------------------
# LLMClientError (base)
# ---------------------------------------------------------------------------

class TestLLMClientError:
    def test_raise_with_message(self):
        with pytest.raises(LLMClientError, match="something broke"):
            raise LLMClientError("something broke")

    def test_catchable_as_trace_error(self):
        with pytest.raises(TraceError):
            raise LLMClientError("caught at base")

    def test_str(self):
        err = LLMClientError("detail")
        assert str(err) == "detail"


# ---------------------------------------------------------------------------
# LLMConfigError
# ---------------------------------------------------------------------------

class TestLLMConfigError:
    def test_raise_with_message(self):
        with pytest.raises(LLMConfigError, match="missing API key"):
            raise LLMConfigError("missing API key")

    def test_catchable_as_llm_client_error(self):
        with pytest.raises(LLMClientError):
            raise LLMConfigError("no key")

    def test_catchable_as_trace_error(self):
        with pytest.raises(TraceError):
            raise LLMConfigError("no key")


# ---------------------------------------------------------------------------
# LLMRateLimitError -- has custom __init__ with retry_after
# ---------------------------------------------------------------------------

class TestLLMRateLimitError:
    def test_default_message(self):
        err = LLMRateLimitError()
        assert str(err) == "Rate limited"
        assert err.retry_after is None

    def test_custom_message_no_retry(self):
        err = LLMRateLimitError("slow down")
        assert str(err) == "slow down"
        assert err.retry_after is None

    def test_retry_after_appended_to_message(self):
        err = LLMRateLimitError("slow down", retry_after=30.0)
        assert "retry after 30.0s" in str(err)
        assert err.retry_after == 30.0

    def test_retry_after_with_default_message(self):
        err = LLMRateLimitError(retry_after=5)
        assert str(err) == "Rate limited (retry after 5s)"
        assert err.retry_after == 5

    def test_retry_after_zero(self):
        err = LLMRateLimitError(retry_after=0)
        # 0 is not None, so message should include it
        assert "retry after 0s" in str(err)
        assert err.retry_after == 0

    def test_catchable_as_llm_client_error(self):
        with pytest.raises(LLMClientError):
            raise LLMRateLimitError()

    def test_catchable_as_trace_error(self):
        with pytest.raises(TraceError):
            raise LLMRateLimitError()


# ---------------------------------------------------------------------------
# LLMAuthError
# ---------------------------------------------------------------------------

class TestLLMAuthError:
    def test_raise_with_message(self):
        with pytest.raises(LLMAuthError, match="401 Unauthorized"):
            raise LLMAuthError("401 Unauthorized")

    def test_catchable_as_llm_client_error(self):
        with pytest.raises(LLMClientError):
            raise LLMAuthError("bad token")

    def test_empty_message(self):
        err = LLMAuthError()
        assert str(err) == ""


# ---------------------------------------------------------------------------
# LLMResponseError
# ---------------------------------------------------------------------------

class TestLLMResponseError:
    def test_raise_with_message(self):
        with pytest.raises(LLMResponseError, match="unexpected JSON"):
            raise LLMResponseError("unexpected JSON")

    def test_catchable_as_llm_client_error(self):
        with pytest.raises(LLMClientError):
            raise LLMResponseError("bad body")


# ---------------------------------------------------------------------------
# LLMToolUseError
# ---------------------------------------------------------------------------

class TestLLMToolUseError:
    def test_raise_with_message(self):
        with pytest.raises(LLMToolUseError, match="truncated"):
            raise LLMToolUseError("Tool call truncated")

    def test_catchable_as_llm_client_error(self):
        with pytest.raises(LLMClientError):
            raise LLMToolUseError("tool_use_failed")

    def test_catchable_as_trace_error(self):
        with pytest.raises(TraceError):
            raise LLMToolUseError("tool_use_failed")

    def test_str(self):
        err = LLMToolUseError("max_tokens too low")
        assert str(err) == "max_tokens too low"


# ---------------------------------------------------------------------------
# Cross-cutting: catch-all handler patterns
# ---------------------------------------------------------------------------

class TestCatchAllPatterns:
    """Verify that a single except LLMClientError catches all leaf types."""

    @pytest.mark.parametrize(
        "exc",
        [
            LLMConfigError("cfg"),
            LLMRateLimitError("rl", retry_after=1),
            LLMAuthError("auth"),
            LLMResponseError("resp"),
            LLMToolUseError("tool"),
        ],
    )
    def test_single_handler_catches_all_leaves(self, exc):
        with pytest.raises(LLMClientError):
            raise exc

    def test_except_order_specificity(self):
        """More specific handler should fire before generic one."""
        try:
            raise LLMRateLimitError(retry_after=10)
        except LLMRateLimitError as e:
            assert e.retry_after == 10
        except LLMClientError:
            pytest.fail("Generic handler should not have fired")
