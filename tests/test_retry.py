"""Tests for the core retry protocol (retry_with_steering).

Tests the generic retry loop, validation, steering, purification,
provenance notes, and error handling -- independent of any specific
Tract operation (chat, compression, etc.).
"""

from __future__ import annotations

import pytest

from tract.exceptions import RetryExhaustedError
from tract.retry import RetryResult, retry_with_steering


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CallTracker:
    """Tracks calls to attempt/validate/steer/head/reset/provenance."""

    def __init__(
        self,
        *,
        results: list | None = None,
        validations: list[tuple[bool, str | None]] | None = None,
    ):
        self.results = results or ["result"]
        self.validations = validations or [(True, None)]
        self._attempt_count = 0
        self._validate_count = 0
        self.steer_calls: list[str] = []
        self.head_calls: int = 0
        self.reset_calls: list[str] = []
        self.provenance_calls: list[tuple[int, list[str]]] = []

    def attempt(self):
        idx = min(self._attempt_count, len(self.results) - 1)
        result = self.results[idx]
        self._attempt_count += 1
        return result

    def validate(self, result):
        idx = min(self._validate_count, len(self.validations) - 1)
        val = self.validations[idx]
        self._validate_count += 1
        return val

    def steer(self, diagnosis: str):
        self.steer_calls.append(diagnosis)

    def head_fn(self) -> str:
        self.head_calls += 1
        return "restore-abc123"

    def reset_fn(self, target: str):
        self.reset_calls.append(target)

    def provenance_note(self, attempts: int, history: list[str]):
        self.provenance_calls.append((attempts, list(history)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetryWithSteering:
    """Core retry protocol tests."""

    def test_first_attempt_succeeds(self):
        """Validator passes on first try -- no retries needed."""
        tracker = CallTracker(
            results=["good"],
            validations=[(True, None)],
        )

        result = retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
        )

        assert isinstance(result, RetryResult)
        assert result.value == "good"
        assert result.attempts == 1
        assert result.history is None
        assert len(tracker.steer_calls) == 0
        assert len(tracker.reset_calls) == 0

    def test_retry_succeeds_on_second(self):
        """Fails once, then succeeds on second attempt."""
        tracker = CallTracker(
            results=["bad", "good"],
            validations=[
                (False, "too short"),
                (True, None),
            ],
        )

        result = retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
        )

        assert result.value == "good"
        assert result.attempts == 2
        assert result.history == ["too short"]
        assert tracker.steer_calls == ["too short"]

    def test_retry_succeeds_on_third(self):
        """Fails twice, then succeeds on third attempt."""
        tracker = CallTracker(
            results=["bad1", "bad2", "good"],
            validations=[
                (False, "error A"),
                (False, "error B"),
                (True, None),
            ],
        )

        result = retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
            max_retries=3,
        )

        assert result.value == "good"
        assert result.attempts == 3
        assert result.history == ["error A", "error B"]
        assert tracker.steer_calls == ["error A", "error B"]

    def test_all_retries_exhausted(self):
        """All attempts fail -- raises RetryExhaustedError."""
        tracker = CallTracker(
            results=["bad"] * 3,
            validations=[(False, "still bad")] * 3,
        )

        with pytest.raises(RetryExhaustedError) as exc_info:
            retry_with_steering(
                attempt=tracker.attempt,
                validate=tracker.validate,
                steer=tracker.steer,
                head_fn=tracker.head_fn,
                reset_fn=tracker.reset_fn,
                max_retries=3,
            )

        err = exc_info.value
        assert err.attempts == 3
        assert err.last_diagnosis == "still bad"
        assert err.last_result == "bad"
        assert "All 3 retry attempts failed" in str(err)

    def test_purify_calls_reset(self):
        """purify=True calls reset_fn on success after retries."""
        tracker = CallTracker(
            results=["bad", "good"],
            validations=[
                (False, "wrong format"),
                (True, None),
            ],
        )

        result = retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
            purify=True,
        )

        assert result.value == "good"
        assert result.attempts == 2
        # reset_fn called with the restore point from head_fn
        assert tracker.reset_calls == ["restore-abc123"]

    def test_purify_no_reset_on_first_success(self):
        """purify=True does NOT reset if first attempt succeeds (nothing to purify)."""
        tracker = CallTracker(
            results=["good"],
            validations=[(True, None)],
        )

        result = retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
            purify=True,
        )

        assert result.value == "good"
        assert result.attempts == 1
        assert tracker.reset_calls == []

    def test_provenance_note_called(self):
        """provenance_note receives correct attempts and history."""
        tracker = CallTracker(
            results=["bad", "good"],
            validations=[
                (False, "quality issue"),
                (True, None),
            ],
        )

        result = retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
            provenance_note=tracker.provenance_note,
        )

        assert result.attempts == 2
        assert len(tracker.provenance_calls) == 1
        attempts, history = tracker.provenance_calls[0]
        assert attempts == 2
        assert history == ["quality issue"]

    def test_provenance_note_on_first_success(self):
        """provenance_note is also called when first attempt succeeds."""
        tracker = CallTracker(
            results=["good"],
            validations=[(True, None)],
        )

        retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
            provenance_note=tracker.provenance_note,
        )

        assert len(tracker.provenance_calls) == 1
        attempts, history = tracker.provenance_calls[0]
        assert attempts == 1
        assert history == []

    def test_steer_called_with_diagnosis(self):
        """steer receives the diagnosis string from validate."""
        tracker = CallTracker(
            results=["bad", "good"],
            validations=[
                (False, "missing required field: name"),
                (True, None),
            ],
        )

        retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
        )

        assert tracker.steer_calls == ["missing required field: name"]

    def test_history_recorded(self):
        """RetryResult.history contains all diagnosis strings."""
        tracker = CallTracker(
            results=["a", "b", "c"],
            validations=[
                (False, "problem 1"),
                (False, "problem 2"),
                (True, None),
            ],
        )

        result = retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
            max_retries=3,
        )

        assert result.history == ["problem 1", "problem 2"]

    def test_steer_not_called_on_last_failure(self):
        """steer is NOT called after the final failed attempt (no point steering)."""
        tracker = CallTracker(
            results=["bad", "bad"],
            validations=[(False, "nope")] * 2,
        )

        with pytest.raises(RetryExhaustedError):
            retry_with_steering(
                attempt=tracker.attempt,
                validate=tracker.validate,
                steer=tracker.steer,
                head_fn=tracker.head_fn,
                reset_fn=tracker.reset_fn,
                max_retries=2,
            )

        # steer called once (after first failure), NOT after second
        assert len(tracker.steer_calls) == 1

    def test_max_retries_one(self):
        """max_retries=1 means only one attempt, no retries."""
        tracker = CallTracker(
            results=["bad"],
            validations=[(False, "fail")],
        )

        with pytest.raises(RetryExhaustedError) as exc_info:
            retry_with_steering(
                attempt=tracker.attempt,
                validate=tracker.validate,
                steer=tracker.steer,
                head_fn=tracker.head_fn,
                reset_fn=tracker.reset_fn,
                max_retries=1,
            )

        assert exc_info.value.attempts == 1

    def test_head_fn_called_once(self):
        """head_fn is called exactly once at the start to get restore point."""
        tracker = CallTracker(
            results=["bad", "good"],
            validations=[(False, "err"), (True, None)],
        )

        retry_with_steering(
            attempt=tracker.attempt,
            validate=tracker.validate,
            steer=tracker.steer,
            head_fn=tracker.head_fn,
            reset_fn=tracker.reset_fn,
        )

        assert tracker.head_calls == 1


class TestRetryResult:
    """Tests for the RetryResult dataclass."""

    def test_frozen(self):
        """RetryResult is frozen (immutable)."""
        result = RetryResult(value="ok", attempts=1)
        with pytest.raises(AttributeError):
            result.value = "changed"  # type: ignore[misc]

    def test_defaults(self):
        """Default history is None."""
        result = RetryResult(value="x", attempts=1)
        assert result.history is None


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError."""

    def test_attributes(self):
        """Error stores attempts, diagnosis, and last_result."""
        err = RetryExhaustedError(5, "bad output", last_result={"text": "nope"})
        assert err.attempts == 5
        assert err.last_diagnosis == "bad output"
        assert err.last_result == {"text": "nope"}

    def test_is_trace_error(self):
        """RetryExhaustedError is a TraceError."""
        from tract.exceptions import TraceError

        err = RetryExhaustedError(1, "failed")
        assert isinstance(err, TraceError)

    def test_str_message(self):
        """String representation includes attempt count and diagnosis."""
        err = RetryExhaustedError(3, "invalid JSON")
        assert "3" in str(err)
        assert "invalid JSON" in str(err)
