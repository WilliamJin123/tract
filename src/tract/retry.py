"""Unified retry protocol for Trace operations.

Provides retry_with_steering() -- a generic retry loop that validates
results, steers the LLM via diagnosis feedback, and optionally purifies
the commit history on success.

The retry protocol works for any operation that produces a result that
can be validated: chat/generate (validate response text), compression
(validate summary quality), or any future LLM-backed operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from tract.exceptions import RetryExhaustedError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    """Result of a retry-guarded operation.

    Attributes:
        value: The successful result value.
        attempts: Total attempts (1 = first try succeeded).
        history: Brief log of failure diagnoses (None if first try succeeded).
    """

    value: T
    attempts: int
    history: list[str] | None = None


def retry_with_steering(
    *,
    attempt: callable,
    validate: callable,
    steer: callable,
    head_fn: callable,
    reset_fn: callable,
    max_retries: int = 3,
    purify: bool = False,
    provenance_note: callable | None = None,
) -> RetryResult:
    """Execute an operation with validation, steering, and optional purification.

    Flow:
        1. Save restore_point = head_fn()
        2. result = attempt()
        3. (ok, diagnosis) = validate(result)
        4. If ok: optionally call provenance_note, return RetryResult
        5. If attempts >= max_retries: raise RetryExhaustedError
        6. steer(diagnosis) -- inject steering feedback
        7. Goto 2

    If purify=True AND success: calls reset_fn(restore_point) before
    returning. The CALLER is responsible for re-committing clean results.

    Args:
        attempt: Callable that produces a result (e.g. LLM call + commit).
        validate: Callable taking the result, returns (ok, diagnosis).
            diagnosis is None on success, a string on failure.
        steer: Callable taking a diagnosis string, injects steering
            feedback (e.g. commits a user message with the diagnosis).
        head_fn: Callable returning the current HEAD hash (restore point).
        reset_fn: Callable taking a hash, resets HEAD to that point.
        max_retries: Maximum total attempts (default 3).
        purify: If True, reset to restore_point on success so caller
            can re-commit clean results without retry artifacts.
        provenance_note: Optional callable(attempts, history) called on
            success to record retry metadata.

    Returns:
        RetryResult with the successful value, attempt count, and history.

    Raises:
        RetryExhaustedError: If all attempts fail validation.
    """
    restore_point = head_fn()
    history: list[str] = []
    last_diagnosis: str | None = None

    for attempt_num in range(1, max_retries + 1):
        result = attempt()
        ok, diagnosis = validate(result)

        if ok:
            if provenance_note is not None:
                provenance_note(attempt_num, history if history else [])

            if purify and attempt_num > 1:
                reset_fn(restore_point)

            return RetryResult(
                value=result,
                attempts=attempt_num,
                history=history if history else None,
            )

        # Failed -- record and steer
        last_diagnosis = diagnosis or "validation failed"
        history.append(last_diagnosis)

        if attempt_num < max_retries:
            steer(last_diagnosis)

    raise RetryExhaustedError(
        attempts=max_retries,
        last_diagnosis=last_diagnosis or "validation failed",
        last_result=result,  # type: ignore[possibly-undefined]
    )
