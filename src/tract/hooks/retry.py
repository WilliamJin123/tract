"""Convenience retry wrapper for hookable operations.

Provides auto_retry() -- a standard validate->retry loop for
Pending objects that support validate() and retry() methods.

Supported types:
- PendingCompress: per-summary validate->retry loop
- PendingMerge: single-shot validate->retry->validate
- Generic fallback: validate once, approve or reject
"""

from __future__ import annotations

from typing import Any

from tract.hooks.compress import PendingCompress
from tract.hooks.merge import PendingMerge
from tract.hooks.validation import HookRejection


def auto_retry(pending: Any, *, max_retries: int = 3) -> Any:
    """Validate and retry a pending operation automatically.

    For PendingCompress: calls validate() on summaries, retries failed
    ones with the diagnosis as guidance, up to max_retries total
    attempts. If all pass, calls approve(). If retries are exhausted,
    rejects with the last diagnosis.

    For PendingMerge: validates all resolutions, retries with diagnosis
    as guidance if any fail, then re-validates once. Approves if all
    pass, rejects otherwise.

    Args:
        pending: A Pending subclass with validate() and retry() methods.
        max_retries: Maximum number of retry attempts per failing item
            (default 3).

    Returns:
        The result of pending.approve() if all validations pass.

    Raises:
        TypeError: If pending does not support validate().
    """
    if isinstance(pending, PendingCompress):
        return _auto_retry_compress(pending, max_retries=max_retries)

    if isinstance(pending, PendingMerge):
        return _auto_retry_merge(pending, max_retries=max_retries)

    # Generic fallback for future Pending types with validate/retry
    if not hasattr(pending, "validate"):
        raise TypeError(
            f"{type(pending).__name__} does not support validate(). "
            f"auto_retry() requires a Pending subclass with validate() and retry()."
        )

    # Generic single-shot: validate once, approve or reject
    result = pending.validate()
    if result.passed:
        return pending.approve()
    else:
        pending.reject(reason=result.diagnosis or "Validation failed")
        return HookRejection(
            reason=result.diagnosis or "Validation failed",
            pending=pending,
            rejection_source="validation",
        )


def _auto_retry_compress(
    pending: PendingCompress, *, max_retries: int = 3
) -> Any:
    """Validate-and-retry loop specialized for PendingCompress.

    Iterates over each summary. For each failing summary, calls
    retry(index, guidance=diagnosis) up to max_retries times. If
    all summaries pass, approves. If any summary still fails after
    retries, rejects with the accumulated diagnosis.

    Args:
        pending: The PendingCompress to validate and retry.
        max_retries: Maximum retry attempts per failing summary.

    Returns:
        CompressResult from approve() if all summaries pass.
    """
    last_diagnosis: str | None = None

    for _attempt in range(max_retries):
        # Validate all summaries
        validation = pending.validate()

        if validation.passed:
            return pending.approve()

        last_diagnosis = validation.diagnosis

        # If validation failed with a specific index, retry just that one
        if validation.index is not None:
            pending.retry(
                validation.index,
                guidance=validation.diagnosis or "",
            )
        else:
            # Whole-operation validation failure -- retry first summary
            # as a heuristic (specific implementations may do better)
            pending.retry(0, guidance=validation.diagnosis or "")

    # Exhausted retries -- reject
    reason = last_diagnosis or "Validation failed after all retries"
    pending.reject(reason=reason)
    return HookRejection(
        reason=reason,
        pending=pending,
        rejection_source="validation",
        metadata={"max_retries": max_retries},
    )


def _auto_retry_merge(
    pending: PendingMerge, *, max_retries: int = 3
) -> Any:
    """Validate-and-retry loop specialized for PendingMerge.

    Single-shot flow: validate -> if failed, retry with diagnosis as
    guidance -> re-validate -> approve or reject. Merge retry re-resolves
    all conflicts at once, so the loop is simpler than compress.

    Args:
        pending: The PendingMerge to validate and retry.
        max_retries: Maximum retry attempts (default 3).

    Returns:
        MergeResult from approve() if all resolutions pass.
    """
    last_diagnosis: str | None = None

    for _attempt in range(max_retries):
        validation = pending.validate()

        if validation.passed:
            return pending.approve()

        last_diagnosis = validation.diagnosis

        # Retry with diagnosis as guidance -- re-resolves all conflicts
        pending.retry(guidance=validation.diagnosis or "")

    # Exhausted retries -- reject
    reason = last_diagnosis or "Validation failed after all retries"
    pending.reject(reason=reason)
    return HookRejection(
        reason=reason,
        pending=pending,
        rejection_source="validation",
        metadata={"max_retries": max_retries},
    )
