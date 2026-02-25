"""Validation and rejection data classes for the hook system.

Provides ValidationResult for per-item validation feedback and
HookRejection for structured rejection reporting back to callers
(e.g. policy feedback loops).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.hooks.pending import Pending


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating one item in a pending operation.

    Used by validate() methods on Pending subclasses and consumed
    by auto_retry() to drive the retry loop.

    Attributes:
        passed: Whether the validation passed.
        diagnosis: Human-readable explanation of the failure, or None
            if validation passed.
        index: Index of the item that failed (e.g. summary index in
            PendingCompress), or None for whole-operation validation.
    """

    passed: bool
    diagnosis: str | None = None
    index: int | None = None


@dataclass(frozen=True)
class HookRejection:
    """Structured rejection information for policy feedback.

    Created when a hook handler or validation rejects a pending
    operation. Routed to policy.on_rejection() for adaptive behavior.

    Attributes:
        reason: Human-readable rejection reason.
        pending: The full Pending object that was rejected.
        rejection_source: Where the rejection originated. One of
            "hook" (handler called reject()), "handler" (handler
            raised an exception), or "validation" (validate() failed).
        metadata: Optional additional context for the rejection.
    """

    reason: str
    pending: Pending
    rejection_source: str  # "hook", "handler", "validation"
    metadata: dict | None = None

    def __post_init__(self) -> None:
        valid_sources = {"hook", "handler", "validation"}
        if self.rejection_source not in valid_sources:
            raise ValueError(
                f"rejection_source must be one of {valid_sources}, "
                f"got {self.rejection_source!r}"
            )
