"""Hook observability event."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HookEvent:
    """A single hook-system event for observability.

    Attributes:
        timestamp: When the event occurred.
        operation: The hookable operation name (e.g. "compress", "gc").
        handler_name: Display name of the handler that ran.
        resolved: Whether the pending was resolved (approved/rejected).
        result: One of "approved", "rejected", "unresolved", "skipped",
            "auto-approved".
    """

    timestamp: datetime
    operation: str
    handler_name: str
    resolved: bool
    result: str  # "approved", "rejected", "unresolved", "skipped", "auto-approved"
