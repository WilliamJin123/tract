"""Pre-built hook handlers for orchestrator-like review patterns.

These demonstrate how to build orchestrator behavior from hooks + policies.
Import and use with ``t.on("policy", handler)`` or ``t.on("compress", handler)``, etc.

The handlers in this module work with any :class:`~tract.hooks.pending.Pending`
subclass.  They call ``pending.approve()`` or ``pending.reject()`` directly,
matching the hook handler contract (``def handler(pending) -> None``).

The ``auto_approve_tool_call`` / ``reject_all_tool_call`` / ``log_and_approve_tool_call``
functions work with the orchestrator's collaborative-mode ``on_tool_call`` callback,
accepting a :class:`~tract.orchestrator.models.ToolCall` and returning a
:class:`~tract.orchestrator.models.ToolCallReview`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.hooks.pending import Pending
    from tract.orchestrator.models import ToolCall, ToolCallReview


# ---------------------------------------------------------------------------
# Hook handlers (for t.on("compress", handler), etc.)
# ---------------------------------------------------------------------------


def auto_approve(pending: Pending) -> None:
    """Auto-approve any pending operation."""
    pending.approve()


def log_and_approve(pending: Pending, *, logger: logging.Logger | None = None) -> None:
    """Log pending details then approve."""
    _logger = logger or logging.getLogger(__name__)
    _logger.info(
        "Auto-approving %s: %s", pending.operation, pending.pending_id
    )
    pending.approve()


def reject_all(pending: Pending, *, reason: str = "Rejected by policy") -> None:
    """Reject all pending operations."""
    pending.reject(reason)


def cli_prompt(pending: Pending) -> None:
    """Interactive CLI prompt for approve/reject/modify.

    Uses ``pending.pprint()`` and ``pending.review()`` for display.
    """
    pending.pprint()
    response = input(
        f"\n[{pending.operation}] Approve? (y/n/m for modify): "
    ).strip().lower()
    if response == "y":
        pending.approve()
    elif response == "m":
        pending.review()  # Interactive editing flow
    else:
        reason = input("Rejection reason: ").strip()
        pending.reject(reason or "Rejected by user")


# ---------------------------------------------------------------------------
# Factory functions for creating parameterized handlers
# ---------------------------------------------------------------------------


def make_log_handler(logger: logging.Logger):
    """Create a log_and_approve handler with a specific logger."""

    def handler(pending: Pending) -> None:
        log_and_approve(pending, logger=logger)

    return handler


def make_reject_handler(reason: str = "Rejected by policy"):
    """Create a reject_all handler with a specific reason."""

    def handler(pending: Pending) -> None:
        reject_all(pending, reason=reason)

    return handler


# ---------------------------------------------------------------------------
# Tool-call review callbacks (for OrchestratorConfig.on_tool_call)
# ---------------------------------------------------------------------------


def auto_approve_tool_call(tool_call: ToolCall) -> ToolCallReview:
    """Approve any tool call automatically.

    For autonomous mode where no human review is needed.
    """
    from tract.orchestrator.models import ToolCallDecision, ToolCallReview as _TCR

    return _TCR(decision=ToolCallDecision.APPROVED)


def log_and_approve_tool_call(tool_call: ToolCall) -> ToolCallReview:
    """Log tool call details then approve automatically.

    For audit trail mode -- all actions are approved but logged
    for later review.
    """
    from tract.orchestrator.models import ToolCallDecision, ToolCallReview as _TCR

    logger = logging.getLogger(__name__)
    logger.info(
        "Orchestrator tool call: action=%s, args=%s",
        tool_call.name,
        tool_call.arguments,
    )
    return _TCR(decision=ToolCallDecision.APPROVED)


def reject_all_tool_call(tool_call: ToolCall) -> ToolCallReview:
    """Reject any tool call automatically.

    For testing and safety -- blocks all orchestrator actions.
    """
    from tract.orchestrator.models import ToolCallDecision, ToolCallReview as _TCR

    return _TCR(decision=ToolCallDecision.REJECTED, reason="Auto-rejected")
