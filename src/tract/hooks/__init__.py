"""Unified hook system for tract.

The hook system provides a single interface for gating operations that
destroy history (gc), transform history (compression, rebase), or
involve judgment calls (merge resolution, policy actions).

Every hookable operation produces a Pending object -- a mutable container
with methods to approve, reject, modify, or retry the planned operation.

Public API:
    Pending              -- base class for all hookable operations
    PendingCompress      -- compression hook object
    PendingGC            -- garbage collection hook object
    PendingRebase        -- rebase hook object
    PendingMerge         -- merge conflict hook object
    PendingPolicy        -- policy action hook object
    GuidanceMixin        -- mixin for two-stage guidance pattern
    ValidationResult     -- per-item validation feedback
    HookRejection        -- structured rejection for policy feedback
    auto_retry           -- convenience validate->retry loop
"""

from tract.hooks.compress import PendingCompress
from tract.hooks.event import HookEvent
from tract.hooks.gc import PendingGC
from tract.hooks.guidance import GuidanceMixin
from tract.hooks.merge import PendingMerge
from tract.hooks.pending import Pending, PendingStatus
from tract.hooks.policy import PendingPolicy
from tract.hooks.rebase import PendingRebase
from tract.hooks.retry import auto_retry
from tract.hooks.tool_result import PendingToolResult
from tract.hooks.validation import HookRejection, ValidationResult

__all__ = [
    "Pending",
    "PendingStatus",
    "PendingCompress",
    "PendingGC",
    "PendingRebase",
    "PendingMerge",
    "PendingPolicy",
    "PendingToolResult",
    "HookEvent",
    "GuidanceMixin",
    "ValidationResult",
    "HookRejection",
    "auto_retry",
]
