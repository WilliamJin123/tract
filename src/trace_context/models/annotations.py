"""Priority annotation model for Trace.

Annotations are lightweight, mutable metadata attached to commits
(like git tags). The annotation table is append-only for provenance.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Priority(str, enum.Enum):
    """Priority levels for commit annotations."""

    SKIP = "skip"
    NORMAL = "normal"
    PINNED = "pinned"


class PriorityAnnotation(BaseModel):
    """SDK-facing priority annotation model."""

    id: Optional[int] = None
    repo_id: str
    target_hash: str
    priority: Priority
    reason: Optional[str] = None
    created_at: datetime


# Default priorities for built-in content types.
# instruction defaults to PINNED; all others default to NORMAL.
DEFAULT_TYPE_PRIORITIES: dict[str, Priority] = {
    "instruction": Priority.PINNED,
    "dialogue": Priority.NORMAL,
    "tool_io": Priority.NORMAL,
    "reasoning": Priority.NORMAL,
    "artifact": Priority.NORMAL,
    "output": Priority.NORMAL,
    "freeform": Priority.NORMAL,
}
