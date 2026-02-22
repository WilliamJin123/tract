"""Priority annotation model for Trace.

Annotations are lightweight, mutable metadata attached to commits
(like git tags). The annotation table is append-only for provenance.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class Priority(str, enum.Enum):
    """Priority levels for commit annotations.

    Ordering: SKIP < NORMAL < IMPORTANT < PINNED.
    """

    SKIP = "skip"
    NORMAL = "normal"
    IMPORTANT = "important"
    PINNED = "pinned"


class RetentionCriteria(BaseModel):
    """Retention criteria for IMPORTANT commits.

    Controls how the compression summarizer treats commits marked IMPORTANT:
    - ``instructions``: Natural-language guidance injected into the summarization
      prompt (fuzzy / LLM-interpreted).
    - ``match_patterns``: Substrings or regexes that MUST appear in the
      compression summary (deterministic validation after summarization).
    - ``match_mode``: How ``match_patterns`` are checked -- ``"substring"``
      (default) or ``"regex"``.
    """

    instructions: str | None = None
    match_patterns: list[str] | None = None
    match_mode: Literal["substring", "regex"] = "substring"


class PriorityAnnotation(BaseModel):
    """SDK-facing priority annotation model."""

    id: Optional[int] = None
    tract_id: str
    target_hash: str
    priority: Priority
    reason: Optional[str] = None
    retention: RetentionCriteria | None = None
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
