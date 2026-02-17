"""Session and spawn models for multi-agent coordination.

Provides:
- SessionContent: Pydantic model for session boundary commits
- SpawnInfo: Frozen dataclass for spawn pointer metadata
- CollapseResult: Frozen dataclass for collapse operation results
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class SessionContent(BaseModel):
    """Content type for session boundary commits.

    Session boundaries mark the start, end, handoff, or checkpoint
    of an agent's work session. They capture what happened, what was
    decided, and what to do next.
    """

    content_type: Literal["session"] = "session"
    session_type: Literal["start", "end", "handoff", "checkpoint"]
    summary: str
    decisions: list[str] = []
    failed_approaches: list[str] = []
    next_steps: list[str] = []


@dataclass(frozen=True)
class SpawnInfo:
    """Metadata about a spawn pointer relationship.

    Immutable snapshot of a spawn pointer for use outside storage layer.
    """

    spawn_id: int
    parent_tract_id: str
    parent_commit_hash: str | None
    child_tract_id: str
    purpose: str
    inheritance_mode: str
    display_name: str | None
    created_at: datetime


@dataclass(frozen=True)
class CollapseResult:
    """Result of a collapse operation (summarizing a child tract back to parent).

    Attributes:
        parent_commit_hash: The summary commit created in the parent; None if
            auto_commit=False.
        child_tract_id: The child tract that was collapsed.
        summary_text: The generated or user-provided summary text; always
            populated so caller can review before committing.
        summary_tokens: Token count of the summary.
        source_tokens: Token count of the source material that was summarized.
        purpose: The original purpose of the spawned task.
    """

    parent_commit_hash: str | None
    child_tract_id: str
    summary_text: str
    summary_tokens: int
    source_tokens: int
    purpose: str
