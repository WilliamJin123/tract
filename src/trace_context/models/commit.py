"""Commit domain model for Trace.

CommitInfo is the SDK-facing model returned when querying commits.
CommitOperation is the enum for commit operations (APPEND, EDIT).
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CommitOperation(str, enum.Enum):
    """Operations that can be performed in a commit."""

    APPEND = "append"
    EDIT = "edit"


class CommitInfo(BaseModel):
    """SDK-facing commit information model.

    This is what users receive when querying commit data.
    Not an ORM model -- used for data transfer only.
    """

    commit_hash: str
    repo_id: str
    parent_hash: Optional[str] = None
    content_hash: str
    content_type: str
    operation: CommitOperation
    reply_to: Optional[str] = None
    message: Optional[str] = None
    token_count: int
    metadata: Optional[dict] = None
    created_at: datetime
