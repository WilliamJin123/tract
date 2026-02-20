"""Commit domain model for Trace.

CommitInfo is the SDK-facing model returned when querying commits.
CommitOperation is the enum for commit operations (APPEND, EDIT).
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator

from tract.models.config import LLMConfig


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
    tract_id: str
    parent_hash: Optional[str] = None
    content_hash: str
    content_type: str
    operation: CommitOperation
    response_to: Optional[str] = None
    message: Optional[str] = None
    token_count: int
    metadata: Optional[dict] = None
    generation_config: Optional[LLMConfig] = None
    created_at: datetime

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("generation_config", mode="before")
    @classmethod
    def _coerce_generation_config(cls, v: object) -> object:
        """Auto-coerce dict input to LLMConfig for backward compatibility."""
        if isinstance(v, dict):
            return LLMConfig.from_dict(v)
        return v
