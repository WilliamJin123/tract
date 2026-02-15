"""Branch domain model for Trace.

BranchInfo is the SDK-facing model returned when listing branches.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BranchInfo(BaseModel):
    """SDK-facing branch information model.

    Returned by Tract.list_branches() and related operations.
    """

    name: str
    commit_hash: str
    is_current: bool = False
    commit_count: Optional[int] = None  # Populated in verbose mode
    description: Optional[str] = None
