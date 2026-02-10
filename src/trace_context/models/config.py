"""Configuration models for Trace.

RepoConfig holds per-repository settings.
TokenBudgetConfig controls token budget enforcement behavior.
"""

from __future__ import annotations

import enum
from typing import Callable, Optional

from pydantic import BaseModel


class BudgetAction(str, enum.Enum):
    """Action to take when token budget is exceeded."""

    WARN = "warn"
    REJECT = "reject"
    CALLBACK = "callback"


class TokenBudgetConfig(BaseModel):
    """Configuration for token budget enforcement."""

    model_config = {"arbitrary_types_allowed": True}

    max_tokens: Optional[int] = None  # None = unlimited
    action: BudgetAction = BudgetAction.WARN
    callback: Optional[Callable[[int, int], None]] = None  # (current, max) -> None


class RepoConfig(BaseModel):
    """Per-repository configuration."""

    model_config = {"arbitrary_types_allowed": True}

    db_path: str = ":memory:"
    tokenizer_encoding: str = "o200k_base"
    token_budget: Optional[TokenBudgetConfig] = None
    default_branch: str = "main"
