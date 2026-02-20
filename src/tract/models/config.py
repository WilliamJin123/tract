"""Configuration models for Trace.

TractConfig holds per-tract settings.
TokenBudgetConfig controls token budget enforcement behavior.
LLMOperationConfig holds per-operation LLM defaults.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
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


class TractConfig(BaseModel):
    """Per-tract configuration."""

    model_config = {"arbitrary_types_allowed": True}

    db_path: str = ":memory:"
    tokenizer_encoding: str = "o200k_base"
    token_budget: Optional[TokenBudgetConfig] = None
    default_branch: str = "main"
    compile_cache_maxsize: int = 8


@dataclass(frozen=True)
class LLMOperationConfig:
    """Per-operation LLM configuration defaults.

    None fields mean 'inherit from tract-level default'.
    Use with Tract.configure_operations() to set different models
    and parameters for each LLM-powered operation.

    Example::

        from tract import LLMOperationConfig
        config = LLMOperationConfig(model="gpt-4o", temperature=0.7)
    """

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra_kwargs: dict | None = None
