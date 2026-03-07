"""Middleware infrastructure for Tract.

Provides MiddlewareContext (the immutable context passed to handlers),
MiddlewareEvent Literal type, and the VALID_EVENTS set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import BaseModel

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.tract import Tract

# ---------------------------------------------------------------------------
# Canonical middleware event type — single source of truth
# ---------------------------------------------------------------------------
MiddlewareEvent = Literal[
    "pre_commit",
    "post_commit",
    "pre_compile",
    "pre_compress",
    "pre_merge",
    "pre_gc",
    "pre_transition",
    "post_transition",
]

VALID_EVENTS: frozenset[str] = frozenset(get_args(MiddlewareEvent))


@dataclass(frozen=True)
class MiddlewareContext:
    """Immutable context passed to middleware handlers."""

    event: MiddlewareEvent
    commit: CommitInfo | None
    tract: Tract
    branch: str
    head: str
    target: str | None = None
    pending: BaseModel | None = None
