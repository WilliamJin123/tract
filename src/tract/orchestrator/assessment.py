"""Context health assessment builder for the orchestrator.

Gathers structural indicators from a Tract instance and formats
them into a prompt for the LLM to assess context health.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.tract import Tract

logger = logging.getLogger(__name__)


def _format_commit_summary(commit_info: CommitInfo) -> str:
    """Format a CommitInfo into a one-line summary for the assessment prompt.

    Args:
        commit_info: A CommitInfo instance.

    Returns:
        Formatted string like "abc12345 append instruction 120t  System prompt".
    """
    msg = commit_info.message or ""
    op_str = commit_info.operation.value if commit_info.operation else "?"
    ct = commit_info.content_type or "?"
    tokens = commit_info.token_count
    ch = commit_info.commit_hash[:8]
    return f"{ch} {op_str:6s} {ct:12s} {tokens:5d}t  {msg}"


def build_context_assessment(
    tract: Tract,
    *,
    task_context: str | None = None,
) -> str:
    """Build the user-side assessment prompt for the orchestrator LLM.

    Gathers context health indicators from the tract without calling
    compile() directly (uses tract.status() which already provides
    token counts from the compile cache).

    Args:
        tract: The Tract instance to assess.
        task_context: Optional task context description for relevance assessment.

    Returns:
        Formatted assessment prompt string.
    """
    from tract.prompts.orchestrator import build_assessment_prompt

    # 1. Get current status (already has token_count, branch_name, etc.)
    status = tract.status()

    # 2. Max tokens from status
    max_tokens = status.token_budget_max or 0

    # 3. Recent commit history for activity log
    recent = tract.log(limit=10)

    # 4. Branch count
    branches = tract.list_branches()
    branch_count = len(branches)

    # 5. Annotation counts via public API
    pinned_count = 0
    skip_count = 0
    try:
        counts = tract.annotation_counts()
        pinned_count = counts["pinned"]
        skip_count = counts["skip"]
    except Exception:
        # Non-critical: proceed with zero counts
        logger.debug("Failed to count annotations for assessment", exc_info=True)

    # 6. Format into assessment prompt
    return build_assessment_prompt(
        token_count=status.token_count,
        max_tokens=max_tokens,
        commit_count=status.commit_count,
        branch_name=status.branch_name or "detached",
        recent_commits=[_format_commit_summary(c) for c in recent],
        task_context=task_context,
        pinned_count=pinned_count,
        skip_count=skip_count,
        branch_count=branch_count,
    )
