"""Built-in tool profiles with curated subsets and scenario-appropriate descriptions.

Three profiles:
- ``SELF_PROFILE``: Tools for an agent managing its OWN context.
- ``SUPERVISOR_PROFILE``: Tools for managing ANOTHER agent's context.
- ``FULL_PROFILE``: All tools with default descriptions.
"""

from __future__ import annotations

import logging

from tract.toolkit.models import ToolConfig, ToolProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All 15 tool names (must match definitions.py)
# ---------------------------------------------------------------------------
_ALL_TOOL_NAMES = [
    "commit",
    "compile",
    "annotate",
    "status",
    "log",
    "diff",
    "compress",
    "branch",
    "switch",
    "merge",
    "reset",
    "checkout",
    "gc",
    "list_branches",
    "get_commit",
]

# ---------------------------------------------------------------------------
# SELF profile: tools for an agent managing its own context
# ---------------------------------------------------------------------------
SELF_PROFILE = ToolProfile(
    name="self",
    tool_configs={
        "commit": ToolConfig(
            enabled=True,
            description=(
                "Record new content into your context. Use this to add your messages, "
                "instructions, tool results, or any content to your conversation history."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile your current context into LLM-ready messages. Check what "
                "your context looks like and how many tokens it uses."
            ),
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Set a priority on one of your commits. Pin important context to "
                "protect it from compression, or skip irrelevant content."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check your current status: which branch you're on, your HEAD commit, "
                "token count, and budget usage."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description=(
                "View your recent commit history. See what content you've recorded "
                "and when."
            ),
        ),
        "compress": ToolConfig(
            enabled=True,
            description=(
                "Compress your context history to reduce token usage. Pinned commits "
                "are preserved. Requires an LLM client via configure_llm()."
            ),
        ),
        "branch": ToolConfig(
            enabled=True,
            description=(
                "Create a new branch to explore an alternative conversation path. "
                "Use branches to organize your context by topic."
            ),
        ),
        "switch": ToolConfig(
            enabled=True,
            description=(
                "Switch to a different branch of your context. Changes your active "
                "conversation history."
            ),
        ),
        "reset": ToolConfig(
            enabled=True,
            description=(
                "Reset your HEAD to a previous commit. Undo recent changes by "
                "moving your branch pointer backward."
            ),
        ),
    },
)

# ---------------------------------------------------------------------------
# SUPERVISOR profile: tools for managing another agent's context
# ---------------------------------------------------------------------------
SUPERVISOR_PROFILE = ToolProfile(
    name="supervisor",
    tool_configs={
        "commit": ToolConfig(
            enabled=True,
            description=(
                "Record new content into the managed agent's context. Add messages, "
                "instructions, or results to their conversation history."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile the managed agent's context into LLM-ready messages. "
                "Review their current context and token usage."
            ),
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Set a priority on a commit in the managed agent's context. "
                "Pin important context or skip irrelevant content."
            ),
        ),
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check the managed agent's current status: branch, HEAD, "
                "token count, and budget usage."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description=(
                "View the managed agent's recent commit history. Understand "
                "what content has been recorded in their context."
            ),
        ),
        "diff": ToolConfig(
            enabled=True,
            description=(
                "Compare two commits in the managed agent's context. Review "
                "edits or understand how their context evolved."
            ),
        ),
        "compress": ToolConfig(
            enabled=True,
            description=(
                "Compress the managed agent's context history to reduce their "
                "token usage. Requires an LLM client via configure_llm()."
            ),
        ),
        "branch": ToolConfig(
            enabled=True,
            description=(
                "Create a new branch in the managed agent's context. Organize "
                "their conversation by topic or create alternative paths."
            ),
        ),
        "switch": ToolConfig(
            enabled=True,
            description=(
                "Switch the managed agent to a different branch. Change which "
                "conversation history they are operating on."
            ),
        ),
        "merge": ToolConfig(
            enabled=True,
            description=(
                "Merge a branch into the managed agent's current branch. Combine "
                "context from separate conversation threads."
            ),
        ),
        "reset": ToolConfig(
            enabled=True,
            description=(
                "Reset the managed agent's HEAD to a previous commit. Undo "
                "their recent changes."
            ),
        ),
        "checkout": ToolConfig(
            enabled=True,
            description=(
                "Checkout a specific commit in the managed agent's context "
                "for read-only inspection."
            ),
        ),
        "gc": ToolConfig(
            enabled=True,
            description=(
                "Run garbage collection on the managed agent's context to "
                "remove unreachable commits and free storage."
            ),
        ),
        "list_branches": ToolConfig(
            enabled=True,
            description=(
                "List all branches in the managed agent's context with their "
                "HEAD commits."
            ),
        ),
        "get_commit": ToolConfig(
            enabled=True,
            description=(
                "Get detailed information about a specific commit in the "
                "managed agent's context."
            ),
        ),
    },
)

# ---------------------------------------------------------------------------
# FULL profile: all tools with default descriptions (no overrides)
# ---------------------------------------------------------------------------
FULL_PROFILE = ToolProfile(
    name="full",
    tool_configs={name: ToolConfig(enabled=True) for name in _ALL_TOOL_NAMES},
)


# ---------------------------------------------------------------------------
# Profile lookup
# ---------------------------------------------------------------------------
_PROFILES: dict[str, ToolProfile] = {
    "self": SELF_PROFILE,
    "supervisor": SUPERVISOR_PROFILE,
    "full": FULL_PROFILE,
}


def get_profile(name: str) -> ToolProfile:
    """Look up a built-in profile by name.

    Args:
        name: Profile name ("self", "supervisor", or "full").

    Returns:
        The matching ToolProfile.

    Raises:
        ValueError: If name is not a recognized profile.
    """
    profile = _PROFILES.get(name)
    if profile is None:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {list(_PROFILES.keys())}"
        )
    return profile
