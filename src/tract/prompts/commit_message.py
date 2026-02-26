"""Commit message generation prompts.

Provides system prompt and user prompt builder for LLM-based
auto-generation of concise one-sentence commit messages.
"""

from __future__ import annotations

COMMIT_MESSAGE_SYSTEM: str = (
    "You generate concise one-sentence commit messages for a context-tracking system. "
    "Each message should start with a verb and capture the semantic essence of the content. "
    "Return ONLY the commit message, nothing else."
)

_MAX_INPUT_CHARS = 2000


def build_commit_message_prompt(content_type: str, text: str) -> str:
    """Build the user prompt for commit message generation.

    Args:
        content_type: The content type discriminator (e.g. "dialogue", "instruction").
        text: The text content to summarize into a commit message.

    Returns:
        The formatted user prompt string.
    """
    truncated = text[:_MAX_INPUT_CHARS]
    if len(text) > _MAX_INPUT_CHARS:
        truncated += "..."
    return (
        f"Content type: {content_type}\n\n"
        f"Generate a one-sentence commit message for this content:\n\n"
        f"{truncated}"
    )
