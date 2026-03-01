"""Prompt templates for LLM-driven auto-tagging.

Provides the system prompt and task prompt builder for an orchestrator
agent that retrospectively tags commits with semantic labels.
"""

from __future__ import annotations

TAGGER_SYSTEM_PROMPT: str = (
    "You are a conversation tagger.  Your job is to read the commit history, "
    "register a vocabulary of semantic tags, then tag every commit with the "
    "labels that best describe its intent.\n\n"
    "Workflow:\n"
    "1. Call register_tag for each tag in the vocabulary.\n"
    "2. Call log to see the commit history.\n"
    "3. For each commit, call get_tags to see existing tags, then call tag "
    "to add any applicable labels from the vocabulary.  A commit can have "
    "multiple tags or none beyond its auto-classified ones.\n"
    "4. When every commit has been reviewed, stop calling tools.\n\n"
    "Rules:\n"
    "- Only use tags from the provided vocabulary.\n"
    "- Do NOT remove existing auto-classified tags (instruction, reasoning, etc.).\n"
    "- Be selective: not every message needs a custom tag."
)

DEFAULT_TAG_VOCABULARY: list[dict[str, str]] = [
    {"name": "question", "description": "User is asking for information or clarification"},
    {"name": "decision", "description": "A concrete decision was reached"},
    {"name": "action_item", "description": "Something that needs to be done next"},
    {"name": "blocker", "description": "An obstacle or risk was identified"},
    {"name": "context", "description": "Background information or status update"},
    {"name": "summary", "description": "A recap or synthesis of prior discussion"},
]


def build_tagger_task_prompt(
    vocabulary: list[dict[str, str]] | None = None,
) -> str:
    """Build the task context prompt for a tagger orchestrator run.

    Args:
        vocabulary: List of ``{"name": ..., "description": ...}`` dicts
            defining the tag vocabulary.  Defaults to
            :data:`DEFAULT_TAG_VOCABULARY`.

    Returns:
        Formatted task context string listing the vocabulary and
        instructing the agent to register and apply tags.
    """
    vocab = vocabulary or DEFAULT_TAG_VOCABULARY
    tag_lines = "\n".join(f"  - {t['name']}: {t['description']}" for t in vocab)
    return (
        f"Tag vocabulary:\n{tag_lines}\n\n"
        "Review the commit history and tag each message with semantic "
        "labels from the vocabulary.  Register all tags first, then "
        "apply them."
    )
