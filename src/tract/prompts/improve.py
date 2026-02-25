"""Content improvement prompts for improve=True feature.

The improve feature pairs human intent with LLM articulation. It uses the
EDIT commit pattern: original committed first, LLM improvement as an EDIT.
restore(version=0) recovers the original.

Improvement style is inferred from context:
- Messages: polish grammar, clarity, conciseness
- Summaries: expand key points, improve structure
- Config: translate natural language to structured format
"""

from __future__ import annotations

IMPROVE_CONTENT_SYSTEM = """You improve text while preserving its meaning and intent.
Adapt your approach based on context:
- Messages: polish grammar, clarity, conciseness
- Summaries: expand key points, improve structure
- Config: translate natural language to structured format
Never change the fundamental meaning. Return ONLY the improved text."""


def build_improve_prompt(original: str, context: str = "message") -> str:
    """Build prompt for content improvement.

    Args:
        original: The original text to improve.
        context: The type of content being improved. One of "message",
            "summary", "config", or any descriptive string.

    Returns:
        The formatted improvement prompt string.
    """
    return f"Improve this {context}. Return only the improved version:\n\n{original}"
