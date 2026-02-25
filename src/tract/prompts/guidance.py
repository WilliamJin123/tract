"""Stage 1 guidance prompts for two-stage operations.

Two-stage operations decompose into:
1. Guidance (judgment): what should the output cover?
2. Execution (production): produce the output given the guidance.

This module provides the prompts for stage 1 (guidance generation).
"""

from __future__ import annotations

COMPRESS_GUIDANCE_SYSTEM = """You are analyzing a conversation to determine what a summary should focus on.
Do NOT write the summary itself. Instead, write guidance about:
1. What key topics/decisions should be preserved
2. What can be safely dropped
3. What level of detail is appropriate
Be concise and specific."""

MERGE_GUIDANCE_SYSTEM = """You are analyzing merge conflicts to determine resolution strategy.
Do NOT resolve the conflicts. Instead, write guidance about:
1. Which version should be preferred for each conflict
2. Whether any conflicts need manual review
3. Suggested resolution approach
Be concise and specific."""


def build_compress_guidance_prompt(
    messages_text: str,
    *,
    instructions: str | None = None,
) -> str:
    """Build the user prompt for compress guidance generation.

    Args:
        messages_text: The conversation text to analyze.
        instructions: Optional user instructions to include.

    Returns:
        The formatted user prompt string.
    """
    prompt = f"Analyze this conversation and provide summarization guidance:\n\n{messages_text}"
    if instructions:
        prompt += f"\n\nUser instructions: {instructions}"
    return prompt


def build_merge_guidance_prompt(
    conflicts_text: str,
    *,
    instructions: str | None = None,
) -> str:
    """Build the user prompt for merge guidance generation.

    Args:
        conflicts_text: The conflict descriptions to analyze.
        instructions: Optional user instructions to include.

    Returns:
        The formatted user prompt string.
    """
    prompt = f"Analyze these merge conflicts and provide resolution guidance:\n\n{conflicts_text}"
    if instructions:
        prompt += f"\n\nUser instructions: {instructions}"
    return prompt
