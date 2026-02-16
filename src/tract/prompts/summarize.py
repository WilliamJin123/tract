"""Default summarization prompt for context compression.

Provides the system prompt and user prompt builder for LLM-based
conversation summarization.
"""

from __future__ import annotations

DEFAULT_SUMMARIZE_SYSTEM: str = (
    "You are a context summarizer for an AI assistant's conversation history. "
    "Your job is to produce a concise summary that preserves the information "
    "most relevant to future conversation quality.\n\n"
    "Guidelines:\n"
    "- Write in third-person narrative prose.\n"
    "- Preserve specific details: names, numbers, code snippets, decisions, "
    "and agreed-upon constraints.\n"
    "- Prioritize information that affects future conversation quality -- "
    "decisions made, preferences expressed, requirements established.\n"
    "- Omit pleasantries, greetings, filler, and meta-conversation "
    "(e.g., 'the user then asked...' is fine, but 'the user said hello' is not).\n"
    "- If a target token count is specified, aim for approximately that length.\n"
    '- Begin your summary with "Previously in this conversation:"'
)


def build_summarize_prompt(
    messages_text: str,
    *,
    target_tokens: int | None = None,
    instructions: str | None = None,
) -> str:
    """Build the user prompt for summarization.

    Args:
        messages_text: The conversation text to summarize.
        target_tokens: Optional target token count for the summary.
        instructions: Optional additional instructions to append.

    Returns:
        The formatted user prompt string.
    """
    prompt = f"Summarize the following conversation segment:\n\n{messages_text}"

    if target_tokens is not None:
        prompt += f"\nTarget approximately {target_tokens} tokens."

    if instructions is not None:
        prompt += f"\nAdditional instructions: {instructions}"

    return prompt
