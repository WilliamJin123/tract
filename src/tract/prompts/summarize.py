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
    retention_instructions: list[str] | None = None,
) -> str:
    """Build the user prompt for summarization.

    Args:
        messages_text: The conversation text to summarize.
        target_tokens: Optional target token count for the summary.
        instructions: Optional additional instructions to append.
        retention_instructions: Optional list of retention instructions from
            IMPORTANT-annotated commits. Each entry is injected as a
            bullet point under a dedicated section.

    Returns:
        The formatted user prompt string.
    """
    prompt = f"Summarize the following conversation segment:\n\n{messages_text}"

    if target_tokens is not None:
        prompt += f"\nTarget approximately {target_tokens} tokens."

    if retention_instructions:
        prompt += (
            "\n\nIMPORTANT: The following content was marked as important. "
            "You MUST preserve these specific details in your summary:"
        )
        for ri in retention_instructions:
            prompt += f"\n- {ri}"

    if instructions is not None:
        prompt += f"\nAdditional instructions: {instructions}"

    return prompt


# ---------------------------------------------------------------------------
# Collapse prompt (for summarizing child tract work back to parent)
# ---------------------------------------------------------------------------

DEFAULT_COLLAPSE_SYSTEM: str = (
    "You are summarizing the work of a subagent that was delegated a specific task. "
    "Your job is to produce a concise report for the parent agent.\n\n"
    "Guidelines:\n"
    "- Focus on OUTCOMES: what was accomplished, decided, or produced.\n"
    "- Include key findings, decisions made, and artifacts created.\n"
    "- Note any failures, blockers, or unresolved issues.\n"
    "- Preserve specific technical details: code snippets, configurations, "
    "exact values, error messages.\n"
    "- Omit the subagent's internal reasoning process unless it contains "
    "important caveats or trade-off analysis.\n"
    "- If a target token count is specified, aim for approximately that length.\n"
    'Begin with: "Subagent completed: [task summary]"'
)


def build_collapse_prompt(
    messages_text: str,
    purpose: str,
    *,
    target_tokens: int | None = None,
    instructions: str | None = None,
) -> str:
    """Build the user prompt for collapse summarization.

    Args:
        messages_text: The child tract's conversation text to summarize.
        purpose: The purpose/task that was delegated to the child.
        target_tokens: Optional target token count for the summary.
        instructions: Optional additional instructions to append.

    Returns:
        The formatted user prompt string.
    """
    prompt = (
        f"The subagent was delegated the following task:\n"
        f"  Purpose: {purpose}\n\n"
        f"Summarize the subagent's work:\n\n{messages_text}"
    )

    if target_tokens is not None:
        prompt += f"\nTarget approximately {target_tokens} tokens."

    if instructions is not None:
        prompt += f"\nAdditional instructions: {instructions}"

    return prompt
