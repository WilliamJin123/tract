"""Summarization prompts for context compression.

Provides system prompts and user prompt builders for LLM-based
summarization.  Three system-prompt variants cover the common cases:

- **DEFAULT_SUMMARIZE_SYSTEM** -- neutral, for compressing any commit
  subset (tool calls, a verbose batch, a section of dialogue, etc.).
- **CONVERSATION_SUMMARIZE_SYSTEM** -- for compressing an entire
  conversation history into a recap.
- **TOOL_SUMMARIZE_SYSTEM** -- for compressing tool-call / tool-result
  sequences while preserving key findings.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# General-purpose (default) -- used by compress() when no system_prompt given
# ---------------------------------------------------------------------------

DEFAULT_SUMMARIZE_SYSTEM: str = (
    "You are a context summarizer for an AI assistant. "
    "Your job is to produce a concise summary of the provided context "
    "segment, preserving information most relevant to future quality.\n\n"
    "Guidelines:\n"
    "- Write in third-person narrative prose.\n"
    "- Preserve specific details: names, numbers, code snippets, decisions, "
    "and agreed-upon constraints.\n"
    "- Prioritize information that affects future interactions -- "
    "decisions made, preferences expressed, requirements established.\n"
    "- Omit filler, pleasantries, and redundant phrasing.\n"
    "- If a target token count is specified, aim for approximately that length."
)

# ---------------------------------------------------------------------------
# Conversation -- for full-conversation compression / auto-compress policies
# ---------------------------------------------------------------------------

CONVERSATION_SUMMARIZE_SYSTEM: str = (
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

# ---------------------------------------------------------------------------
# Tool calls -- for compressing tool-call / tool-result sequences
# ---------------------------------------------------------------------------

TOOL_SUMMARIZE_SYSTEM: str = (
    "You are summarizing tool-call interactions from an AI agent's workflow. "
    "The content contains tool calls (function invocations) and their results. "
    "Your job is to distill the sequence into a concise summary of what "
    "happened and what was found.\n\n"
    "Guidelines:\n"
    "- Focus on OUTCOMES: what was searched/fetched, what was found or "
    "returned, and any errors encountered.\n"
    "- Omit raw file contents, full directory listings, verbose API "
    "responses, and other bulk output -- summarize what they contained.\n"
    "- Preserve key findings: specific values, line numbers, file paths, "
    "error messages, and decisions made based on results.\n"
    "- If multiple tool calls were made, summarize the sequence of actions "
    "and their cumulative result, not each call individually.\n"
    "- If a target token count is specified, aim for approximately that length."
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
        instructions: Extra guidance appended to the default user prompt.
            The base summarization prompt is preserved; this is added as
            "Additional instructions: ..." at the end.
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
        instructions: Extra guidance appended to the default collapse prompt.
            The base prompt is preserved; this is added at the end.

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
