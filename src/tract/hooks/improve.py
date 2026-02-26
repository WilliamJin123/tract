"""Content improvement helpers using EDIT commit pattern.

The improve feature uses two patterns:
1. Content improvement (messages, summaries): Original committed first,
   LLM improvement is an EDIT. restore(version=0) recovers original.
2. Instruction improvement (operational metadata): Both original and
   improved stored on OperationEventRow.

Phase 4 implements the structure and helpers. Full wiring into Tract
methods (improve=True parameter) is deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.tract import Tract


def _improve_content(tract: Tract, original_hash: str, improved_text: str) -> str:
    """Apply improved text as an EDIT commit on top of the original.

    Original committed first, LLM improvement is an EDIT.
    restore(version=0) recovers original.

    Args:
        tract: The Tract instance to commit on.
        original_hash: Hash of the original commit to improve.
        improved_text: The improved text from the LLM.

    Returns:
        Hash of the EDIT commit.

    Note:
        This helper assumes the original commit already exists.
        The caller is responsible for committing the original first,
        then calling this function with the improved text.
        Full wiring into Tract.user(improve=True) etc. is deferred.
    """
    from tract.models.commit import CommitOperation
    from tract.models.content import DialogueContent

    # Look up the original commit to get its role
    original = tract._commit_repo.get_commit(original_hash)
    role = "user"
    if original and original.role:
        role = original.role

    result = tract.commit(
        DialogueContent(role=role, text=improved_text),
        operation=CommitOperation.EDIT,
        edit_target=original_hash,
        message=f"improve: {role} message",
    )
    return result.hash


def _improve_instructions(original: str, improved: str) -> dict:
    """Store both original and improved instructions.

    For operational metadata (not commits), store both versions.
    This is used for instructions on compress/merge operations where
    the instructions are not committed but stored on OperationEventRow.

    Args:
        original: The original instruction text from the user.
        improved: The LLM-improved instruction text.

    Returns:
        Dict with original_instructions and effective_instructions keys.
    """
    return {
        "original_instructions": original,
        "effective_instructions": improved,
    }
