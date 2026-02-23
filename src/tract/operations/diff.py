"""Diff operations: structured comparison between two compiled contexts.

Provides compute_diff() which compares two sets of compiled messages and
returns a structured DiffResult with per-message unified diffs, role changes,
token deltas, and generation config changes.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from tract.protocols import Message


@dataclass(frozen=True)
class MessageDiff:
    """Diff for a single message position.

    Attributes:
        index: Position in the output diff list.
        status: One of "added", "removed", "modified", "unchanged".
        role_a: Role in commit A (None if added).
        role_b: Role in commit B (None if removed).
        content_diff_lines: Unified diff output lines for modified messages.
        token_delta: Token count change (positive = more tokens in B).
    """

    index: int
    status: Literal["added", "removed", "modified", "unchanged"]
    role_a: str | None = None  # role in commit A (None if added)
    role_b: str | None = None  # role in commit B (None if removed)
    content_diff_lines: list[str] = field(default_factory=list)  # unified diff lines
    token_delta: int = 0


@dataclass(frozen=True)
class DiffStat:
    """Summary statistics for a diff.

    Attributes:
        messages_added: Number of messages only in commit B.
        messages_removed: Number of messages only in commit A.
        messages_modified: Number of messages changed between commits.
        messages_unchanged: Number of identical messages.
        total_token_delta: Net token count change (B - A).
    """

    messages_added: int = 0
    messages_removed: int = 0
    messages_modified: int = 0
    messages_unchanged: int = 0
    total_token_delta: int = 0


@dataclass(frozen=True)
class DiffResult:
    """Structured diff between two commits.

    Attributes:
        commit_a: Hash of the first (older) commit.
        commit_b: Hash of the second (newer) commit.
        message_diffs: Per-message diff entries.
        stat: Summary statistics.
        generation_config_changes: Map of field_name -> (old_value, new_value)
            for generation config fields that changed.
    """

    commit_a: str  # hash
    commit_b: str  # hash
    message_diffs: list[MessageDiff] = field(default_factory=list)
    stat: DiffStat = field(default_factory=DiffStat)
    generation_config_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    # field_name -> (old_value, new_value)

    def pprint(self, *, stat_only: bool = False) -> None:
        """Pretty-print this diff with colored unified diff output."""
        from tract.formatting import pprint_diff_result
        pprint_diff_result(self, stat_only=stat_only)


def _serialize_message(msg: Message) -> str:
    """Serialize a Message to diffable text representation.

    Format:
        role: {role}
        name: {name}  (only if name is set)
        ---
        {content}
    """
    lines = []
    lines.append(f"role: {msg.role}")
    if msg.name:
        lines.append(f"name: {msg.name}")
    lines.append("---")
    lines.append(msg.content)
    return "\n".join(lines)


def _compute_generation_config_changes(
    configs_a: list,
    configs_b: list,
) -> dict[str, tuple[Any, Any]]:
    """Compare the last generation config in each chain.

    Accepts either list[dict] or list[LLMConfig | None].
    Returns a dict mapping field names to (old_value, new_value) tuples
    for fields that differ between the two configs.
    """
    def _to_dict(c: object) -> dict:
        if c is None:
            return {}
        if isinstance(c, dict):
            return c
        # LLMConfig or similar with to_dict()
        return c.to_dict() if hasattr(c, "to_dict") else {}

    # Use the last non-empty config from each side as "active" config
    config_a: dict[str, Any] = {}
    for c in reversed(configs_a):
        d = _to_dict(c)
        if d:
            config_a = d
            break

    config_b: dict[str, Any] = {}
    for c in reversed(configs_b):
        d = _to_dict(c)
        if d:
            config_b = d
            break

    if not config_a and not config_b:
        return {}

    # Find all keys across both configs
    all_keys = set(config_a.keys()) | set(config_b.keys())
    changes: dict[str, tuple[Any, Any]] = {}

    for key in all_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            changes[key] = (val_a, val_b)

    return changes


def compute_diff(
    commit_a_hash: str,
    commit_b_hash: str,
    messages_a: list[Message],
    messages_b: list[Message],
    configs_a: list,
    configs_b: list,
    token_counts_a: list[int] | None = None,
    token_counts_b: list[int] | None = None,
) -> DiffResult:
    """Compute a structured diff between two compiled message lists.

    Uses difflib.SequenceMatcher to align messages between A and B,
    then classifies each position as added/removed/modified/unchanged.

    Args:
        commit_a_hash: Hash of the first commit (or "(empty)").
        commit_b_hash: Hash of the second commit.
        messages_a: Compiled messages from commit A.
        messages_b: Compiled messages from commit B.
        configs_a: Generation configs from commit A chain.
        configs_b: Generation configs from commit B chain.
        token_counts_a: Optional per-message token counts for A.
        token_counts_b: Optional per-message token counts for B.

    Returns:
        DiffResult with per-message diffs and summary statistics.
    """
    # Serialize messages for alignment
    serialized_a = [_serialize_message(m) for m in messages_a]
    serialized_b = [_serialize_message(m) for m in messages_b]

    # Use SequenceMatcher to align message lists
    matcher = difflib.SequenceMatcher(None, serialized_a, serialized_b)

    message_diffs: list[MessageDiff] = []
    diff_index = 0
    added = 0
    removed = 0
    modified = 0
    unchanged = 0
    total_token_delta = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                msg_a = messages_a[i1 + k]
                msg_b = messages_b[j1 + k]
                message_diffs.append(MessageDiff(
                    index=diff_index,
                    status="unchanged",
                    role_a=msg_a.role,
                    role_b=msg_b.role,
                    token_delta=0,
                ))
                diff_index += 1
                unchanged += 1

        elif tag == "replace":
            # Matched region where content changed
            a_len = i2 - i1
            b_len = j2 - j1
            paired = min(a_len, b_len)

            for k in range(paired):
                msg_a = messages_a[i1 + k]
                msg_b = messages_b[j1 + k]
                old_lines = serialized_a[i1 + k].splitlines(keepends=True)
                new_lines = serialized_b[j1 + k].splitlines(keepends=True)
                diff_lines = list(difflib.unified_diff(
                    old_lines, new_lines,
                    fromfile=f"commit {commit_a_hash[:8]}",
                    tofile=f"commit {commit_b_hash[:8]}",
                    lineterm="",
                ))

                tok_a = token_counts_a[i1 + k] if token_counts_a else 0
                tok_b = token_counts_b[j1 + k] if token_counts_b else 0
                delta = tok_b - tok_a

                message_diffs.append(MessageDiff(
                    index=diff_index,
                    status="modified",
                    role_a=msg_a.role,
                    role_b=msg_b.role,
                    content_diff_lines=diff_lines,
                    token_delta=delta,
                ))
                diff_index += 1
                modified += 1
                total_token_delta += delta

            # Extra messages in A (removed)
            for k in range(paired, a_len):
                msg_a = messages_a[i1 + k]
                tok_a = token_counts_a[i1 + k] if token_counts_a else 0
                message_diffs.append(MessageDiff(
                    index=diff_index,
                    status="removed",
                    role_a=msg_a.role,
                    token_delta=-tok_a,
                ))
                diff_index += 1
                removed += 1
                total_token_delta -= tok_a

            # Extra messages in B (added)
            for k in range(paired, b_len):
                msg_b = messages_b[j1 + k]
                tok_b = token_counts_b[j1 + k] if token_counts_b else 0
                message_diffs.append(MessageDiff(
                    index=diff_index,
                    status="added",
                    role_b=msg_b.role,
                    token_delta=tok_b,
                ))
                diff_index += 1
                added += 1
                total_token_delta += tok_b

        elif tag == "delete":
            for k in range(i2 - i1):
                msg_a = messages_a[i1 + k]
                tok_a = token_counts_a[i1 + k] if token_counts_a else 0
                message_diffs.append(MessageDiff(
                    index=diff_index,
                    status="removed",
                    role_a=msg_a.role,
                    token_delta=-tok_a,
                ))
                diff_index += 1
                removed += 1
                total_token_delta -= tok_a

        elif tag == "insert":
            for k in range(j2 - j1):
                msg_b = messages_b[j1 + k]
                tok_b = token_counts_b[j1 + k] if token_counts_b else 0
                message_diffs.append(MessageDiff(
                    index=diff_index,
                    status="added",
                    role_b=msg_b.role,
                    token_delta=tok_b,
                ))
                diff_index += 1
                added += 1
                total_token_delta += tok_b

    # Compute generation config changes
    gen_config_changes = _compute_generation_config_changes(configs_a, configs_b)

    stat = DiffStat(
        messages_added=added,
        messages_removed=removed,
        messages_modified=modified,
        messages_unchanged=unchanged,
        total_token_delta=total_token_delta,
    )

    return DiffResult(
        commit_a=commit_a_hash,
        commit_b=commit_b_hash,
        message_diffs=message_diffs,
        stat=stat,
        generation_config_changes=gen_config_changes,
    )
