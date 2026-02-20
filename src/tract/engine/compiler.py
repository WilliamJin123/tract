"""Default context compiler for Trace.

Converts a commit chain into LLM-ready structured messages.
Handles edit resolution, priority filtering, time-travel compilation,
and type-to-role mapping.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from tract.models.annotations import DEFAULT_TYPE_PRIORITIES, Priority
from tract.models.config import LLMConfig
from tract.models.content import BUILTIN_TYPE_HINTS
from tract.protocols import CompiledContext, Message

if TYPE_CHECKING:
    from tract.protocols import TokenCounter
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
    )
    from tract.storage.schema import CommitRow

logger = logging.getLogger(__name__)


def _normalize_dt(dt: datetime) -> datetime:
    """Strip timezone info for comparison (SQLite stores naive datetimes)."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


class DefaultContextCompiler:
    """Default implementation of the ContextCompiler protocol.

    Walks the commit chain from head to root, resolves edits, filters
    by priority, maps content types to LLM roles, and produces a
    structured message list.

    Note on token counts:
    - Per-commit token_count in the database reflects raw content tokens.
    - CompiledContext.token_count reflects the formatted output including
      message overhead (per-message tokens, response primer, etc.).
    """

    def __init__(
        self,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        annotation_repo: AnnotationRepository,
        token_counter: TokenCounter,
        type_to_role_map: dict[str, str] | None = None,
        parent_repo: CommitParentRepository | None = None,
    ) -> None:
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._annotation_repo = annotation_repo
        self._token_counter = token_counter
        self._type_to_role_override = type_to_role_map or {}
        self._parent_repo = parent_repo

    def compile(
        self,
        tract_id: str,
        head_hash: str,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
        include_edit_annotations: bool = False,
    ) -> CompiledContext:
        """Compile commits into structured messages for LLM consumption.

        Args:
            tract_id: Tract identifier (used for annotation lookups).
            head_hash: Hash of the HEAD commit to start walking from.
            at_time: Only include commits created at or before this datetime.
            at_commit: Only include commits up to and including this commit hash.
            include_edit_annotations: If True, append '[edited]' marker to
                content that was replaced by an edit.

        Returns:
            CompiledContext with messages, token count, and metadata.

        Raises:
            ValueError: If both at_time and at_commit are provided.
        """
        if at_time is not None and at_commit is not None:
            raise ValueError("Cannot specify both at_time and at_commit; use one or the other.")

        # Step 1: Walk commit chain (head -> root), then reverse to root -> head
        commits = self._walk_chain(head_hash, at_time=at_time, at_commit=at_commit)

        if not commits:
            return CompiledContext(messages=[], token_count=0, commit_count=0, token_source="")

        # Step 2: Build edit resolution map
        edit_map = self._build_edit_map(commits, at_time=at_time)

        # Step 3: Build priority map
        priority_map = self._build_priority_map(commits, at_time=at_time)

        # Step 4: Build effective commit list
        effective_commits = self._build_effective_commits(commits, edit_map, priority_map)

        # Step 4b: Extract commit hashes for effective commits (parallel to messages)
        effective_commit_hashes = [c.commit_hash for c in effective_commits]

        # Step 4c: Collect generation configs for effective commits
        generation_configs: list[LLMConfig | None] = []
        for c in effective_commits:
            # If this commit was edited, prefer the edit's config;
            # fall back to the original commit's config if the edit has none.
            edit_commit = edit_map.get(c.commit_hash)
            if edit_commit is not None and edit_commit.generation_config_json is not None:
                config = LLMConfig.from_dict(edit_commit.generation_config_json)
            else:
                config = LLMConfig.from_dict(c.generation_config_json) if c.generation_config_json else None
            generation_configs.append(config)

        # Step 5-6: Map to messages
        messages = self._build_messages(effective_commits, edit_map, include_edit_annotations)

        # Step 7: Count tokens on compiled output
        messages_dicts = [
            {"role": m.role, "content": m.content}
            if m.name is None
            else {"role": m.role, "content": m.content, "name": m.name}
            for m in messages
        ]
        token_count = self._token_counter.count_messages(messages_dicts)

        encoding_name = getattr(self._token_counter, "encoding_name", "unknown")
        token_source = f"tiktoken:{encoding_name}" if encoding_name != "unknown" else ""

        return CompiledContext(
            messages=messages,
            token_count=token_count,
            commit_count=len(effective_commits),
            token_source=token_source,
            generation_configs=generation_configs,
            commit_hashes=effective_commit_hashes,
        )

    def _walk_chain(
        self,
        head_hash: str,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
    ) -> list[CommitRow]:
        """Walk parent chain from head to root, apply time filters, return root-to-head order.

        When parent_repo is available, detects merge commits and includes
        commits from both parents using "branch blocks" ordering: all
        first-parent commits in order, then second-parent's unique commits
        before the merge point, in order.
        """
        # First-parent walk (linear chain)
        ancestors = self._commit_repo.get_ancestors(head_hash)
        # ancestors is head-first (newest first), reverse to root-first
        commits = list(reversed(ancestors))

        # If parent_repo is available, handle merge commits
        if self._parent_repo is not None:
            commits = self._walk_with_merge_parents(commits)

        # Apply at_commit filter: include only up to and including the specified hash
        if at_commit is not None:
            filtered = []
            for c in commits:
                filtered.append(c)
                if c.commit_hash == at_commit:
                    break
            commits = filtered

        # Apply at_time filter: include only commits at or before the datetime
        if at_time is not None:
            at_time_naive = _normalize_dt(at_time)
            commits = [c for c in commits if _normalize_dt(c.created_at) <= at_time_naive]

        return commits

    def _walk_with_merge_parents(
        self,
        first_parent_commits: list[CommitRow],
    ) -> list[CommitRow]:
        """Expand a first-parent commit list to include merge parent branches.

        Uses "branch blocks" ordering: for each merge commit found,
        the second parent's unique commits (not already in the list)
        are inserted before the merge commit in chronological order.
        """
        assert self._parent_repo is not None

        seen: set[str] = {c.commit_hash for c in first_parent_commits}
        result: list[CommitRow] = []

        for commit in first_parent_commits:
            # Check if this commit is a merge commit
            parents = self._parent_repo.get_parents(commit.commit_hash)

            if len(parents) >= 2:
                # Walk the second parent's chain to find unique commits
                second_parent_hash = parents[1]
                second_branch_commits = self._collect_unique_ancestors(
                    second_parent_hash, seen
                )
                # Insert second branch's commits before the merge commit
                for sc in second_branch_commits:
                    seen.add(sc.commit_hash)
                result.extend(second_branch_commits)

            result.append(commit)

        return result

    def _collect_unique_ancestors(
        self,
        start_hash: str,
        seen: set[str],
    ) -> list[CommitRow]:
        """Collect ancestors from start_hash that are not in 'seen'.

        Returns commits in chronological order (root to tip).
        Stops when hitting a commit already in 'seen'.
        """
        unique: list[CommitRow] = []
        current_hash: str | None = start_hash

        while current_hash is not None and current_hash not in seen:
            commit = self._commit_repo.get(current_hash)
            if commit is None:
                break
            unique.append(commit)
            current_hash = commit.parent_hash

        # Reverse to chronological order (root first)
        unique.reverse()
        return unique

    def _build_edit_map(
        self,
        commits: list[CommitRow],
        *,
        at_time: datetime | None = None,
    ) -> dict[str, CommitRow]:
        """Build map of response_to -> latest edit commit.

        If multiple edits target the same commit, the latest one (by created_at) wins.
        """
        from tract.models.commit import CommitOperation

        edit_map: dict[str, CommitRow] = {}
        for c in commits:
            if c.operation == CommitOperation.EDIT and c.response_to is not None:
                # Only include edits within the at_time boundary
                if at_time is not None and _normalize_dt(c.created_at) > _normalize_dt(at_time):
                    continue
                existing = edit_map.get(c.response_to)
                if existing is None or c.created_at > existing.created_at:
                    edit_map[c.response_to] = c
        return edit_map

    def _build_priority_map(
        self,
        commits: list[CommitRow],
        *,
        at_time: datetime | None = None,
    ) -> dict[str, Priority]:
        """Build map of commit_hash -> effective priority.

        Uses annotations if available, otherwise falls back to
        DEFAULT_TYPE_PRIORITIES based on content_type.
        """
        commit_hashes = [c.commit_hash for c in commits]
        annotations = self._annotation_repo.batch_get_latest(commit_hashes)

        priority_map: dict[str, Priority] = {}
        for c in commits:
            annotation = annotations.get(c.commit_hash)
            if annotation is not None:
                # If at_time is set, only consider annotations within that boundary
                if at_time is not None and _normalize_dt(annotation.created_at) > _normalize_dt(at_time):
                    annotation = None

            if annotation is not None:
                priority_map[c.commit_hash] = annotation.priority
            else:
                priority_map[c.commit_hash] = DEFAULT_TYPE_PRIORITIES.get(
                    c.content_type, Priority.NORMAL
                )

        return priority_map

    def _build_effective_commits(
        self,
        commits: list[CommitRow],
        edit_map: dict[str, CommitRow],
        priority_map: dict[str, Priority],
    ) -> list[CommitRow]:
        """Build the effective commit list after edit resolution and priority filtering."""
        from tract.models.commit import CommitOperation

        effective: list[CommitRow] = []
        for c in commits:
            # Skip EDIT commits (they are substitutions, not standalone messages)
            if c.operation == CommitOperation.EDIT:
                continue
            # Skip commits with SKIP priority
            if priority_map.get(c.commit_hash) == Priority.SKIP:
                continue
            # Include the commit (possibly with substituted content via edit_map)
            effective.append(c)

        return effective

    def build_message_for_commit(self, commit_row: CommitRow) -> Message:
        """Build a single Message from a commit's blob content.

        Loads the blob, parses JSON, maps content_type to role,
        extracts text. This is the single-commit equivalent of the
        loop body in _build_messages().

        Args:
            commit_row: The source commit row (after edit resolution).

        Returns:
            Message with role, content, and optional name.
        """
        blob = self._blob_repo.get(commit_row.content_hash)
        if blob is None:
            logger.warning("Blob not found for commit %s", commit_row.commit_hash)
            return Message(role="system", content="[missing content]")

        content_data = json.loads(blob.payload_json)
        content_type = content_data.get("content_type", "unknown")
        role = self._map_role(content_type, content_data)
        text = self._extract_message_text(content_type, content_data)
        name = content_data.get("name") if content_type == "dialogue" else None
        return Message(role=role, content=text, name=name)

    def _build_messages(
        self,
        effective_commits: list[CommitRow],
        edit_map: dict[str, CommitRow],
        include_edit_annotations: bool,
    ) -> list[Message]:
        """Convert effective commits to Message objects."""
        messages: list[Message] = []

        for c in effective_commits:
            # Determine which commit's content to use
            source_commit = edit_map.get(c.commit_hash, c)

            msg = self.build_message_for_commit(source_commit)

            # Add edit annotation if requested
            if include_edit_annotations and c.commit_hash in edit_map:
                msg = Message(role=msg.role, content=msg.content + " [edited]", name=msg.name)

            messages.append(msg)

        return messages

    def _map_role(self, content_type: str, content_data: dict) -> str:
        """Map content type to LLM message role.

        Priority order:
        1. type_to_role_map override
        2. DialogueContent: use the role field from content itself
        3. ToolIOContent: always "tool"
        4. BUILTIN_TYPE_HINTS default_role
        5. Fallback: "assistant"
        """
        # Check override map first
        if content_type in self._type_to_role_override:
            return self._type_to_role_override[content_type]

        # Special case: DialogueContent uses its own role field
        if content_type == "dialogue":
            return content_data.get("role", "user")

        # Special case: ToolIOContent always maps to "tool"
        if content_type == "tool_io":
            return "tool"

        # Use builtin type hints
        hints = BUILTIN_TYPE_HINTS.get(content_type)
        if hints is not None:
            return hints.default_role

        return "assistant"

    def _extract_message_text(self, content_type: str, content_data: dict) -> str:
        """Extract the display text from parsed content data."""
        if content_type == "tool_io":
            tool_name = content_data.get("tool_name", "unknown")
            direction = content_data.get("direction", "call")
            payload = content_data.get("payload", {})
            status = content_data.get("status")
            header = f"Tool {direction}: {tool_name}"
            if status:
                header += f" ({status})"
            return f"{header}\n{json.dumps(payload, indent=2)}"

        if content_type == "freeform":
            return json.dumps(content_data.get("payload", {}), indent=2)

        # For types with 'text' field
        if "text" in content_data:
            return content_data["text"]

        # ArtifactContent uses 'content' field
        if "content" in content_data:
            return content_data["content"]

        return json.dumps(content_data)

