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


_TOOL_IO_CALL_MAX_LEN = 200  # truncate compact call repr


def _render_tool_io(
    tool_name: str,
    direction: str,
    payload: dict,
    status: str | None,
) -> str:
    """Render tool_io content compactly to reduce token waste.

    Results:
      success with single "result" key  -> ``tool_name -> value``
      error   with single "error" key   -> ``tool_name x value``
      other payloads                     -> ``tool_name (status)\n{compact json}``

    Calls:
      ``tool_name(arg=val, ...)``  (single line, truncated)
    """
    if direction == "result":
        # Single-key shortcut: just emit the value string directly
        keys = list(payload)
        if len(keys) == 1 and keys[0] in ("result", "error", "text"):
            value = str(payload[keys[0]])
            if status == "error":
                return f"{tool_name} \u2717 {value}"
            # success / None
            return f"{tool_name} \u2192 {value}"

        # Multi-key payload -- compact JSON, no pretty-print
        header = tool_name
        if status == "error":
            header += " \u2717"
        elif status:
            header += " \u2192"
        return f"{header}\n{json.dumps(payload, separators=(',', ':'))}"

    # direction == "call"
    args_parts: list[str] = []
    for k, v in payload.items():
        if isinstance(v, str):
            # Short strings inline, long ones truncated
            if len(v) > 60:
                v_repr = repr(v[:57] + "...")
            else:
                v_repr = repr(v)
        else:
            v_repr = json.dumps(v, separators=(",", ":"))
        args_parts.append(f"{k}={v_repr}")
    compact = f"{tool_name}({', '.join(args_parts)})"
    if len(compact) > _TOOL_IO_CALL_MAX_LEN:
        compact = compact[: _TOOL_IO_CALL_MAX_LEN - 3] + "..."
    return compact


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
        self.tool_result_format: str = "minimal"

    def compile(
        self,
        tract_id: str,
        head_hash: str,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
        include_edit_annotations: bool = False,
        include_reasoning: bool = False,
        strategy: str = "full",
        strategy_k: int = 5,
        recent_ratio: float | None = None,
    ) -> CompiledContext:
        """Compile commits into structured messages for LLM consumption.

        Args:
            tract_id: Tract identifier (used for annotation lookups).
            head_hash: Hash of the HEAD commit to start walking from.
            at_time: Only include commits created at or before this datetime.
            at_commit: Only include commits up to and including this commit hash.
            include_edit_annotations: If True, append '[edited]' marker to
                content that was replaced by an edit.
            include_reasoning: If True, promote reasoning commits from
                default SKIP to NORMAL priority so they appear in output.
                Explicit annotations always take precedence.
            strategy: Compile strategy. ``"full"`` (default) compiles all
                commits with full content. ``"messages"`` emits only commit
                messages (lightweight). ``"adaptive"`` keeps the last
                ``strategy_k`` commits at full detail and earlier ones as
                messages only.
            strategy_k: Number of recent commits to keep at full detail
                when using the ``"adaptive"`` strategy. Default 5.
            recent_ratio: If set, compute ``strategy_k`` as a ratio of the
                total effective commits instead of using a fixed count.
                Must be between 0.0 and 1.0 inclusive. For example,
                ``recent_ratio=0.7`` keeps the last 70% of commits at full
                detail and summarizes the first 30%. Overrides
                ``strategy_k`` when both are provided. Only used when
                ``strategy`` is ``"adaptive"``.

        Returns:
            CompiledContext with messages, token count, and metadata.

        Raises:
            ValueError: If both at_time and at_commit are provided.
            ValueError: If recent_ratio is not between 0.0 and 1.0.
        """
        if at_time is not None and at_commit is not None:
            raise ValueError("Cannot specify both at_time and at_commit; use one or the other.")

        if recent_ratio is not None and not (0.0 <= recent_ratio <= 1.0):
            raise ValueError(
                f"recent_ratio must be between 0.0 and 1.0, got {recent_ratio}"
            )

        _valid_strategies = ("full", "messages", "adaptive")
        if strategy not in _valid_strategies:
            raise ValueError(
                f"Invalid compile strategy {strategy!r}; must be one of {_valid_strategies}"
            )

        # Step 1: Walk commit chain (head -> root), then reverse to root -> head
        commits = self._walk_chain(head_hash, at_time=at_time, at_commit=at_commit)

        if not commits:
            return CompiledContext(messages=[], token_count=0, commit_count=0, token_source="")

        # Step 2: Build edit resolution map
        edit_map = self._build_edit_map(commits, at_time=at_time)

        # Step 3: Build priority map
        priority_map = self._build_priority_map(commits, at_time=at_time, include_reasoning=include_reasoning)

        # Step 4: Build effective commit list
        effective_commits, parsed_blob_cache = self._build_effective_commits(commits, edit_map, priority_map)

        # Step 4b: Extract commit hashes for effective commits (parallel to messages)
        effective_commit_hashes = [c.commit_hash for c in effective_commits]

        # Step 4c: Collect priorities for effective commits (parallel to messages)
        effective_priorities = [
            priority_map.get(c.commit_hash, Priority.NORMAL).value
            for c in effective_commits
        ]

        # Step 4d: Collect generation configs for effective commits
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

        # Compute effective strategy_k from recent_ratio when applicable
        effective_strategy_k = strategy_k
        if recent_ratio is not None and strategy == "adaptive":
            effective_strategy_k = max(1, int(len(effective_commits) * recent_ratio))

        # Step 5-6: Map to messages
        messages = self._build_messages(
            effective_commits, edit_map, include_edit_annotations,
            strategy=strategy, strategy_k=effective_strategy_k,
            parsed_blob_cache=parsed_blob_cache,
        )

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
            priorities=effective_priorities,
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
        # _walk_with_merge_parents is only called when _parent_repo is set
        if self._parent_repo is None:
            raise RuntimeError("_parent_repo must be set for merge-aware compilation")

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
        """Build map of edit_target -> latest edit commit.

        If multiple edits target the same commit, the latest one (by created_at) wins.
        """
        from tract.models.commit import CommitOperation

        edit_map: dict[str, CommitRow] = {}
        for c in commits:
            if c.operation == CommitOperation.EDIT and c.edit_target is not None:
                # Only include edits within the at_time boundary
                if at_time is not None and _normalize_dt(c.created_at) > _normalize_dt(at_time):
                    continue
                existing = edit_map.get(c.edit_target)
                if existing is None or c.created_at > existing.created_at:
                    edit_map[c.edit_target] = c
        return edit_map

    def _build_priority_map(
        self,
        commits: list[CommitRow],
        *,
        at_time: datetime | None = None,
        include_reasoning: bool = False,
    ) -> dict[str, Priority]:
        """Build map of commit_hash -> effective priority.

        Uses annotations if available, otherwise falls back to
        DEFAULT_TYPE_PRIORITIES based on content_type.

        When include_reasoning is True, reasoning commits that would
        get default SKIP priority are promoted to NORMAL instead.
        Explicit annotations always take precedence.
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
                priority = annotation.priority
                # Promote reasoning SKIP -> NORMAL when include_reasoning is set,
                # but only for auto-generated default annotations (not explicit
                # user annotations set via t.annotate()).
                is_auto_default = (
                    annotation.reason is not None
                    and annotation.reason.startswith("Default priority for")
                )
                if (
                    include_reasoning
                    and c.content_type == "reasoning"
                    and priority == Priority.SKIP
                    and is_auto_default
                ):
                    priority = Priority.NORMAL
                priority_map[c.commit_hash] = priority
            else:
                default = DEFAULT_TYPE_PRIORITIES.get(
                    c.content_type, Priority.NORMAL
                )
                # Promote reasoning SKIP -> NORMAL when include_reasoning is set
                if include_reasoning and c.content_type == "reasoning" and default == Priority.SKIP:
                    default = Priority.NORMAL
                priority_map[c.commit_hash] = default

        return priority_map

    def _build_effective_commits(
        self,
        commits: list[CommitRow],
        edit_map: dict[str, CommitRow],
        priority_map: dict[str, Priority],
    ) -> tuple[list[CommitRow], dict[str, dict]]:
        """Build the effective commit list after edit resolution and priority filtering.

        Also filters out commits whose content type has ``compilable=False``
        in the built-in type hints.

        Deduplicates named InstructionContent (directive override-by-name):
        same name -> closest to HEAD wins.

        Returns:
            Tuple of (effective_commits, parsed_blob_cache) where
            *parsed_blob_cache* maps ``content_hash -> parsed dict`` for
            blobs that were already fetched and JSON-parsed during the
            instruction dedup step.  Downstream consumers (e.g.
            ``_build_messages``) can use this to avoid redundant blob
            fetches and JSON parses.
        """
        import json as _json

        from tract.models.commit import CommitOperation
        from tract.models.content import ContentTypeHints as _CTH

        effective: list[CommitRow] = []
        for c in commits:
            # Skip EDIT commits (they are substitutions, not standalone messages)
            if c.operation == CommitOperation.EDIT:
                continue
            # Skip commits with SKIP priority
            if priority_map.get(c.commit_hash) == Priority.SKIP:
                continue
            # Skip commits whose content type is not compilable
            hints = BUILTIN_TYPE_HINTS.get(c.content_type, _CTH())
            if not hints.compilable:
                continue
            # Include the commit (possibly with substituted content via edit_map)
            effective.append(c)

        # Deduplicate named InstructionContent (directive override-by-name)
        # Batch-fetch all blobs needed for instruction dedup in one query
        instruction_hashes: list[str] = []
        for c in effective:
            if c.content_type == "instruction":
                row = edit_map.get(c.commit_hash, c)
                instruction_hashes.append(row.content_hash)

        blob_cache = self._blob_repo.batch_get(instruction_hashes) if instruction_hashes else {}

        # Build a parsed-blob cache: content_hash -> parsed JSON dict.
        # This avoids re-fetching and re-parsing the same blobs in
        # _build_messages() / build_message_for_commit().
        parsed_blob_cache: dict[str, dict] = {}
        for content_hash, blob in blob_cache.items():
            parsed_blob_cache[content_hash] = _json.loads(blob.payload_json)

        seen_names: dict[str, int] = {}
        remove_indices: set[int] = set()
        for i in range(len(effective) - 1, -1, -1):  # walk HEAD -> root
            if effective[i].content_type != "instruction":
                continue
            # Resolve blob (use edit_map if available)
            row = edit_map.get(effective[i].commit_hash, effective[i])
            payload = parsed_blob_cache.get(row.content_hash)
            if payload is None:
                continue
            name = payload.get("name")
            if name:
                if name in seen_names:
                    remove_indices.add(i)
                else:
                    seen_names[name] = i
        if remove_indices:
            effective = [c for i, c in enumerate(effective) if i not in remove_indices]

        return effective, parsed_blob_cache

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

        # Extract tool data from commit metadata (if any)
        from tract.protocols import ToolCall as _ToolCall

        meta = commit_row.metadata_json or {}
        tool_calls = None
        tool_call_id = None

        if "tool_calls" in meta:
            tool_calls = [_ToolCall.from_dict(tc) for tc in meta["tool_calls"]]
        if "tool_call_id" in meta:
            tool_call_id = meta["tool_call_id"]
        # For tool result messages, prefer the function name from metadata
        if role == "tool" and "name" in meta:
            name = meta["name"]

        return Message(
            role=role, content=text, name=name,
            tool_calls=tool_calls, tool_call_id=tool_call_id,
            content_type=content_type,
        )

    def _build_message_from_parsed(
        self,
        commit_row: CommitRow,
        content_data: dict,
    ) -> Message:
        """Build a Message from a commit using an already-parsed blob dict.

        This is the cache-friendly variant of :meth:`build_message_for_commit`.
        It skips the blob fetch and JSON parse since *content_data* was
        already loaded during the instruction dedup step.
        """
        from tract.protocols import ToolCall as _ToolCall

        content_type = content_data.get("content_type", "unknown")
        role = self._map_role(content_type, content_data)
        text = self._extract_message_text(content_type, content_data)
        name = content_data.get("name") if content_type == "dialogue" else None

        meta = commit_row.metadata_json or {}
        tool_calls = None
        tool_call_id = None

        if "tool_calls" in meta:
            tool_calls = [_ToolCall.from_dict(tc) for tc in meta["tool_calls"]]
        if "tool_call_id" in meta:
            tool_call_id = meta["tool_call_id"]
        if role == "tool" and "name" in meta:
            name = meta["name"]

        return Message(
            role=role, content=text, name=name,
            tool_calls=tool_calls, tool_call_id=tool_call_id,
            content_type=content_type,
        )

    def _build_messages(
        self,
        effective_commits: list[CommitRow],
        edit_map: dict[str, CommitRow],
        include_edit_annotations: bool,
        *,
        strategy: str = "full",
        strategy_k: int = 5,
        parsed_blob_cache: dict[str, dict] | None = None,
    ) -> list[Message]:
        """Convert effective commits to Message objects.

        When *strategy* is ``"messages"``, every commit is rendered as a
        lightweight message containing only the commit message (or a
        fallback ``[content_type] commit`` string).

        When *strategy* is ``"adaptive"``, the last *strategy_k* commits
        keep full content and earlier commits get the messages-only
        treatment.

        Args:
            parsed_blob_cache: Optional mapping of ``content_hash`` to
                already-parsed JSON dicts (populated by
                ``_build_effective_commits`` during instruction dedup).
                Avoids redundant blob fetches and JSON parses for blobs
                that were already loaded.
        """
        if parsed_blob_cache is None:
            parsed_blob_cache = {}

        messages: list[Message] = []

        # For adaptive strategy, determine the index where full detail starts
        if strategy == "adaptive":
            full_start = max(0, len(effective_commits) - strategy_k)
        else:
            full_start = 0

        # Batch-fetch blobs needed for messages-only commits (role resolution),
        # but skip any already present in parsed_blob_cache.
        messages_only_hashes: list[str] = []
        for i, c in enumerate(effective_commits):
            is_messages_only = (
                strategy == "messages"
                or (strategy == "adaptive" and i < full_start)
            )
            if is_messages_only:
                source_commit = edit_map.get(c.commit_hash, c)
                if source_commit.content_hash not in parsed_blob_cache:
                    messages_only_hashes.append(source_commit.content_hash)

        messages_blob_cache = (
            self._blob_repo.batch_get(messages_only_hashes)
            if messages_only_hashes
            else {}
        )

        for i, c in enumerate(effective_commits):
            # Decide whether this commit gets messages-only treatment
            messages_only = (
                strategy == "messages"
                or (strategy == "adaptive" and i < full_start)
            )

            if messages_only:
                # Lightweight: just the commit message
                summary = c.message or f"[{c.content_type}] commit"
                # Still need to resolve the role from the source commit
                source_commit = edit_map.get(c.commit_hash, c)
                # Check parsed_blob_cache first, then raw blob cache
                if source_commit.content_hash in parsed_blob_cache:
                    content_data = parsed_blob_cache[source_commit.content_hash]
                    role = self._map_role(source_commit.content_type, content_data)
                else:
                    blob = messages_blob_cache.get(source_commit.content_hash)
                    if blob is not None:
                        content_data = json.loads(blob.payload_json)
                        role = self._map_role(source_commit.content_type, content_data)
                    else:
                        role = "assistant"
                msg = Message(role=role, content=summary, content_type=c.content_type)
            else:
                # Full content -- use parsed_blob_cache when available
                source_commit = edit_map.get(c.commit_hash, c)
                if source_commit.content_hash in parsed_blob_cache:
                    msg = self._build_message_from_parsed(
                        source_commit, parsed_blob_cache[source_commit.content_hash],
                    )
                else:
                    msg = self.build_message_for_commit(source_commit)

            # Add edit annotation if requested
            if include_edit_annotations and c.commit_hash in edit_map:
                msg = Message(
                    role=msg.role, content=msg.content + " [edited]", name=msg.name,
                    tool_calls=msg.tool_calls, tool_call_id=msg.tool_call_id,
                    content_type=msg.content_type,
                )

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
            fmt = getattr(self, "tool_result_format", "minimal")
            if fmt == "verbose":
                header = f"Tool {direction}: {tool_name}"
                if status:
                    header += f" ({status})"
                return f"{header}\n{json.dumps(payload, indent=2)}"
            if fmt == "json":
                header = f"Tool {direction}: {tool_name}"
                if status:
                    header += f" ({status})"
                return f"{header}\n{json.dumps(payload, separators=(',', ':'))}"
            # "minimal" (default)
            return _render_tool_io(tool_name, direction, payload, status)

        if content_type == "freeform":
            return json.dumps(content_data.get("payload", {}), indent=2)

        # For types with 'text' field
        if "text" in content_data:
            return content_data["text"]

        # ArtifactContent uses 'content' field
        if "content" in content_data:
            return content_data["content"]

        return json.dumps(content_data)

