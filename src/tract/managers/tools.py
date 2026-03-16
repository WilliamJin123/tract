"""Tool tracking manager for Tract.

Extracted from ToolMixin (_tools.py) into a standalone class with explicit
constructor dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tract.models.commit import CommitInfo
    from tract.models.compression import ToolDropResult
    from tract.protocols import CompiledContext, ToolTurn
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitRepository,
        ParentRepository,
        RefRepository,
        ToolSchemaRepository,
    )


class ToolManager:
    """Tool tracking: set, get, get_for_commit, find/drop tool turns."""

    def __init__(
        self,
        tract_id: str,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        ref_repo: RefRepository,
        parent_repo: ParentRepository,
        annotation_repo: AnnotationRepository,
        tool_schema_repo: ToolSchemaRepository | None,
        check_open: Callable[[], None],
        commit_session: Callable[[], None],
        get_head: Callable[[], str | None],
        log_fn: Callable,  # Tract.log
        annotate_fn: Callable,  # AnnotationManager.set
        row_to_info: Callable,
    ) -> None:
        self._tract_id = tract_id
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._ref_repo = ref_repo
        self._parent_repo = parent_repo
        self._annotation_repo = annotation_repo
        self._tool_schema_repo = tool_schema_repo
        self._check_open = check_open
        self._commit_session = commit_session
        self._get_head = get_head
        self._log_fn = log_fn
        self._annotate_fn = annotate_fn
        self._row_to_info = row_to_info

        # Owned state
        self._active_tools: list[dict] | None = None

    def set(self, tools: list[dict] | None) -> None:
        """Set active tool definitions for subsequent commits.

        When set, every subsequent ``commit()`` will automatically link these
        tool definitions unless overridden by passing ``tools=`` explicitly.

        Pass ``None`` to clear.
        """
        self._active_tools = tools

    def get(self) -> list[dict] | None:
        """Get the currently active tool definitions."""
        return self._active_tools

    def get_for_commit(self, commit_hash: str) -> list[dict]:
        """Get tool definitions linked to a specific commit."""
        if self._tool_schema_repo is None:
            return []
        rows = self._tool_schema_repo.get_for_commit(commit_hash)
        return [row.schema_json for row in rows]

    def find_results(
        self,
        name: str | None = None,
        after: str | None = None,
        limit: int = 500,
    ) -> list[CommitInfo]:
        """Find tool result commits on the current branch."""
        entries = self._log_fn(limit=limit)
        entries.reverse()  # oldest-first

        results = []
        after_found = after is None

        for ci in entries:
            if not after_found:
                if ci.commit_hash == after:
                    after_found = True
                continue

            meta = ci.metadata or {}
            if "tool_call_id" not in meta:
                continue
            if name is not None and meta.get("name") != name:
                continue
            results.append(ci)

        return results

    def find_calls(
        self,
        name: str | None = None,
        limit: int = 500,
    ) -> list[CommitInfo]:
        """Find assistant commits that requested tool calls."""
        entries = self._log_fn(limit=limit)
        entries.reverse()  # oldest-first

        results = []
        for ci in entries:
            meta = ci.metadata or {}
            tool_calls = meta.get("tool_calls")
            if not tool_calls:
                continue
            if name is not None:
                if not any(tc.get("name") == name for tc in tool_calls):
                    continue
            results.append(ci)

        return results

    def find_turns(
        self,
        name: str | None = None,
        limit: int = 500,
    ) -> list[ToolTurn]:
        """Find paired tool-call + tool-result commit groups."""
        from tract.protocols import ToolTurn

        entries = self._log_fn(limit=limit)
        entries.reverse()  # oldest-first

        result_index: dict[str, list[CommitInfo]] = {}
        for ci in entries:
            meta = ci.metadata or {}
            tcid = meta.get("tool_call_id")
            if tcid:
                result_index.setdefault(tcid, []).append(ci)

        turns = []
        for ci in entries:
            meta = ci.metadata or {}
            tool_calls_data = meta.get("tool_calls")
            if not tool_calls_data:
                continue

            all_results = []
            turn_names = []
            for tc_raw in tool_calls_data:
                turn_names.append(tc_raw["name"])
                all_results.extend(result_index.get(tc_raw.get("id", ""), []))

            if name is not None and name not in turn_names:
                continue

            turns.append(ToolTurn(
                call=ci,
                results=all_results,
                tool_names=turn_names,
            ))

        return turns

    def drop_failed_turns(
        self,
        name: str | None = None,
    ) -> ToolDropResult:
        """Drop tool turns that contain error results from the compiled context."""
        from tract.models.annotations import Priority
        from tract.models.compression import ToolDropResult

        turns = self.find_turns(name=name)

        turns_dropped = 0
        commits_skipped = 0
        tokens_freed = 0
        dropped_names: set[str] = set()

        for turn in turns:
            has_error = False
            for r in turn.results:
                meta = r.metadata or {}
                if meta.get("is_error", False):
                    has_error = True
                    break

            if not has_error:
                continue

            turns_dropped += 1
            dropped_names.update(turn.tool_names)

            self._annotate_fn(turn.call.commit_hash, Priority.SKIP)
            commits_skipped += 1
            tokens_freed += turn.call.token_count

            for r in turn.results:
                self._annotate_fn(r.commit_hash, Priority.SKIP)
                commits_skipped += 1
                tokens_freed += r.token_count

        return ToolDropResult(
            turns_dropped=turns_dropped,
            commits_skipped=commits_skipped,
            tokens_freed=tokens_freed,
            tool_names=tuple(sorted(dropped_names)),
        )

    def _store_and_link(
        self, commit_hash: str, tools: list[dict]
    ) -> None:
        """Store tool schemas (content-addressed) and link to a commit."""
        from tract.models.tools import hash_tool_schema

        now = datetime.now(timezone.utc)
        for position, tool in enumerate(tools):
            content_hash = hash_tool_schema(tool)
            name = ""
            if isinstance(tool, dict):
                func = tool.get("function", {})
                if isinstance(func, dict):
                    name = func.get("name", "")
                if not name:
                    name = tool.get("name", "")
            self._tool_schema_repo.store(content_hash, name, tool, now)
            self._tool_schema_repo.link_to_commit(commit_hash, content_hash, position)

    def _gather_for_compile(self) -> list[dict]:
        """Gather tools from the last commit that has tools linked."""
        if self._tool_schema_repo is None:
            return []

        current_head = self._get_head()
        if current_head is None:
            return []

        ancestors = self._commit_repo.get_ancestors(current_head)
        all_hashes = [r.commit_hash for r in ancestors]
        tool_map = self._tool_schema_repo.batch_get_commit_tool_hashes(all_hashes)

        for commit_row in ancestors:
            if commit_row.commit_hash in tool_map:
                rows = self._tool_schema_repo.get_for_commit(
                    commit_row.commit_hash
                )
                return [row.schema_json for row in rows]

        return []
