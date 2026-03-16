"""Search and query manager extracted from SearchMixin."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tract.models.annotations import Priority
from tract.models.commit import CommitOperation

if TYPE_CHECKING:
    from typing import Any, Callable

    from tract.models.commit import CommitInfo
    from tract.models.config import LLMConfig, Operator, TractConfig
    from tract.operations.diff import DiffResult
    from tract.operations.health import HealthReport
    from tract.operations.history import StatusInfo
    from tract.protocols import CompiledContext

logger = logging.getLogger(__name__)


class SearchManager:
    """Search, query, status, and commit inspection operations."""

    def __init__(
        self,
        tract_id: str,
        commit_repo,
        blob_repo,
        ref_repo,
        annotation_repo,
        parent_repo,
        event_repo,
        commit_engine,  # CommitEngine
        token_counter,
        compiler,  # ContextCompiler
        config: TractConfig,
        custom_type_registry: dict,
        check_open: Callable | None = None,  # Callable
        enrich: Callable | None = None,  # Callable - AnnotationManager._enrich_with_priorities
        get_head: Callable | None = None,  # Callable -> str|None
        get_ancestors: Callable | None = None,  # Callable - _get_merge_aware_ancestors
        row_to_info: Callable | None = None,  # Callable - commit_engine._row_to_info
        compile_fn: Callable | None = None,  # Callable - Tract.compile
        compile_at_fn: Callable | None = None,  # Callable - Tract._compile_at
        resolve_commit_fn: Callable | None = None,  # Callable - BranchManager.resolve
        get_config_fn: Callable | None = None,  # Callable - ConfigManager.get (for get_config)
        commit_fn: Callable | None = None,  # Callable - Tract.commit
        tag_annotation_repo=None,
        tract_ref: Any = None,  # The Tract instance (for build_manifest)
    ) -> None:
        self._tract_id = tract_id
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._ref_repo = ref_repo
        self._annotation_repo = annotation_repo
        self._parent_repo = parent_repo
        self._event_repo = event_repo
        self._commit_engine = commit_engine
        self._token_counter = token_counter
        self._compiler = compiler
        self._config = config
        self._custom_type_registry = custom_type_registry
        self._check_open_fn = check_open or (lambda: None)
        self._enrich = enrich or (lambda entries: entries)
        self._get_head = get_head or (lambda: self._ref_repo.get_head(self._tract_id))
        self._get_ancestors = get_ancestors
        self._row_to_info = row_to_info or self._commit_engine._row_to_info
        self._compile_fn = compile_fn
        self._compile_at_fn = compile_at_fn
        self._resolve_commit_fn = resolve_commit_fn
        self._get_config_fn = get_config_fn
        self._commit_fn = commit_fn
        self._tag_annotation_repo = tag_annotation_repo
        self._tract_ref = tract_ref

    # ------------------------------------------------------------------
    # Log / ancestry
    # ------------------------------------------------------------------

    def log(
        self,
        limit: int = 20,
        *,
        op_filter: CommitOperation | None = None,
        tags: list[str] | None = None,
        tag_match: str = "any",
    ) -> list[CommitInfo]:
        """Walk commit history from HEAD backward.

        Each returned :class:`CommitInfo` has ``effective_priority`` set
        (``"skip"``, ``"normal"``, ``"important"``, or ``"pinned"``).

        Args:
            limit: Maximum number of commits to return.  Default 20.
            op_filter: If set, only include commits with this operation type.
            tags: If set, only include commits that have these tags.
            tag_match: ``"any"`` (OR) or ``"all"`` (AND).  Default ``"any"``.

        Returns:
            List of :class:`CommitInfo` in reverse chronological order.
        """
        current_head = self._get_head()
        if current_head is None:
            return []

        if tags is None:
            # No tag filter -- use fast path
            ancestors = self._get_ancestors(
                current_head, limit=limit, op_filter=op_filter,
            )
            entries = [self._row_to_info(row) for row in ancestors]
            return self._enrich(entries)

        # Tag filtering: walk more commits and filter
        ancestors = self._get_ancestors(
            current_head, limit=500, op_filter=op_filter,
        )
        tag_set = set(tags)

        # Batch-fetch annotation tags for all ancestor commits
        annotation_map: dict[str, list[str]] = {}
        if self._tag_annotation_repo is not None:
            all_hashes = [r.commit_hash for r in ancestors]
            annotation_map = self._tag_annotation_repo.batch_get_tags(all_hashes)

        results: list[CommitInfo] = []
        for row in ancestors:
            # Combine immutable + mutable tags
            commit_tags = set(row.tags_json) if row.tags_json else set()
            commit_tags.update(annotation_map.get(row.commit_hash, []))

            if tag_match == "any":
                if commit_tags & tag_set:
                    results.append(self._row_to_info(row))
            else:  # "all"
                if tag_set <= commit_tags:
                    results.append(self._row_to_info(row))

            if len(results) >= limit:
                break
        return self._enrich(results)

    # ------------------------------------------------------------------
    # Find / search
    # ------------------------------------------------------------------

    def find(
        self,
        *,
        content: str | None = None,
        pattern: str | None = None,
        tag: str | None = None,
        content_type: str | None = None,
        metadata_key: str | None = None,
        metadata_value: str | None = None,
        branch: str | None = None,
        limit: int = 50,
    ) -> list[CommitInfo]:
        """Search commits by content, tags, content type, or metadata.

        Walks the ancestry of the specified branch (or current HEAD) and
        returns commits matching **all** provided criteria (AND logic).
        """
        self._check_open_fn()
        import re

        from tract.exceptions import BranchNotFoundError

        # Resolve starting commit hash
        if branch is not None:
            start_hash = self._ref_repo.get_branch(self._tract_id, branch)
            if start_hash is None:
                raise BranchNotFoundError(branch)
        else:
            start_hash = self._get_head()

        if start_hash is None:
            return []

        # Pre-compile regex if provided
        compiled_re = re.compile(pattern) if pattern is not None else None

        # Walk a generous window of ancestors for filtering
        scan_limit = max(limit * 10, 500)
        ancestors = self._get_ancestors(start_hash, limit=scan_limit)

        # Batch-fetch annotation tags if tag filter is active
        annotation_map: dict[str, list[str]] = {}
        if tag is not None and self._tag_annotation_repo is not None:
            all_hashes = [r.commit_hash for r in ancestors]
            annotation_map = self._tag_annotation_repo.batch_get_tags(all_hashes)

        results: list[CommitInfo] = []
        for row in ancestors:
            # --- content_type filter ---
            if content_type is not None and row.content_type != content_type:
                continue

            # --- metadata filters ---
            if metadata_key is not None:
                md = row.metadata_json
                if not isinstance(md, dict) or metadata_key not in md:
                    continue
                if metadata_value is not None and md[metadata_key] != metadata_value:
                    continue

            # --- tag filter (immutable + mutable) ---
            if tag is not None:
                commit_tags: set[str] = set(row.tags_json) if row.tags_json else set()
                commit_tags.update(annotation_map.get(row.commit_hash, []))
                if tag not in commit_tags:
                    continue

            # --- content / pattern filters (load blob lazily) ---
            if content is not None or compiled_re is not None:
                blob = self._blob_repo.get(row.content_hash)
                if blob is None:
                    continue
                try:
                    blob_text = blob.payload_json
                except Exception:
                    logger.debug(
                        "Skipping blob %s: payload unreadable", row.content_hash,
                        exc_info=True,
                    )
                    continue

                if content is not None and content not in blob_text:
                    continue
                if compiled_re is not None and not compiled_re.search(blob_text):
                    continue

            info = self._row_to_info(row)
            results.append(info)

            if len(results) >= limit:
                break

        return self._enrich(results)

    def find_one(
        self,
        *,
        content: str | None = None,
        pattern: str | None = None,
        tag: str | None = None,
        content_type: str | None = None,
        metadata_key: str | None = None,
        metadata_value: str | None = None,
        branch: str | None = None,
    ) -> CommitInfo | None:
        """Search commits and return the first match, or ``None``."""
        hits = self.find(
            content=content,
            pattern=pattern,
            tag=tag,
            content_type=content_type,
            metadata_key=metadata_key,
            metadata_value=metadata_value,
            branch=branch,
            limit=1,
        )
        return hits[0] if hits else None

    def query_by_config(
        self,
        field_or_config: str | LLMConfig | None = None,
        operator: Operator | None = None,
        value: Any = None,
        *,
        conditions: list[tuple[str, Operator, Any]] | None = None,
    ) -> list[CommitInfo]:
        """Query commits by generation config values."""
        from tract.models.config import LLMConfig

        if isinstance(field_or_config, LLMConfig):
            conds: list[tuple[str, str, object]] = []
            for k, v in field_or_config.non_none_fields().items():
                if isinstance(v, tuple):
                    v = list(v)
                conds.append((k, "=", v))
            if not conds:
                return []
            rows = self._commit_repo.get_by_config_multi(self._tract_id, conds)
        elif conditions is not None:
            rows = self._commit_repo.get_by_config_multi(self._tract_id, conditions)
        elif isinstance(field_or_config, str) and operator is not None:
            rows = self._commit_repo.get_by_config_multi(
                self._tract_id, [(field_or_config, operator, value)]
            )
        else:
            raise TypeError(
                "query_by_config requires either: "
                "(field, operator, value), "
                "conditions=[...], "
                "or an LLMConfig object"
            )
        return [self._row_to_info(row) for row in rows]

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    def skipped(self, limit: int = 100) -> list[CommitInfo]:
        """Return commits with effective priority SKIP."""
        entries = self.log(limit=limit)
        return [e for e in entries if e.effective_priority == Priority.SKIP.value]

    def pinned(self, limit: int = 100) -> list[CommitInfo]:
        """Return commits with effective priority PINNED."""
        entries = self.log(limit=limit)
        return [e for e in entries if e.effective_priority == Priority.PINNED.value]

    # ------------------------------------------------------------------
    # Status / health / manifest
    # ------------------------------------------------------------------

    def status(self) -> StatusInfo:
        """Get current tract status."""
        from tract.operations.history import StatusInfo

        current_head = self._get_head()
        branch_name = self._ref_repo.get_current_branch(self._tract_id)
        is_detached = self._ref_repo.is_detached(self._tract_id)

        token_count = 0
        token_source = ""
        commit_count = 0
        if current_head is not None:
            compiled = self._compile_fn()
            token_count = compiled.token_count
            token_source = compiled.token_source
            commit_count = compiled.commit_count

        token_budget_max = None
        if self._config.token_budget and self._config.token_budget.max_tokens:
            token_budget_max = self._config.token_budget.max_tokens

        recent = self.log(limit=3)

        return StatusInfo(
            head_hash=current_head,
            branch_name=branch_name,
            is_detached=is_detached,
            commit_count=commit_count,
            token_count=token_count,
            token_budget_max=token_budget_max,
            token_source=token_source,
            recent_commits=recent,
        )

    def health(self) -> HealthReport:
        """Run health checks on this tract's DAG."""
        from tract.operations.health import check_health

        return check_health(
            self._tract_id,
            self._commit_repo,
            self._blob_repo,
            self._ref_repo,
            self._parent_repo,
        )

    def manifest(self, max_log_entries: int = 30) -> str:
        """Build a lightweight text manifest of current context."""
        from tract.gate import build_manifest
        return build_manifest(self._tract_ref, max_log_entries)

    # ------------------------------------------------------------------
    # Diff / compare
    # ------------------------------------------------------------------

    def diff(
        self,
        commit_a: str | None = None,
        commit_b: str | None = None,
    ) -> DiffResult:
        """Compare two commits and return structured diff."""
        self._check_open_fn()
        from tract.exceptions import CommitNotFoundError, TraceError
        from tract.operations.diff import compute_diff
        from tract.protocols import CompiledContext

        # Default commit_b to HEAD
        if commit_b is None:
            current_head = self._get_head()
            if current_head is None:
                raise TraceError("No commits to diff")
            commit_b = current_head
        else:
            commit_b = self._resolve_commit_fn(commit_b)

        # Look up commit_b row
        row_b = self._commit_repo.get(commit_b)
        if row_b is None:
            raise CommitNotFoundError(commit_b)

        # Auto-resolve commit_a
        if commit_a is None:
            if row_b.operation == CommitOperation.EDIT and row_b.edit_target:
                commit_a = row_b.edit_target
            elif row_b.parent_hash:
                commit_a = row_b.parent_hash
            else:
                commit_a = None
        else:
            commit_a = self._resolve_commit_fn(commit_a)

        # Compile both commits to get their messages
        if commit_a is not None:
            compiled_a = self._compile_at_fn(commit_a)
        else:
            compiled_a = CompiledContext(
                messages=[], token_count=0, commit_count=0,
                token_source="", generation_configs=[], commit_hashes=[],
            )

        compiled_b = self._compile_at_fn(commit_b)

        return compute_diff(
            commit_a_hash=commit_a or "(empty)",
            commit_b_hash=commit_b,
            messages_a=compiled_a.messages,
            messages_b=compiled_b.messages,
            configs_a=compiled_a.generation_configs,
            configs_b=compiled_b.generation_configs,
        )

    def compare(
        self,
        branch_a: str | None = None,
        branch_b: str | None = None,
        *,
        commit_a: str | None = None,
        commit_b: str | None = None,
    ) -> DiffResult:
        """Compare compiled contexts between two branches or commits."""
        self._check_open_fn()
        from tract.exceptions import TraceError
        from tract.operations.diff import compute_diff

        # --- Validate mutual exclusivity ---
        if branch_a is not None and commit_a is not None:
            raise ValueError("Cannot specify both branch_a and commit_a; use one or the other.")
        if branch_b is not None and commit_b is not None:
            raise ValueError("Cannot specify both branch_b and commit_b; use one or the other.")

        # --- Resolve side A ---
        if commit_a is not None:
            hash_a = self._resolve_commit_fn(commit_a)
        elif branch_a is not None:
            hash_a = self._resolve_commit_fn(branch_a)
        else:
            current_head = self._get_head()
            if current_head is None:
                raise TraceError("No commits on current branch to use as side A")
            hash_a = current_head

        # --- Resolve side B ---
        if commit_b is not None:
            hash_b = self._resolve_commit_fn(commit_b)
        elif branch_b is not None:
            hash_b = self._resolve_commit_fn(branch_b)
        else:
            raise ValueError("Must specify branch_b or commit_b for the comparison target.")

        compiled_a = self._compile_at_fn(hash_a)
        compiled_b = self._compile_at_fn(hash_b)

        return compute_diff(
            commit_a_hash=hash_a,
            commit_b_hash=hash_b,
            messages_a=compiled_a.messages,
            messages_b=compiled_b.messages,
            configs_a=compiled_a.generation_configs,
            configs_b=compiled_b.generation_configs,
        )

    # ------------------------------------------------------------------
    # Edit history / restore
    # ------------------------------------------------------------------

    def edit_history(self, commit_hash: str) -> list[CommitInfo]:
        """Get the full edit chain for a commit."""
        self._check_open_fn()
        from tract.exceptions import CommitNotFoundError

        resolved = self._resolve_commit_fn(commit_hash)
        row = self._commit_repo.get(resolved)
        if row is None:
            raise CommitNotFoundError(resolved)

        original_hash = row.edit_target if row.edit_target else resolved
        rows = self._commit_repo.get_edits_for(original_hash, self._tract_id)
        if not rows:
            raise CommitNotFoundError(resolved)

        return [self._row_to_info(r) for r in rows]

    def restore(
        self,
        commit_hash: str,
        version: int = 0,
        *,
        message: str | None = None,
    ) -> CommitInfo:
        """Restore a previous version of a commit by creating a new EDIT."""
        self._check_open_fn()
        from tract.exceptions import CommitNotFoundError
        from tract.models.content import validate_content

        history = self.edit_history(commit_hash)
        if version < 0 or version >= len(history):
            raise IndexError(
                f"Version {version} out of range (0..{len(history) - 1})"
            )

        source = history[version]
        original = history[0]

        if message is None:
            message = f"restore to version {version}"

        blob_row = self._blob_repo.get(source.content_hash)
        if blob_row is None:
            raise CommitNotFoundError(source.commit_hash)

        content_data = json.loads(blob_row.payload_json)
        content = validate_content(
            content_data, custom_registry=self._custom_type_registry
        )

        return self._commit_fn(
            content,
            operation=CommitOperation.EDIT,
            edit_target=original.commit_hash,
            message=message,
            generation_config=source.generation_config.to_dict()
            if source.generation_config
            else None,
        )

    # ------------------------------------------------------------------
    # Commit inspection (from tract.py, not previously in a mixin)
    # ------------------------------------------------------------------

    def get_commit(self, commit_hash: str) -> CommitInfo | None:
        """Fetch a commit by its hash.

        Returns:
            :class:`CommitInfo` if found, *None* otherwise.
        """
        return self._commit_engine.get_commit(commit_hash)

    def get_content(self, commit_or_hash: CommitInfo | str) -> str | dict | None:
        """Load the content for a commit.

        For simple content types (dialogue, instruction, etc.), returns the
        text string.  For structured content types that carry additional
        metadata (reasoning, freeform), returns the full parsed dict so
        callers can inspect fields like ``format``.

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.

        Returns:
            The content text (str), full content dict, or *None* if the
            commit or blob is not found.
        """
        if isinstance(commit_or_hash, str):
            row = self._commit_repo.get(commit_or_hash)
            if row is None:
                return None
            content_hash = row.content_hash
        else:
            content_hash = commit_or_hash.content_hash

        blob = self._blob_repo.get(content_hash)
        if blob is None:
            return None

        try:
            data = json.loads(blob.payload_json)
        except (json.JSONDecodeError, TypeError):
            return blob.payload_json

        # Structured content types: return the full dict so callers
        # can access all fields (e.g. format, payload, etc.)
        _STRUCTURED_TYPES = {"reasoning", "freeform"}
        if isinstance(data, dict) and data.get("content_type") in _STRUCTURED_TYPES:
            return data

        # Extract text from known content shapes
        for key in ("text", "content"):
            if key in data:
                return data[key]
        if "payload" in data:
            val = data["payload"]
            return json.dumps(val) if isinstance(val, dict) else str(val)
        return blob.payload_json

    def get_metadata(self, commit_or_hash: CommitInfo | str) -> dict | None:
        """Load the metadata dict for a commit.

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.

        Returns:
            The metadata dict, or *None* if the commit is not found or
            has no metadata.
        """
        if isinstance(commit_or_hash, str):
            row = self._commit_repo.get(commit_or_hash)
            if row is None:
                return None
            return row.metadata_json
        return commit_or_hash.metadata

    def show(self, commit_or_hash: CommitInfo | str) -> None:
        """Pretty-print a commit with its full content.

        Like ``git show`` -- displays commit metadata and the complete
        content text.  For metadata-only output, use
        ``info.pprint()`` instead.

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.
        """
        self._check_open_fn()
        from tract.formatting import pprint_commit_info

        if isinstance(commit_or_hash, str):
            info = self.get_commit(commit_or_hash)
            if info is None:
                raise ValueError(f"Commit not found: {commit_or_hash}")
        else:
            info = commit_or_hash

        content = self.get_content(info)
        pprint_commit_info(info, content=content)

    # ------------------------------------------------------------------
    # Internal: merge-aware ancestor walk
    # ------------------------------------------------------------------

    def _get_merge_aware_ancestors(
        self,
        start_hash: str,
        limit: int | None = None,
        *,
        op_filter: object | None = None,
    ) -> list:
        """Walk ancestry from start_hash following ALL parents (primary + merge).

        This is the default implementation used when no ``get_ancestors``
        callback is provided at construction time.
        """
        if self._parent_repo is None:
            return list(
                self._commit_repo.get_ancestors(
                    start_hash, limit=limit, op_filter=op_filter,
                )
            )

        fetch_limit = None if op_filter is not None else limit
        all_rows = list(
            self._commit_repo.get_ancestors_with_merges(start_hash, limit=fetch_limit)
        )

        if op_filter is not None:
            all_rows = [r for r in all_rows if r.operation == op_filter]
            if limit is not None:
                all_rows = all_rows[:limit]

        return all_rows
