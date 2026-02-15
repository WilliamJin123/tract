"""Tract -- the public SDK entry point for Trace.

Ties together storage, commit engine, and context compiler into a clean,
user-facing API.  Users interact with ``Tract.open()``, ``t.commit()``,
``t.compile()``, etc.

Not thread-safe in v1.  Each thread should open its own ``Tract``.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from tract.engine.commit import CommitEngine
from tract.engine.compiler import DefaultContextCompiler
from tract.engine.tokens import TiktokenCounter
from tract.models.annotations import Priority, PriorityAnnotation
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.config import TractConfig
from tract.models.content import validate_content
from tract.exceptions import (
    BranchNotFoundError,
    CommitNotFoundError,
    ContentValidationError,
    DetachedHeadError,
    TraceError,
)
from tract.protocols import CompiledContext, CompileSnapshot, ContextCompiler, Message, TokenCounter, TokenUsage
from tract.storage.engine import create_session_factory, create_trace_engine, init_db
from tract.storage.sqlite import (
    SqliteAnnotationRepository,
    SqliteBlobRepository,
    SqliteCommitParentRepository,
    SqliteCommitRepository,
    SqliteRefRepository,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Engine
    from sqlalchemy.orm import Session

    from tract.models.branch import BranchInfo
    from tract.models.merge import MergeResult
    from tract.operations.diff import DiffResult
    from tract.operations.history import StatusInfo
    from tract.storage.schema import CommitRow


class Tract:
    """Primary entry point for Trace -- git-like version control for LLM context.

    Create a tract via :meth:`Tract.open` (recommended) or
    :meth:`Tract.from_components` (testing / DI).

    Example::

        with Tract.open() as t:
            t.commit(InstructionContent(text="You are helpful."))
            t.commit(DialogueContent(role="user", text="Hi"))
            result = t.compile()
            print(result.messages)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        engine: Engine | None,
        session: Session,
        commit_engine: CommitEngine,
        compiler: ContextCompiler,
        tract_id: str,
        config: TractConfig,
        commit_repo: SqliteCommitRepository,
        blob_repo: SqliteBlobRepository,
        ref_repo: SqliteRefRepository,
        annotation_repo: SqliteAnnotationRepository,
        token_counter: TokenCounter,
        parent_repo: SqliteCommitParentRepository | None = None,
        verify_cache: bool = False,
    ) -> None:
        self._engine = engine
        self._session = session
        self._commit_engine = commit_engine
        self._compiler = compiler
        self._tract_id = tract_id
        self._config = config
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._ref_repo = ref_repo
        self._annotation_repo = annotation_repo
        self._token_counter = token_counter
        self._parent_repo = parent_repo
        self._custom_type_registry: dict[str, type[BaseModel]] = {}
        self._snapshot_cache: OrderedDict[str, CompileSnapshot] = OrderedDict()
        self._snapshot_cache_maxsize: int = config.compile_cache_maxsize
        self._verify_cache: bool = verify_cache
        self._closed = False

    @classmethod
    def open(
        cls,
        path: str = ":memory:",
        *,
        tract_id: str | None = None,
        config: TractConfig | None = None,
        tokenizer: TokenCounter | None = None,
        compiler: ContextCompiler | None = None,
        verify_cache: bool = False,
    ) -> Tract:
        """Open (or create) a Trace repository.

        Args:
            path: SQLite path.  ``":memory:"`` for in-memory (default).
            tract_id: Unique tract identifier.  Generated if not provided.
            config: Tract configuration.  Defaults created if *None*.
            tokenizer: Pluggable token counter.  TiktokenCounter by default.
            compiler: Pluggable context compiler.  DefaultContextCompiler by default.
            verify_cache: If True, cross-check every cache hit against a
                full recompile (oracle testing).  Default False.

        Returns:
            A ready-to-use ``Tract`` instance.
        """
        if tract_id is None:
            tract_id = uuid.uuid4().hex

        if config is None:
            config = TractConfig(db_path=path)

        # Engine / session
        engine = create_trace_engine(path)
        init_db(engine)
        session_factory = create_session_factory(engine)
        session = session_factory()

        # Repositories
        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        ref_repo = SqliteRefRepository(session)
        annotation_repo = SqliteAnnotationRepository(session)
        parent_repo = SqliteCommitParentRepository(session)

        # Token counter
        token_counter = tokenizer or TiktokenCounter(
            encoding_name=config.tokenizer_encoding,
        )

        # Commit engine
        commit_engine = CommitEngine(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            tract_id=tract_id,
            token_budget=config.token_budget,
            parent_repo=parent_repo,
        )

        # Context compiler
        ctx_compiler: ContextCompiler = compiler or DefaultContextCompiler(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            parent_repo=parent_repo,
        )

        # Ensure "main" branch ref exists (idempotent)
        head = ref_repo.get_head(tract_id)
        if head is None:
            # No HEAD yet -- that is fine, first commit will set it.
            pass

        return cls(
            engine=engine,
            session=session,
            commit_engine=commit_engine,
            compiler=ctx_compiler,
            tract_id=tract_id,
            config=config,
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            parent_repo=parent_repo,
            verify_cache=verify_cache,
        )

    @classmethod
    def from_components(
        cls,
        *,
        engine: Engine | None = None,
        session: Session,
        commit_repo: SqliteCommitRepository,
        blob_repo: SqliteBlobRepository,
        ref_repo: SqliteRefRepository,
        annotation_repo: SqliteAnnotationRepository,
        token_counter: TokenCounter,
        compiler: ContextCompiler,
        tract_id: str,
        config: TractConfig | None = None,
        verify_cache: bool = False,
    ) -> Tract:
        """Create a ``Tract`` from pre-built components.

        Skips engine/session creation.  Useful for testing and DI.
        """
        if config is None:
            config = TractConfig()

        commit_engine = CommitEngine(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            tract_id=tract_id,
            token_budget=config.token_budget,
        )

        return cls(
            engine=engine,
            session=session,
            commit_engine=commit_engine,
            compiler=compiler,
            tract_id=tract_id,
            config=config,
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            verify_cache=verify_cache,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def tract_id(self) -> str:
        """The tract identifier."""
        return self._tract_id

    @property
    def head(self) -> str | None:
        """Current HEAD commit hash, or *None* if no commits yet."""
        return self._ref_repo.get_head(self._tract_id)

    @property
    def config(self) -> TractConfig:
        """The tract configuration."""
        return self._config

    @property
    def is_detached(self) -> bool:
        """Whether HEAD is in detached state (pointing directly at a commit)."""
        return self._ref_repo.is_detached(self._tract_id)

    @property
    def current_branch(self) -> str | None:
        """The current branch name, or *None* if HEAD is detached."""
        return self._ref_repo.get_current_branch(self._tract_id)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def commit(
        self,
        content: BaseModel | dict,
        *,
        operation: CommitOperation = CommitOperation.APPEND,
        message: str | None = None,
        response_to: str | None = None,
        metadata: dict | None = None,
        generation_config: dict | None = None,
    ) -> CommitInfo:
        """Create a new commit.

        Args:
            content: A Pydantic content model *or* a dict (auto-validated).
            operation: ``APPEND`` (default) or ``EDIT``.
            message: Optional human-readable message.
            response_to: For ``EDIT``, the hash of the commit being replaced.
            metadata: Optional arbitrary metadata.
            generation_config: Optional LLM generation config (temperature,
                model, top_p, etc.).  Immutable once committed.

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        # Guard: detached HEAD blocks commits
        if self._ref_repo.is_detached(self._tract_id):
            raise DetachedHeadError()

        # Auto-validate dicts through the content type system
        if isinstance(content, dict):
            content = validate_content(content, custom_registry=self._custom_type_registry)

        prev_head = self.head

        info = self._commit_engine.create_commit(
            content=content,
            operation=operation,
            message=message,
            response_to=response_to,
            metadata=metadata,
            generation_config=generation_config,
        )

        # Persist to database
        self._session.commit()

        # Update compile cache: incremental extend for APPEND,
        # in-memory patching for EDIT, otherwise next compile() rebuilds
        if (
            operation == CommitOperation.APPEND
            and isinstance(self._compiler, DefaultContextCompiler)
        ):
            parent_snapshot = self._cache_get(prev_head) if prev_head else None
            if parent_snapshot is not None:
                self._extend_snapshot_for_append(info, parent_snapshot)
            # If no parent snapshot, next compile() builds from scratch
        elif (
            operation == CommitOperation.EDIT
            and isinstance(self._compiler, DefaultContextCompiler)
        ):
            parent_snapshot = self._cache_get(prev_head) if prev_head else None
            if parent_snapshot is not None:
                edit_row = self._commit_repo.get(info.commit_hash)
                if edit_row is not None:
                    patched = self._patch_snapshot_for_edit(
                        parent_snapshot, info.commit_hash, edit_row
                    )
                    if patched is not None:
                        self._cache_put(info.commit_hash, patched)
            # Do NOT clear cache -- other entries at different HEADs remain valid

        return info

    def compile(
        self,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
        include_edit_annotations: bool = False,
    ) -> CompiledContext:
        """Compile the current context into LLM-ready messages.

        Args:
            at_time: Only include commits at or before this datetime.
            at_commit: Only include commits up to this hash.
            include_edit_annotations: Append ``[edited]`` markers.

        Returns:
            :class:`CompiledContext` with messages and token counts.
        """
        current_head = self.head
        if current_head is None:
            return CompiledContext(messages=[], token_count=0, commit_count=0, token_source="")

        # Time-travel and edit annotations: always full compile, don't touch snapshot
        if at_time is not None or at_commit is not None or include_edit_annotations:
            return self._compiler.compile(
                self._tract_id,
                current_head,
                at_time=at_time,
                at_commit=at_commit,
                include_edit_annotations=include_edit_annotations,
            )

        # Cache hit: snapshot exists for current head in LRU cache
        cached = self._cache_get(current_head)
        if cached is not None:
            result = self._snapshot_to_compiled(cached)
            if self._verify_cache:
                fresh = self._compiler.compile(self._tract_id, current_head)
                assert result.messages == fresh.messages, (
                    f"Cache message mismatch: cached {len(result.messages)} msgs, "
                    f"fresh {len(fresh.messages)} msgs"
                )
                assert result.token_count == fresh.token_count, (
                    f"Cache token mismatch: cached {result.token_count}, "
                    f"fresh {fresh.token_count}"
                )
            return result

        # Cache miss: full compile and build snapshot
        result = self._compiler.compile(self._tract_id, current_head)
        snapshot = self._build_snapshot_from_compiled(current_head, result)
        if snapshot is not None:
            self._cache_put(current_head, snapshot)
        return result

    def get_commit(self, commit_hash: str) -> CommitInfo | None:
        """Fetch a commit by its hash.

        Returns:
            :class:`CommitInfo` if found, *None* otherwise.
        """
        return self._commit_engine.get_commit(commit_hash)

    def annotate(
        self,
        target_hash: str,
        priority: Priority,
        *,
        reason: str | None = None,
    ) -> PriorityAnnotation:
        """Create a priority annotation on a commit.

        Args:
            target_hash: Hash of the commit to annotate.
            priority: Priority level (``SKIP``, ``NORMAL``, ``PINNED``).
            reason: Optional reason for the annotation.

        Returns:
            :class:`PriorityAnnotation` model.
        """
        annotation = self._commit_engine.annotate(target_hash, priority, reason)
        self._session.commit()

        # Annotations affect ALL cached snapshots that include the target commit.
        # Strategy: clear everything, then optionally re-add a patched current HEAD.
        if isinstance(self._compiler, DefaultContextCompiler):
            current_head = self.head
            patched = None
            if current_head:
                snapshot = self._cache_get(current_head)
                if snapshot is not None:
                    patched = self._patch_snapshot_for_annotate(
                        snapshot, target_hash, priority
                    )
            self._cache_clear()  # Clear ALL entries (other HEADs may contain the annotated commit)
            if patched is not None:
                self._cache_put(current_head, patched)  # Re-add patched current HEAD
        else:
            self._cache_clear()

        return annotation

    def get_annotations(self, target_hash: str) -> list[PriorityAnnotation]:
        """Get the full annotation history for a commit.

        Returns:
            List of :class:`PriorityAnnotation` in chronological order.
        """
        rows = self._annotation_repo.get_history(target_hash)
        return [
            PriorityAnnotation(
                id=row.id,
                tract_id=row.tract_id,
                target_hash=row.target_hash,
                priority=row.priority,
                reason=row.reason,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def log(
        self,
        limit: int = 20,
        *,
        op_filter: CommitOperation | None = None,
    ) -> list[CommitInfo]:
        """Walk commit history from HEAD backward.

        Args:
            limit: Maximum number of commits to return.  Default 20.
            op_filter: If set, only include commits with this operation type.

        Returns:
            List of :class:`CommitInfo` in reverse chronological order
            (newest first).  Empty list if no commits.
        """
        current_head = self.head
        if current_head is None:
            return []

        ancestors = self._commit_repo.get_ancestors(
            current_head, limit=limit, op_filter=op_filter,
        )
        return [self._commit_engine._row_to_info(row) for row in ancestors]

    def status(self) -> StatusInfo:
        """Get current tract status.

        Returns :class:`StatusInfo` with HEAD position, branch name,
        detached state, compiled token count, budget info, and last 3 commits.
        """
        from tract.operations.history import StatusInfo

        current_head = self.head
        branch_name = self._ref_repo.get_current_branch(self._tract_id)
        is_detached = self._ref_repo.is_detached(self._tract_id)

        # Get compiled token count (uses cache if available)
        token_count = 0
        token_source = ""
        commit_count = 0
        if current_head is not None:
            compiled = self.compile()
            token_count = compiled.token_count
            token_source = compiled.token_source
            commit_count = compiled.commit_count

        # Get token budget max
        token_budget_max = None
        if self._config.token_budget and self._config.token_budget.max_tokens:
            token_budget_max = self._config.token_budget.max_tokens

        # Get last 3 commits for preview
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

    def _compile_at(self, commit_hash: str) -> CompiledContext:
        """Compile at a specific commit, using LRU cache if available.

        Args:
            commit_hash: The commit hash to compile at.

        Returns:
            CompiledContext for the given commit.
        """
        # Check LRU cache first
        cached = self._cache_get(commit_hash)
        if cached is not None:
            return self._snapshot_to_compiled(cached)
        # Cache miss: compile fresh
        return self._compiler.compile(self._tract_id, commit_hash)

    def diff(
        self,
        commit_a: str | None = None,
        commit_b: str | None = None,
    ) -> DiffResult:
        """Compare two commits and return structured diff.

        Args:
            commit_a: First commit (hash or prefix).  If None and commit_b is
                an EDIT commit, auto-resolves to the edit target (response_to).
                If None and commit_b is not EDIT, uses commit_b's parent.
            commit_b: Second commit (hash or prefix).  Defaults to HEAD.

        Returns:
            :class:`DiffResult` with per-message diffs, token deltas,
            and config changes.

        Raises:
            TraceError: If no commits exist.
            CommitNotFoundError: If references can't be resolved.
        """
        from tract.operations.diff import compute_diff

        # Default commit_b to HEAD
        if commit_b is None:
            current_head = self.head
            if current_head is None:
                raise TraceError("No commits to diff")
            commit_b = current_head
        else:
            commit_b = self.resolve_commit(commit_b)

        # Look up commit_b row
        row_b = self._commit_repo.get(commit_b)
        if row_b is None:
            raise CommitNotFoundError(commit_b)

        # Auto-resolve commit_a
        if commit_a is None:
            if row_b.operation == CommitOperation.EDIT and row_b.response_to:
                commit_a = row_b.response_to
            elif row_b.parent_hash:
                commit_a = row_b.parent_hash
            else:
                # First commit, diff against empty
                commit_a = None
        else:
            commit_a = self.resolve_commit(commit_a)

        # Compile both commits to get their messages
        if commit_a is not None:
            compiled_a = self._compile_at(commit_a)
        else:
            compiled_a = CompiledContext(
                messages=[], token_count=0, commit_count=0,
                token_source="", generation_configs=[], commit_hashes=[],
            )

        compiled_b = self._compile_at(commit_b)

        return compute_diff(
            commit_a_hash=commit_a or "(empty)",
            commit_b_hash=commit_b,
            messages_a=compiled_a.messages,
            messages_b=compiled_b.messages,
            configs_a=compiled_a.generation_configs,
            configs_b=compiled_b.generation_configs,
        )

    def query_by_config(
        self,
        field: str,
        operator: str,
        value: object,
    ) -> list[CommitInfo]:
        """Query commits by generation config values.

        Uses SQL-side json_extract() for efficient filtering.

        Args:
            field: JSON field name in the generation config
                (e.g., ``"temperature"``, ``"model"``).
            operator: Comparison operator
                (``"="``, ``"!="``, ``">"``, ``"<"``, ``">="``, ``"<="``).
            value: Value to compare against.

        Returns:
            List of :class:`CommitInfo` matching the condition,
            ordered by created_at.
        """
        rows = self._commit_repo.get_by_config(
            self._tract_id, field, operator, value
        )
        return [self._commit_engine._row_to_info(row) for row in rows]

    def resolve_commit(self, ref_or_prefix: str) -> str:
        """Resolve a commit reference to a full commit hash.

        Resolution order:
        1. Full commit hash (exact match)
        2. Branch name
        3. Hash prefix (min 4 chars)

        Args:
            ref_or_prefix: A commit hash, branch name, or hash prefix.

        Returns:
            The full commit hash.

        Raises:
            CommitNotFoundError: If no commit can be resolved.
            AmbiguousPrefixError: If a prefix matches multiple commits.
        """
        from tract.operations.navigation import resolve_commit as _resolve

        return _resolve(
            ref_or_prefix, self._tract_id, self._commit_repo, self._ref_repo
        )

    def reset(
        self,
        target: str,
        *,
        mode: str = "soft",
    ) -> str:
        """Reset HEAD to a target commit.

        Stores the current HEAD as ORIG_HEAD before moving.

        Args:
            target: A commit hash, branch name, or hash prefix.
            mode: ``"soft"`` (default) or ``"hard"``.  In Trace both behave
                identically (no working directory to clean).

        Returns:
            The resolved target commit hash (new HEAD).

        Raises:
            CommitNotFoundError: If target cannot be resolved.
        """
        from tract.operations.navigation import reset as _reset

        resolved = self.resolve_commit(target)
        result = _reset(resolved, mode, self._tract_id, self._ref_repo)  # type: ignore[arg-type]
        self._session.commit()
        return result

    def checkout(self, target: str) -> str:
        """Checkout a commit or branch.

        - Branch name: attach HEAD to that branch (enables commits).
        - Commit hash/prefix: detach HEAD (read-only inspection).
        - ``"-"``: return to previous position via PREV_HEAD.

        Stores the current HEAD as PREV_HEAD before switching.

        Args:
            target: A branch name, commit hash, hash prefix, or ``"-"``.

        Returns:
            The resolved commit hash at the new HEAD position.

        Raises:
            CommitNotFoundError: If the target cannot be resolved.
            TraceError: If ``"-"`` is used but no PREV_HEAD exists.
        """
        from tract.operations.navigation import checkout as _checkout

        commit_hash, _is_detached = _checkout(
            target, self._tract_id, self._commit_repo, self._ref_repo
        )
        self._session.commit()
        return commit_hash

    def branch(
        self,
        name: str,
        *,
        source: str | None = None,
        switch: bool = True,
    ) -> str:
        """Create a new branch.

        Args:
            name: Branch name (git-style naming rules apply).
            source: Commit hash to branch from.  Defaults to HEAD.
            switch: If True (default), switch HEAD to the new branch.

        Returns:
            The commit hash the new branch points to.

        Raises:
            BranchExistsError: If branch name already exists.
            InvalidBranchNameError: If branch name is invalid.
            TraceError: If no commits exist and no source specified.
        """
        from tract.operations.branch import create_branch

        result = create_branch(
            name,
            self._tract_id,
            self._ref_repo,
            self._commit_repo,
            source=source,
            switch=switch,
        )
        self._session.commit()
        return result

    def switch(self, target: str) -> str:
        """Switch to a branch (branch-only, unlike checkout).

        Unlike :meth:`checkout`, this method ONLY accepts branch names.
        It will not silently detach HEAD on commit hashes -- use
        :meth:`checkout` for that.

        Args:
            target: A branch name.

        Returns:
            The commit hash at the target branch HEAD.

        Raises:
            BranchNotFoundError: If target is not a valid branch name.
        """
        # Validate that target is a branch
        branch_hash = self._ref_repo.get_branch(self._tract_id, target)
        if branch_hash is None:
            raise BranchNotFoundError(target)

        from tract.operations.navigation import checkout as _checkout

        commit_hash, _is_detached = _checkout(
            target, self._tract_id, self._commit_repo, self._ref_repo
        )
        self._session.commit()
        return commit_hash

    def list_branches(self) -> list[BranchInfo]:
        """List all branches with current branch indicator.

        Returns:
            List of :class:`BranchInfo` with ``is_current=True`` for
            the active branch.
        """
        from tract.models.branch import BranchInfo
        from tract.operations.branch import list_branches

        branch_names = list_branches(self._tract_id, self._ref_repo)
        current = self._ref_repo.get_current_branch(self._tract_id)

        branches: list[BranchInfo] = []
        for name in branch_names:
            commit_hash = self._ref_repo.get_branch(self._tract_id, name)
            if commit_hash is not None:
                branches.append(
                    BranchInfo(
                        name=name,
                        commit_hash=commit_hash,
                        is_current=(name == current),
                    )
                )
        return branches

    def delete_branch(self, name: str, *, force: bool = False) -> None:
        """Delete a branch.

        Args:
            name: Branch name to delete.
            force: If True, delete even if branch has unmerged commits.

        Raises:
            BranchNotFoundError: If branch doesn't exist.
            TraceError: If trying to delete the current branch.
            UnmergedBranchError: If branch has unmerged commits (without force).
        """
        from tract.operations.branch import delete_branch

        delete_branch(
            name,
            self._tract_id,
            self._ref_repo,
            self._commit_repo,
            self._parent_repo,
            force=force,
        )
        self._session.commit()

    def configure_llm(self, client: object) -> None:
        """Configure the LLM client for semantic operations.

        Stores the client and creates a default OpenAIResolver from it.

        Args:
            client: An LLM client conforming to the LLMClient protocol.
        """
        from tract.llm.resolver import OpenAIResolver

        self._llm_client = client
        self._default_resolver = OpenAIResolver(client)

    def merge(
        self,
        source_branch: str,
        *,
        resolver: object | None = None,
        strategy: str = "auto",
        no_ff: bool = False,
        auto_commit: bool = False,
        model: str | None = None,
        delete_branch: bool = False,
    ) -> MergeResult:
        """Merge a source branch into the current branch.

        Args:
            source_branch: Name of the branch to merge.
            resolver: Optional conflict resolver (ResolverCallable).
                Falls back to self._default_resolver if configured.
            strategy: ``"auto"`` (default) or ``"semantic"``.
            no_ff: If True, always create a merge commit (no fast-forward).
            auto_commit: If True, auto-commit even with resolved conflicts.
            model: Override model for the default resolver.
            delete_branch: If True, delete the source branch after merge.

        Returns:
            :class:`MergeResult` describing the outcome.
        """
        from tract.models.merge import MergeResult
        from tract.operations.merge import merge_branches

        # Determine resolver
        effective_resolver = resolver
        if effective_resolver is None:
            effective_resolver = getattr(self, "_default_resolver", None)

        # If model override and using default resolver, create new one with that model
        if model is not None and effective_resolver is getattr(self, "_default_resolver", None):
            if hasattr(self, "_llm_client"):
                from tract.llm.resolver import OpenAIResolver

                effective_resolver = OpenAIResolver(self._llm_client, model=model)

        result = merge_branches(
            tract_id=self._tract_id,
            source_branch=source_branch,
            commit_repo=self._commit_repo,
            ref_repo=self._ref_repo,
            parent_repo=self._parent_repo,
            blob_repo=self._blob_repo,
            annotation_repo=self._annotation_repo,
            commit_engine=self._commit_engine,
            token_counter=self._token_counter,
            resolver=effective_resolver,
            strategy=strategy,
            no_ff=no_ff,
        )

        # Auto-commit resolved conflicts
        if (
            auto_commit
            and result.merge_type == "conflict"
            and result.resolutions
            and len(result.resolutions) >= len(result.conflicts)
        ):
            return self.commit_merge(result)

        # Persist any changes (branch pointer moves, merge commits)
        self._session.commit()

        # Delete source branch if requested and merge was committed
        if delete_branch and (result.committed or result.merge_type == "fast_forward"):
            from tract.operations.branch import delete_branch as _delete_branch

            _delete_branch(
                source_branch,
                self._tract_id,
                self._ref_repo,
                self._commit_repo,
                self._parent_repo,
                force=True,
            )
            self._session.commit()

        # Clear compile cache (merge changes HEAD)
        self._cache_clear()

        return result

    def commit_merge(self, result: MergeResult) -> MergeResult:
        """Finalize a conflict merge after reviewing/editing resolutions.

        Args:
            result: A MergeResult from a previous merge() call with conflicts.

        Returns:
            Updated MergeResult with committed=True and merge_commit_hash set.

        Raises:
            MergeError: If not all conflicts are resolved.
        """
        from tract.exceptions import MergeError
        from tract.models.content import FreeformContent
        from tract.operations.merge import create_merge_commit

        # Validate all conflicts have resolutions
        unresolved = []
        for conflict in result.conflicts:
            target_key = conflict.target_hash or conflict.commit_b.commit_hash
            if target_key not in result.resolutions:
                unresolved.append(target_key)

        if unresolved:
            raise MergeError(
                f"Cannot commit merge: {len(unresolved)} unresolved conflict(s)"
            )

        # Build merge content from resolutions
        resolution_text = "\n\n".join(
            f"Resolution for {k}:\n{v}" for k, v in result.resolutions.items()
        )
        merge_content = FreeformContent(
            payload={
                "message": f"Merged {result.source_branch} into {result.target_branch}",
                "resolutions": result.resolutions,
            }
        )

        # Determine parent hashes
        source_hash = result._source_tip_hash
        target_hash = result._target_tip_hash
        if source_hash is None or target_hash is None:
            # Fallback: resolve from ref_repo
            source_hash = self._ref_repo.get_branch(self._tract_id, result.source_branch)
            target_hash = self._ref_repo.get_head(self._tract_id)

        parent_hashes = [target_hash, source_hash] if target_hash and source_hash else []

        # Collect generation config from resolver resolutions
        gen_config = None
        if result.generation_configs:
            gen_config = next(iter(result.generation_configs.values()), None)

        merge_info = create_merge_commit(
            commit_engine=self._commit_engine,
            parent_repo=self._parent_repo,
            content=merge_content,
            parent_hashes=parent_hashes,
            message=f"Merge branch '{result.source_branch}' into {result.target_branch}",
            generation_config=gen_config,
        )

        self._session.commit()

        result.committed = True
        result.merge_commit_hash = merge_info.commit_hash

        # Clear compile cache
        self._cache_clear()

        return result

    def record_usage(
        self,
        usage: TokenUsage | dict,
        *,
        head_hash: str | None = None,
    ) -> CompiledContext:
        """Record API-reported token usage, updating cached compilation.

        Accepts either a :class:`TokenUsage` dataclass or a provider-specific
        dict.  Supported dict formats:

        - **OpenAI:** ``{prompt_tokens, completion_tokens, total_tokens}``
        - **Anthropic:** ``{input_tokens, output_tokens}``

        Args:
            usage: :class:`TokenUsage` or dict with token counts.
            head_hash: Associate with a specific HEAD.  Defaults to current HEAD.

        Returns:
            Updated :class:`CompiledContext` with API-reported token count.

        Raises:
            TraceError: If no commits exist or *head_hash* doesn't match.
            ContentValidationError: If dict format is unrecognised.
        """
        # Normalize input
        if isinstance(usage, dict):
            usage = self._normalize_usage_dict(usage)
        elif not isinstance(usage, TokenUsage):
            raise ContentValidationError(
                f"Expected TokenUsage or dict, got {type(usage).__name__}"
            )

        target_hash = head_hash or self.head
        if target_hash is None:
            raise TraceError("Cannot record usage: no commits exist")

        # Validate explicit head_hash matches current HEAD
        if head_hash is not None and head_hash != self.head:
            raise TraceError(
                f"Cannot record usage: head_hash {head_hash} "
                f"does not match current HEAD {self.head}"
            )

        # If no snapshot yet, trigger a compile to populate one
        snapshot = self._cache_get(target_hash)
        if snapshot is None:
            self.compile()
            snapshot = self._cache_get(target_hash)

        # Update snapshot with API-reported counts
        if snapshot is not None:
            token_source = f"api:{usage.prompt_tokens}+{usage.completion_tokens}"
            updated = replace(snapshot, token_count=usage.prompt_tokens, token_source=token_source)
            self._cache_put(target_hash, updated)
            return self._snapshot_to_compiled(updated)

        # Fallback (custom compiler, no snapshot): return minimal result
        token_source = f"api:{usage.prompt_tokens}+{usage.completion_tokens}"
        return CompiledContext(
            messages=[],
            token_count=usage.prompt_tokens,
            commit_count=0,
            token_source=token_source,
        )

    def _normalize_usage_dict(self, usage_dict: dict) -> TokenUsage:
        """Normalise provider-specific usage dicts to :class:`TokenUsage`.

        Supports OpenAI (``prompt_tokens``) and Anthropic (``input_tokens``)
        formats.
        """
        if "prompt_tokens" in usage_dict:
            return TokenUsage(
                prompt_tokens=usage_dict.get("prompt_tokens", 0),
                completion_tokens=usage_dict.get("completion_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
            )
        elif "input_tokens" in usage_dict:
            input_t = usage_dict.get("input_tokens", 0)
            output_t = usage_dict.get("output_tokens", 0)
            return TokenUsage(
                prompt_tokens=input_t,
                completion_tokens=output_t,
                total_tokens=input_t + output_t,
            )
        else:
            raise ContentValidationError(
                f"Unrecognized usage dict format. "
                f"Expected 'prompt_tokens' (OpenAI) or 'input_tokens' (Anthropic). "
                f"Got keys: {list(usage_dict.keys())}"
            )

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Context manager for atomic multi-commit batches.

        Defers the session ``commit()`` until the batch exits successfully.
        Rolls back on exception.

        Example::

            with t.batch():
                t.commit(InstructionContent(text="System prompt"))
                t.commit(DialogueContent(role="user", text="Hi"))
        """
        # Invalidate compile cache on batch entry
        self._cache_clear()

        # Stash the real session.commit and replace with a no-op
        _real_commit = self._session.commit

        def _noop_commit() -> None:
            pass

        self._session.commit = _noop_commit  # type: ignore[assignment]
        try:
            yield
            # Success: flush pending and commit once
            _real_commit()
        except Exception:
            self._session.rollback()
            raise
        finally:
            self._session.commit = _real_commit  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # LRU cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, head_hash: str) -> CompileSnapshot | None:
        """Get snapshot from LRU cache.  Returns None on miss."""
        if head_hash not in self._snapshot_cache:
            return None
        self._snapshot_cache.move_to_end(head_hash)
        return self._snapshot_cache[head_hash]

    def _cache_put(self, head_hash: str, snapshot: CompileSnapshot) -> None:
        """Store snapshot in LRU cache, evicting LRU entry if at capacity."""
        if head_hash in self._snapshot_cache:
            self._snapshot_cache.move_to_end(head_hash)
        self._snapshot_cache[head_hash] = snapshot
        while len(self._snapshot_cache) > self._snapshot_cache_maxsize:
            self._snapshot_cache.popitem(last=False)

    def _cache_clear(self) -> None:
        """Clear all cached snapshots."""
        self._snapshot_cache.clear()

    # ------------------------------------------------------------------
    # Incremental compile helpers
    # ------------------------------------------------------------------

    def _snapshot_to_compiled(self, snapshot: CompileSnapshot) -> CompiledContext:
        """Convert a CompileSnapshot to a CompiledContext for return.

        Uses copy-on-output for generation_configs to prevent user mutations
        of the returned CompiledContext from corrupting the cached snapshot.
        """
        return CompiledContext(
            messages=list(snapshot.messages),
            token_count=snapshot.token_count,
            commit_count=snapshot.commit_count,
            token_source=snapshot.token_source,
            generation_configs=[dict(c) for c in snapshot.generation_configs],
            commit_hashes=list(snapshot.commit_hashes),
        )

    def _build_snapshot_from_compiled(
        self, head_hash: str, result: CompiledContext
    ) -> CompileSnapshot | None:
        """Build a CompileSnapshot from a full compile result.

        Returns None if the compiler is not a DefaultContextCompiler
        (custom compilers bypass incremental cache).
        """
        if not isinstance(self._compiler, DefaultContextCompiler):
            return None
        return CompileSnapshot(
            head_hash=head_hash,
            messages=tuple(result.messages),
            commit_count=result.commit_count,
            token_count=result.token_count,
            token_source=result.token_source,
            generation_configs=tuple(dict(c) for c in result.generation_configs),
            commit_hashes=tuple(result.commit_hashes),
        )

    def _tiktoken_source(self) -> str:
        """Return the token_source string for tiktoken-based counts."""
        if isinstance(self._token_counter, TiktokenCounter):
            return f"tiktoken:{self._token_counter._encoding_name}"
        return ""

    def _extend_snapshot_for_append(
        self, commit_info: CommitInfo, parent_snapshot: CompileSnapshot
    ) -> None:
        """Incrementally extend a cached snapshot for an APPEND commit.

        Builds the message for the new commit, appends it (no aggregation),
        and recounts tokens.  The parent snapshot stays in the LRU cache
        under its own HEAD (useful for future checkout back).
        """
        commit_row = self._commit_repo.get(commit_info.commit_hash)
        if commit_row is None:
            return

        assert isinstance(self._compiler, DefaultContextCompiler)
        new_message = self._compiler.build_message_for_commit(commit_row)
        new_config = dict(commit_row.generation_config_json or {})

        new_messages = parent_snapshot.messages + (new_message,)
        new_commit_hashes = parent_snapshot.commit_hashes + (commit_info.commit_hash,)

        # Recount tokens on all messages
        messages_dicts = [
            {"role": m.role, "content": m.content}
            if m.name is None
            else {"role": m.role, "content": m.content, "name": m.name}
            for m in new_messages
        ]
        new_token_count = self._token_counter.count_messages(messages_dicts)

        self._cache_put(
            commit_info.commit_hash,
            CompileSnapshot(
                head_hash=commit_info.commit_hash,
                messages=new_messages,
                commit_count=parent_snapshot.commit_count + 1,
                token_count=new_token_count,
                token_source=self._tiktoken_source(),
                generation_configs=parent_snapshot.generation_configs + (new_config,),
                commit_hashes=new_commit_hashes,
            ),
        )

    def _patch_snapshot_for_edit(
        self,
        parent_snapshot: CompileSnapshot,
        new_head_hash: str,
        edit_row: CommitRow,
    ) -> CompileSnapshot | None:
        """Patch a cached snapshot for an EDIT commit in-memory.

        Finds the message corresponding to the edited target (via response_to),
        replaces it with the new message, and recounts tokens.

        Returns None if patching is not possible (missing commit_hashes, target
        not found), signaling caller to fall back to full recompile on next
        compile().
        """
        if not parent_snapshot.commit_hashes:
            return None

        target_hash = edit_row.response_to
        if target_hash is None:
            return None

        # Find position of the target commit in the snapshot
        try:
            target_idx = list(parent_snapshot.commit_hashes).index(target_hash)
        except ValueError:
            return None  # Target not in snapshot

        assert isinstance(self._compiler, DefaultContextCompiler)
        new_message = self._compiler.build_message_for_commit(edit_row)

        # Replace message at target position
        new_messages = list(parent_snapshot.messages)
        new_messages[target_idx] = new_message

        # Handle generation_config: edit-inherits-original rule
        new_configs = list(parent_snapshot.generation_configs)
        if edit_row.generation_config_json is not None:
            new_configs[target_idx] = dict(edit_row.generation_config_json)  # copy-on-input
        # else: keep original config at target_idx (edit-inherits-original)

        # Recount tokens
        messages_dicts = [
            {"role": m.role, "content": m.content}
            if m.name is None
            else {"role": m.role, "content": m.content, "name": m.name}
            for m in new_messages
        ]
        new_token_count = self._token_counter.count_messages(messages_dicts)

        return CompileSnapshot(
            head_hash=new_head_hash,
            messages=tuple(new_messages),
            commit_count=parent_snapshot.commit_count,  # Same count (EDIT replaces, doesn't add)
            token_count=new_token_count,
            token_source=self._tiktoken_source(),
            generation_configs=tuple(new_configs),
            commit_hashes=parent_snapshot.commit_hashes,  # Same positions
        )

    def _patch_snapshot_for_annotate(
        self,
        snapshot: CompileSnapshot,
        target_hash: str,
        new_priority: Priority,
    ) -> CompileSnapshot | None:
        """Patch a cached snapshot for an annotation change.

        SKIP: remove the target's message from the snapshot.
        NORMAL/PINNED on already-included commit: no change needed.
        NORMAL/PINNED on previously-SKIP commit: return None (full recompile).
        """
        if not snapshot.commit_hashes:
            return None

        # Find target position
        target_idx = None
        for i, ch in enumerate(snapshot.commit_hashes):
            if ch == target_hash:
                target_idx = i
                break

        if new_priority == Priority.SKIP:
            if target_idx is None:
                return snapshot  # Already not in snapshot

            # Remove message, config, and hash at target position
            new_messages = list(snapshot.messages)
            new_configs = list(snapshot.generation_configs)
            new_hashes = list(snapshot.commit_hashes)
            del new_messages[target_idx]
            del new_configs[target_idx]
            del new_hashes[target_idx]

            # Recount tokens
            messages_dicts = [
                {"role": m.role, "content": m.content}
                if m.name is None
                else {"role": m.role, "content": m.content, "name": m.name}
                for m in new_messages
            ]
            new_token_count = self._token_counter.count_messages(messages_dicts)

            return CompileSnapshot(
                head_hash=snapshot.head_hash,
                messages=tuple(new_messages),
                commit_count=snapshot.commit_count - 1,
                token_count=new_token_count,
                token_source=self._tiktoken_source(),
                generation_configs=tuple(new_configs),
                commit_hashes=tuple(new_hashes),
            )
        else:
            # NORMAL or PINNED
            if target_idx is not None:
                return snapshot  # Already included, no change
            else:
                return None  # Was skipped, need full recompile (don't have message content)

    def register_content_type(self, name: str, model: type[BaseModel]) -> None:
        """Register a custom content type for this tract instance.

        Args:
            name: The ``content_type`` discriminator value.
            model: A Pydantic ``BaseModel`` subclass.
        """
        self._custom_type_registry[name] = model

    def close(self) -> None:
        """Close the session and dispose the engine."""
        if self._closed:
            return
        self._closed = True
        self._session.close()
        if self._engine is not None:
            self._engine.dispose()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Tract:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._closed:
            return f"Tract(tract_id='{self._tract_id}', closed=True)"
        return f"Tract(tract_id='{self._tract_id}', head='{self.head}')"
