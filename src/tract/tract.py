"""Tract -- the public SDK entry point for Trace.

Ties together storage, commit engine, and context compiler into a clean,
user-facing API.  Users interact with ``Tract.open()``, ``t.commit()``,
``t.compile()``, etc.

Not thread-safe in v1.  Each thread should open its own ``Tract``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from tract.engine.cache import CacheManager
from tract.engine.commit import CommitEngine
from tract.engine.compiler import DefaultContextCompiler
from tract.engine.tokens import TiktokenCounter
from tract.models.annotations import Priority, PriorityAnnotation, RetentionCriteria
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.config import LLMConfig, Operator, OperationClients, OperationConfigs, TractConfig
from tract.models.content import validate_content
from tract.exceptions import (
    BranchNotFoundError,
    CommitNotFoundError,
    ContentValidationError,
    DetachedHeadError,
    TraceError,
)
from tract.protocols import ChatResponse, CompiledContext, ContextCompiler, TokenCounter, TokenUsage
from tract.storage.engine import create_session_factory, create_trace_engine, init_db
from tract.storage.sqlite import (
    SqliteAnnotationRepository,
    SqliteBlobRepository,
    SqliteCommitParentRepository,
    SqliteCommitRepository,
    SqliteCompileRecordRepository,
    SqliteOperationEventRepository,
    SqliteRefRepository,
    SqliteSpawnPointerRepository,
    SqliteToolSchemaRepository,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from sqlalchemy import Engine
    from sqlalchemy.orm import Session

    from tract.models.branch import BranchInfo
    from tract.models.compression import CompressResult, GCResult, PendingCompression
    from tract.models.merge import ImportResult, MergeResult, RebaseResult
    from tract.models.policy import PolicyProposal
    from tract.models.session import SpawnInfo
    from tract.operations.diff import DiffResult
    from tract.operations.history import StatusInfo
    from tract.orchestrator.config import AutonomyLevel, OrchestratorConfig
    from tract.orchestrator.loop import Orchestrator
    from tract.orchestrator.models import OrchestratorResult
    from tract.policy.evaluator import PolicyEvaluator
    from tract.storage.schema import CommitRow
    from tract.storage.sqlite import SqlitePolicyRepository


# ------------------------------------------------------------------
# Auto-message generation helper
# ------------------------------------------------------------------


_MAX_AUTO_MSG_LEN = 500


def _auto_message(content_type: str, text: str) -> str:
    """Generate a descriptive commit message from content text.

    The content_type is available separately on the commit; the message
    is a preview of the text (max 500 chars, newlines flattened).
    Full content is available via ``Tract.show()`` or
    ``Tract.get_content()``.

    Args:
        content_type: The content type discriminator (kept for the
            empty-text fallback).
        text: The text content of the commit.

    Returns:
        A text preview (max 500 chars), or the content_type if text
        is empty.
    """
    preview = text.strip().replace("\n", " ")
    if not preview:
        return content_type
    if len(preview) > _MAX_AUTO_MSG_LEN:
        preview = preview[: _MAX_AUTO_MSG_LEN - 3] + "..."
    return preview


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
        event_repo: SqliteOperationEventRepository | None = None,
        compile_record_repo: SqliteCompileRecordRepository | None = None,
        tool_schema_repo: object | None = None,
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
        self._event_repo = event_repo
        self._compile_record_repo = compile_record_repo
        self._tool_schema_repo = tool_schema_repo
        self._spawn_repo: SqliteSpawnPointerRepository | None = None
        self._custom_type_registry: dict[str, type[BaseModel]] = {}
        self._cache = CacheManager(
            maxsize=config.compile_cache_maxsize,
            compiler=compiler,
            token_counter=token_counter,
            commit_repo=commit_repo,
        )
        self._verify_cache: bool = verify_cache
        self._in_batch: bool = False
        self._closed = False
        self._policy_evaluator: PolicyEvaluator | None = None
        self._policy_repo: SqlitePolicyRepository | None = None
        self._orchestrating: bool = False
        self._orchestrator: Orchestrator | None = None  # type: ignore[assignment]
        self._trigger_commit_count: int = 0
        self._token_trigger_fired: bool = False
        self._owns_llm_client: bool = False
        self._default_config: LLMConfig | None = None
        self._operation_configs: OperationConfigs = OperationConfigs()
        self._operation_clients: OperationClients = OperationClients()
        self._active_tools: list[dict] | None = None

    @classmethod
    def open(
        cls,
        path: str = ":memory:",
        *,
        url: str | None = None,
        engine: Engine | None = None,
        tract_id: str | None = None,
        config: TractConfig | None = None,
        tokenizer: TokenCounter | None = None,
        compiler: ContextCompiler | None = None,
        verify_cache: bool = False,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        llm_client: object | None = None,
        default_config: LLMConfig | None = None,
        operations: OperationConfigs | None = None,
        operation_configs: dict[str, LLMConfig] | None = None,  # deprecated: use operations=
        tokenizer_encoding: str | None = None,
    ) -> Tract:
        """Open (or create) a Trace repository.

        Supports three database modes:

        1. **SQLite shorthand** (default): ``Tract.open(":memory:")`` or
           ``Tract.open("/path/to/file.db")``.
        2. **Full URL**: ``Tract.open(url="postgresql://user:pass@host/db")``
           for any SQLAlchemy-supported backend.
        3. **Pre-built engine**: ``Tract.open(engine=my_engine)`` for full
           control over connection pooling, echo, etc.

        Args:
            path: SQLite path.  ``":memory:"`` for in-memory (default).
                Ignored when *url* or *engine* is provided.
            url: Full SQLAlchemy database URL.  Use this for non-SQLite
                backends (PostgreSQL, MySQL, etc.) or for explicit SQLite
                URLs like ``"sqlite:///path/to/file.db"``.
            engine: A pre-built SQLAlchemy ``Engine``.  When provided,
                Tract skips engine creation entirely and uses this engine
                directly.  The caller owns the engine lifecycle.
            tract_id: Unique tract identifier.  Generated if not provided.
            config: Tract configuration.  Defaults created if *None*.
            tokenizer: Pluggable token counter.  TiktokenCounter by default.
            compiler: Pluggable context compiler.  DefaultContextCompiler by default.
            verify_cache: If True, cross-check every cache hit against a
                full recompile (oracle testing).  Default False.
            api_key: If provided, auto-configures an OpenAI-compatible LLM
                client.  Enables :meth:`chat` and :meth:`generate`.
            model: Default model for LLM calls (default ``"gpt-4o-mini"``).
                Only used when *api_key* is provided.
            base_url: API base URL override for OpenAI-compatible APIs.
                Only used when *api_key* is provided.
            llm_client: A custom LLM client conforming to the
                :class:`~tract.llm.protocols.LLMClient` protocol.  Mutually
                exclusive with *api_key*.  The caller owns the client lifecycle
                (Tract will **not** close it).
            default_config: Default LLM config for all operations.  Mutually
                exclusive with *model=*.  Use for full control over defaults.
            operations: Per-operation LLM configuration defaults as a typed
                :class:`OperationConfigs` instance.
            operation_configs: *Deprecated:* Per-operation LLM configuration
                defaults as a dict.  Use *operations=* instead.
            tokenizer_encoding: Tiktoken encoding name for token counting.
                Common values: ``"o200k_base"`` (GPT-4o/o1/o3, default),
                ``"cl100k_base"`` (GPT-4/3.5-turbo).  Overrides
                ``config.tokenizer_encoding`` when both are provided.

        Returns:
            A ready-to-use ``Tract`` instance.

        Raises:
            ValueError: If mutually exclusive params are combined
                (e.g. *url* + *engine*, *model* + *default_config*).
        """
        # Validate mutual exclusivity of database params
        _has_path = path != ":memory:"
        _has_url = url is not None
        _has_engine = engine is not None
        if sum([_has_path, _has_url, _has_engine]) > 1:
            raise ValueError(
                "Specify at most one of: path= (SQLite shorthand), "
                "url= (full SQLAlchemy URL), or engine= (pre-built engine)."
            )

        if tract_id is None:
            tract_id = uuid.uuid4().hex

        if config is None:
            if url is not None:
                config = TractConfig(db_url=url)
            else:
                config = TractConfig(db_path=path)

        # Engine / session
        if engine is None:
            if url is not None:
                engine = create_trace_engine(url=url)
            else:
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
        event_repo = SqliteOperationEventRepository(session)
        compile_record_repo = SqliteCompileRecordRepository(session)
        tool_schema_repo = SqliteToolSchemaRepository(session)

        # Token counter (tokenizer_encoding= overrides config when both provided)
        encoding = tokenizer_encoding or config.tokenizer_encoding
        token_counter = tokenizer or TiktokenCounter(
            encoding_name=encoding,
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

        # Spawn pointer repository
        spawn_repo = SqliteSpawnPointerRepository(session)

        # Policy repository
        from tract.storage.sqlite import SqlitePolicyRepository as _SqlitePolicyRepository
        policy_repo = _SqlitePolicyRepository(session)

        # Ensure "main" branch ref exists (idempotent)
        head = ref_repo.get_head(tract_id)
        if head is None:
            # No HEAD yet -- that is fine, first commit will set it.
            pass

        tract = cls(
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
            event_repo=event_repo,
            compile_record_repo=compile_record_repo,
            tool_schema_repo=tool_schema_repo,
            verify_cache=verify_cache,
        )
        tract._spawn_repo = spawn_repo
        tract._policy_repo = policy_repo

        # Auto-load persisted policy config (if any)
        saved_config = tract.load_policy_config()
        if saved_config is not None:
            from tract.policy.builtin import (
                ArchivePolicy as _ArchivePolicy,
                BranchPolicy as _BranchPolicy,
                CompressPolicy as _CompressPolicy,
                PinPolicy as _PinPolicy,
            )

            _policy_type_map: dict[str, type] = {
                "auto-compress": _CompressPolicy,
                "auto-pin": _PinPolicy,
                "auto-branch": _BranchPolicy,
                "auto-archive": _ArchivePolicy,
                # Backward compat for saved configs with old name
                "auto-rebase": _ArchivePolicy,
            }
            policies = []
            for entry in saved_config.get("policies", []):
                policy_cls = _policy_type_map.get(entry.get("name"))
                if policy_cls is not None and entry.get("enabled", True):
                    policies.append(policy_cls.from_config(entry))
            if policies:
                tract.configure_policies(policies=policies)

        # Validate: model= and default_config= are mutually exclusive
        if model is not None and default_config is not None:
            raise ValueError(
                "Cannot specify both model= and default_config=. "
                "Use default_config=LLMConfig(model=...) for full control."
            )

        # Validate: api_key= and llm_client= are mutually exclusive
        if api_key is not None and llm_client is not None:
            raise ValueError(
                "Cannot specify both api_key= and llm_client=. "
                "Use api_key= for built-in OpenAI client, or "
                "llm_client= for a custom client."
            )

        # Auto-configure LLM if api_key provided
        if api_key is not None:
            from tract.llm.client import OpenAIClient

            client = OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                default_model=model or "gpt-4o-mini",
            )
            tract.configure_llm(client)
            tract._owns_llm_client = True
            if model is not None:
                tract._default_config = LLMConfig(model=model)

        # Configure externally-provided LLM client (caller owns lifecycle)
        elif llm_client is not None:
            tract.configure_llm(llm_client)
            tract._owns_llm_client = False

        # Apply default_config if provided (without api_key)
        if default_config is not None and tract._default_config is None:
            tract._default_config = default_config

        # Apply per-operation configs (new typed path)
        if operations is not None:
            tract._operation_configs = operations
        # Apply per-operation configs (legacy dict path)
        elif operation_configs is not None:
            tract.configure_operations(**operation_configs)

        return tract

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
        llm_client: object | None = None,
        event_repo: SqliteOperationEventRepository | None = None,
        compile_record_repo: SqliteCompileRecordRepository | None = None,
        tool_schema_repo: object | None = None,
    ) -> Tract:
        """Create a ``Tract`` from pre-built components.

        Skips engine/session creation.  Useful for testing and DI.

        Args:
            llm_client: Optional custom LLM client.  Caller owns lifecycle.
            compile_record_repo: Optional compile record repository.
            tool_schema_repo: Optional tool schema repository.
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

        tract = cls(
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
            event_repo=event_repo,
            compile_record_repo=compile_record_repo,
            tool_schema_repo=tool_schema_repo,
        )
        if llm_client is not None:
            tract.configure_llm(llm_client)
            tract._owns_llm_client = False
        return tract

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

    @property
    def spawn_repo(self) -> SqliteSpawnPointerRepository | None:
        """Expose spawn repo for internal use by Session."""
        return self._spawn_repo

    @property
    def policy_evaluator(self) -> PolicyEvaluator | None:
        """The policy evaluator, or None if not configured."""
        return self._policy_evaluator

    # ------------------------------------------------------------------
    # Spawn relationship helpers
    # ------------------------------------------------------------------

    def parent(self) -> SpawnInfo | None:
        """Get the spawn info for this tract's parent.

        Returns:
            SpawnInfo if this tract was spawned from a parent, None for
            root tracts or tracts without a spawn_repo.
        """
        if self._spawn_repo is None:
            return None
        row = self._spawn_repo.get_by_child(self._tract_id)
        if row is None:
            return None
        from tract.operations.spawn import _row_to_spawn_info
        return _row_to_spawn_info(row)

    def children(self) -> list[SpawnInfo]:
        """Get spawn info for all child tracts spawned from this tract.

        Returns:
            List of SpawnInfo for each child, in chronological order.
            Empty list if no children or no spawn_repo.
        """
        if self._spawn_repo is None:
            return []
        rows = self._spawn_repo.get_children(self._tract_id)
        from tract.operations.spawn import _row_to_spawn_info
        return [_row_to_spawn_info(row) for row in rows]

    # ------------------------------------------------------------------
    # Tool tracking
    # ------------------------------------------------------------------

    def set_tools(self, tools: list[dict] | None) -> None:
        """Set active tool definitions for subsequent commits.

        When set, every subsequent ``commit()`` (including ``system()``,
        ``user()``, ``assistant()``, etc.) will automatically link these
        tool definitions unless overridden by passing ``tools=`` to
        ``commit()`` explicitly.

        Pass ``None`` to clear.

        Args:
            tools: List of tool definition dicts, or None to clear.
        """
        self._active_tools = tools

    def get_tools(self) -> list[dict] | None:
        """Get the currently active tool definitions.

        Returns:
            The list set via ``set_tools()``, or None if not set.
        """
        return self._active_tools

    def get_commit_tools(self, commit_hash: str) -> list[dict]:
        """Get tool definitions linked to a specific commit.

        Args:
            commit_hash: The commit to query.

        Returns:
            List of tool definition dicts in position order, or empty
            list if no tools are linked or repo is not available.
        """
        if self._tool_schema_repo is None:
            return []
        rows = self._tool_schema_repo.get_for_commit(commit_hash)
        return [row.schema_json for row in rows]

    def _store_and_link_tools(
        self, commit_hash: str, tools: list[dict]
    ) -> None:
        """Store tool schemas (content-addressed) and link to a commit.

        Args:
            commit_hash: The commit to link tools to.
            tools: List of tool definition dicts.
        """
        from datetime import datetime as _dt, timezone as _tz
        from tract.models.tools import hash_tool_schema

        now = _dt.now(_tz.utc)
        for position, tool in enumerate(tools):
            content_hash = hash_tool_schema(tool)
            name = ""
            # Extract name from common tool definition formats
            if isinstance(tool, dict):
                # OpenAI format: {"type": "function", "function": {"name": ...}}
                func = tool.get("function", {})
                if isinstance(func, dict):
                    name = func.get("name", "")
                # Anthropic format: {"name": ..., "input_schema": ...}
                if not name:
                    name = tool.get("name", "")
            self._tool_schema_repo.store(content_hash, name, tool, now)
            self._tool_schema_repo.link_to_commit(commit_hash, content_hash, position)

    def _gather_tools_for_compile(self) -> list[dict]:
        """Gather tools from the last commit that has tools linked.

        Walks the commit chain from HEAD backwards to find the most recent
        commit with tool definitions, and returns those tools.

        Returns:
            List of tool definition dicts, or empty list if none found.
        """
        if self._tool_schema_repo is None:
            return []

        current_head = self.head
        if current_head is None:
            return []

        # Walk ancestor chain looking for commits with tools
        ancestors = self._commit_repo.get_ancestors(current_head)
        for commit_row in ancestors:
            tool_hashes = self._tool_schema_repo.get_commit_tool_hashes(
                commit_row.commit_hash
            )
            if tool_hashes:
                rows = self._tool_schema_repo.get_for_commit(
                    commit_row.commit_hash
                )
                return [row.schema_json for row in rows]

        return []

    def _inject_tools(self, result: CompiledContext) -> CompiledContext:
        """Inject tool definitions into a compiled context.

        Gathers tools from the last commit with tools linked and creates
        a new CompiledContext with the tools field populated. Returns the
        original result unchanged if no tools are found.
        """
        tools = self._gather_tools_for_compile()
        if not tools:
            return result
        # CompiledContext is frozen, so create new instance with tools
        return CompiledContext(
            messages=result.messages,
            token_count=result.token_count,
            commit_count=result.commit_count,
            token_source=result.token_source,
            generation_configs=result.generation_configs,
            commit_hashes=result.commit_hashes,
            tools=tools,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def commit(
        self,
        content: BaseModel | dict,
        *,
        operation: CommitOperation = CommitOperation.APPEND,
        message: str | None = None,
        edit_target: str | None = None,
        metadata: dict | None = None,
        generation_config: dict | None = None,
        tools: list[dict] | None = None,
    ) -> CommitInfo:
        """Create a new commit.

        Args:
            content: A Pydantic content model *or* a dict (auto-validated).
            operation: ``APPEND`` (default) or ``EDIT``.
            message: Optional human-readable message.
            edit_target: For ``EDIT``, the hash of the commit being replaced.
            metadata: Optional arbitrary metadata.
            generation_config: Optional LLM generation config (temperature,
                model, top_p, etc.).  Immutable once committed.
            tools: Optional list of tool definitions (JSON schema dicts) to
                associate with this commit.  When ``set_tools()`` has been
                called, those tools are auto-linked to every subsequent commit
                unless overridden by passing ``tools=`` explicitly.

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        # Guard: detached HEAD blocks commits
        if self._ref_repo.is_detached(self._tract_id):
            raise DetachedHeadError()

        # Auto-validate dicts through the content type system
        if isinstance(content, dict):
            content = validate_content(content, custom_registry=self._custom_type_registry)

        # Auto-generate commit message if not provided
        if message is None and isinstance(content, BaseModel):
            from tract.engine.commit import extract_text_from_content as _extract_text

            _text = _extract_text(content)
            _ctype = content.model_dump(mode="json").get("content_type", "unknown")
            message = _auto_message(_ctype, _text)

        prev_head = self.head

        info = self._commit_engine.create_commit(
            content=content,
            operation=operation,
            message=message,
            edit_target=edit_target,
            metadata=metadata,
            generation_config=generation_config,
        )

        # Link tool schemas to this commit
        effective_tools = tools if tools is not None else self._active_tools
        if effective_tools is not None and self._tool_schema_repo is not None:
            self._store_and_link_tools(info.commit_hash, effective_tools)

        # Persist to database
        self._session.commit()

        # Update compile cache: incremental extend for APPEND,
        # in-memory patching for EDIT, otherwise next compile() rebuilds.
        # Skip cache updates during batch() -- cache was cleared on entry
        # and will be rebuilt on next compile() after the batch completes.
        if not self._in_batch:
            if operation == CommitOperation.APPEND and self._cache.uses_default_compiler:
                parent_snapshot = self._cache.get(prev_head) if prev_head else None
                if parent_snapshot is not None:
                    self._cache.extend_for_append(info, parent_snapshot)
                # If no parent snapshot, next compile() builds from scratch
            elif operation == CommitOperation.EDIT and self._cache.uses_default_compiler:
                parent_snapshot = self._cache.get(prev_head) if prev_head else None
                if parent_snapshot is not None:
                    edit_row = self._commit_repo.get(info.commit_hash)
                    if edit_row is not None:
                        patched = self._cache.patch_for_edit(
                            parent_snapshot, info.commit_hash, edit_row
                        )
                        if patched is not None:
                            self._cache.put(info.commit_hash, patched)
                # Do NOT clear cache -- other entries at different HEADs remain valid

        # Evaluate commit-triggered policies (after commit)
        if (
            self._policy_evaluator is not None
            and not self._in_batch
            and not self._orchestrating
        ):
            self._policy_evaluator.evaluate(trigger="commit")

        # Check orchestrator triggers (after policy evaluation)
        if not self._orchestrating and not self._in_batch:
            self._check_orchestrator_triggers("commit")

        return info

    def system(
        self,
        text: str,
        *,
        edit: str | None = None,
        message: str | None = None,
        metadata: dict | None = None,
        priority: Priority | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
    ) -> CommitInfo:
        """Commit a system instruction.

        Shorthand for ``commit(InstructionContent(text=text))``.

        System instructions are **PINNED by default** — they survive
        compression unchanged.  Pass ``priority=`` to override (e.g.
        ``Priority.NORMAL`` to allow compression).

        Args:
            text: The instruction text.
            edit: If provided, the hash of the commit to replace (EDIT
                operation).  Omit for a normal APPEND.
            message: Optional commit message.
            metadata: Optional commit metadata.
            priority: Optional priority annotation to set on the commit.
                Overrides the default PINNED annotation.
            retain: Fuzzy retention instructions (for IMPORTANT priority).
            retain_match: Deterministic retention patterns (for IMPORTANT priority).

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        from tract.models.content import InstructionContent

        info = self.commit(
            InstructionContent(text=text),
            operation=CommitOperation.EDIT if edit else CommitOperation.APPEND,
            edit_target=edit,
            message=message,
            metadata=metadata,
        )
        if priority is not None:
            self.annotate(
                info.commit_hash, priority,
                retain=retain, retain_match=retain_match,
            )
        return info

    def user(
        self,
        text: str,
        *,
        edit: str | None = None,
        message: str | None = None,
        name: str | None = None,
        metadata: dict | None = None,
        priority: Priority | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
    ) -> CommitInfo:
        """Commit a user message.

        Shorthand for ``commit(DialogueContent(role='user', text=text))``.

        Args:
            text: The message text.
            edit: If provided, the hash of the commit to replace (EDIT
                operation).  Omit for a normal APPEND.
            message: Optional commit message.
            name: Optional speaker name.
            metadata: Optional commit metadata.
            priority: Optional priority annotation to set on the commit.
            retain: Fuzzy retention instructions (for IMPORTANT priority).
            retain_match: Deterministic retention patterns (for IMPORTANT priority).

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        from tract.models.content import DialogueContent

        info = self.commit(
            DialogueContent(role="user", text=text, name=name),
            operation=CommitOperation.EDIT if edit else CommitOperation.APPEND,
            edit_target=edit,
            message=message,
            metadata=metadata,
        )
        if priority is not None:
            self.annotate(
                info.commit_hash, priority,
                retain=retain, retain_match=retain_match,
            )
        return info

    def assistant(
        self,
        text: str,
        *,
        edit: str | None = None,
        message: str | None = None,
        name: str | None = None,
        metadata: dict | None = None,
        generation_config: dict | None = None,
        priority: Priority | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
    ) -> CommitInfo:
        """Commit an assistant response.

        Shorthand for ``commit(DialogueContent(role='assistant', text=text))``.

        Args:
            text: The response text.
            edit: If provided, the hash of the commit to replace (EDIT
                operation).  Omit for a normal APPEND.
            message: Optional commit message.
            name: Optional speaker name.
            metadata: Optional commit metadata.
            generation_config: Optional LLM generation config.
            priority: Optional priority annotation to set on the commit.
            retain: Fuzzy retention instructions (for IMPORTANT priority).
            retain_match: Deterministic retention patterns (for IMPORTANT priority).

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        from tract.models.content import DialogueContent

        info = self.commit(
            DialogueContent(role="assistant", text=text, name=name),
            operation=CommitOperation.EDIT if edit else CommitOperation.APPEND,
            edit_target=edit,
            message=message,
            metadata=metadata,
            generation_config=generation_config,
        )
        if priority is not None:
            self.annotate(
                info.commit_hash, priority,
                retain=retain, retain_match=retain_match,
            )
        return info

    # ------------------------------------------------------------------
    # Conversation layer (chat/generate)
    # ------------------------------------------------------------------

    def _resolve_llm_config(
        self,
        operation: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        **kwargs: object,
    ) -> dict:
        """Resolve effective LLM config: sugar > llm_config > operation > tract default.

        Four-level resolution chain for each field:
        1. Sugar params (model=, temperature=, max_tokens=) -- highest priority
        2. llm_config fields (if provided and field is not None)
        3. Operation-level config (from configure_operations)
        4. Tract-level default config (_default_config)

        Returns a dict of kwargs to pass to llm_client.chat(). Only includes
        keys that have a non-None value at some level in the chain.

        Args:
            operation: Operation name ("chat", "merge", "compress", "orchestrate").
            model: Call-level model override (sugar).
            temperature: Call-level temperature override (sugar).
            max_tokens: Call-level max_tokens override (sugar).
            llm_config: Full LLMConfig override for this call.
            **kwargs: Additional call-level kwargs (highest priority).
        """
        op_config = getattr(self._operation_configs, operation, None)
        default = self._default_config

        # Sugar params dict (only the 3 convenience overrides)
        sugar: dict = {}
        if model is not None:
            sugar["model"] = model
        if temperature is not None:
            sugar["temperature"] = temperature
        if max_tokens is not None:
            sugar["max_tokens"] = max_tokens

        resolved: dict = {}

        # Resolve each typed field through 4-level chain
        _TYPED_FIELDS = (
            "model", "temperature", "max_tokens", "top_p",
            "frequency_penalty", "presence_penalty", "top_k",
            "seed", "stop_sequences",
        )
        for field_name in _TYPED_FIELDS:
            # Level 1: Sugar param
            val = sugar.get(field_name)
            if val is not None:
                resolved[field_name] = val
                continue
            # Level 2: llm_config
            if llm_config is not None:
                val = getattr(llm_config, field_name, None)
                if val is not None:
                    resolved[field_name] = val
                    continue
            # Level 3: Operation config
            if op_config is not None:
                val = getattr(op_config, field_name, None)
                if val is not None:
                    resolved[field_name] = val
                    continue
            # Level 4: Tract default
            if default is not None:
                val = getattr(default, field_name, None)
                if val is not None:
                    resolved[field_name] = val

        # Translate canonical names to OpenAI-compatible API names
        if "stop_sequences" in resolved:
            val = resolved.pop("stop_sequences")
            resolved["stop"] = list(val) if isinstance(val, tuple) else val

        # Merge extra kwargs: tract default < operation < llm_config < call kwargs
        # (each level's extra overrides the previous)
        if default is not None and default.extra:
            resolved.update(dict(default.extra))
        if op_config is not None and op_config.extra:
            resolved.update(dict(op_config.extra))
        if llm_config is not None and llm_config.extra:
            resolved.update(dict(llm_config.extra))
        resolved.update(kwargs)

        return resolved

    def _build_generation_config(self, response: dict, *, resolved: dict) -> dict:
        """Build generation_config from the full resolved LLM kwargs.

        Captures ALL fields that were sent to the LLM (model, temperature,
        top_p, seed, etc.) so they can be queried via query_by_config().

        The response's model field is authoritative (actual model used may
        differ from requested model due to aliases/routing).

        Args:
            response: Raw LLM response dict.
            resolved: The full resolved kwargs dict from _resolve_llm_config().
        """
        config = dict(resolved)
        # Response model is authoritative
        if "model" in response:
            config["model"] = response["model"]
        return config

    def _extract_content(self, response: dict, *, client: object | None = None) -> str:
        """Extract content from LLM response, dispatching to the client's method.

        Falls back to OpenAI-format extraction for duck-typed clients that
        don't implement ``extract_content()``.

        Args:
            response: Raw LLM response dict.
            client: The LLM client that produced the response.  If None,
                falls back to the tract-level default.
        """
        c = client if client is not None else getattr(self, "_llm_client", None)
        if c is not None and hasattr(c, "extract_content"):
            return c.extract_content(response)
        # Default: OpenAI format
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            from tract.llm.errors import LLMResponseError

            raise LLMResponseError(
                f"Cannot extract content from response: {exc}. "
                f"Implement extract_content() on your client for custom formats."
            ) from exc

    def _extract_usage(self, response: dict, *, client: object | None = None) -> dict | None:
        """Extract usage from LLM response, dispatching to the client's method.

        Falls back to OpenAI-format extraction for duck-typed clients that
        don't implement ``extract_usage()``.

        Args:
            response: Raw LLM response dict.
            client: The LLM client that produced the response.  If None,
                falls back to the tract-level default.
        """
        c = client if client is not None else getattr(self, "_llm_client", None)
        if c is not None and hasattr(c, "extract_usage"):
            return c.extract_usage(response)
        # Default: OpenAI format
        return response.get("usage")

    def generate(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        metadata: dict | None = None,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        max_retries: int = 3,
        purify: bool = False,
        provenance_note: bool = False,
        retry_prompt: str | None = None,
    ) -> ChatResponse:
        """Compile context, call LLM, commit assistant response, record usage.

        Assumes the conversation context (system prompt, user messages) has
        already been committed. Use :meth:`chat` for the all-in-one path.

        Args:
            model: Model override for this call.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            llm_config: Full LLMConfig override for this call.
            message: Optional commit message for the assistant commit.
            metadata: Optional metadata for the assistant commit.
            validator: Optional callable that validates the response text.
                Takes the response text, returns (ok, diagnosis). When provided,
                the generate call is wrapped with retry_with_steering.
            max_retries: Maximum retry attempts when validator is set (default 3).
            purify: If True, reset to pre-retry state on success and re-commit
                clean results (no retry artifacts in history).
            provenance_note: If True, commit a meta note recording retry count.
            retry_prompt: Custom steering prompt template. The diagnosis string
                is appended to this. Defaults to a standard steering message.

        Returns:
            :class:`ChatResponse` with text, usage, commit_info, generation_config.

        Raises:
            LLMConfigError: If no LLM client is configured.
            TraceError: If called inside batch().
            RetryExhaustedError: If all retry attempts fail validation.
        """
        if not self._has_llm_client("chat"):
            from tract.llm.errors import LLMConfigError

            raise LLMConfigError(
                "No LLM client configured. Pass api_key= or llm_client= "
                "to Tract.open(), or call configure_llm(client)."
            )

        if self._in_batch:
            raise TraceError("chat()/generate() cannot be used inside batch()")

        if validator is None:
            return self._generate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
            )

        # Retry-guarded path
        from tract.retry import RetryResult, retry_with_steering

        def _attempt() -> ChatResponse:
            return self._generate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
            )

        def _validate(resp: ChatResponse) -> tuple[bool, str | None]:
            return validator(resp.text)

        def _steer(diagnosis: str) -> None:
            prompt = retry_prompt or "The previous response did not pass validation."
            self.user(f"{prompt}\nDiagnosis: {diagnosis}")

        def _head_fn() -> str:
            return self.head or ""

        def _reset_fn(restore_point: str) -> None:
            if restore_point:
                self.reset(restore_point)

        def _provenance(attempts: int, history: list[str]) -> None:
            if provenance_note and attempts > 1:
                note = f"[retry] Succeeded on attempt {attempts}/{max_retries}"
                if history:
                    note += f". Previous failures: {'; '.join(history)}"
                self.user(note, message="retry provenance note")

        retry_result: RetryResult[ChatResponse] = retry_with_steering(
            attempt=_attempt,
            validate=_validate,
            steer=_steer,
            head_fn=_head_fn,
            reset_fn=_reset_fn,
            max_retries=max_retries,
            purify=purify,
            provenance_note=_provenance if provenance_note else None,
        )

        chat_response = retry_result.value

        # If purify was active and we had retries, re-commit clean result
        if purify and retry_result.attempts > 1:
            commit_info = self.assistant(
                chat_response.text,
                message=message,
                metadata=metadata,
                generation_config=(
                    chat_response.generation_config.to_dict()
                    if chat_response.generation_config
                    else None
                ),
            )
            from tract.protocols import ChatResponse as _CR

            chat_response = _CR(
                text=chat_response.text,
                usage=chat_response.usage,
                commit_info=commit_info,
                generation_config=chat_response.generation_config,
            )

        return chat_response

    def _generate_once(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        metadata: dict | None = None,
    ) -> ChatResponse:
        """Single generate attempt (no retry). Internal helper.

        Contains the original generate() logic: compile, LLM call, commit, usage.
        """
        from tract.protocols import ChatResponse

        # 1. Compile context
        compiled = self.compile()
        messages = compiled.to_dicts()

        # 1b. Persist compile record (SC-3: chat/generate auto-create)
        if self._compile_record_repo is not None:
            import uuid as _uuid
            from datetime import datetime as _dt, timezone as _tz

            record_id = _uuid.uuid4().hex
            current_head = self.head
            self._compile_record_repo.save_record(
                record_id=record_id,
                tract_id=self._tract_id,
                head_hash=current_head or "",
                token_count=compiled.token_count,
                commit_count=compiled.commit_count,
                token_source=compiled.token_source,
                params_json=None,  # No compile params for standard calls
                created_at=_dt.now(_tz.utc),
            )
            for pos, commit_hash in enumerate(compiled.commit_hashes):
                self._compile_record_repo.add_effective(
                    record_id, commit_hash, pos
                )

        # 2. Call LLM (resolve per-operation client and config)
        chat_client = self._resolve_llm_client("chat")
        llm_kwargs = self._resolve_llm_config(
            "chat", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
        )
        response = chat_client.chat(messages, **llm_kwargs)

        # 3. Extract content and usage (dispatch to client methods, with
        #    OpenAI-format defaults for duck-typed clients that lack them)
        text = self._extract_content(response, client=chat_client)
        usage_dict = self._extract_usage(response, client=chat_client)

        # 4. Build generation_config (use resolved kwargs for accurate tracking)
        gen_config = self._build_generation_config(response, resolved=llm_kwargs)

        # 5. Commit assistant response
        commit_info = self.assistant(
            text, message=message, metadata=metadata, generation_config=gen_config
        )

        # 6. Record usage
        usage = None
        if usage_dict:
            usage = self._normalize_usage_dict(usage_dict)
            self.record_usage(usage)

        return ChatResponse(
            text=text,
            usage=usage,
            commit_info=commit_info,
            generation_config=LLMConfig.from_dict(gen_config) or LLMConfig(),
        )

    def chat(
        self,
        text: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        name: str | None = None,
        metadata: dict | None = None,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        max_retries: int = 3,
        purify: bool = False,
        provenance_note: bool = False,
        retry_prompt: str | None = None,
    ) -> ChatResponse:
        """Send a user message and get an LLM response in one call.

        Commits the user message, compiles context, calls the LLM,
        commits the assistant response, and records usage. Equivalent to::

            t.user(text, message=message, name=name, metadata=metadata)
            response = t.generate(model=model, temperature=temperature, ...)

        Args:
            text: The user message text.
            model: Model override for this call.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            llm_config: Full LLMConfig override for this call.
            message: Optional commit message for the user commit.
            name: Optional speaker name for the user commit.
            metadata: Optional metadata for the user commit.
            validator: Optional callable that validates the response text.
                Takes the response text, returns (ok, diagnosis). When provided,
                the generate call is wrapped with retry_with_steering.
            max_retries: Maximum retry attempts when validator is set (default 3).
            purify: If True, reset to pre-retry state on success and re-commit
                clean results (no retry artifacts in history).
            provenance_note: If True, commit a meta note recording retry count.
            retry_prompt: Custom steering prompt template. The diagnosis string
                is appended to this. Defaults to a standard steering message.

        Returns:
            :class:`ChatResponse` with text, usage, commit_info, generation_config.

        Raises:
            LLMConfigError: If no LLM client is configured.
            DetachedHeadError: If HEAD is detached.
            TraceError: If called inside batch().
            RetryExhaustedError: If all retry attempts fail validation.
        """
        # Commit user message
        self.user(text, message=message, name=name, metadata=metadata)
        # Delegate to generate
        import dataclasses as _dc
        response = self.generate(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config=llm_config,
            validator=validator,
            max_retries=max_retries,
            purify=purify,
            provenance_note=provenance_note,
            retry_prompt=retry_prompt,
        )
        return _dc.replace(response, prompt=text)

    # ------------------------------------------------------------------
    # Compile record accessors
    # ------------------------------------------------------------------

    def compile_records(self, limit: int = 100) -> list:
        """Get compile records for this tract, newest first.

        Returns list of CompileRecordRow objects, or empty list if
        compile record repository is not available.
        """
        if self._compile_record_repo is None:
            return []
        records = self._compile_record_repo.get_all(self._tract_id)
        return list(reversed(records))[:limit]  # newest first

    def compile_record_commits(self, record_id: str) -> list[str]:
        """Get the ordered commit hashes for a compile record.

        Returns list of commit hashes in compilation order, or empty list
        if record not found or compile record repository not available.
        """
        if self._compile_record_repo is None:
            return []
        effectives = self._compile_record_repo.get_effectives(record_id)
        return [e.commit_hash for e in effectives]

    def compile(
        self,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
        include_edit_annotations: bool = False,
        order: list[str] | None = None,
        check_safety: bool = True,
    ) -> CompiledContext | tuple[CompiledContext, list]:
        """Compile the current context into LLM-ready messages.

        Args:
            at_time: Only include commits at or before this datetime.
            at_commit: Only include commits up to this hash.
            include_edit_annotations: Append ``[edited]`` markers.
            order: If provided, reorder compiled messages so that commits
                appear in this order. Commits not in ``order`` are appended
                at their original relative positions after the ordered ones.
                When ``order`` is provided, the compile cache is bypassed
                and the return type is ``(CompiledContext, list[ReorderWarning])``.
            check_safety: If True (default) and ``order`` is provided, run
                structural safety checks that warn about EDIT-before-target
                and broken response chains.

        Returns:
            :class:`CompiledContext` when ``order`` is None (default).
            ``(CompiledContext, list[ReorderWarning])`` when ``order`` is provided.
        """
        # Evaluate compile-triggered policies (before compilation)
        if (
            self._policy_evaluator is not None
            and not self._in_batch
            and not self._orchestrating
        ):
            self._policy_evaluator.evaluate(trigger="compile")

        # Check orchestrator triggers (after policy evaluation)
        if not self._orchestrating and not self._in_batch:
            self._check_orchestrator_triggers("compile")

        current_head = self.head
        if current_head is None:
            empty = CompiledContext(messages=[], token_count=0, commit_count=0, token_source="")
            if order is not None:
                return empty, []
            return empty

        # If order provided, bypass cache entirely and do a full compile + reorder
        if order is not None:
            result = self._compiler.compile(self._tract_id, current_head)
            warnings = []
            if check_safety:
                from tract.operations.compression import check_reorder_safety
                warnings = check_reorder_safety(order, self._commit_repo, self._blob_repo)
            reordered = self._reorder_compiled(result, order)
            reordered = self._inject_tools(reordered)
            return reordered, warnings

        # Time-travel and edit annotations: always full compile, don't touch snapshot
        if at_time is not None or at_commit is not None or include_edit_annotations:
            result = self._compiler.compile(
                self._tract_id,
                current_head,
                at_time=at_time,
                at_commit=at_commit,
                include_edit_annotations=include_edit_annotations,
            )
            # Apply API-reported token override if available for the
            # resolved head commit (tiktoken is temporary; API is truth).
            if result.commit_hashes:
                resolved_head = result.commit_hashes[-1]
                api_override = self._cache.get_api_override(resolved_head)
                if api_override is not None:
                    result = replace(result, token_count=api_override[0], token_source=api_override[1])
            return self._inject_tools(result)

        # Cache hit: snapshot exists for current head in LRU cache
        cached = self._cache.get(current_head)
        if cached is not None:
            result = self._cache.to_compiled(cached)
            if self._verify_cache:
                fresh = self._compiler.compile(self._tract_id, current_head)
                # Compare messages ignoring per-message token_count (computed
                # by cache, not compiler — fresh Messages have token_count=0)
                cached_core = [(m.role, m.content, m.name) for m in result.messages]
                fresh_core = [(m.role, m.content, m.name) for m in fresh.messages]
                assert cached_core == fresh_core, (
                    f"Cache message mismatch: cached {len(result.messages)} msgs, "
                    f"fresh {len(fresh.messages)} msgs"
                )
                # Skip token_count assertion when API-sourced (record_usage
                # calibrates to API totals which legitimately differ from tiktoken)
                if not result.token_source.startswith("api:"):
                    assert result.token_count == fresh.token_count, (
                        f"Cache token mismatch: cached {result.token_count}, "
                        f"fresh {fresh.token_count}"
                    )
                assert result.commit_count == fresh.commit_count, (
                    f"Cache commit_count mismatch: cached {result.commit_count}, "
                    f"fresh {fresh.commit_count}"
                )
                assert result.generation_configs == fresh.generation_configs, (
                    f"Cache generation_configs mismatch: "
                    f"cached {len(result.generation_configs)}, "
                    f"fresh {len(fresh.generation_configs)}"
                )
                assert result.commit_hashes == fresh.commit_hashes, (
                    f"Cache commit_hashes mismatch: "
                    f"cached {len(result.commit_hashes)}, "
                    f"fresh {len(fresh.commit_hashes)}"
                )
            return self._inject_tools(result)

        # Cache miss: full compile and build snapshot
        result = self._compiler.compile(self._tract_id, current_head)
        snapshot = self._cache.build_snapshot(current_head, result)
        if snapshot is not None:
            # Restore API-reported token override if it survived cache eviction
            api_override = self._cache.get_api_override(current_head)
            if api_override is not None:
                snapshot = replace(snapshot, token_count=api_override[0], token_source=api_override[1])
            self._cache.put(current_head, snapshot)
            result = self._cache.to_compiled(snapshot)
        return self._inject_tools(result)

    def _reorder_compiled(
        self, result: CompiledContext, order: list[str]
    ) -> CompiledContext:
        """Reorder a compiled context according to the given commit hash order.

        Commits in ``order`` appear first (in the specified order).
        Commits not in ``order`` are appended after, preserving their
        original relative positions.

        Args:
            result: The compiled context to reorder.
            order: List of commit hashes specifying the desired order.

        Returns:
            A new CompiledContext with reordered messages.

        Raises:
            CommitNotFoundError: If any hash in ``order`` is not in the
                compiled result's commit_hashes.
        """
        # Build mapping from hash to index in the compiled result
        hash_to_idx: dict[str, int] = {
            h: i for i, h in enumerate(result.commit_hashes)
        }

        # Validate all hashes in order exist in the result
        for h in order:
            if h not in hash_to_idx:
                raise CommitNotFoundError(h)

        # Build final index ordering
        new_indices = [hash_to_idx[h] for h in order]
        ordered_set = set(new_indices)
        remaining = [i for i in range(len(result.messages)) if i not in ordered_set]
        final_order = new_indices + remaining

        # Reorder all parallel arrays
        new_messages = [result.messages[i] for i in final_order]
        new_configs = (
            [result.generation_configs[i] for i in final_order]
            if result.generation_configs
            else []
        )
        new_hashes = [result.commit_hashes[i] for i in final_order]

        # Recount tokens for the new message order
        token_count = self._token_counter.count_messages(
            [{"role": m.role, "content": m.content} for m in new_messages]
        )

        return CompiledContext(
            messages=new_messages,
            token_count=token_count,
            commit_count=result.commit_count,
            token_source=result.token_source,
            generation_configs=new_configs,
            commit_hashes=new_hashes,
        )

    def get_commit(self, commit_hash: str) -> CommitInfo | None:
        """Fetch a commit by its hash.

        Returns:
            :class:`CommitInfo` if found, *None* otherwise.
        """
        return self._commit_engine.get_commit(commit_hash)

    def get_content(self, commit_or_hash: CommitInfo | str) -> str | None:
        """Load the full content text for a commit.

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.

        Returns:
            The content text, or *None* if the commit or blob is not found.
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

        import json
        try:
            data = json.loads(blob.payload_json)
        except (json.JSONDecodeError, TypeError):
            return blob.payload_json

        # Extract text from known content shapes
        for key in ("text", "content"):
            if key in data:
                return data[key]
        if "payload" in data:
            val = data["payload"]
            return json.dumps(val) if isinstance(val, dict) else str(val)
        return blob.payload_json

    def show(self, commit_or_hash: CommitInfo | str) -> None:
        """Pretty-print a commit with its full content.

        Like ``git show`` — displays commit metadata and the complete
        content text.  For metadata-only output, use
        ``info.pprint()`` instead.

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.
        """
        from tract.formatting import pprint_commit_info

        if isinstance(commit_or_hash, str):
            info = self.get_commit(commit_or_hash)
            if info is None:
                raise ValueError(f"Commit not found: {commit_or_hash}")
        else:
            info = commit_or_hash

        content = self.get_content(info)
        pprint_commit_info(info, content=content)

    def annotate(
        self,
        target_hash: str,
        priority: Priority,
        *,
        reason: str | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
        retain_match_mode: str = "substring",
    ) -> PriorityAnnotation:
        """Create a priority annotation on a commit.

        Args:
            target_hash: Hash of the commit to annotate.
            priority: Priority level (``SKIP``, ``NORMAL``, ``IMPORTANT``, ``PINNED``).
            reason: Optional reason for the annotation.
            retain: Fuzzy retention instructions (NL guidance for the LLM
                summarizer). Only meaningful for ``IMPORTANT`` priority.
            retain_match: Deterministic retention patterns -- substrings or
                regexes that MUST appear in compression summaries.
            retain_match_mode: How ``retain_match`` patterns are checked:
                ``"substring"`` (default) or ``"regex"``.

        Returns:
            :class:`PriorityAnnotation` model.
        """
        retention = None
        if retain is not None or retain_match is not None:
            retention = RetentionCriteria(
                instructions=retain,
                match_patterns=retain_match,
                match_mode=retain_match_mode,
            )
        annotation = self._commit_engine.annotate(
            target_hash, priority, reason, retention=retention
        )
        self._session.commit()

        # Annotations affect ALL cached snapshots that include the target commit.
        # Strategy: clear everything, then optionally re-add a patched current HEAD.
        # Exception: if patch returns the same snapshot object (NORMAL/PINNED on
        # already-included commit), the annotation is a no-op for compiled output
        # and we can skip the clear entirely.
        if self._cache.uses_default_compiler:
            current_head = self.head
            patched = None
            if current_head:
                snapshot = self._cache.get(current_head)
                if snapshot is not None:
                    patched = self._cache.patch_for_annotate(
                        snapshot, target_hash, priority
                    )
            if patched is not None and patched is snapshot:
                pass  # No-op: annotation didn't change compiled output
            else:
                self._cache.clear()
                if patched is not None:
                    self._cache.put(current_head, patched)
        else:
            self._cache.clear()

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
                retention=RetentionCriteria(**row.retention_json)
                if row.retention_json else None,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def annotation_counts(self, limit: int = 500) -> dict[str, int]:
        """Count pinned and skipped annotations across recent commits.

        Args:
            limit: Maximum number of commits to scan. Default 500.

        Returns:
            Dict with ``"pinned"`` and ``"skip"`` integer counts.
        """
        entries = self.log(limit=limit)
        commit_hashes = [e.commit_hash for e in entries]
        pinned = 0
        skip = 0
        if commit_hashes:
            annotations = self._annotation_repo.batch_get_latest(commit_hashes)
            for _hash, ann_row in annotations.items():
                if ann_row.priority == Priority.PINNED:
                    pinned += 1
                elif ann_row.priority == Priority.SKIP:
                    skip += 1
        return {"pinned": pinned, "skip": skip}

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
        cached = self._cache.get(commit_hash)
        if cached is not None:
            return self._cache.to_compiled(cached)
        # Cache miss: compile fresh and store for future hits
        result = self._compiler.compile(self._tract_id, commit_hash)
        snapshot = self._cache.build_snapshot(commit_hash, result)
        if snapshot is not None:
            self._cache.put(commit_hash, snapshot)
        return result

    def diff(
        self,
        commit_a: str | None = None,
        commit_b: str | None = None,
    ) -> DiffResult:
        """Compare two commits and return structured diff.

        Args:
            commit_a: First commit (hash or prefix).  If None and commit_b is
                an EDIT commit, auto-resolves to the edit target (edit_target).
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
            if row_b.operation == CommitOperation.EDIT and row_b.edit_target:
                commit_a = row_b.edit_target
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
        field_or_config: str | LLMConfig | None = None,
        operator: Operator | None = None,
        value: object = None,
        *,
        conditions: list[tuple[str, Operator, object]] | None = None,
    ) -> list[CommitInfo]:
        """Query commits by generation config values.

        Supports three calling patterns:

        1. **Single field**::

            t.query_by_config("model", "=", "gpt-4o")
            t.query_by_config("temperature", ">", 0.5)

        2. **Multi-field AND** — all conditions must match::

            t.query_by_config(conditions=[
                ("model", "=", "gpt-4o"),
                ("temperature", ">", 0.5),
            ])

        3. **Whole-config match**::

            t.query_by_config(LLMConfig(model="gpt-4o", temperature=0.7))
            # Finds commits matching ALL non-None fields with "=" semantics

        Supported operators:

        - ``"="``  — equal
        - ``"!="`` — not equal
        - ``">"``  — greater than
        - ``"<"``  — less than
        - ``">="`` — greater than or equal
        - ``"<="`` — less than or equal
        - ``"in"`` — set membership (value is a list)
        - ``"not in"`` — negated set membership (value is a list)
        - ``"between"`` — inclusive range (value is ``[low, high]``)
        - ``"not between"`` — outside inclusive range (value is ``[low, high]``)

        Examples::

            # Set membership and its negation
            t.query_by_config("model", "in", ["gpt-4o", "gpt-4o-mini"])
            t.query_by_config("model", "not in", ["gpt-4o", "gpt-4o-mini"])

            # Inclusive range and its negation
            t.query_by_config("temperature", "between", [0.3, 0.8])
            t.query_by_config("temperature", "not between", [0.3, 0.8])

            # Compose multiple conditions (AND)
            t.query_by_config(conditions=[
                ("temperature", "between", [0.3, 0.8]),
                ("model", "!=", "gpt-3.5-turbo"),
            ])

        Args:
            field_or_config: A field name (str) for single-field query, or
                an LLMConfig object for whole-config matching.
            operator: Comparison operator (single-field mode only).
                One of ``=``, ``!=``, ``>``, ``<``, ``>=``, ``<=``,
                ``in``, ``not in``, ``between``, ``not between``.
            value: Value to compare against (single-field mode only).
                For ``in``, pass a list. For ``between``, pass
                ``[low, high]`` (inclusive).
            conditions: List of ``(field, operator, value)`` tuples for
                multi-field AND queries.

        Returns:
            List of :class:`CommitInfo` matching the condition(s),
            ordered by created_at.
        """
        if isinstance(field_or_config, LLMConfig):
            # Whole-config match: convert non-None fields to AND conditions
            conds: list[tuple[str, str, object]] = []
            for k, v in field_or_config.non_none_fields().items():
                if isinstance(v, tuple):
                    v = list(v)  # SQLite expects list for JSON arrays
                conds.append((k, "=", v))
            if not conds:
                return []
            rows = self._commit_repo.get_by_config_multi(self._tract_id, conds)
        elif conditions is not None:
            # Multi-field AND
            rows = self._commit_repo.get_by_config_multi(self._tract_id, conditions)
        elif isinstance(field_or_config, str) and operator is not None:
            # Single-field (backward compatible)
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

    def configure_llm(
        self,
        client: object,
        *,
        resolver: object | None = None,
    ) -> None:
        """Configure the LLM client for semantic operations.

        Args:
            client: An LLM client conforming to the
                :class:`~tract.llm.protocols.LLMClient` protocol.
            resolver: Optional conflict resolver.  If *None*, an
                :class:`~tract.llm.resolver.OpenAIResolver` is created
                from *client* (suitable for OpenAI-compatible APIs).
                Pass a custom resolver for non-OpenAI clients.
        """
        self._llm_client = client
        if resolver is not None:
            self._default_resolver = resolver
        else:
            from tract.llm.resolver import OpenAIResolver

            self._default_resolver = OpenAIResolver(client)

    def configure_operations(
        self,
        _configs: OperationConfigs | None = None,
        /,
        **operation_configs: LLMConfig,
    ) -> None:
        """Set per-operation LLM defaults.

        Accepts either an OperationConfigs instance (new style) or keyword
        arguments (backward compatible).

        Args:
            _configs: OperationConfigs instance with typed fields.
            **operation_configs: Operation name -> LLMConfig mappings.
                Valid names: ``"chat"``, ``"merge"``, ``"compress"``,
                ``"orchestrate"``.

        Raises:
            TypeError: If both positional and keyword arguments provided,
                or if a keyword value is not an LLMConfig.

        Example::

            from tract import LLMConfig, OperationConfigs
            # New style:
            t.configure_operations(OperationConfigs(
                chat=LLMConfig(model="gpt-4o"),
                compress=LLMConfig(model="gpt-3.5-turbo"),
            ))
            # Backward compatible:
            t.configure_operations(
                chat=LLMConfig(model="gpt-4o"),
                compress=LLMConfig(model="gpt-3.5-turbo"),
            )
        """
        if _configs is not None and operation_configs:
            raise TypeError(
                "Cannot mix positional OperationConfigs with keyword arguments"
            )
        if _configs is not None:
            if not isinstance(_configs, OperationConfigs):
                raise TypeError(
                    f"Expected OperationConfigs, got {type(_configs).__name__}"
                )
            self._operation_configs = _configs
            return
        # Keyword path: validate and construct OperationConfigs
        _valid_ops = {"chat", "merge", "compress", "orchestrate"}
        for name, cfg in operation_configs.items():
            if not isinstance(cfg, LLMConfig):
                raise TypeError(
                    f"Expected LLMConfig for '{name}', "
                    f"got {type(cfg).__name__}"
                )
            if name not in _valid_ops:
                raise ValueError(
                    f"Unknown operation '{name}'. "
                    f"Valid operations: {', '.join(sorted(_valid_ops))}"
                )
        # Merge with existing: only replace fields that are provided
        from dataclasses import replace as _replace
        updates = {}
        for name, cfg in operation_configs.items():
            updates[name] = cfg
        self._operation_configs = _replace(self._operation_configs, **updates)

    @property
    def operation_configs(self) -> OperationConfigs:
        """Current per-operation LLM configurations (read-only, frozen)."""
        return self._operation_configs

    def configure_clients(
        self,
        _clients: OperationClients | None = None,
        /,
        **operation_clients: object,
    ) -> None:
        """Set per-operation LLM client overrides.

        Each operation can use a different LLM client (e.g. OpenAI for chat,
        Ollama for compression).  Operations without a per-operation client
        fall back to the tract-level default set via ``configure_llm()`` or
        ``Tract.open(api_key=...)``.

        Accepts either an OperationClients instance or keyword arguments.

        Args:
            _clients: OperationClients instance with typed fields.
            **operation_clients: Operation name -> client mappings.
                Valid names: ``"chat"``, ``"merge"``, ``"compress"``,
                ``"orchestrate"``.

        Raises:
            TypeError: If both positional and keyword arguments provided.
            ValueError: If an unknown operation name is given.

        Example::

            t.configure_clients(
                chat=openai_client,
                compress=ollama_client,
            )
        """
        if _clients is not None and operation_clients:
            raise TypeError(
                "Cannot mix positional OperationClients with keyword arguments"
            )
        if _clients is not None:
            if not isinstance(_clients, OperationClients):
                raise TypeError(
                    f"Expected OperationClients, got {type(_clients).__name__}"
                )
            self._operation_clients = _clients
            return
        _valid_ops = {"chat", "merge", "compress", "orchestrate"}
        for name in operation_clients:
            if name not in _valid_ops:
                raise ValueError(
                    f"Unknown operation '{name}'. "
                    f"Valid operations: {', '.join(sorted(_valid_ops))}"
                )
        from dataclasses import replace as _replace
        self._operation_clients = _replace(self._operation_clients, **operation_clients)

    @property
    def operation_clients(self) -> OperationClients:
        """Current per-operation LLM client overrides (read-only, frozen)."""
        return self._operation_clients

    def _resolve_llm_client(self, operation: str) -> object:
        """Resolve the LLM client for a given operation.

        Two-level lookup: per-operation client > tract-level default.

        Args:
            operation: Operation name (``"chat"``, ``"merge"``, etc.).

        Returns:
            The resolved LLM client.

        Raises:
            AttributeError: If no client is configured at any level.
        """
        client = getattr(self._operation_clients, operation, None)
        if client is not None:
            return client
        # Fall back to tract-level default (AttributeError if not set)
        return self._llm_client

    def _has_llm_client(self, operation: str | None = None) -> bool:
        """Check if an LLM client is available.

        Args:
            operation: If given, also checks per-operation client.

        Returns:
            True if a client is available at any level.
        """
        if operation is not None:
            op_client = getattr(self._operation_clients, operation, None)
            if op_client is not None:
                return True
        return hasattr(self, "_llm_client")

    def merge(
        self,
        source_branch: str,
        *,
        resolver: object | None = None,
        strategy: str = "auto",
        no_ff: bool = False,
        auto_commit: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        delete_branch: bool = False,
        message: str | None = None,
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
            temperature: Override temperature for the default resolver.
            max_tokens: Override max_tokens for the default resolver.
            llm_config: Full LLMConfig override for this call.
            delete_branch: If True, delete the source branch after merge.
            message: Optional merge commit message. If not provided, a
                default message is generated.

        Returns:
            :class:`MergeResult` describing the outcome.
        """
        from tract.models.merge import MergeResult
        from tract.operations.merge import merge_branches

        # Determine resolver
        effective_resolver = resolver
        if effective_resolver is None:
            effective_resolver = getattr(self, "_default_resolver", None)

        # Resolve per-operation config for merge
        merge_config = self._resolve_llm_config(
            "merge", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
        )

        # If using default resolver AND config differs from default, create tailored resolver
        if effective_resolver is getattr(self, "_default_resolver", None) and merge_config:
            if self._has_llm_client("merge"):
                from tract.llm.resolver import OpenAIResolver

                effective_resolver = OpenAIResolver(
                    self._resolve_llm_client("merge"),
                    model=merge_config.get("model"),
                    temperature=merge_config.get("temperature", 0.3),
                    max_tokens=merge_config.get("max_tokens", 2048),
                )

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
            return self.commit_merge(result, message=message)

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
        self._cache.clear()

        return result

    def commit_merge(
        self,
        result: MergeResult,
        *,
        message: str | None = None,
    ) -> MergeResult:
        """Finalize a conflict merge after reviewing/editing resolutions.

        Args:
            result: A MergeResult from a previous merge() call with conflicts.
            message: Optional merge commit message. If not provided, a
                default message is generated.

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
        merge_content = FreeformContent(
            payload={
                "message": f"Merged {result.source_branch} into {result.target_branch}",
                "resolutions": result.resolutions,
            }
        )

        # Determine parent hashes
        source_hash = result.source_tip_hash
        target_hash = result.target_tip_hash
        if source_hash is None or target_hash is None:
            # Fallback: resolve from ref_repo
            source_hash = self._ref_repo.get_branch(self._tract_id, result.source_branch)
            target_hash = self._ref_repo.get_head(self._tract_id)

        parent_hashes = [target_hash, source_hash] if target_hash and source_hash else []

        # Collect generation config from resolver resolutions
        gen_config = None
        if result.generation_configs:
            gen_config = next(iter(result.generation_configs.values()), None)

        default_msg = f"Merge branch '{result.source_branch}' into {result.target_branch}"
        merge_info = create_merge_commit(
            commit_engine=self._commit_engine,
            parent_repo=self._parent_repo,
            content=merge_content,
            parent_hashes=parent_hashes,
            message=message or default_msg,
            generation_config=gen_config,
        )

        self._session.commit()

        result.committed = True
        result.merge_commit_hash = merge_info.commit_hash

        # Clear compile cache
        self._cache.clear()

        return result

    def import_commit(
        self,
        commit_hash: str,
        *,
        resolver: object | None = None,
    ) -> ImportResult:
        """Import a commit onto the current branch (replaces cherry-pick).

        Creates a new commit with the same content but different hash and
        parentage (current HEAD as parent).

        Args:
            commit_hash: Hash (or prefix) of the commit to import.
            resolver: Optional resolver for handling issues (e.g., EDIT
                target missing on current branch).  Falls back to
                ``self._default_resolver`` if configured via
                :meth:`configure_llm`.

        Returns:
            :class:`ImportResult` describing the outcome.

        Raises:
            ImportCommitError: If issues detected and no resolver, or
                resolver aborts.
        """
        from tract.models.merge import ImportResult
        from tract.operations.rebase import import_commit as _import_commit

        # Resolve commit hash (supports prefixes and branch names)
        resolved = self.resolve_commit(commit_hash)

        # Determine resolver
        effective_resolver = resolver
        if effective_resolver is None:
            effective_resolver = getattr(self, "_default_resolver", None)

        result = _import_commit(
            commit_hash=resolved,
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            ref_repo=self._ref_repo,
            blob_repo=self._blob_repo,
            commit_engine=self._commit_engine,
            parent_repo=self._parent_repo,
            resolver=effective_resolver,
            event_repo=self._event_repo,
        )

        self._session.commit()

        # Clear compile cache (import changes HEAD)
        self._cache.clear()

        return result

    def rebase(
        self,
        target_branch: str,
        *,
        resolver: object | None = None,
    ) -> RebaseResult:
        """Rebase the current branch onto a target branch.

        Replays current branch's commits on top of the target branch tip,
        producing new commits with new hashes.

        Args:
            target_branch: Name of the branch to rebase onto.
            resolver: Optional resolver for semantic safety warnings.
                Falls back to ``self._default_resolver`` if configured
                via :meth:`configure_llm`.

        Returns:
            :class:`RebaseResult` describing the outcome.

        Raises:
            RebaseError: On merge commits in range, resolver abort, etc.
            SemanticSafetyError: If safety warnings and no resolver.
        """
        from tract.models.merge import RebaseResult
        from tract.operations.rebase import rebase as _rebase

        # Determine resolver
        effective_resolver = resolver
        if effective_resolver is None:
            effective_resolver = getattr(self, "_default_resolver", None)

        result = _rebase(
            tract_id=self._tract_id,
            target_branch=target_branch,
            commit_repo=self._commit_repo,
            ref_repo=self._ref_repo,
            parent_repo=self._parent_repo,
            blob_repo=self._blob_repo,
            commit_engine=self._commit_engine,
            resolver=effective_resolver,
            event_repo=self._event_repo,
        )

        self._session.commit()

        # Clear compile cache (rebase changes HEAD and commit hashes)
        self._cache.clear()

        return result

    def compress(
        self,
        *,
        commits: list[str] | None = None,
        from_commit: str | None = None,
        to_commit: str | None = None,
        target_tokens: int | None = None,
        preserve: list[str] | None = None,
        auto_commit: bool = True,
        content: str | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        max_retries: int = 3,
    ) -> CompressResult | PendingCompression:
        """Compress commit chains into summaries.

        Supports three modes:
        - **Manual** (``content`` provided): Uses your text as the summary.
        - **LLM** (``configure_llm()`` called): Uses LLM for summarization.
        - **Collaborative** (``auto_commit=False``): Returns PendingCompression
          for review; call ``.approve()`` or :meth:`approve_compression` to finalize.

        PINNED commits survive verbatim. SKIP commits are excluded.
        Original commits remain in DB (non-destructive).

        Args:
            commits: Explicit list of commit hashes to compress.
            from_commit: Start of range (inclusive).
            to_commit: End of range (inclusive).
            target_tokens: Target token count for summaries.
            preserve: Hashes to treat as temporarily PINNED.
            auto_commit: If True, commit immediately.
            content: Manual summary text (bypasses LLM).
            instructions: Additional LLM instructions.
            system_prompt: Custom system prompt for LLM.
            model: Override model for LLM summarization.
            temperature: Override temperature for LLM summarization.
            max_tokens: Override max_tokens for LLM summarization.
            llm_config: Full LLMConfig override for this call.
            validator: Optional callable that validates the summary text.
                Takes the summary text, returns (ok, diagnosis). When provided,
                each LLM summarization is wrapped with retry_with_steering.
            max_retries: Maximum retry attempts when validator is set (default 3).

        Returns:
            :class:`CompressResult` or :class:`PendingCompression`.

        Raises:
            DetachedHeadError: If HEAD is detached.
            CompressionError: On various error conditions.
            LLMConfigError: If explicit LLM params given without client.
            RetryExhaustedError: If all retry attempts fail validation.
        """
        from tract.models.compression import PendingCompression as _PendingCompression
        from tract.operations.compression import compress_range

        # Guard: detached HEAD blocks compression
        if self._ref_repo.is_detached(self._tract_id):
            raise DetachedHeadError()

        if self._event_repo is None:
            from tract.exceptions import CompressionError
            raise CompressionError("Compression repository not available")

        has_client = self._has_llm_client("compress")
        llm_client = self._resolve_llm_client("compress") if has_client else None

        # Guard: explicit LLM config without LLM client
        has_explicit_llm = (
            model is not None
            or temperature is not None
            or max_tokens is not None
            or llm_config is not None
        )
        if has_explicit_llm and llm_client is None and content is None:
            from tract.llm.errors import LLMConfigError
            raise LLMConfigError(
                "LLM parameters provided (model, temperature, max_tokens, or "
                "llm_config) but no LLM client is configured. Call "
                "configure_llm() or pass api_key to Tract.open(), or provide "
                "content= for manual compression."
            )

        # Resolve per-operation LLM config for compress
        llm_kwargs = self._resolve_llm_config(
            "compress", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
        ) if llm_client is not None else {}

        # Use savepoint for atomic rollback on partial failure
        nested = self._session.begin_nested()
        try:
            result = compress_range(
                tract_id=self._tract_id,
                commit_repo=self._commit_repo,
                blob_repo=self._blob_repo,
                annotation_repo=self._annotation_repo,
                ref_repo=self._ref_repo,
                commit_engine=self._commit_engine,
                token_counter=self._token_counter,
                event_repo=self._event_repo,
                parent_repo=self._parent_repo,
                commits=commits,
                from_commit=from_commit,
                to_commit=to_commit,
                target_tokens=target_tokens,
                preserve=preserve,
                auto_commit=auto_commit,
                llm_client=llm_client,
                llm_kwargs=llm_kwargs,
                generation_config=llm_kwargs if llm_kwargs else None,
                content=content,
                instructions=instructions,
                system_prompt=system_prompt,
                type_registry=self._custom_type_registry,
                validator=validator,
                max_retries=max_retries,
            )
        except Exception:
            nested.rollback()
            raise

        if isinstance(result, _PendingCompression):
            # No DB writes in collaborative path; rollback the unnecessary savepoint
            nested.rollback()
            # Set the commit function for later approval
            result._commit_fn = self._finalize_compression
            return result

        # Auto-commit mode: persist to DB
        self._session.commit()

        # Clear compile cache
        self._cache.clear()

        return result

    def _finalize_compression(
        self, pending: PendingCompression
    ) -> CompressResult:
        """Finalize a pending compression by creating commits.

        Called by PendingCompression.approve() via the _commit_fn closure,
        or directly by approve_compression().

        Args:
            pending: The PendingCompression to finalize.

        Returns:
            CompressResult with committed compression details.
        """
        from tract.operations.compression import _commit_compression

        if self._event_repo is None:
            from tract.exceptions import CompressionError
            raise CompressionError("Compression repository not available")

        # Use savepoint for atomic rollback on partial failure
        nested = self._session.begin_nested()
        try:
            result = _commit_compression(
                tract_id=self._tract_id,
                commit_repo=self._commit_repo,
                blob_repo=self._blob_repo,
                ref_repo=self._ref_repo,
                commit_engine=self._commit_engine,
                token_counter=self._token_counter,
                event_repo=self._event_repo,
                summaries=pending.summaries,
                range_commits=pending._range_commits,
                pinned_commits=pending._pinned_commits,
                normal_commits=pending._normal_commits,
                pinned_hashes=pending._pinned_hashes,
                skip_hashes=pending._skip_hashes,
                groups=pending._groups,
                original_tokens=pending.original_tokens,
                target_tokens=pending._target_tokens,
                instructions=pending._instructions,
                branch_name=pending._branch_name,
                type_registry=self._custom_type_registry,
                expected_head=pending._head_hash,
                generation_config=getattr(pending, '_generation_config', None),
            )
        except Exception:
            nested.rollback()
            raise

        self._session.commit()
        self._cache.clear()
        return result

    def approve_compression(
        self, pending: PendingCompression
    ) -> CompressResult:
        """Finalize a pending compression (alternative to pending.approve()).

        Args:
            pending: The PendingCompression to finalize.

        Returns:
            CompressResult with committed compression details.
        """
        return self._finalize_compression(pending)

    def gc(
        self,
        *,
        orphan_retention_days: int = 7,
        archive_retention_days: int | None = None,
        branch: str | None = None,
    ) -> GCResult:
        """Garbage-collect unreachable commits.

        Removes commits not reachable from any branch tip, subject to
        configurable retention policies.

        Args:
            orphan_retention_days: Days before orphaned commits become
                eligible for removal. Default 7.
            archive_retention_days: If set, days before archived (compression
                source) commits become eligible for removal. None (default)
                means archives are never removed.
            branch: If set, only check this branch for reachability.
                WARNING: commits reachable from other branches may be removed.

        Returns:
            :class:`GCResult` with removal counts and duration.

        Raises:
            CompressionError: If compression repository is not available.
        """
        from tract.exceptions import CompressionError
        from tract.models.compression import GCResult as _GCResult
        from tract.operations.compression import gc as _gc

        if self._event_repo is None:
            raise CompressionError("Compression repository not available")

        if self._parent_repo is None:
            raise CompressionError("Parent repository not available")

        result = _gc(
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            ref_repo=self._ref_repo,
            parent_repo=self._parent_repo,
            blob_repo=self._blob_repo,
            event_repo=self._event_repo,
            orphan_retention_days=orphan_retention_days,
            archive_retention_days=archive_retention_days,
            branch=branch,
        )

        self._cache.clear()
        self._session.commit()
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
        snapshot = self._cache.get(target_hash)
        if snapshot is None:
            self.compile()
            snapshot = self._cache.get(target_hash)

        # Update snapshot with API-reported counts.
        # Use prompt + completion because the compiled context at HEAD
        # includes the assistant response (committed before record_usage).
        if snapshot is not None:
            context_tokens = usage.prompt_tokens + usage.completion_tokens
            token_source = f"api:{usage.prompt_tokens}+{usage.completion_tokens}"
            updated = replace(snapshot, token_count=context_tokens, token_source=token_source)
            self._cache.put(target_hash, updated)
            # Persist override so it survives cache eviction
            self._cache.store_api_override(target_hash, context_tokens, token_source)
            return self._cache.to_compiled(updated)

        # Fallback (custom compiler, no snapshot): return minimal result
        context_tokens = usage.prompt_tokens + usage.completion_tokens
        token_source = f"api:{usage.prompt_tokens}+{usage.completion_tokens}"
        return CompiledContext(
            messages=[],
            token_count=context_tokens,
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
        # Invalidate compile cache on batch entry and set _in_batch flag
        # so commit() skips cache updates for intermediate states.
        self._cache.clear()
        self._in_batch = True

        # Stash the real session.commit and replace with a no-op so that
        # individual commit() calls inside the batch don't flush to the database.
        # NOTE: We intentionally monkey-patch instead of using
        # session.begin_nested() (SAVEPOINT) because SAVEPOINTs still flush
        # intermediate state, while we want the entire batch committed atomically.
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
            self._in_batch = False
            self._session.commit = _real_commit  # type: ignore[assignment]


    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def configure_policies(
        self,
        policies: list | None = None,
        *,
        on_proposal: Callable[[PolicyProposal], None] | None = None,
        cooldown_seconds: float = 0,
    ) -> None:
        """Configure the policy evaluator.

        Creates a PolicyEvaluator with the given policies, enabling
        automatic policy evaluation on compile() and commit() calls.

        Args:
            policies: List of Policy instances to register.
            on_proposal: Callback invoked when a collaborative policy
                creates a proposal.
            cooldown_seconds: Minimum seconds between re-evaluations
                of the same policy. Default 0 (no cooldown).
        """
        from tract.policy.evaluator import PolicyEvaluator as _PolicyEvaluator

        self._policy_evaluator = _PolicyEvaluator(
            tract=self,
            policies=policies,
            policy_repo=self._policy_repo,
            on_proposal=on_proposal,
            cooldown_seconds=cooldown_seconds,
        )

    def register_policy(self, policy: object) -> None:
        """Register a single policy with the evaluator.

        If no evaluator exists, one is created automatically.

        Args:
            policy: A Policy instance to register.
        """
        if self._policy_evaluator is None:
            self.configure_policies()
        self._policy_evaluator.register(policy)  # type: ignore[union-attr]

    def unregister_policy(self, policy_name: str) -> None:
        """Remove a policy by name.

        No-op if no evaluator exists.

        Args:
            policy_name: The name of the policy to remove.
        """
        if self._policy_evaluator is not None:
            self._policy_evaluator.unregister(policy_name)

    def pause_all_policies(self) -> None:
        """Pause all policy evaluation (emergency kill switch).

        No-op if no evaluator exists.
        """
        if self._policy_evaluator is not None:
            self._policy_evaluator.pause()

    def resume_all_policies(self) -> None:
        """Resume all policy evaluation.

        No-op if no evaluator exists.
        """
        if self._policy_evaluator is not None:
            self._policy_evaluator.resume()

    def get_pending_proposals(self) -> list[PolicyProposal]:
        """Get all pending policy proposals.

        Returns:
            List of PolicyProposal objects with status="pending".
            Empty list if no evaluator configured.
        """
        if self._policy_evaluator is None:
            return []
        return self._policy_evaluator.get_pending_proposals()

    def approve_proposal(self, proposal_id: str) -> object:
        """Approve and execute a pending policy proposal.

        Args:
            proposal_id: The ID of the proposal to approve.

        Returns:
            Result of executing the action.

        Raises:
            PolicyExecutionError: If no evaluator or proposal not found.
        """
        if self._policy_evaluator is None:
            from tract.exceptions import PolicyExecutionError

            raise PolicyExecutionError("No policy evaluator configured")
        return self._policy_evaluator.approve_proposal(proposal_id)

    def reject_proposal(self, proposal_id: str, reason: str = "") -> None:
        """Reject a pending policy proposal.

        Args:
            proposal_id: The ID of the proposal to reject.
            reason: Optional reason for rejection.

        Raises:
            PolicyExecutionError: If no evaluator or proposal not found.
        """
        if self._policy_evaluator is None:
            from tract.exceptions import PolicyExecutionError

            raise PolicyExecutionError("No policy evaluator configured")
        self._policy_evaluator.reject_proposal(proposal_id, reason)

    def save_policy_config(self, config_data: dict) -> None:
        """Persist policy configuration to _trace_meta.

        Args:
            config_data: Dictionary of policy configuration to persist.
        """
        import json as _json

        from sqlalchemy import select as _select

        from tract.storage.schema import TraceMetaRow

        stmt = _select(TraceMetaRow).where(TraceMetaRow.key == "policy_config")
        existing = self._session.execute(stmt).scalar_one_or_none()
        if existing is None:
            self._session.add(
                TraceMetaRow(key="policy_config", value=_json.dumps(config_data))
            )
        else:
            existing.value = _json.dumps(config_data)
        self._session.commit()

    def load_policy_config(self) -> dict | None:
        """Load policy configuration from _trace_meta.

        Returns:
            Dictionary of policy configuration, or None if not set.
        """
        import json as _json

        from sqlalchemy import select as _select

        from tract.storage.schema import TraceMetaRow

        stmt = _select(TraceMetaRow).where(TraceMetaRow.key == "policy_config")
        row = self._session.execute(stmt).scalar_one_or_none()
        if row is None:
            return None
        return _json.loads(row.value)

    def register_content_type(self, name: str, model: type[BaseModel]) -> None:
        """Register a custom content type for this tract instance.

        Args:
            name: The ``content_type`` discriminator value.
            model: A Pydantic ``BaseModel`` subclass.
        """
        self._custom_type_registry[name] = model

    # ------------------------------------------------------------------
    # Agent toolkit
    # ------------------------------------------------------------------

    def as_tools(
        self,
        *,
        profile: str | object = "self",
        overrides: dict[str, str] | None = None,
        format: str = "openai",
    ) -> list[dict]:
        """Get tool definitions for this tract in LLM-consumable format.

        Combines tool definitions, profile filtering, optional description
        overrides, and format conversion in one call.

        Args:
            profile: A profile name (``"self"``, ``"supervisor"``, ``"full"``)
                or a :class:`~tract.toolkit.models.ToolProfile` instance.
            overrides: Optional dict mapping tool names to replacement
                descriptions.  Applied on top of the profile's descriptions.
            format: Output format -- ``"openai"`` (default) or ``"anthropic"``.

        Returns:
            List of tool definition dicts in the requested format.
        """
        from tract.toolkit.definitions import get_all_tools
        from tract.toolkit.models import ToolProfile as _ToolProfile
        from tract.toolkit.profiles import get_profile

        all_tools = get_all_tools(self)

        # Resolve profile
        if isinstance(profile, str):
            resolved_profile = get_profile(profile)
        elif isinstance(profile, _ToolProfile):
            resolved_profile = profile
        else:
            raise TypeError(
                f"profile must be a string or ToolProfile, got {type(profile).__name__}"
            )

        # Apply profile filtering
        filtered = resolved_profile.filter_tools(all_tools)

        # Apply overrides
        if overrides:
            from dataclasses import replace as _replace

            new_filtered = []
            for tool in filtered:
                if tool.name in overrides:
                    tool = _replace(tool, description=overrides[tool.name])
                new_filtered.append(tool)
            filtered = new_filtered

        # Convert to requested format
        if format == "openai":
            return [tool.to_openai() for tool in filtered]
        elif format == "anthropic":
            return [tool.to_anthropic() for tool in filtered]
        else:
            raise ValueError(
                f"Unknown format '{format}'. Supported: 'openai', 'anthropic'."
            )

    # ------------------------------------------------------------------
    # Orchestrator facade
    # ------------------------------------------------------------------

    def _set_orchestrating(self, flag: bool) -> None:
        """Set the orchestrating recursion guard flag.

        Called by the Orchestrator to prevent policy evaluation from
        re-triggering orchestrator runs.

        Args:
            flag: True when orchestrating, False when done.
        """
        self._orchestrating = flag

    def _check_orchestrator_triggers(self, trigger: str) -> None:
        """Check if orchestrator triggers should fire.

        Called from compile() and commit() after policy evaluation,
        guarded by ``not self._orchestrating and not self._in_batch``.

        Args:
            trigger: The trigger type ("compile" or "commit").
        """
        if self._orchestrator is None:
            return
        # Lazy import to avoid circular dependency
        from tract.orchestrator.loop import Orchestrator as _Orchestrator

        if not isinstance(self._orchestrator, _Orchestrator):
            return

        orch: _Orchestrator = self._orchestrator  # type: ignore[assignment]
        triggers = orch._config.triggers
        if triggers is None:
            return

        try:
            _trig_autonomy = triggers.autonomy

            if trigger == "compile" and triggers.on_compile:
                self.orchestrate(trigger_autonomy=_trig_autonomy)
                return  # Only fire once per trigger check

            if trigger == "commit":
                fired = False

                if triggers.on_commit_count is not None:
                    self._trigger_commit_count += 1
                    if self._trigger_commit_count >= triggers.on_commit_count:
                        self._trigger_commit_count = 0
                        self.orchestrate(trigger_autonomy=_trig_autonomy)
                        fired = True

                if not fired and triggers.on_token_threshold is not None:
                    status = self.status()
                    if (
                        status.token_budget_max
                        and status.token_budget_max > 0
                    ):
                        pct = status.token_count / status.token_budget_max
                        if pct >= triggers.on_token_threshold:
                            if not self._token_trigger_fired:
                                self._token_trigger_fired = True
                                self.orchestrate(
                                    trigger_autonomy=_trig_autonomy
                                )
                        else:
                            # Reset cooldown when usage drops below threshold
                            self._token_trigger_fired = False
        except Exception:
            # Trigger errors must not break commit/compile
            import logging

            logging.getLogger(__name__).debug(
                "Orchestrator trigger error", exc_info=True
            )

    def configure_orchestrator(
        self,
        config: OrchestratorConfig | None = None,
        llm_callable: Callable | None = None,
    ) -> None:
        """Configure the orchestrator for this tract.

        Creates an Orchestrator instance and stores it for later use
        by :meth:`orchestrate` and trigger checks.

        Args:
            config: An OrchestratorConfig instance, or None for defaults.
            llm_callable: Optional callable for LLM calls. If not
                provided and no tract LLM client is configured, a
                warning is logged.
        """
        from tract.orchestrator import Orchestrator as _Orchestrator
        from tract.orchestrator import OrchestratorConfig as _OrchestratorConfig

        if config is None:
            config = _OrchestratorConfig()

        self._orchestrator = _Orchestrator(
            self, config=config, llm_callable=llm_callable
        )

        if llm_callable is None and not self._has_llm_client("orchestrate"):
            import logging

            logging.getLogger(__name__).warning(
                "Orchestrator configured without LLM. "
                "Call configure_llm() or provide llm_callable."
            )

    def orchestrate(
        self,
        *,
        config: OrchestratorConfig | None = None,
        llm_callable: Callable | None = None,
        trigger_autonomy: AutonomyLevel | None = None,
    ) -> OrchestratorResult:
        """Run the orchestrator agent loop.

        Convenience method that creates or reuses an Orchestrator
        and runs it.

        Args:
            config: Optional OrchestratorConfig override.
            llm_callable: Optional LLM callable override.
            trigger_autonomy: Optional autonomy override from a trigger.
                When set, effective autonomy is min(ceiling, trigger_autonomy).

        Returns:
            OrchestratorResult from the orchestrator run.
        """
        from tract.orchestrator import Orchestrator as _Orchestrator

        # Step 1: Resolve per-operation config BEFORE the three-way branch
        orch_resolved = self._resolve_llm_config("orchestrate")

        # Step 2: If operation-level config found, apply to config (mutation-safe)
        if orch_resolved:
            if config is not None:
                # Caller provided config -- only fill in None/default fields
                # Use dataclasses.replace() to avoid mutating the caller's object
                overrides: dict = {}
                if config.model is None and "model" in orch_resolved:
                    overrides["model"] = orch_resolved["model"]
                if config.temperature == 0.0 and "temperature" in orch_resolved:
                    overrides["temperature"] = orch_resolved["temperature"]
                if config.max_tokens is None and "max_tokens" in orch_resolved:
                    overrides["max_tokens"] = orch_resolved["max_tokens"]
                # Collect remaining resolved fields into extra_llm_kwargs
                _orch_known = {"model", "temperature", "max_tokens"}
                extra = {k: v for k, v in orch_resolved.items() if k not in _orch_known}
                if extra and config.extra_llm_kwargs is None:
                    overrides["extra_llm_kwargs"] = extra
                if overrides:
                    config = replace(config, **overrides)
            else:
                # No caller config -- create one from operation defaults
                from tract.orchestrator.config import OrchestratorConfig as _OrchestratorConfig
                _orch_known = {"model", "temperature", "max_tokens"}
                extra = {k: v for k, v in orch_resolved.items() if k not in _orch_known}
                config = _OrchestratorConfig(
                    model=orch_resolved.get("model"),
                    temperature=orch_resolved.get("temperature", 0.0),
                    max_tokens=orch_resolved.get("max_tokens"),
                    extra_llm_kwargs=extra if extra else None,
                )

        # Step 3: Three-way branch (now with resolved config)
        # If overrides provided, create a new orchestrator
        if config is not None or llm_callable is not None:
            orch = _Orchestrator(
                self, config=config, llm_callable=llm_callable
            )
            return orch.run(trigger_autonomy=trigger_autonomy)

        # If existing orchestrator, reuse it
        if self._orchestrator is not None:
            orch_inst: _Orchestrator = self._orchestrator  # type: ignore[assignment]
            orch_inst.reset()
            return orch_inst.run(trigger_autonomy=trigger_autonomy)

        # Create one with defaults
        orch = _Orchestrator(self)
        return orch.run(trigger_autonomy=trigger_autonomy)

    def stop_orchestrator(self) -> None:
        """Stop the running orchestrator immediately.

        No-op if no orchestrator is configured.
        """
        if self._orchestrator is not None:
            from tract.orchestrator.loop import Orchestrator as _Orchestrator

            if isinstance(self._orchestrator, _Orchestrator):
                self._orchestrator.stop()  # type: ignore[union-attr]

    def pause_orchestrator(self) -> None:
        """Pause the running orchestrator gracefully.

        No-op if no orchestrator is configured.
        """
        if self._orchestrator is not None:
            from tract.orchestrator.loop import Orchestrator as _Orchestrator

            if isinstance(self._orchestrator, _Orchestrator):
                self._orchestrator.pause()  # type: ignore[union-attr]

    def close(self) -> None:
        """Close the session and dispose the engine."""
        if self._closed:
            return
        self._closed = True
        # Close internally-created LLM client (not externally-provided ones)
        if self._owns_llm_client and hasattr(self, "_llm_client"):
            self._llm_client.close()
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
