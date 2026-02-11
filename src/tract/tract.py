"""Tract -- the public SDK entry point for Trace.

Ties together storage, commit engine, and context compiler into a clean,
user-facing API.  Users interact with ``Tract.open()``, ``t.commit()``,
``t.compile()``, etc.

Not thread-safe in v1.  Each thread should open its own ``Tract``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
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
from tract.exceptions import ContentValidationError, TraceError
from tract.protocols import CompiledContext, CompileSnapshot, ContextCompiler, Message, TokenCounter, TokenUsage
from tract.storage.engine import create_session_factory, create_trace_engine, init_db
from tract.storage.sqlite import (
    SqliteAnnotationRepository,
    SqliteBlobRepository,
    SqliteCommitRepository,
    SqliteRefRepository,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy import Engine
    from sqlalchemy.orm import Session


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
        self._custom_type_registry: dict[str, type[BaseModel]] = {}
        self._compile_snapshot: CompileSnapshot | None = None
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
    ) -> Tract:
        """Open (or create) a Trace repository.

        Args:
            path: SQLite path.  ``":memory:"`` for in-memory (default).
            tract_id: Unique tract identifier.  Generated if not provided.
            config: Tract configuration.  Defaults created if *None*.
            tokenizer: Pluggable token counter.  TiktokenCounter by default.
            compiler: Pluggable context compiler.  DefaultContextCompiler by default.

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
        )

        # Context compiler
        ctx_compiler: ContextCompiler = compiler or DefaultContextCompiler(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
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
        # Auto-validate dicts through the content type system
        if isinstance(content, dict):
            content = validate_content(content, custom_registry=self._custom_type_registry)

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

        # Update compile snapshot: incremental extend for APPEND with
        # DefaultContextCompiler, otherwise invalidate
        if (
            operation == CommitOperation.APPEND
            and self._compile_snapshot is not None
            and isinstance(self._compiler, DefaultContextCompiler)
        ):
            self._extend_snapshot_for_append(info)
        else:
            self._compile_snapshot = None

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

        # Cache hit: snapshot exists for current head
        if self._compile_snapshot is not None and self._compile_snapshot.head_hash == current_head:
            return self._snapshot_to_compiled(self._compile_snapshot)

        # Cache miss: full compile and build snapshot
        result = self._compiler.compile(self._tract_id, current_head)
        self._compile_snapshot = self._build_snapshot_from_compiled(current_head, result)
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
        self._compile_snapshot = None
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

    def log(self, limit: int = 10) -> list[CommitInfo]:
        """Walk commit history from HEAD backward.

        Args:
            limit: Maximum number of commits to return.

        Returns:
            List of :class:`CommitInfo` in reverse chronological order
            (newest first).  Empty list if no commits.
        """
        current_head = self.head
        if current_head is None:
            return []

        ancestors = self._commit_repo.get_ancestors(current_head, limit=limit)
        return [self._commit_engine._row_to_info(row) for row in ancestors]

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
        if self._compile_snapshot is None or self._compile_snapshot.head_hash != target_hash:
            self.compile()

        # Update snapshot with API-reported counts
        if self._compile_snapshot is not None and self._compile_snapshot.head_hash == target_hash:
            token_source = f"api:{usage.prompt_tokens}+{usage.completion_tokens}"
            self._compile_snapshot = CompileSnapshot(
                head_hash=self._compile_snapshot.head_hash,
                raw_messages=self._compile_snapshot.raw_messages,
                aggregated_messages=self._compile_snapshot.aggregated_messages,
                effective_hashes=self._compile_snapshot.effective_hashes,
                commit_count=self._compile_snapshot.commit_count,
                token_count=usage.prompt_tokens,
                token_source=token_source,
                generation_configs=self._compile_snapshot.generation_configs,
            )
            return self._snapshot_to_compiled(self._compile_snapshot)

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
        # Invalidate compile snapshot on batch entry
        self._compile_snapshot = None

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
    # Incremental compile helpers
    # ------------------------------------------------------------------

    def _snapshot_to_compiled(self, snapshot: CompileSnapshot) -> CompiledContext:
        """Convert a CompileSnapshot to a CompiledContext for return.

        Uses copy-on-output for generation_configs to prevent user mutations
        of the returned CompiledContext from corrupting the cached snapshot.
        """
        return CompiledContext(
            messages=list(snapshot.aggregated_messages),
            token_count=snapshot.token_count,
            commit_count=snapshot.commit_count,
            token_source=snapshot.token_source,
            generation_configs=[dict(c) for c in snapshot.generation_configs],
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
            raw_messages=tuple(result.messages),
            aggregated_messages=tuple(result.messages),
            effective_hashes=frozenset(),
            commit_count=result.commit_count,
            token_count=result.token_count,
            token_source=result.token_source,
            generation_configs=tuple(dict(c) for c in result.generation_configs),
        )

    def _tiktoken_source(self) -> str:
        """Return the token_source string for tiktoken-based counts."""
        if isinstance(self._token_counter, TiktokenCounter):
            return f"tiktoken:{self._token_counter._encoding_name}"
        return ""

    def _extend_snapshot_for_append(self, commit_info: CommitInfo) -> None:
        """Incrementally extend the cached snapshot for an APPEND commit.

        Builds the message for the new commit, applies tail aggregation
        if the new message has the same role as the last aggregated message,
        and recounts tokens.
        """
        snapshot = self._compile_snapshot
        if snapshot is None:
            return

        commit_row = self._commit_repo.get(commit_info.commit_hash)
        if commit_row is None:
            self._compile_snapshot = None
            return

        assert isinstance(self._compiler, DefaultContextCompiler)
        new_message = self._compiler.build_message_for_commit(commit_row)
        new_config = commit_row.generation_config_json or {}

        new_raw = snapshot.raw_messages + (new_message,)

        # Tail aggregation: merge if same role as last aggregated message
        if (
            snapshot.aggregated_messages
            and new_message.role == snapshot.aggregated_messages[-1].role
        ):
            last = snapshot.aggregated_messages[-1]
            merged = Message(
                role=last.role,
                content=last.content + "\n\n" + new_message.content,
                name=last.name,
            )
            new_aggregated = snapshot.aggregated_messages[:-1] + (merged,)
        else:
            new_aggregated = snapshot.aggregated_messages + (new_message,)

        # Recount tokens on the aggregated messages
        messages_dicts = [
            {"role": m.role, "content": m.content}
            if m.name is None
            else {"role": m.role, "content": m.content, "name": m.name}
            for m in new_aggregated
        ]
        new_token_count = self._token_counter.count_messages(messages_dicts)

        self._compile_snapshot = CompileSnapshot(
            head_hash=commit_info.commit_hash,
            raw_messages=new_raw,
            aggregated_messages=new_aggregated,
            effective_hashes=snapshot.effective_hashes | {commit_info.commit_hash},
            commit_count=snapshot.commit_count + 1,
            token_count=new_token_count,
            token_source=self._tiktoken_source(),
            generation_configs=snapshot.generation_configs + (new_config,),
        )

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
