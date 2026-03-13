"""Tract -- the public SDK entry point for Trace.

Ties together storage, commit engine, and context compiler into a clean,
user-facing API.  Users interact with ``Tract.open()``, ``t.commit()``,
``t.compile()``, etc.

Not thread-safe in v1.  Each thread should open its own ``Tract``.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import fields as dc_fields, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, overload

from pydantic import BaseModel

from tract.engine.cache import CacheManager
from tract.engine.commit import CommitEngine
from tract.engine.compiler import DefaultContextCompiler
from tract.engine.tokens import TiktokenCounter
from tract.models.annotations import DEFAULT_TYPE_PRIORITIES, Priority, PriorityAnnotation, RetentionCriteria
from tract.models.commit import CommitInfo, CommitMetadata, CommitOperation
from tract.models.config import LLMConfig, Operator, OperationClients, OperationConfigs, OperationPrompts, RetryConfig, TractConfig
from tract.models.content import validate_content
from tract.exceptions import (
    BlockedError,
    BranchNotFoundError,
    ClosedError,
    CommitNotFoundError,
    ContentValidationError,
    DetachedHeadError,
    TagNotRegisteredError,
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
    SqlitePersistenceRepository,
    SqliteRefRepository,
    SqliteSpawnPointerRepository,
    SqliteTagAnnotationRepository,
    SqliteTagRegistryRepository,
    SqliteToolSchemaRepository,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Any

    from sqlalchemy import Engine
    from sqlalchemy.orm import Session

    from tract.llm.protocols import LLMClient, ResolverCallable
    from tract.loop import LoopResult
    from tract.middleware import MiddlewareEvent
    from tract.toolkit.models import ToolProfile
    from tract.models.branch import BranchInfo
    from tract.models.compression import CompressResult, GCResult, ReorderWarning, ToolCompactResult, ToolDropResult
    from tract.models.merge import ImportResult, MergeResult, MergeStrategy, RebaseResult
    from tract.models.session import SpawnInfo
    from tract.operations.diff import DiffResult
    from tract.operations.health import HealthReport
    from tract.operations.history import StatusInfo
    from tract.operations.config_index import ConfigIndex
    from tract.toolkit.executor import ToolExecutor
    from tract.toolkit.models import ToolName
    from tract.toolkit.profiles import ProfileName
    from tract.protocols import ToolTurn
    from tract.models.config import ToolSummarizationConfig

logger = logging.getLogger(__name__)


def _resolve_text(
    text: str | None = None,
    path: str | Path | None = None,
    *,
    label: str = "text",
    prompt_dir: str | Path | None = None,
) -> str:
    """Return *text* directly or read it from *path*.

    Exactly one of *text* or *path* must be provided.

    Args:
        text: Inline text content.
        path: Path to a file whose contents will be read (UTF-8).
        label: Name used in error messages (e.g. ``"text"``, ``"directive"``).
        prompt_dir: Base directory for resolving relative *path* values.
            When set and *path* is relative, ``prompt_dir / path`` is tried
            first.  Falls back to *path* as-is if the resolved file does
            not exist.

    Raises:
        ValueError: If both or neither argument is supplied, or file is missing.
    """
    from pathlib import Path as _Path

    if text is not None and path is not None:
        raise ValueError(f"Pass either {label} or path=, not both.")
    if text is None and path is None:
        raise ValueError(f"Either {label} or path= is required.")
    if path is not None:
        p = _Path(path)
        # Resolve relative paths against prompt_dir when available
        if not p.is_absolute() and prompt_dir is not None:
            candidate = _Path(prompt_dir) / p
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        if not p.is_file():
            raise ValueError(f"File not found: {p}")
        return p.read_text(encoding="utf-8")
    return text  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Compile strategy type
# ---------------------------------------------------------------------------
CompileStrategy = Literal["full", "messages", "adaptive"]

# ---------------------------------------------------------------------------
# Valid operation names for configure_operations / configure_clients
# ---------------------------------------------------------------------------
_VALID_OPERATION_NAMES: frozenset[str] = frozenset({"chat", "merge", "compress", "message"})
_VALID_PROMPT_NAMES: frozenset[str] = frozenset({"compress", "merge", "message", "commit_message"})


# ------------------------------------------------------------------
# Auto-message generation helper
# ------------------------------------------------------------------


_MAX_AUTO_MSG_LEN = 500


def _fallback_message(content_type: str, text: str) -> str:
    """Generate a fallback commit message by truncating content text.

    Used when no LLM is available for auto-summarization, or when
    auto_message is disabled.

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


def _detect_provider(
    base_url: str | None, model: str | None,
) -> Literal["openai", "anthropic"]:
    """Auto-detect LLM provider from base_url or model name."""
    if base_url and "anthropic" in base_url.lower():
        return "anthropic"
    if model and model.startswith("claude"):
        return "anthropic"
    return "openai"


# ------------------------------------------------------------------
# Retry helpers (exponential backoff for LLM calls)
# ------------------------------------------------------------------


def _calculate_backoff(attempt: int, retry_config: RetryConfig) -> float:
    """Calculate backoff delay for a given retry attempt.

    Args:
        attempt: Zero-based attempt index (0 = first retry).
        retry_config: Retry configuration with delay/factor/jitter settings.

    Returns:
        Delay in seconds (with optional jitter applied).
    """
    import random

    delay = min(
        retry_config.initial_delay * (retry_config.backoff_factor ** attempt),
        retry_config.max_delay,
    )
    if retry_config.jitter:
        delay *= 0.5 + random.random()
    return delay


def _is_retryable(e: Exception, retry_config: RetryConfig) -> bool:
    """Check whether *e* should be retried per *retry_config*.

    :class:`ContentValidationError` and :class:`BlockedError` are never
    retried.  When ``retry_config.retryable_errors`` is non-empty, only
    those exception types are retried.
    """
    if isinstance(e, (ContentValidationError, BlockedError)):
        return False
    if retry_config.retryable_errors and not isinstance(
        e, retry_config.retryable_errors
    ):
        return False
    return True


def _retry_with_backoff(
    func: Callable,
    retry_config: RetryConfig | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute *func* with exponential backoff on transient failure.

    Validation errors (:class:`ContentValidationError`,
    :class:`BlockedError`) are never retried.  When
    ``retry_config.retryable_errors`` is non-empty, only those exception
    types are retried; all others propagate immediately.
    """
    import time

    if not retry_config or retry_config.max_retries <= 0:
        return func(*args, **kwargs)

    last_error: Exception | None = None
    for attempt in range(retry_config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _is_retryable(e, retry_config):
                raise
            last_error = e
            if attempt < retry_config.max_retries:
                delay = _calculate_backoff(attempt, retry_config)
                logger.debug(
                    "Retry attempt %d/%d after %s: %s (delay=%.2fs)",
                    attempt + 1, retry_config.max_retries,
                    type(e).__name__, e, delay,
                )
                time.sleep(delay)
    raise last_error  # type: ignore[misc]  # guaranteed non-None after loop


async def _aretry_with_backoff(
    func: Callable,
    retry_config: RetryConfig | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Async version of :func:`_retry_with_backoff`."""
    import asyncio

    if not retry_config or retry_config.max_retries <= 0:
        return await func(*args, **kwargs)

    last_error: Exception | None = None
    for attempt in range(retry_config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if not _is_retryable(e, retry_config):
                raise
            last_error = e
            if attempt < retry_config.max_retries:
                delay = _calculate_backoff(attempt, retry_config)
                logger.debug(
                    "Async retry attempt %d/%d after %s: %s (delay=%.2fs)",
                    attempt + 1, retry_config.max_retries,
                    type(e).__name__, e, delay,
                )
                await asyncio.sleep(delay)
    raise last_error  # type: ignore[misc]  # guaranteed non-None after loop


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
        tool_schema_repo: SqliteToolSchemaRepository | None = None,
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
        self._tag_annotation_repo: SqliteTagAnnotationRepository | None = None
        self._tag_registry_repo: SqliteTagRegistryRepository | None = None
        self._strict_tags: bool = True
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
        self._creating_thread = threading.current_thread().ident
        self._owns_llm_client: bool = False
        self._llm_client: LLMClient | None = None  # type: ignore[assignment]
        self._default_config: LLMConfig | None = None
        self._operation_configs: OperationConfigs = OperationConfigs()
        self._operation_prompts: OperationPrompts = OperationPrompts()
        self._operation_clients: OperationClients = OperationClients()
        self._active_tools: list[dict] | None = None
        self._tool_executor: ToolExecutor | None = None  # type: ignore[assignment]  # lazy runner import
        self._tool_summarization_config: ToolSummarizationConfig | None = None
        self._commit_reasoning: bool = True
        self._auto_message_enabled: bool = False
        self._retry_config: RetryConfig | None = None

        # Tool defaults (set via open() or set_tool_profile/set_tool_result_format)
        self._tool_profile: str | ToolProfile | None = None
        self._tool_result_format: Literal["minimal", "json", "verbose"] = "minimal"

        # Config index (per-key resolution from DAG ancestry)
        self._config_index: ConfigIndex | None = None

        # Middleware state
        self._middleware: dict[str, list[tuple[str, Callable]]] = {}
        self._in_middleware_events: set[str] = set()

        # Persistence state
        self._db_path: str = ":memory:"
        self._persistence_repo: SqlitePersistenceRepository | None = None
        self._quarantined: list[str] = []

        # Workflow profile state
        self._active_profile: object | None = None  # WorkflowProfile when loaded

        # Custom tools registered via @t.tool decorator
        self._custom_tools: dict[str, Any] = {}  # name -> ToolDefinition

        # Prompt file directory (auto-discovered or explicit)
        self._prompt_dir: str | Path | None = None

    def _check_open(self) -> None:
        """Raise :class:`ClosedError` if closed, or :class:`ThreadSafetyError` if wrong thread."""
        if self._closed:
            raise ClosedError()
        current = threading.current_thread().ident
        if current != self._creating_thread:
            from tract.exceptions import ThreadSafetyError
            creating_name = str(self._creating_thread)
            current_name = str(current)
            raise ThreadSafetyError(creating_name, current_name)

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
        llm_client: LLMClient | None = None,
        default_config: LLMConfig | None = None,
        operations: OperationConfigs | None = None,
        operation_configs: dict[str, LLMConfig] | None = None,  # deprecated: use operations=
        tokenizer_encoding: str | None = None,
        commit_reasoning: bool = True,
        auto_message: bool | str | LLMConfig = False,
        provider: Literal["openai", "anthropic"] | None = None,
        tool_profile: str | ToolProfile | None = None,
        tool_result_format: Literal["minimal", "json", "verbose"] | None = None,
        retry: RetryConfig | None = None,
        prompt_dir: str | Path | None = None,
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
            auto_message: Enable LLM-generated commit messages.  Accepts:
                - ``False`` (default): truncated content preview.
                - ``True``: use the tract-level default LLM client/model.
                - ``"model-name"``: use a specific model (e.g. ``"llama3.1-8b"``).
                - :class:`LLMConfig`: full control over message generation config.
            provider: LLM provider to use when *api_key* is provided.
                ``"openai"`` (default) or ``"anthropic"``.  When ``None``,
                auto-detects from *base_url* or *model* name: base URLs
                containing ``"anthropic"`` or models starting with ``"claude"``
                select the Anthropic client.
            tool_profile: Default tool profile for :meth:`run`, :meth:`as_tools`,
                and :meth:`as_callable_tools`.  A profile name (``"compact"``,
                ``"self"``, ``"supervisor"``, ``"full"``) or a
                :class:`~tract.toolkit.models.ToolProfile` instance.  When
                ``None`` (default), methods use ``"compact"``.  Set once here
                to avoid passing ``profile=`` on every call.
            tool_result_format: How tool results render in compiled context.
                ``"minimal"`` (default): compact single-line output.
                ``"json"``: compact JSON (no indentation).
                ``"verbose"``: full JSON with ``indent=2`` (legacy behavior).
            retry: Default retry configuration for LLM calls.  When
                provided, transient errors (network, rate limit, server)
                are retried with exponential backoff.  Per-call overrides
                are accepted by :meth:`generate`, :meth:`chat`, and their
                async counterparts.
            prompt_dir: Base directory for resolving relative ``path=``
                arguments in :meth:`system`, :meth:`user`, :meth:`directive`,
                and :meth:`assistant`.  When ``None`` (default), the
                ``.tract/prompts/`` directory is used if it exists.

        Returns:
            A ready-to-use ``Tract`` instance.

        Raises:
            ValueError: If mutually exclusive params are combined
                (e.g. *url* + *engine*, *model* + *default_config*).
        """
        # Auto-discover .tract/tract.db when no explicit path/url/engine
        import os as _os
        if (
            path == ":memory:"
            and url is None
            and engine is None
            and not _os.environ.get("TRACT_NO_AUTO_DISCOVER")
        ):
            from pathlib import Path as _Path
            auto_db = _Path(".tract") / "tract.db"
            if auto_db.is_file():
                path = str(auto_db)

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

        # Tag repositories
        tag_annotation_repo = SqliteTagAnnotationRepository(session)
        tag_registry_repo = SqliteTagRegistryRepository(session)

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
        tract._tag_annotation_repo = tag_annotation_repo
        tract._tag_registry_repo = tag_registry_repo

        # Seed base tags (idempotent)
        tract._seed_base_tags()

        # Persistence repository + file-based state
        persistence_repo = SqlitePersistenceRepository(session)
        tract._persistence_repo = persistence_repo
        tract._db_path = path
        tract._load_persisted_state()

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
            detected = provider or _detect_provider(base_url, model)
            if detected == "anthropic":
                from tract.llm.anthropic_client import AnthropicClient

                client = AnthropicClient(
                    api_key=api_key,
                    base_url=base_url,
                    default_model=model or "claude-sonnet-4-6",
                )
            else:
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

        # Reasoning commit config
        tract._commit_reasoning = commit_reasoning

        # Auto-message config
        if auto_message is not False:
            tract._auto_message_enabled = True
            if isinstance(auto_message, str):
                tract._operation_configs = replace(
                    tract._operation_configs,
                    message=LLMConfig(model=auto_message, temperature=0.0),
                )
            elif isinstance(auto_message, LLMConfig):
                tract._operation_configs = replace(
                    tract._operation_configs, message=auto_message,
                )
            # auto_message=True: no operation config needed, uses default

        # Tool defaults
        if tool_profile is not None:
            tract._tool_profile = tool_profile
        if tool_result_format is not None:
            tract._tool_result_format = tool_result_format
        # Sync format to compiler
        if hasattr(tract._compiler, "tool_result_format"):
            tract._compiler.tool_result_format = tract._tool_result_format

        # Retry config
        if retry is not None:
            tract._retry_config = retry

        # Prompt directory: explicit > auto-discover .tract/prompts/
        if prompt_dir is not None:
            tract._prompt_dir = prompt_dir
        else:
            from pathlib import Path as _Path
            auto = _Path(".tract") / "prompts"
            if auto.is_dir():
                tract._prompt_dir = auto

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
        llm_client: LLMClient | None = None,
        event_repo: SqliteOperationEventRepository | None = None,
        compile_record_repo: SqliteCompileRecordRepository | None = None,
        tool_schema_repo: SqliteToolSchemaRepository | None = None,
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
    def config_index(self) -> ConfigIndex:
        """Get the current config index (built/cached from DAG ancestry)."""
        from tract.operations.config_index import ConfigIndex as _ConfigIndex

        if self._config_index is None or self._config_index.is_stale:
            head = self.head
            if head is None:
                return _ConfigIndex()
            self._config_index = _ConfigIndex.build(
                self._commit_repo, self._blob_repo, head,
                parent_repo=self._parent_repo,
            )
        return self._config_index

    def get_config(self, key: str, default: Any = None) -> Any:
        """Resolve a config value from DAG.

        Uses DAG precedence: closest to HEAD wins.
        """
        return self.config_index.get(key, default=default)

    def get_all_configs(self) -> dict[str, Any]:
        """Resolve all config key-value pairs from DAG."""
        return self.config_index.get_all()

    # Well-known config key type validators
    _WELL_KNOWN_CONFIG_TYPES: dict[str, type | tuple[type, ...]] = {
        "model": (str,),
        "temperature": (int, float),
        "max_tokens": (int,),
        "max_commit_tokens": (int,),
        "auto_compress_threshold": (int,),
        "compact_tools": (dict,),
        "compile_strategy": (str,),
        "compile_strategy_k": (int,),
        "handoff_summary_k": (int,),
    }

    def configure(self, **settings: Any) -> CommitInfo:
        """Commit config to DAG. Well-known keys are type-checked.

        Raises ValueError if a well-known key has the wrong type.
        Unknown keys pass through without validation.
        None values are valid (unset semantics).
        """
        self._check_open()
        from tract.models.content import ConfigContent

        for key, value in settings.items():
            if value is not None and key in self._WELL_KNOWN_CONFIG_TYPES:
                expected = self._WELL_KNOWN_CONFIG_TYPES[key]
                if not isinstance(value, expected):
                    raise ValueError(
                        f"Config '{key}' expects {expected}, got {type(value).__name__}"
                    )
        content = ConfigContent(settings=settings)
        info = self.commit(content, message=f"configure: {', '.join(settings)}")
        if self._config_index is not None:
            self._config_index.invalidate()
        return info

    def directive(
        self,
        name: str,
        text: str | None = None,
        *,
        path: str | Path | None = None,
        priority: Priority | None = None,
        message: str | None = None,
        tags: list[str] | None = None,
    ) -> CommitInfo:
        """Commit a named standing instruction (compiled, override-by-name).

        Default priority is PINNED. The compiler deduplicates by name:
        same name -> closest to HEAD wins.

        Pass *text* inline or load from *path* (mutually exclusive).
        """
        self._check_open()
        from tract.models.annotations import Priority as _Priority
        from tract.models.content import InstructionContent

        resolved = _resolve_text(text, path, label="text", prompt_dir=self._prompt_dir)
        content = InstructionContent(text=resolved, name=name)
        info = self.commit(
            content,
            message=message or f"directive: {name}",
            tags=tags,
        )
        # Apply priority annotation
        # Default is PINNED (matching InstructionContent type hint).
        # When user explicitly requests NORMAL, annotate to override the
        # type-hint default of PINNED.
        actual_priority = priority if priority is not None else _Priority.PINNED
        if priority is not None and actual_priority != _Priority.PINNED:
            self.annotate(info.commit_hash, actual_priority)
        return info

    def apply_template(
        self, name: str, *, directive_name: str | None = None, **params: object
    ) -> CommitInfo:
        """Apply a directive template by name with parameters.

        Args:
            name: Template name (built-in or custom registered)
            directive_name: Override the directive name (defaults to template name)
            **params: Template parameters to fill in placeholders

        Returns:
            CommitInfo for the directive commit
        """
        from tract.templates import get_template

        template = get_template(name)
        content = template.render(**params)
        return self.directive(directive_name or template.name, content)

    def list_templates(self) -> list:
        """List all available directive templates."""
        from tract.templates import list_templates

        return list_templates()

    # ------------------------------------------------------------------
    # Workflow profiles
    # ------------------------------------------------------------------

    def load_profile(self, name: str, *, apply_directives: bool = True) -> None:
        """Load a workflow profile, applying its config and directives.

        Args:
            name: Profile name (``"coding"``, ``"research"``, ``"ecommerce"``,
                or a custom-registered name).
            apply_directives: Whether to apply the profile's directives
                (default ``True``).

        Raises:
            KeyError: If the profile name is not found.
        """
        from tract.profiles import get_profile as _get_workflow_profile

        profile = _get_workflow_profile(name)

        # Apply config
        if profile.config:
            self.configure(**profile.config)

        # Apply directives
        if apply_directives:
            for dir_name, content in profile.directives.items():
                self.directive(dir_name, content)

        # Apply directive templates
        if profile.directive_templates:
            for tmpl_name, params in profile.directive_templates.items():
                self.apply_template(tmpl_name, **params)

        # Store profile reference for stage transitions
        self._active_profile = profile

    def apply_stage(self, stage_name: str) -> None:
        """Apply stage-specific config from the active workflow profile.

        Overrides configuration values for the given stage while keeping
        non-overridden settings from the base profile config.

        Args:
            stage_name: Stage name (must exist in the profile's ``stages`` dict).

        Raises:
            ValueError: If no profile is loaded or the stage name is unknown.
        """
        if self._active_profile is None:
            raise ValueError("No workflow profile loaded. Call load_profile() first.")
        profile = self._active_profile
        if stage_name not in profile.stages:
            available = ", ".join(sorted(profile.stages.keys()))
            raise ValueError(
                f"Stage '{stage_name}' not in profile '{profile.name}'. "
                f"Available: {available}"
            )
        stage_config = profile.stages[stage_name]
        self.configure(**stage_config)

    @property
    def active_profile(self) -> object | None:
        """The currently loaded workflow profile, or None."""
        return self._active_profile

    def add_middleware(self, event: MiddlewareEvent, handler: Callable) -> str:
        """Register middleware. Returns handler ID for removal."""
        from tract.middleware import VALID_EVENTS

        if event not in VALID_EVENTS:
            raise ValueError(
                f"Unknown middleware event '{event}'. "
                f"Valid events: {sorted(VALID_EVENTS)}"
            )
        handler_id = uuid.uuid4().hex[:12]
        self._middleware.setdefault(event, []).append((handler_id, handler))
        return handler_id

    # Backward-compatible alias
    use = add_middleware

    def remove_middleware(self, handler_id: str) -> None:
        """Remove a registered middleware handler."""
        for event, handlers in self._middleware.items():
            for i, (hid, _fn) in enumerate(handlers):
                if hid == handler_id:
                    handlers.pop(i)
                    return
        raise ValueError(f"Middleware handler '{handler_id}' not found")

    def _run_middleware(self, event: str, **kwargs: Any) -> None:
        """Run middleware handlers for an event.

        Raises BlockedError if a handler blocks (pre_* events only).
        """
        if event in self._in_middleware_events:
            return  # recursion guard
        handlers = self._middleware.get(event, [])
        if not handlers:
            return
        self._in_middleware_events.add(event)
        try:
            from tract.middleware import MiddlewareContext

            ctx = MiddlewareContext(
                event=event,
                commit=kwargs.get("commit"),
                tract=self,
                branch=self.current_branch or "",
                head=self.head or "",
                target=kwargs.get("target"),
                pending=kwargs.get("pending"),
            )
            for _id, fn in handlers:
                fn(ctx)
        finally:
            self._in_middleware_events.discard(event)

    def transition(
        self,
        target: str,
        *,
        handoff: Literal["full", "summary", "none"] | str = "none",
        **kwargs: Any,
    ) -> CommitInfo | None:
        """Transition to target branch with optional handoff.

        Args:
            target: Branch name.
            handoff: ``"full"`` (compile all), ``"summary"`` (adaptive),
                ``"none"``, or custom text string.

        Returns:
            CommitInfo of the handoff commit on the target, or None if no handoff.
        """
        self._run_middleware("pre_transition", target=target)

        payload = None
        if handoff == "full":
            payload = str(self.compile().to_dicts())
        elif handoff == "summary":
            k = self.get_config("handoff_summary_k") or 3
            payload = str(self.compile(strategy="adaptive", strategy_k=k).to_dicts())
        elif handoff != "none":
            payload = handoff

        existing = {b.name for b in self.list_branches()}
        if target not in existing:
            from tract.operations.branch import create_branch

            create_branch(target, self._tract_id, self._ref_repo, self._commit_repo)
            self._commit_session()
        self.switch(target)

        result = None
        if payload:
            result = self.system(f"Context handoff:\n{payload}", message=f"handoff to {target}")

        self._run_middleware("post_transition", target=target)
        return result

    @property
    def spawn_repo(self) -> SqliteSpawnPointerRepository | None:
        """Expose spawn repo for internal use by Session."""
        return self._spawn_repo

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
        from tract.models.tools import hash_tool_schema

        now = datetime.now(timezone.utc)
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
        a new CompiledContext with the tools field populated.  Also adds
        the estimated token cost of the tool schemas to ``token_count``
        so pre-call budget checks account for them.

        When the token count is API-calibrated (source starts with ``"api:"``),
        the API's ``prompt_tokens`` already includes tool definitions, so we
        skip adding tiktoken-estimated tool tokens to avoid double-counting.

        Returns the original result unchanged if no tools are found.
        """
        tools = self._gather_tools_for_compile()
        if not tools:
            return result
        # Only add tiktoken tool tokens when source is NOT API-calibrated
        # (API prompt_tokens already includes tool definition costs)
        token_count = result.token_count
        if not result.token_source.startswith("api:"):
            tools_text = json.dumps(tools)
            tools_tokens = self._token_counter.count_text(tools_text)
            token_count += tools_tokens
        # CompiledContext is frozen, so create new instance with tools
        return CompiledContext(
            messages=result.messages,
            token_count=token_count,
            commit_count=result.commit_count,
            token_source=result.token_source,
            generation_configs=result.generation_configs,
            commit_hashes=result.commit_hashes,
            priorities=result.priorities,
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
        tags: list[str] | None = None,
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
            tags: Optional list of immutable tag names to attach to the commit.
                These are set at commit time and cannot be changed later.
                Merged with auto-classified tags (deduplicated).

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        self._check_open()
        # Guard: detached HEAD blocks commits
        if self._ref_repo.is_detached(self._tract_id):
            raise DetachedHeadError()

        # Auto-validate dicts through the content type system
        if isinstance(content, dict):
            content = validate_content(content, custom_registry=self._custom_type_registry)

        # Determine role and content_type for auto-classification
        _ctype = getattr(content, "content_type", "unknown") if isinstance(content, BaseModel) else "unknown"
        _role = getattr(content, "role", None) if isinstance(content, BaseModel) else None

        # Extract text once (reused for auto-message and max_commit_tokens)
        _text: str | None = None
        if isinstance(content, BaseModel):
            from tract.engine.commit import extract_text_from_content as _extract_text
            _text = _extract_text(content)

        # Auto-generate commit message and auto-classify tags
        if message is None and isinstance(content, BaseModel):
            auto_msg, auto_tags = self._auto_classify(
                _ctype, _text, role=_role, operation=operation, metadata=metadata,
            )
            message = auto_msg
        else:
            # Message provided: only classify tags (no LLM call for message)
            auto_tags = self._classify_tags(
                _ctype, role=_role, operation=operation, metadata=metadata,
            )

        # Merge explicit tags with auto-classified tags (deduplicated)
        explicit_tags = list(tags) if tags else []
        all_tags = list(dict.fromkeys(explicit_tags + auto_tags))  # preserves order, deduplicates

        # Validate tags against registry in strict mode
        if all_tags:
            self._validate_tags(all_tags)

        # Pre-commit middleware (can block)
        self._run_middleware("pre_commit", pending=content)

        # Config enforcement: max_commit_tokens
        max_commit_tokens = self.get_config("max_commit_tokens")
        if max_commit_tokens is not None and _text is not None:
            _est_tokens = self._token_counter.count_text(_text) if _text else 0
            if _est_tokens > int(max_commit_tokens):
                raise BlockedError(
                    "pre_commit",
                    f"Exceeds max_commit_tokens ({max_commit_tokens})",
                )

        prev_head = self.head

        info = self._commit_engine.create_commit(
            content=content,
            operation=operation,
            message=message,
            edit_target=edit_target,
            metadata=metadata,
            generation_config=generation_config,
            tags=all_tags if all_tags else None,
        )

        # Link tool schemas to this commit
        effective_tools = tools if tools is not None else self._active_tools
        if effective_tools is not None and self._tool_schema_repo is not None:
            self._store_and_link_tools(info.commit_hash, effective_tools)

        # Persist to database
        self._commit_session()

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

        # Fire post-commit middleware
        self._run_middleware("post_commit", commit=info)

        return info

    def metadata(
        self,
        kind: str,
        data: dict | str,
        *,
        path: str | None = None,
        message: str | None = None,
        tags: list[str] | None = None,
    ) -> CommitInfo:
        """Create or update a metadata entry.

        Args:
            kind: Freeform label ("file_tree", "project_plan", etc.).
            data: Structured or text content.
            path: Optional filesystem path for export/sync.
            message: Optional commit message.
            tags: Optional tags.

        Returns:
            CommitInfo for the metadata commit.
        """
        from tract.models.content import MetadataContent

        # MetadataContent.data is dict; if str passed, wrap it
        if isinstance(data, str):
            data_dict = {"text": data}
        else:
            data_dict = data
        content = MetadataContent(kind=kind, data=data_dict, path=path)
        return self.commit(
            content,
            message=message or f"metadata: {kind}",
            tags=tags,
        )

    def _commit_dialogue(
        self,
        role: str,
        text: str | None,
        *,
        path: str | Path | None,
        edit: str | None,
        message: str | None,
        name: str | None,
        metadata: dict | None,
        generation_config: dict | None,
        priority: Priority | None,
        retain: str | None,
        retain_match: list[str] | None,
        improve: bool,
        tags: list[str] | None,
    ) -> CommitInfo:
        """Shared implementation for :meth:`system`, :meth:`user`, and :meth:`assistant`.

        Handles text resolution, content construction, commit, priority
        annotation, and optional LLM improvement.  Role-specific
        differences (content class, extra kwargs) are parameterised.
        """
        from tract.models.content import DialogueContent, InstructionContent

        text = _resolve_text(text, path, label="text", prompt_dir=self._prompt_dir)

        if role == "system":
            content = InstructionContent(text=text)
        else:
            content = DialogueContent(role=role, text=text, name=name)

        commit_kwargs: dict[str, Any] = dict(
            operation=CommitOperation.EDIT if edit else CommitOperation.APPEND,
            edit_target=edit,
            message=message,
            metadata=metadata,
            tags=tags,
        )
        if generation_config is not None:
            commit_kwargs["generation_config"] = generation_config

        info = self.commit(content, **commit_kwargs)

        if priority is not None:
            self.annotate(
                info.commit_hash, priority,
                retain=retain, retain_match=retain_match,
            )
        if improve:
            if not self._has_llm_client("improve"):
                raise ValueError(
                    "improve=True requires an LLM client. "
                    "Call configure_llm() or pass api_key to Tract.open()."
                )
            improved = self._improve_commit(info, text, role)
            if improved is not None:
                info = improved
        return info

    def system(
        self,
        text: str | None = None,
        *,
        path: str | Path | None = None,
        edit: str | None = None,
        message: str | None = None,
        metadata: dict | None = None,
        priority: Priority | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
        improve: bool = False,
        tags: list[str] | None = None,
    ) -> CommitInfo:
        """Commit a system instruction.

        Shorthand for ``commit(InstructionContent(text=text))``.

        System instructions are **PINNED by default** — they survive
        compression unchanged.  Pass ``priority=`` to override (e.g.
        ``Priority.NORMAL`` to allow compression).

        Pass *text* inline or load from *path* (mutually exclusive).

        Args:
            text: The instruction text.
            path: Path to a file whose contents will be used as text.
            edit: If provided, the hash of the commit to replace (EDIT
                operation).  Omit for a normal APPEND.
            message: Optional commit message.
            metadata: Optional commit metadata.
            priority: Optional priority annotation to set on the commit.
                Overrides the default PINNED annotation.
            retain: Fuzzy retention instructions (for IMPORTANT priority).
            retain_match: Deterministic retention patterns (for IMPORTANT priority).
            improve: If True, use LLM to improve the text and apply as
                an EDIT commit on top of the original.  Requires an LLM
                client to be configured.
            tags: Optional list of immutable tags to attach.

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        self._check_open()
        return self._commit_dialogue(
            "system", text,
            path=path, edit=edit, message=message, name=None,
            metadata=metadata, generation_config=None, priority=priority,
            retain=retain, retain_match=retain_match, improve=improve,
            tags=tags,
        )

    def user(
        self,
        text: str | None = None,
        *,
        path: str | Path | None = None,
        edit: str | None = None,
        message: str | None = None,
        name: str | None = None,
        metadata: dict | None = None,
        priority: Priority | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
        improve: bool = False,
        tags: list[str] | None = None,
    ) -> CommitInfo:
        """Commit a user message.

        Shorthand for ``commit(DialogueContent(role='user', text=text))``.

        Pass *text* inline or load from *path* (mutually exclusive).

        Args:
            text: The message text.
            path: Path to a file whose contents will be used as text.
            edit: If provided, the hash of the commit to replace (EDIT
                operation).  Omit for a normal APPEND.
            message: Optional commit message.
            name: Optional speaker name.
            metadata: Optional commit metadata.
            priority: Optional priority annotation to set on the commit.
            retain: Fuzzy retention instructions (for IMPORTANT priority).
            retain_match: Deterministic retention patterns (for IMPORTANT priority).
            improve: If True, use LLM to improve the text and apply as
                an EDIT commit on top of the original.  Requires an LLM
                client to be configured.
            tags: Optional list of immutable tags to attach.

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        self._check_open()
        return self._commit_dialogue(
            "user", text,
            path=path, edit=edit, message=message, name=name,
            metadata=metadata, generation_config=None, priority=priority,
            retain=retain, retain_match=retain_match, improve=improve,
            tags=tags,
        )

    def assistant(
        self,
        text: str | None = None,
        *,
        path: str | Path | None = None,
        edit: str | None = None,
        message: str | None = None,
        name: str | None = None,
        metadata: dict | None = None,
        generation_config: dict | None = None,
        priority: Priority | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
        improve: bool = False,
        tags: list[str] | None = None,
    ) -> CommitInfo:
        """Commit an assistant response.

        Shorthand for ``commit(DialogueContent(role='assistant', text=text))``.

        Pass *text* inline or load from *path* (mutually exclusive).

        Args:
            text: The response text.
            path: Path to a file whose contents will be used as text.
            edit: If provided, the hash of the commit to replace (EDIT
                operation).  Omit for a normal APPEND.
            message: Optional commit message.
            name: Optional speaker name.
            metadata: Optional commit metadata.
            generation_config: Optional LLM generation config.
            priority: Optional priority annotation to set on the commit.
            retain: Fuzzy retention instructions (for IMPORTANT priority).
            retain_match: Deterministic retention patterns (for IMPORTANT priority).
            improve: If True, use LLM to improve the text and apply as
                an EDIT commit on top of the original.  Requires an LLM
                client to be configured.
            tags: Optional list of immutable tags to attach.

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        self._check_open()
        return self._commit_dialogue(
            "assistant", text,
            path=path, edit=edit, message=message, name=name,
            metadata=metadata, generation_config=generation_config,
            priority=priority, retain=retain, retain_match=retain_match,
            improve=improve, tags=tags,
        )

    def _improve_commit(
        self,
        original_info: CommitInfo,
        text: str,
        role: str,
    ) -> CommitInfo | None:
        """LLM-improve text and apply as EDIT commit.

        Args:
            original_info: The :class:`CommitInfo` of the original commit.
            text: The original text content.
            role: The message role (``"system"``, ``"user"``, ``"assistant"``).

        Returns:
            :class:`CommitInfo` for the EDIT commit, or ``None`` if
            improvement was skipped (LLM failure, empty result, or
            identical text).
        """
        from tract.prompts.improve import IMPROVE_CONTENT_SYSTEM, build_improve_prompt

        try:
            client = self._resolve_llm_client("improve")
            llm_kwargs = (
                self._resolve_llm_config("improve")
                if self._has_llm_client("improve")
                else {}
            )
            messages = [
                {"role": "system", "content": IMPROVE_CONTENT_SYSTEM},
                {"role": "user", "content": build_improve_prompt(text, context=role)},
            ]
            response = client.chat(messages, **llm_kwargs)
            improved_text = self._extract_content(response, client=client).strip()
            if not improved_text or improved_text == text:
                return None  # No improvement or same text
        except Exception:
            import warnings

            warnings.warn(
                f"LLM improvement failed for {role} message; keeping original.",
                stacklevel=3,
            )
            return None

        # Apply as EDIT
        from tract.models.content import DialogueContent, InstructionContent

        if role == "system":
            content = InstructionContent(text=improved_text)
        else:
            content = DialogueContent(role=role, text=improved_text)

        edit_info = self.commit(
            content,
            operation=CommitOperation.EDIT,
            edit_target=original_info.commit_hash,
            message=f"improve: {role} message",
        )
        return edit_info

    def reasoning(
        self,
        text: str,
        *,
        format: str = "parsed",
        message: str | None = None,
        metadata: dict | None = None,
    ) -> CommitInfo:
        """Commit reasoning/chain-of-thought content.

        Shorthand for ``commit(ReasoningContent(text=text, format=format))``.

        Reasoning commits are **SKIP by default** — excluded from
        ``compile()`` unless ``include_reasoning=True`` is passed.

        Args:
            text: The reasoning text.
            format: Extraction source format (``"parsed"``, ``"raw"``,
                ``"think_tags"``, ``"anthropic"``).
            message: Optional commit message.
            metadata: Optional commit metadata.

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        from tract.models.content import ReasoningContent

        return self.commit(
            ReasoningContent(text=text, format=format),
            message=message or self._auto_message("reasoning", text),
            metadata=metadata,
        )

    def tool_result(
        self,
        tool_call_id: str,
        name: str,
        content: str,
        *,
        edit: str | None = None,
        message: str | None = None,
        metadata: dict | None = None,
        is_error: bool = False,
    ) -> CommitInfo:
        """Commit a tool execution result.

        Shorthand for committing a tool result message in OpenAI-compatible
        format.  The ``tool_call_id`` links this result back to the
        :class:`ToolCall` that requested it.

        Args:
            tool_call_id: The ID from the originating ToolCall.
            name: The function/tool name.
            content: The result text.
            edit: If set, the commit hash of a previous tool result to replace.
                Uses EDIT operation instead of APPEND. The original is preserved
                in history for provenance.
            message: Optional commit message (auto-generated if omitted).
            metadata: Optional extra metadata (tool_call_id and name are
                added automatically).
            is_error: If True, mark this result as a failed tool call by
                storing ``is_error: True`` in commit metadata. Used by
                :meth:`drop_failed_tool_turns` to identify and skip
                error turns.

        Returns:
            :class:`CommitInfo` for the new commit.
        """
        from tract.models.content import DialogueContent

        meta = {**(metadata or {}), "tool_call_id": tool_call_id, "name": name}
        if is_error:
            meta["is_error"] = True

        operation = CommitOperation.EDIT if edit else CommitOperation.APPEND
        return self.commit(
            DialogueContent(role="tool", text=content),
            operation=operation,
            edit_target=edit,
            message=message or f"tool result: {name}",
            metadata=meta,
        )

    def configure_tool_summarization(
        self,
        instructions: dict[str, str] | None = None,
        *,
        auto_threshold: int | None = None,
        default_instructions: str | None = None,
        include_context: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        """Configure automatic tool result summarization.

        Stores config for later use by tool result processing.

        Args:
            instructions: Per-tool summarization instructions. Keys are
                tool names, values are instruction strings passed to the
                summarization LLM. For example::

                    {"grep": "summarize to matching filenames only",
                     "bash": "preserve exit code and last 10 lines"}

            auto_threshold: Token count threshold. Tool results exceeding
                this count are automatically summarized (using per-tool
                instructions if available, or ``default_instructions``).
                Results under the threshold pass through unchanged.
            default_instructions: Fallback instructions for tools not
                listed in ``instructions`` but over the threshold.
            include_context: If True, compile the current conversation
                context and pass it to the summarization LLM so it can
                filter tool results based on conversational relevance.
            system_prompt: Override the default system prompt for
                summarization. When ``include_context=True`` and no
                explicit system_prompt is given,
                ``TOOL_CONTEXT_SUMMARIZE_SYSTEM`` is used.

        Example::

            t.configure_tool_summarization(
                instructions={
                    "grep": "summarize to filenames and line numbers",
                    "read_file": "keep first 20 lines, summarize rest",
                },
                auto_threshold=500,
            )
        """
        from tract.models.config import ToolSummarizationConfig

        self._tool_summarization_config = ToolSummarizationConfig(
            instructions=instructions or {},
            auto_threshold=auto_threshold,
            default_instructions=default_instructions,
            include_context=include_context,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Tool query API
    # ------------------------------------------------------------------

    def find_tool_results(
        self,
        name: str | None = None,
        after: str | None = None,
    ) -> list[CommitInfo]:
        """Find tool result commits on the current branch.

        Walks the commit history and returns commits where
        ``metadata["tool_call_id"]`` exists (indicating a tool result).

        Args:
            name: If set, only return results where ``metadata["name"]``
                matches this value.
            after: If set, only return results that come after this commit
                hash in history (exclusive). "After" means more recent.

        Returns:
            List of :class:`CommitInfo` for matching tool result commits,
            in chronological order (oldest first).
        """
        entries = self.log(limit=10000)
        entries.reverse()  # oldest-first

        results = []
        after_found = after is None  # if no after filter, include all

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

    def find_tool_calls(
        self,
        name: str | None = None,
    ) -> list[CommitInfo]:
        """Find assistant commits that requested tool calls.

        Walks the commit history and returns commits where
        ``metadata["tool_calls"]`` exists (indicating the assistant
        requested one or more tool calls).

        Args:
            name: If set, only return commits where at least one tool
                call in ``metadata["tool_calls"]`` has a matching name.

        Returns:
            List of :class:`CommitInfo` for matching assistant commits,
            in chronological order (oldest first).
        """
        entries = self.log(limit=10000)
        entries.reverse()  # oldest-first

        results = []
        for ci in entries:
            meta = ci.metadata or {}
            tool_calls = meta.get("tool_calls")
            if not tool_calls:
                continue
            if name is not None:
                # Direct dict access avoids ToolCall.from_dict overhead
                if not any(tc.get("name") == name for tc in tool_calls):
                    continue
            results.append(ci)

        return results

    def find_tool_turns(
        self,
        name: str | None = None,
    ) -> list["ToolTurn"]:
        """Find paired tool-call + tool-result commit groups.

        Walks the branch, matches tool results to their originating
        assistant tool-call message by ``tool_call_id``. Returns
        :class:`ToolTurn` instances with the call and its result(s)
        grouped together.

        Args:
            name: If set, only return turns where at least one tool
                in the turn matches this name.

        Returns:
            List of :class:`ToolTurn` in chronological order.
        """

        from tract.protocols import ToolTurn

        entries = self.log(limit=10000)
        entries.reverse()  # oldest-first

        # Build index: tool_call_id -> list of result CommitInfos
        result_index: dict[str, list[CommitInfo]] = {}
        for ci in entries:
            meta = ci.metadata or {}
            tcid = meta.get("tool_call_id")
            if tcid:
                result_index.setdefault(tcid, []).append(ci)

        # Walk tool-call commits and pair with their results
        turns = []
        for ci in entries:
            meta = ci.metadata or {}
            tool_calls_data = meta.get("tool_calls")
            if not tool_calls_data:
                continue

            # Direct dict access avoids ToolCall.from_dict overhead per item
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

    def drop_failed_tool_turns(
        self,
        name: str | None = None,
    ) -> "ToolDropResult":
        """Drop tool turns that contain error results from the compiled context.

        Walks :meth:`find_tool_turns` and checks each result's metadata for
        ``is_error: True``.  If *any* result in a turn is an error, the
        entire turn (call commit + all result commits) is annotated with
        :attr:`Priority.SKIP` so it no longer appears in :meth:`compile`.

        Args:
            name: If set, only consider turns matching this tool name.

        Returns:
            :class:`ToolDropResult` with stats on what was dropped.
        """
        from tract.models.compression import ToolDropResult

        turns = self.find_tool_turns(name=name)

        turns_dropped = 0
        commits_skipped = 0
        tokens_freed = 0
        dropped_names: set[str] = set()

        for turn in turns:
            # Check if any result in this turn has is_error
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

            # Skip the call commit
            self.annotate(turn.call.commit_hash, Priority.SKIP)
            commits_skipped += 1
            tokens_freed += turn.call.token_count

            # Skip all result commits
            for r in turn.results:
                self.annotate(r.commit_hash, Priority.SKIP)
                commits_skipped += 1
                tokens_freed += r.token_count

        return ToolDropResult(
            turns_dropped=turns_dropped,
            commits_skipped=commits_skipped,
            tokens_freed=tokens_freed,
            tool_names=tuple(sorted(dropped_names)),
        )

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
        include_sources: bool = False,
        **kwargs: Any,
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
            operation: Operation name ("chat", "merge", "compress").
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
        sources: dict = {}

        for field_name in _TYPED_FIELDS:
            # Level 1: Sugar param
            val = sugar.get(field_name)
            if val is not None:
                resolved[field_name] = val
                if include_sources:
                    sources[field_name] = "sugar"
                continue
            # Level 2: llm_config
            if llm_config is not None:
                val = getattr(llm_config, field_name, None)
                if val is not None:
                    resolved[field_name] = val
                    if include_sources:
                        sources[field_name] = "llm_config"
                    continue
            # Level 3: Operation config
            if op_config is not None:
                val = getattr(op_config, field_name, None)
                if val is not None:
                    resolved[field_name] = val
                    if include_sources:
                        sources[field_name] = f"operation:{operation}"
                    continue
            # Level 4: Tract default
            if default is not None:
                val = getattr(default, field_name, None)
                if val is not None:
                    resolved[field_name] = val
                    if include_sources:
                        sources[field_name] = "tract_default"

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

        if include_sources:
            resolved["_resolution_sources"] = sources

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

    def _extract_content(self, response: dict, *, client: LLMClient | None = None) -> str:
        """Extract content from LLM response, dispatching to the client's method.

        Falls back to OpenAI-format extraction for duck-typed clients that
        don't implement ``extract_content()``.

        Args:
            response: Raw LLM response dict.
            client: The LLM client that produced the response.  If None,
                falls back to the tract-level default.
        """
        c = client if client is not None else self._llm_client
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

    def _extract_usage(self, response: dict, *, client: LLMClient | None = None) -> dict | None:
        """Extract usage from LLM response, dispatching to the client's method.

        Falls back to OpenAI-format extraction for duck-typed clients that
        don't implement ``extract_usage()``.

        Args:
            response: Raw LLM response dict.
            client: The LLM client that produced the response.  If None,
                falls back to the tract-level default.
        """
        c = client if client is not None else self._llm_client
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
        reasoning: bool = True,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        max_retries: int = 3,
        hide_retries: bool = True,
        retry_prompt: str | None = None,
        retry: RetryConfig | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Compile context, call LLM, commit assistant response, record usage.

        Assumes the conversation context (system prompt, user messages) has
        already been committed. Use :meth:`chat` for the all-in-one path.

        When ``validator`` is provided, failed attempts are committed and
        SKIP-annotated (if ``hide_retries=True``), so they exist in the
        chain for audit/debugging but don't pollute the compiled context.
        Retry metadata (attempt count) is automatically attached to the
        final commit.

        Args:
            model: Model override for this call.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            llm_config: Full LLMConfig override for this call.
            message: Optional commit message for the assistant commit.
            metadata: Optional metadata for the assistant commit.
            reasoning: If True (default), auto-commit reasoning traces
                extracted from the LLM response. Set to False to skip
                reasoning commits for this call. Global opt-out via
                ``commit_reasoning=False`` on ``Tract.open()``.
            validator: Optional callable that validates the response text.
                Takes the response text, returns (ok, diagnosis). When
                provided, retries with steering on validation failure.
            max_retries: Maximum retry attempts when validator is set (default 3).
            hide_retries: If True (default), SKIP-annotate failed attempts
                and steering messages so they don't appear in compiled
                context. If False, all retry artifacts remain visible.
            retry_prompt: Custom steering prompt template. The diagnosis string
                is appended to this. Defaults to a standard steering message.
            retry: Per-call :class:`RetryConfig` override.  When provided,
                the raw LLM call is retried with exponential backoff on
                transient errors.  Overrides the tract-level retry config.
            **kwargs: Extra provider-specific parameters passed through to the
                LLM client (e.g. ``reasoning_effort="high"``).  Highest
                priority in the config resolution chain.

        Returns:
            :class:`ChatResponse` with text, usage, commit_info, generation_config.

        Raises:
            LLMConfigError: If no LLM client is configured.
            TraceError: If called inside batch().
            RetryExhaustedError: If all retry attempts fail validation.
        """
        self._check_open()
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
                reasoning=reasoning, retry=retry, **kwargs,
            )

        # Validation retry loop
        import dataclasses as _dc

        intermediate_hashes: list[str] = []
        last_diagnosis: str | None = None

        for attempt in range(max_retries + 1):
            response = self._generate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
                reasoning=reasoning, retry=retry, **kwargs,
            )

            ok, diagnosis = validator(response.text)
            if ok:
                # SKIP-annotate intermediate commits if hiding retries
                if hide_retries and intermediate_hashes:
                    for h in intermediate_hashes:
                        self.annotate(h, Priority.SKIP,
                                      reason="retry: hidden intermediate")

                # Attach retry metadata to the final commit
                if attempt > 0 and response.commit_info:
                    retry_meta = {"retry_attempts": attempt}
                    existing = response.commit_info.metadata or {}
                    merged = {**existing, **retry_meta}
                    self._commit_repo.update_metadata(
                        response.commit_info.commit_hash, merged
                    )
                    self._commit_session()
                    updated_info = response.commit_info.model_copy(
                        update={"metadata": merged}
                    )
                    response = _dc.replace(response, commit_info=updated_info)

                return response

            last_diagnosis = diagnosis

            # Record failed response hash for deferred SKIP
            failed_hash = self.head
            if failed_hash:
                intermediate_hashes.append(failed_hash)

            # Steer with diagnosis if not last attempt
            if attempt < max_retries:
                steering = retry_prompt or "Your previous response did not pass validation. Please try again."
                if diagnosis:
                    steering = f"{steering}\n\nDiagnosis: {diagnosis}"
                steering_info = self.user(steering)
                intermediate_hashes.append(steering_info.commit_hash)

        # All retries exhausted
        from tract.exceptions import RetryExhaustedError
        raise RetryExhaustedError(
            attempts=max_retries + 1,
            last_diagnosis=last_diagnosis or "validation failed",
            last_result=response.text,
        )

    def _generate_once(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        metadata: dict | None = None,
        reasoning: bool = True,
        retry: RetryConfig | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Single generate attempt (no retry). Internal helper.

        Contains the original generate() logic: compile, LLM call, commit, usage.
        When *retry* is provided, the raw LLM call is wrapped in
        :func:`_retry_with_backoff` to handle transient failures.
        """
        from tract.protocols import ChatResponse

        # 1. Compile context
        compiled = self.compile()
        messages = compiled.to_dicts()

        # 1b. Persist compile record (SC-3: chat/generate auto-create)
        if self._compile_record_repo is not None:
            self._save_compile_record(
                self.head or "",
                compiled.token_count,
                compiled.commit_count,
                compiled.token_source,
                compiled.commit_hashes,
            )

        # 2. Call LLM (resolve per-operation client and config)
        chat_client = self._resolve_llm_client("chat")
        llm_kwargs = self._resolve_llm_config(
            "chat", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config, **kwargs,
        )
        if compiled.tools:
            llm_kwargs["tools"] = compiled.tools

        # Pre-generate middleware (can block)
        self._run_middleware(
            "pre_generate",
            pending={"messages": messages, "config": llm_kwargs},
        )

        effective_retry = retry or self._retry_config
        response = _retry_with_backoff(
            chat_client.chat, effective_retry, messages, **llm_kwargs,
        )

        # 3. Extract content and usage (dispatch to client methods, with
        #    OpenAI-format defaults for duck-typed clients that lack them)
        text = self._extract_content(response, client=chat_client)
        usage_dict = self._extract_usage(response, client=chat_client)

        # Post-generate middleware (informational, cannot block)
        self._run_middleware(
            "post_generate",
            pending={
                "response": text or "",
                "tokens_used": (
                    usage_dict.get("total_tokens", 0) if usage_dict else 0
                ),
            },
        )

        # 3a. Extract reasoning (duck-typed optional on LLM client)
        reasoning_text: str | None = None
        reasoning_format: str = "parsed"
        if hasattr(chat_client, "extract_reasoning"):
            reasoning_result = chat_client.extract_reasoning(response)
            if reasoning_result is not None:
                reasoning_text, reasoning_format = reasoning_result
                # When <think> tags were extracted, strip them from the
                # assistant text so ChatResponse.text is clean
                if reasoning_format == "think_tags":
                    import re as _re
                    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

        # 3b. Parse tool_calls from the raw response (if any)
        from tract.protocols import ToolCall as _ToolCall

        raw_tool_calls = None
        try:
            raw_tool_calls = response["choices"][0]["message"].get("tool_calls")
        except (KeyError, IndexError, TypeError):
            pass
        tool_calls = (
            [_ToolCall.from_openai(tc) for tc in raw_tool_calls]
            if raw_tool_calls
            else None
        )

        # 4. Build generation_config (use resolved kwargs for accurate tracking)
        gen_config = self._build_generation_config(response, resolved=llm_kwargs)

        # 5a. Commit reasoning trace (if extracted and enabled)
        reasoning_commit_info: CommitInfo | None = None
        if reasoning_text and self._commit_reasoning and reasoning:
            reasoning_commit_info = self.reasoning(
                reasoning_text,
                format=reasoning_format,
            )

        # 5b. Commit assistant response (include tool_calls in metadata for provenance)
        commit_meta = metadata
        if tool_calls:
            commit_meta = {**(metadata or {}), "tool_calls": [tc.to_dict() for tc in tool_calls]}
        commit_info = self.assistant(
            text, message=message, metadata=commit_meta, generation_config=gen_config
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
            reasoning=reasoning_text,
            reasoning_commit=reasoning_commit_info,
            tool_calls=tool_calls,
            raw_response=response,
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
        reasoning: bool = True,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        max_retries: int = 3,
        hide_retries: bool = True,
        retry_prompt: str | None = None,
        retry: RetryConfig | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a user message and get an LLM response in one call.

        Commits the user message, compiles context, calls the LLM,
        commits the assistant response, and records usage. Equivalent to::

            t.user(text, message=message, name=name, metadata=metadata)
            response = t.generate(model=model, temperature=temperature, ...)

        When ``validator`` is provided, retries with steering on failure.
        See :meth:`generate` for details on retry behavior.

        Args:
            text: The user message text.
            model: Model override for this call.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            llm_config: Full LLMConfig override for this call.
            message: Optional commit message for the user commit.
            name: Optional speaker name for the user commit.
            metadata: Optional metadata for the user commit.
            reasoning: If True (default), auto-commit reasoning traces
                extracted from the LLM response. Set to False to skip
                reasoning commits for this call.
            validator: Optional callable that validates the response text.
                Takes the response text, returns (ok, diagnosis). When
                provided, retries with steering on validation failure.
            max_retries: Maximum retry attempts when validator is set (default 3).
            hide_retries: If True (default), SKIP-annotate failed attempts
                and steering messages so they don't appear in compiled context.
            retry_prompt: Custom steering prompt template. The diagnosis string
                is appended to this. Defaults to a standard steering message.
            **kwargs: Extra provider-specific parameters passed through to the
                LLM client (e.g. ``reasoning_effort="high"``).  Highest
                priority in the config resolution chain.

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
            reasoning=reasoning,
            validator=validator,
            max_retries=max_retries,
            hide_retries=hide_retries,
            retry_prompt=retry_prompt,
            retry=retry,
            **kwargs,
        )
        return _dc.replace(response, prompt=text)

    _TOOLS_SENTINEL = object()
    _PROFILE_SENTINEL = object()

    def run(
        self,
        task: str | None = None,
        *,
        max_steps: int = 50,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[dict] | None | object = _TOOLS_SENTINEL,
        profile: str | ToolProfile | object = _PROFILE_SENTINEL,
        tool_names: list[str] | None = None,
        tool_handlers: dict[str, Callable] | None = None,
        llm_client: LLMClient | None = None,
        on_step: Callable | None = None,
        on_token: Callable | None = None,
        on_tool_result: Callable | None = None,
        stream: bool = False,
        step_budget: int | None = None,
        tool_validator: Callable | None = None,
        auto_compress_threshold: float | None = None,
    ) -> LoopResult:  # noqa: F821
        """Run the default agent loop on this tract.

        Convenience wrapper around :func:`tract.loop.run_loop`.

        Args:
            task: Task description (committed as user message).
            max_steps: Maximum loop iterations.
            max_tokens: Maximum tokens per LLM response.  Passed to
                ``client.chat(max_tokens=...)``.  None means no limit
                (provider default).
            system_prompt: System prompt prepended to context.
            tools: Tool definitions (OpenAI format). Pass an explicit empty
                list ``[]`` to send no tools. When omitted, tools are built
                from ``profile`` / ``tool_names``.
            profile: Tool profile name (``"compact"``, ``"self"``,
                ``"supervisor"``, ``"full"``) or a :class:`ToolProfile`
                instance. Only used when ``tools`` is not provided.
                Falls back to ``tool_profile`` from :meth:`open`, then
                ``"compact"``.
            tool_names: Subset of tool names to include. Only used when
                ``tools`` is not provided. Filters the profile's tools to
                just the named ones.
            tool_handlers: Mapping of custom tool names to callables.
                When the LLM calls a tool in this dict, the function is
                called with the tool arguments as keyword args. Tools not
                in this dict are dispatched to tract's built-in executor.
            llm_client: LLM client override.
            on_step: Step callback ``(step_num, response) -> None``.
            on_token: Streaming callback ``(text_chunk) -> None``.  When
                provided and the LLM client supports streaming, each text
                delta is passed to this callback as it arrives.
            on_tool_result: Tool result callback ``(tool_name, output, status) -> None``.
                Called after each tool execution with the tool name, output text,
                and ``"success"`` or ``"error"`` status.
            stream: Enable streaming even without on_token callback.
            step_budget: Maximum total tokens across all loop steps.
                When exceeded, the loop stops gracefully with
                ``result.budget_exhausted == True``.
            tool_validator: Callable ``(tool_name, args_dict) -> (ok, error_msg)``
                that validates tool arguments before execution. Invalid
                calls are committed as errors without executing the tool.
            auto_compress_threshold: Float 0.0-1.0. When compiled context
                exceeds this fraction of ``max_tokens``, the loop
                auto-compresses before the next LLM call.

        Returns:
            LoopResult with status, reason, steps, and tool_calls.
        """
        from tract.loop import LoopConfig, run_loop

        # Resolve tools
        if tools is self._TOOLS_SENTINEL:
            effective_profile = (
                self._tool_profile or "compact"
            ) if profile is self._PROFILE_SENTINEL else profile
            resolved_tools = self.as_tools(
                profile=effective_profile,
                tool_names=tool_names,
                format="openai",
            )
        else:
            resolved_tools = tools  # type: ignore[assignment]  # user-supplied tools passthrough

        # Merge custom tool handlers from @t.tool into tool_handlers
        if self._custom_tools:
            merged_handlers = {
                name: td.handler for name, td in self._custom_tools.items()
            }
            if tool_handlers:
                merged_handlers.update(tool_handlers)  # explicit overrides win
            tool_handlers = merged_handlers

        config = LoopConfig(
            max_steps=max_steps,
            system_prompt=system_prompt,
            stream=stream,
            max_tokens=max_tokens,
            step_budget=step_budget,
            tool_validator=tool_validator,
            auto_compress_threshold=auto_compress_threshold,
        )
        return run_loop(
            self,
            task=task,
            config=config,
            llm_client=llm_client,
            tools=resolved_tools,
            tool_handlers=tool_handlers,
            on_step=on_step,
            on_token=on_token,
            on_tool_result=on_tool_result,
        )

    def revise(
        self,
        commit_hash: str,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        reasoning: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Ask the LLM to revise a previous commit and apply as an EDIT.

        Convenience wrapper that combines chat + edit + skip into one call.
        Internally: sends ``prompt`` via :meth:`chat` to get the LLM's
        revised text, creates an EDIT commit targeting ``commit_hash``
        with that text, and SKIPs the intermediate user/assistant commits
        so they don't appear in compiled context.

        Args:
            commit_hash: Hash (or prefix) of the commit to revise.  Must
                be an APPEND commit (system, user, or assistant).
            prompt: Instruction telling the LLM how to revise the content.
            model: Model override for this call.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            llm_config: Full LLMConfig override for this call.
            message: Optional commit message for the EDIT commit.
            reasoning: If True (default), auto-commit reasoning traces.
            **kwargs: Extra provider-specific parameters.

        Returns:
            :class:`ChatResponse` whose ``commit_info`` is the EDIT commit
            and whose ``text`` is the revised content.

        Raises:
            LLMConfigError: If no LLM client is configured.
            EditTargetError: If ``commit_hash`` cannot be found or is
                itself an EDIT commit.
        """
        import dataclasses as _dc

        # Step 1: Get the LLM's revised text via a normal chat call.
        response = self.chat(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config=llm_config,
            reasoning=reasoning,
            **kwargs,
        )

        # Step 2: Resolve the target to determine the content role.
        resolved = self.resolve_commit(commit_hash)
        target_row = self._commit_repo.get(resolved)
        if target_row is None:
            from tract.exceptions import EditTargetError
            raise EditTargetError(f"EDIT target commit not found: {resolved}")

        # Determine the role of the target commit so we use the right
        # shorthand (system / user / assistant).
        ct = target_row.content_type
        if ct == "instruction":
            role = "system"
        elif ct == "dialogue":
            # Load the blob to get the role field.
            blob = self._blob_repo.get(target_row.content_hash)
            data = json.loads(blob.payload_json) if blob else {}
            role = data.get("role", "assistant")
        else:
            role = "assistant"

        # Step 3: Apply as an EDIT commit.
        shorthand = {"system": self.system, "user": self.user, "assistant": self.assistant}
        edit_fn = shorthand.get(role, self.assistant)
        edit_info = edit_fn(
            response.text,
            edit=resolved,
            message=message or f"revise: {prompt[:60]}",
        )

        # Step 4: SKIP the intermediate user + assistant commits from
        # the chat() call so only the EDIT survives in compiled context.
        self.annotate(response.commit_info.parent_hash, Priority.SKIP)
        self.annotate(response.commit_info.commit_hash, Priority.SKIP)
        if response.reasoning_commit is not None:
            self.annotate(response.reasoning_commit.commit_hash, Priority.SKIP)

        # Return a ChatResponse pointing to the EDIT commit.
        return _dc.replace(
            response,
            commit_info=edit_info,
            prompt=prompt,
        )

    # ------------------------------------------------------------------
    # Async LLM methods
    # ------------------------------------------------------------------

    async def _agenerate_once(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        metadata: dict | None = None,
        reasoning: bool = True,
        retry: RetryConfig | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async version of :meth:`_generate_once`."""
        from tract.llm.protocols import acall_llm
        from tract.protocols import ChatResponse

        # 1. Compile context (sync — local operation)
        compiled = self.compile()
        messages = compiled.to_dicts()

        if self._compile_record_repo is not None:
            self._save_compile_record(
                self.head or "",
                compiled.token_count,
                compiled.commit_count,
                compiled.token_source,
                compiled.commit_hashes,
            )

        # 2. Call LLM (async)
        chat_client = self._resolve_llm_client("chat")
        llm_kwargs = self._resolve_llm_config(
            "chat", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config, **kwargs,
        )
        if compiled.tools:
            llm_kwargs["tools"] = compiled.tools

        # Pre-generate middleware (can block)
        self._run_middleware(
            "pre_generate",
            pending={"messages": messages, "config": llm_kwargs},
        )

        effective_retry = retry or self._retry_config

        async def _do_llm_call() -> Any:
            return await acall_llm(chat_client, messages, **llm_kwargs)

        response = await _aretry_with_backoff(_do_llm_call, effective_retry)

        # 3. Extract content and usage (sync)
        text = self._extract_content(response, client=chat_client)
        usage_dict = self._extract_usage(response, client=chat_client)

        # Post-generate middleware (informational, cannot block)
        self._run_middleware(
            "post_generate",
            pending={
                "response": text or "",
                "tokens_used": (
                    usage_dict.get("total_tokens", 0) if usage_dict else 0
                ),
            },
        )

        # 3a. Extract reasoning
        reasoning_text: str | None = None
        reasoning_format: str = "parsed"
        if hasattr(chat_client, "extract_reasoning"):
            reasoning_result = chat_client.extract_reasoning(response)
            if reasoning_result is not None:
                reasoning_text, reasoning_format = reasoning_result
                if reasoning_format == "think_tags":
                    import re as _re
                    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

        # 3b. Parse tool_calls
        from tract.protocols import ToolCall as _ToolCall

        raw_tool_calls = None
        try:
            raw_tool_calls = response["choices"][0]["message"].get("tool_calls")
        except (KeyError, IndexError, TypeError):
            pass
        tool_calls = (
            [_ToolCall.from_openai(tc) for tc in raw_tool_calls]
            if raw_tool_calls
            else None
        )

        # 4. Build generation_config
        gen_config = self._build_generation_config(response, resolved=llm_kwargs)

        # 5a. Commit reasoning (sync)
        reasoning_commit_info: CommitInfo | None = None
        if reasoning_text and self._commit_reasoning and reasoning:
            reasoning_commit_info = self.reasoning(
                reasoning_text,
                format=reasoning_format,
            )

        # 5b. Commit assistant response (sync)
        commit_meta = metadata
        if tool_calls:
            commit_meta = {**(metadata or {}), "tool_calls": [tc.to_dict() for tc in tool_calls]}
        commit_info = self.assistant(
            text, message=message, metadata=commit_meta, generation_config=gen_config
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
            reasoning=reasoning_text,
            reasoning_commit=reasoning_commit_info,
            tool_calls=tool_calls,
            raw_response=response,
        )

    async def agenerate(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        metadata: dict | None = None,
        reasoning: bool = True,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        max_retries: int = 3,
        hide_retries: bool = True,
        retry_prompt: str | None = None,
        retry: RetryConfig | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async version of :meth:`generate`.

        Compile context, call LLM asynchronously, commit assistant response.
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
            return await self._agenerate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
                reasoning=reasoning, retry=retry, **kwargs,
            )

        # Validation retry loop
        import dataclasses as _dc

        intermediate_hashes: list[str] = []
        last_diagnosis: str | None = None

        for attempt in range(max_retries + 1):
            response = await self._agenerate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
                reasoning=reasoning, retry=retry, **kwargs,
            )

            ok, diagnosis = validator(response.text)
            if ok:
                if hide_retries and intermediate_hashes:
                    for h in intermediate_hashes:
                        self.annotate(h, Priority.SKIP,
                                      reason="retry: hidden intermediate")

                if attempt > 0 and response.commit_info:
                    retry_meta = {"retry_attempts": attempt}
                    existing = response.commit_info.metadata or {}
                    merged = {**existing, **retry_meta}
                    self._commit_repo.update_metadata(
                        response.commit_info.commit_hash, merged
                    )
                    self._commit_session()
                    updated_info = response.commit_info.model_copy(
                        update={"metadata": merged}
                    )
                    response = _dc.replace(response, commit_info=updated_info)

                return response

            last_diagnosis = diagnosis
            failed_hash = self.head
            if failed_hash:
                intermediate_hashes.append(failed_hash)

            if attempt < max_retries:
                steering = retry_prompt or "Your previous response did not pass validation. Please try again."
                if diagnosis:
                    steering = f"{steering}\n\nDiagnosis: {diagnosis}"
                steering_info = self.user(steering)
                intermediate_hashes.append(steering_info.commit_hash)

        from tract.exceptions import RetryExhaustedError
        raise RetryExhaustedError(
            attempts=max_retries + 1,
            last_diagnosis=last_diagnosis or "validation failed",
            last_result=response.text,
        )

    async def achat(
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
        reasoning: bool = True,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        max_retries: int = 3,
        hide_retries: bool = True,
        retry_prompt: str | None = None,
        retry: RetryConfig | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async version of :meth:`chat`.

        Commits the user message (sync), then awaits the LLM response.
        """
        import dataclasses as _dc

        self.user(text, message=message, name=name, metadata=metadata)
        response = await self.agenerate(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config=llm_config,
            reasoning=reasoning,
            validator=validator,
            max_retries=max_retries,
            hide_retries=hide_retries,
            retry_prompt=retry_prompt,
            retry=retry,
            **kwargs,
        )
        return _dc.replace(response, prompt=text)

    async def arun(
        self,
        task: str | None = None,
        *,
        max_steps: int = 50,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: list[dict] | None | object = _TOOLS_SENTINEL,
        profile: str | ToolProfile | object = _PROFILE_SENTINEL,
        tool_names: list[str] | None = None,
        tool_handlers: dict[str, Callable] | None = None,
        llm_client: LLMClient | None = None,
        on_step: Callable | None = None,
        on_token: Callable | None = None,
        on_tool_result: Callable | None = None,
        stream: bool = False,
        step_budget: int | None = None,
        tool_validator: Callable | None = None,
        auto_compress_threshold: float | None = None,
    ) -> LoopResult:
        """Async version of :meth:`run`.

        Runs the agent loop with async LLM calls and non-blocking tool execution.
        See :meth:`run` for full parameter documentation.
        """
        from tract.loop import LoopConfig, arun_loop

        # Resolve tools (same logic as sync run)
        if tools is self._TOOLS_SENTINEL:
            effective_profile = (
                self._tool_profile or "compact"
            ) if profile is self._PROFILE_SENTINEL else profile
            resolved_tools = self.as_tools(
                profile=effective_profile,
                tool_names=tool_names,
                format="openai",
            )
        else:
            resolved_tools = tools  # type: ignore[assignment]

        # Merge custom tool handlers from @t.tool into tool_handlers
        if self._custom_tools:
            merged_handlers = {
                name: td.handler for name, td in self._custom_tools.items()
            }
            if tool_handlers:
                merged_handlers.update(tool_handlers)
            tool_handlers = merged_handlers

        config = LoopConfig(
            max_steps=max_steps,
            system_prompt=system_prompt,
            stream=stream,
            max_tokens=max_tokens,
            step_budget=step_budget,
            tool_validator=tool_validator,
            auto_compress_threshold=auto_compress_threshold,
        )
        return await arun_loop(
            self,
            task=task,
            config=config,
            llm_client=llm_client,
            tools=resolved_tools,
            tool_handlers=tool_handlers,
            on_step=on_step,
            on_token=on_token,
            on_tool_result=on_tool_result,
        )

    async def arevise(
        self,
        commit_hash: str,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        message: str | None = None,
        reasoning: bool = True,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async version of :meth:`revise`."""
        import dataclasses as _dc

        response = await self.achat(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config=llm_config,
            reasoning=reasoning,
            **kwargs,
        )

        resolved = self.resolve_commit(commit_hash)
        target_row = self._commit_repo.get(resolved)
        if target_row is None:
            from tract.exceptions import EditTargetError
            raise EditTargetError(f"EDIT target commit not found: {resolved}")

        ct = target_row.content_type
        if ct == "instruction":
            role = "system"
        elif ct == "dialogue":
            blob = self._blob_repo.get(target_row.content_hash)
            data = json.loads(blob.payload_json) if blob else {}
            role = data.get("role", "assistant")
        else:
            role = "assistant"

        shorthand = {"system": self.system, "user": self.user, "assistant": self.assistant}
        edit_fn = shorthand.get(role, self.assistant)
        edit_info = edit_fn(
            response.text,
            edit=resolved,
            message=message or f"revise: {prompt[:60]}",
        )

        self.annotate(response.commit_info.parent_hash, Priority.SKIP)
        self.annotate(response.commit_info.commit_hash, Priority.SKIP)
        if response.reasoning_commit is not None:
            self.annotate(response.reasoning_commit.commit_hash, Priority.SKIP)

        return _dc.replace(
            response,
            commit_info=edit_info,
            prompt=prompt,
        )

    async def acompress(
        self,
        *,
        commits: list[str] | None = None,
        from_commit: str | None = None,
        to_commit: str | None = None,
        target_tokens: int | None = None,
        preserve: list[str] | None = None,
        content: str | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        two_stage: bool | None = None,
        triggered_by: str | None = None,
        strategy: str = "default",
        window_size: int = 5,
    ) -> CompressResult:
        """Async version of :meth:`compress`.

        The LLM summarization is awaited; commit finalization is sync.
        """
        from tract.operations.compression import (
            _classify_by_priority,
            _commit_compression,
            _partition_around_pinned,
            _reconstruct_content,
            _resolve_commit_range,
            acompress_range,
            sliding_window_compress,
        )

        if self._ref_repo.is_detached(self._tract_id):
            raise DetachedHeadError()

        if self._event_repo is None:
            from tract.exceptions import CompressionError
            raise CompressionError("Compression repository not available")

        has_client = self._has_llm_client("compress")
        llm_client = self._resolve_llm_client("compress") if has_client else None

        has_explicit_llm = (
            model is not None
            or temperature is not None
            or max_tokens is not None
            or llm_config is not None
        )
        if has_explicit_llm and llm_client is None and content is None:
            from tract.llm.errors import LLMConfigError
            raise LLMConfigError(
                "LLM parameters provided but no LLM client is configured. "
                "Call configure_llm() or pass api_key to Tract.open(), or "
                "provide content= for manual compression."
            )

        llm_kwargs = self._resolve_llm_config(
            "compress", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
        ) if llm_client is not None else {}

        effective_system_prompt = system_prompt
        if effective_system_prompt is None and self._operation_prompts.compress is not None:
            effective_system_prompt = self._operation_prompts.compress

        self._run_middleware("pre_compress")

        # --- Sliding window strategy (delegates to sync helper) ---
        if strategy == "sliding_window":
            import asyncio
            return await asyncio.to_thread(
                self._compress_sliding_window,
                window_size=window_size,
                target_tokens=target_tokens,
                preserve=preserve,
                content=content,
                instructions=instructions,
                system_prompt=effective_system_prompt,
                llm_client=llm_client,
                llm_kwargs=llm_kwargs,
                two_stage=two_stage,
                sliding_window_compress_fn=sliding_window_compress,
                _classify_by_priority_fn=_classify_by_priority,
                _commit_compression_fn=_commit_compression,
                _partition_around_pinned_fn=_partition_around_pinned,
                _reconstruct_content_fn=_reconstruct_content,
            )

        # Step 1: Async LLM summarization
        range_result = await acompress_range(
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
            llm_client=llm_client,
            llm_kwargs=llm_kwargs,
            generation_config=llm_kwargs if llm_kwargs else None,
            content=content,
            instructions=instructions,
            system_prompt=effective_system_prompt,
            type_registry=self._custom_type_registry,
            two_stage=two_stage or False,
        )

        # Step 2: Finalize (sync — same as sync compress)
        head_hash = self._ref_repo.get_head(self._tract_id)
        branch_name = self._ref_repo.get_current_branch(self._tract_id)
        range_commits = _resolve_commit_range(
            self._commit_repo, self._ref_repo, self._annotation_repo,
            self._tract_id, head_hash,
            commits=commits, from_commit=from_commit, to_commit=to_commit,
        )
        pinned_commits, _important, normal_commits, skip_commits = (
            _classify_by_priority(range_commits, self._annotation_repo, preserve=preserve)
        )
        normal_commits = normal_commits + _important
        pinned_hashes = {r.commit_hash for r in pinned_commits}
        skip_hashes = {r.commit_hash for r in skip_commits}
        groups = _partition_around_pinned(range_commits, pinned_hashes, skip_hashes)
        original_tokens = sum(c.token_count for c in normal_commits)

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
                summaries=range_result.summary_commits,
                range_commits=range_commits,
                pinned_commits=pinned_commits,
                normal_commits=normal_commits,
                pinned_hashes=pinned_hashes,
                skip_hashes=skip_hashes,
                groups=groups,
                original_tokens=original_tokens,
                target_tokens=target_tokens,
                instructions=instructions,
                system_prompt=effective_system_prompt,
                branch_name=branch_name,
                type_registry=self._custom_type_registry,
                expected_head=head_hash,
                generation_config=range_result.generation_config,
            )
        except Exception:
            nested.rollback()
            raise

        self._commit_session()
        self._cache.clear()

        if llm_kwargs:
            import dataclasses as _dc
            result = _dc.replace(result, config=LLMConfig.from_dict(llm_kwargs))

        return result

    async def acompress_tool_calls(
        self,
        commits: list[str] | None = None,
        *,
        name: str | None = None,
        target_tokens: int | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        triggered_by: str | None = None,
    ) -> ToolCompactResult:
        """Async version of :meth:`compress_tool_calls`."""
        from tract.exceptions import CompressionError
        from tract.llm.protocols import acall_llm
        from tract.models.compression import ToolCompactResult
        from tract.operations.compression import build_role_label
        from tract.prompts.summarize import (
            TOOL_COMPACT_SYSTEM,
            build_tool_compact_prompt,
        )

        turns = self.find_tool_turns(name=name)

        if commits is not None:
            commit_set = set(commits)
            turns = [
                turn for turn in turns
                if any(h in commit_set for h in turn.all_hashes)
            ]

        if not turns:
            raise CompressionError("No tool turns found to compact")

        results_to_compact: list[CommitInfo] = []
        parts: list[str] = []

        for turn in turns:
            call_meta = turn.call.metadata or {}
            call_text = self.get_content(turn.call) or ""
            parts.append(f"{build_role_label('assistant', call_meta)}: {call_text}")

            for r in turn.results:
                r_meta = r.metadata or {}
                r_text = self.get_content(r) or ""
                parts.append(f"{build_role_label('tool', r_meta)}: {r_text}")
                results_to_compact.append(r)

        if not results_to_compact:
            raise CompressionError("No tool results found to compact")

        sequence_text = "\n".join(parts)

        prompt = build_tool_compact_prompt(
            sequence_text,
            result_count=len(results_to_compact),
            target_tokens=target_tokens,
            instructions=instructions,
        )
        sys_prompt = (
            system_prompt if system_prompt is not None else TOOL_COMPACT_SYSTEM
        )

        llm_kwargs_resolved: dict = {}
        if any(v is not None for v in (model, temperature, max_tokens, llm_config)):
            resolved = self._resolve_llm_config(
                "compress", model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
            )
            if resolved:
                llm_kwargs_resolved = resolved

        llm = self._resolve_llm_client("compress")

        # Async LLM call
        response = await acall_llm(
            llm,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            **llm_kwargs_resolved,
        )

        raw_content = response["choices"][0]["message"]["content"]
        try:
            summaries = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CompressionError(
                f"LLM returned invalid JSON for tool compaction: {exc}\n"
                f"Response: {raw_content[:200]}"
            ) from exc

        if not isinstance(summaries, list) or len(summaries) != len(results_to_compact):
            raise CompressionError(
                f"Expected {len(results_to_compact)} summaries, "
                f"got {len(summaries) if isinstance(summaries, list) else type(summaries).__name__}"
            )

        original_tokens = 0
        compacted_tokens = 0
        edit_commits: list[str] = []
        source_commits: list[str] = []

        for result_ci, summary in zip(results_to_compact, summaries):
            r_meta = result_ci.metadata or {}
            original_tokens += result_ci.token_count
            source_commits.append(result_ci.commit_hash)

            edited = self.tool_result(
                tool_call_id=r_meta.get("tool_call_id", ""),
                name=r_meta.get("name", ""),
                content=str(summary),
                edit=result_ci.commit_hash,
            )
            compacted_tokens += edited.token_count
            edit_commits.append(edited.commit_hash)

        all_tool_names = sorted({n for turn in turns for n in turn.tool_names})

        effective_config = LLMConfig.from_dict(
            self._resolve_llm_config(
                "compress", model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
            )
        )

        return ToolCompactResult(
            edit_commits=tuple(edit_commits),
            source_commits=tuple(source_commits),
            tool_names=tuple(all_tool_names),
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            turn_count=len(turns),
            config=effective_config,
        )

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

    def token_checkpoints(self, limit: int = 100) -> list:
        """API-calibrated token checkpoints, newest first.

        Returns compile records where ``token_source`` starts with ``"api:"``,
        i.e. records created by :meth:`record_usage`. Use ``limit=0`` to
        return all matching records.
        """
        if self._compile_record_repo is None:
            return []
        all_records = list(reversed(self._compile_record_repo.get_all(self._tract_id)))
        api_records = [r for r in all_records if r.token_source.startswith("api:")]
        if limit == 0:
            return api_records
        return api_records[:limit]

    @overload
    def compile(
        self,
        *,
        at_time: datetime | None = ...,
        at_commit: str | None = ...,
        include_edit_annotations: bool = ...,
        include_reasoning: bool = ...,
        order: None = None,
        check_safety: bool = ...,
        strategy: str = ...,
        strategy_k: int = ...,
    ) -> CompiledContext: ...

    @overload
    def compile(
        self,
        *,
        at_time: datetime | None = ...,
        at_commit: str | None = ...,
        include_edit_annotations: bool = ...,
        include_reasoning: bool = ...,
        order: list[str],
        check_safety: bool = ...,
        strategy: str = ...,
        strategy_k: int = ...,
    ) -> tuple[CompiledContext, list[ReorderWarning]]: ...

    def compile(
        self,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
        include_edit_annotations: bool = False,
        include_reasoning: bool = False,
        order: list[str] | None = None,
        check_safety: bool = True,
        strategy: str = "full",
        strategy_k: int = 5,
    ) -> CompiledContext | tuple[CompiledContext, list[ReorderWarning]]:
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
            strategy: Compile strategy. ``"full"`` (default) compiles all
                commits with full content. ``"messages"`` emits only commit
                messages (lightweight). ``"adaptive"`` keeps the last
                ``strategy_k`` commits at full detail and earlier ones as
                messages only.
            strategy_k: Number of recent commits to keep at full detail
                when using the ``"adaptive"`` strategy. Default 5.

        Returns:
            :class:`CompiledContext` when ``order`` is None (default).
            ``(CompiledContext, list[ReorderWarning])`` when ``order`` is provided.
        """
        self._check_open()
        current_head = self.head
        if current_head is None:
            empty = CompiledContext(messages=[], token_count=0, commit_count=0, token_source="")
            if order is not None:
                return empty, []
            return empty

        # Pre-compile middleware (can block via BlockedError)
        self._run_middleware("pre_compile")

        # Strategy kwargs to forward to the compiler
        _strategy_kw = dict(strategy=strategy, strategy_k=strategy_k)

        # If order provided, bypass cache entirely and do a full compile + reorder
        if order is not None:
            result = self._compiler.compile(
                self._tract_id, current_head,
                include_reasoning=include_reasoning, **_strategy_kw,
            )
            warnings = []
            if check_safety:
                from tract.operations.compression import check_reorder_safety
                warnings = check_reorder_safety(order, self._commit_repo, self._blob_repo)
            reordered = self._reorder_compiled(result, order)
            reordered = self._inject_tools(reordered)
            return reordered, warnings

        # Non-default strategy, time-travel, edit annotations, and
        # include_reasoning: always full compile, don't touch snapshot
        _bypass_cache = (
            strategy != "full"
            or at_time is not None
            or at_commit is not None
            or include_edit_annotations
            or include_reasoning
        )
        if _bypass_cache:
            result = self._compiler.compile(
                self._tract_id,
                current_head,
                at_time=at_time,
                at_commit=at_commit,
                include_edit_annotations=include_edit_annotations,
                include_reasoning=include_reasoning,
                **_strategy_kw,
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
                # by cache, not compiler -- fresh Messages have token_count=0)
                cached_core = [(m.role, m.content, m.name, m.tool_calls, m.tool_call_id) for m in result.messages]
                fresh_core = [(m.role, m.content, m.name, m.tool_calls, m.tool_call_id) for m in fresh.messages]
                if cached_core != fresh_core:
                    raise RuntimeError(
                        f"Cache message mismatch: cached {len(result.messages)} msgs, "
                        f"fresh {len(fresh.messages)} msgs"
                    )
                # Skip token_count check when API-sourced (record_usage
                # calibrates to API totals which legitimately differ from tiktoken)
                if not result.token_source.startswith("api:"):
                    if result.token_count != fresh.token_count:
                        raise RuntimeError(
                            f"Cache token mismatch: cached {result.token_count}, "
                            f"fresh {fresh.token_count}"
                        )
                if result.commit_count != fresh.commit_count:
                    raise RuntimeError(
                        f"Cache commit_count mismatch: cached {result.commit_count}, "
                        f"fresh {fresh.commit_count}"
                    )
                if result.generation_configs != fresh.generation_configs:
                    raise RuntimeError(
                        f"Cache generation_configs mismatch: "
                        f"cached {len(result.generation_configs)}, "
                        f"fresh {len(fresh.generation_configs)}"
                    )
                if result.commit_hashes != fresh.commit_hashes:
                    raise RuntimeError(
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

        import json
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
        self._check_open()
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
        self._commit_session()

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

    # ------------------------------------------------------------------
    # Tag system
    # ------------------------------------------------------------------

    def tag(self, target_hash: str, tag_name: str) -> None:
        """Add a mutable tag annotation to a commit.

        Unlike immutable tags set at commit time, annotation tags can be
        added retrospectively.

        Args:
            target_hash: Hash of the commit to tag.
            tag_name: Tag name to add.

        Raises:
            CommitNotFoundError: If the commit doesn't exist.
            TagNotRegisteredError: If strict mode is on and tag is not registered.
        """
        self._check_open()
        commit = self._commit_repo.get(target_hash)
        if commit is None:
            raise CommitNotFoundError(target_hash)
        if self._strict_tags and self._tag_registry_repo is not None:
            if not self._tag_registry_repo.is_registered(self._tract_id, tag_name):
                raise TagNotRegisteredError(tag_name)
        if self._tag_annotation_repo is not None:
            from datetime import timezone
            now = datetime.now(timezone.utc)
            self._tag_annotation_repo.add_tag(
                self._tract_id, target_hash, tag_name, now,
            )
            self._commit_session()

    def untag(self, target_hash: str, tag_name: str) -> bool:
        """Remove a mutable tag annotation from a commit.

        Args:
            target_hash: Hash of the commit to untag.
            tag_name: Tag name to remove.

        Returns:
            True if the tag was removed, False if it didn't exist.
        """
        if self._tag_annotation_repo is None:
            return False
        result = self._tag_annotation_repo.remove_tag(
            self._tract_id, target_hash, tag_name,
        )
        self._commit_session()
        return result

    def get_tags(self, target_hash: str) -> list[str]:
        """Get all tags for a commit (immutable + mutable combined).

        Args:
            target_hash: Hash of the commit.

        Returns:
            Deduplicated list of tag names.
        """
        tags: set[str] = set()
        # Immutable tags from CommitRow
        commit_row = self._commit_repo.get(target_hash)
        if commit_row is not None and commit_row.tags_json:
            tags.update(commit_row.tags_json)
        # Mutable annotation tags
        if self._tag_annotation_repo is not None:
            annotation_tags = self._tag_annotation_repo.get_tags(target_hash)
            tags.update(annotation_tags)
        return sorted(tags)

    def register_tag(self, name: str, description: str | None = None) -> None:
        """Register a new tag name.

        Registered tags can be used with ``tag()`` and ``commit(tags=[...])``.
        In strict mode (default), only registered tags are allowed.

        Args:
            name: Tag name.
            description: Optional description of the tag.
        """
        if self._tag_registry_repo is None:
            return
        from datetime import timezone
        now = datetime.now(timezone.utc)
        self._tag_registry_repo.register(
            self._tract_id, name, description, auto_created=False, created_at=now,
        )
        self._commit_session()

    def list_tags(self) -> list[dict]:
        """List all registered tags with descriptions and usage counts.

        Returns:
            List of dicts with ``name``, ``description``, ``auto_created``,
            and ``count`` keys.
        """
        if self._tag_registry_repo is None:
            return []
        rows = self._tag_registry_repo.list_all(self._tract_id)
        result = []
        for row in rows:
            # Count usage from both immutable and annotation tags
            count = 0
            if self._tag_annotation_repo is not None:
                annotation_hashes = self._tag_annotation_repo.get_commits_by_tag(
                    self._tract_id, row.tag_name,
                )
                count += len(annotation_hashes)
            # Count immutable tags from commits (walk recent history)
            head = self.head
            if head is not None:
                ancestors = self._get_merge_aware_ancestors(head, limit=500)
                for ancestor in ancestors:
                    if ancestor.tags_json and row.tag_name in ancestor.tags_json:
                        count += 1
            result.append({
                "name": row.tag_name,
                "description": row.description,
                "auto_created": bool(row.auto_created),
                "count": count,
            })
        return result

    def query_by_tags(
        self,
        tags: list[str],
        *,
        match: str = "any",
        limit: int = 100,
    ) -> list[CommitInfo]:
        """Query commits by tags (combining immutable and mutable tags).

        Args:
            tags: Tag names to filter by.
            match: ``"any"`` (OR -- commit has at least one tag) or
                ``"all"`` (AND -- commit has every listed tag).
            limit: Maximum results.

        Returns:
            List of :class:`CommitInfo` matching the tag criteria.
        """
        if not tags:
            return []

        # Collect candidate commit hashes from both sources.
        # For "all" match, we first gather candidates with "any" match,
        # then re-check that ALL tags are present across both sources.
        candidate_hashes: set[str] = set()
        collect_match = "any" if match == "all" else match

        # Source 1: Annotation tags
        if self._tag_annotation_repo is not None:
            annotation_matches = self._tag_annotation_repo.get_commits_by_tags(
                self._tract_id, tags, match=collect_match,
            )
            candidate_hashes.update(annotation_matches)

        # Source 2: Immutable tags from CommitRow (walk history)
        head = self.head
        if head is not None:
            ancestors = self._get_merge_aware_ancestors(head, limit=500)
            for row in ancestors:
                if row.tags_json:
                    commit_tags = set(row.tags_json)
                    if commit_tags & set(tags):
                        candidate_hashes.add(row.commit_hash)

        # For "all" match, re-check: each hash must have ALL tags
        # when combining immutable + annotation sources
        if match == "all":
            final_hashes: set[str] = set()
            for h in candidate_hashes:
                all_tags = set(self.get_tags(h))
                if set(tags) <= all_tags:
                    final_hashes.add(h)
            candidate_hashes = final_hashes

        # Convert to CommitInfo, in reverse chronological order
        results: list[CommitInfo] = []
        if head is not None:
            ancestors = self._get_merge_aware_ancestors(head, limit=500)
            for row in ancestors:
                if row.commit_hash in candidate_hashes:
                    results.append(self._commit_engine._row_to_info(row))
                    if len(results) >= limit:
                        break
        return results

    def _seed_base_tags(self) -> None:
        """Seed the tag registry with base tags (idempotent)."""
        if self._tag_registry_repo is None:
            return

        from datetime import timezone
        now = datetime.now(timezone.utc)

        base_tags = {
            "instruction": "System messages / instructions",
            "tool_call": "Messages containing tool calls",
            "tool_result": "Tool result messages",
            "reasoning": "Assistant reasoning without tool calls",
            "revision": "EDIT operations",
            "observation": "User messages with data / observations",
            "decision": "Assistant messages with explicit choices",
            "summary": "Compression output / summaries",
        }
        for tag_name, description in base_tags.items():
            self._tag_registry_repo.register(
                self._tract_id, tag_name, description,
                auto_created=True, created_at=now,
            )
        self._commit_session()

    def _validate_tags(self, tags: list[str]) -> None:
        """Validate tags against registry in strict mode.

        Raises:
            TagNotRegisteredError: If any tag is not registered (reports all
                unregistered tags at once).
        """
        if not self._strict_tags or self._tag_registry_repo is None:
            return
        unregistered = [
            tag for tag in tags
            if not self._tag_registry_repo.is_registered(self._tract_id, tag)
        ]
        if unregistered:
            raise TagNotRegisteredError(unregistered)

    def _classify_tags(
        self,
        content_type: str,
        *,
        role: str | None = None,
        operation: CommitOperation | None = None,
        metadata: dict | None = None,
    ) -> list[str]:
        """Heuristic-based tag classification (no LLM call).

        Args:
            content_type: The content type discriminator.
            role: The message role (if dialogue).
            operation: The commit operation.
            metadata: The commit metadata.

        Returns:
            Deduplicated list of tag names.
        """
        tags: list[str] = []

        # Classify based on content type and role
        if content_type == "instruction" or role == "system":
            tags.append("instruction")
        if role == "assistant":
            if metadata and metadata.get("tool_calls"):
                tags.append("tool_call")
            else:
                tags.append("reasoning")
        if role == "user":
            if metadata and metadata.get("tool_call_id"):
                tags.append("tool_result")
        if role == "tool":
            tags.append("tool_result")
        if content_type == "tool_io":
            if "tool_call" not in tags and "tool_result" not in tags:
                tags.append("tool_call")
        if operation == CommitOperation.EDIT:
            tags.append("revision")
        if content_type == "session" or (content_type and "session" in content_type):
            tags.append("observation")

        # Deduplicate preserving order
        seen: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags

    def _auto_classify(
        self,
        content_type: str,
        text: str,
        *,
        role: str | None = None,
        operation: CommitOperation | None = None,
        metadata: dict | None = None,
    ) -> tuple[str, list[str]]:
        """Generate a commit message and auto-classify tags.

        Combines ``_auto_message()`` (may use LLM) with ``_classify_tags()``
        (pure heuristic, no LLM).

        Args:
            content_type: The content type discriminator.
            text: The text content.
            role: The message role (if dialogue).
            operation: The commit operation.
            metadata: The commit metadata.

        Returns:
            Tuple of (message, tags).
        """
        tags = self._classify_tags(
            content_type, role=role, operation=operation, metadata=metadata,
        )
        message = self._auto_message(content_type, text)
        return message, tags

    def _enrich_with_priorities(self, entries: list[CommitInfo]) -> list[CommitInfo]:
        """Enrich CommitInfo entries with effective_priority.

        Resolves each commit's effective priority by checking explicit
        annotations first, then falling back to DEFAULT_TYPE_PRIORITIES.
        """
        if not entries:
            return entries
        hashes = [e.commit_hash for e in entries]
        annotations = self._annotation_repo.batch_get_latest(hashes)
        enriched: list[CommitInfo] = []
        for entry in entries:
            ann = annotations.get(entry.commit_hash)
            if ann is not None:
                priority = ann.priority
            else:
                priority = DEFAULT_TYPE_PRIORITIES.get(
                    entry.content_type, Priority.NORMAL,
                )
            enriched.append(entry.model_copy(update={"effective_priority": priority.value}))
        return enriched

    def _get_merge_aware_ancestors(
        self,
        start_hash: str,
        limit: int | None = None,
        *,
        op_filter: object | None = None,
    ) -> list:
        """Walk ancestry from start_hash following ALL parents (primary + merge).

        Returns commits in reverse chronological order (newest first).
        Falls back to get_ancestors() when no parent_repo is available.

        Performance: when *limit* is provided and no *op_filter*, BFS stops
        after visiting ``limit * 3`` nodes (heuristic overshoot to ensure
        correct chronological ordering after sort+truncate).  This avoids
        materialising the entire DAG for common bounded queries.
        """
        from collections import deque

        if self._parent_repo is None:
            return list(
                self._commit_repo.get_ancestors(
                    start_hash, limit=limit, op_filter=op_filter,
                )
            )

        # Early-exit cap: when limit is set and there is no op_filter that
        # could discard most rows, we stop BFS after collecting enough nodes
        # to guarantee the top-limit results by created_at.  The 3x
        # multiplier accounts for BFS visiting nodes in graph order (not
        # chronological order).
        visit_cap = (limit * 3) if (limit is not None and op_filter is None) else None

        # BFS collecting reachable commits
        visited: set[str] = set()
        queue: deque[str] = deque([start_hash])
        all_rows = []

        while queue:
            if visit_cap is not None and len(all_rows) >= visit_cap:
                break
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            row = self._commit_repo.get(current)
            if row is None:
                continue
            all_rows.append(row)
            # Follow primary parent
            if row.parent_hash:
                queue.append(row.parent_hash)
            # Follow merge parents
            for extra in self._parent_repo.get_parents(current):
                if extra not in visited:
                    queue.append(extra)

        # Sort newest-first by created_at
        all_rows.sort(key=lambda r: r.created_at, reverse=True)

        # Apply op_filter
        if op_filter is not None:
            all_rows = [r for r in all_rows if r.operation == op_filter]

        # Apply limit
        if limit is not None:
            all_rows = all_rows[:limit]

        return all_rows

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
            tags: If set, only include commits that have these tags
                (combining immutable commit tags and mutable annotation tags).
            tag_match: ``"any"`` (OR -- at least one tag matches) or
                ``"all"`` (AND -- all tags must match).  Default ``"any"``.

        Returns:
            List of :class:`CommitInfo` in reverse chronological order
            (newest first).  Empty list if no commits.
        """
        current_head = self.head
        if current_head is None:
            return []

        if tags is None:
            # No tag filter -- use fast path
            ancestors = self._get_merge_aware_ancestors(
                current_head, limit=limit, op_filter=op_filter,
            )
            entries = [self._commit_engine._row_to_info(row) for row in ancestors]
            return self._enrich_with_priorities(entries)

        # Tag filtering: walk more commits and filter
        ancestors = self._get_merge_aware_ancestors(
            current_head, limit=500, op_filter=op_filter,
        )
        results: list[CommitInfo] = []
        tag_set = set(tags)
        for row in ancestors:
            # Combine immutable + mutable tags
            commit_tags = set(row.tags_json) if row.tags_json else set()
            if self._tag_annotation_repo is not None:
                annotation_tags = self._tag_annotation_repo.get_tags(row.commit_hash)
                commit_tags.update(annotation_tags)

            if tag_match == "any":
                if commit_tags & tag_set:
                    results.append(self._commit_engine._row_to_info(row))
            else:  # "all"
                if tag_set <= commit_tags:
                    results.append(self._commit_engine._row_to_info(row))

            if len(results) >= limit:
                break
        return self._enrich_with_priorities(results)

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

        Args:
            content: Substring match in commit content text.
            pattern: Regex match in commit content text.
            tag: Match commits that have this tag (immutable or mutable).
            content_type: Match commits with this content type.
            metadata_key: Match commits that have this key in metadata.
            metadata_value: When used with ``metadata_key``, match commits
                where ``metadata[metadata_key] == metadata_value``.
            branch: Search a specific branch.  Defaults to current HEAD.
            limit: Maximum number of results.  Default 50.

        Returns:
            List of matching :class:`CommitInfo` in reverse chronological
            order (newest first).
        """
        import re

        # Resolve starting commit hash
        if branch is not None:
            start_hash = self._ref_repo.get_branch(self._tract_id, branch)
            if start_hash is None:
                raise BranchNotFoundError(branch)
        else:
            start_hash = self.head

        if start_hash is None:
            return []

        # Pre-compile regex if provided
        compiled_re = re.compile(pattern) if pattern is not None else None

        # Walk a generous window of ancestors for filtering
        scan_limit = max(limit * 10, 500)
        ancestors = self._get_merge_aware_ancestors(start_hash, limit=scan_limit)

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
                if self._tag_annotation_repo is not None:
                    commit_tags.update(
                        self._tag_annotation_repo.get_tags(row.commit_hash)
                    )
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
                    # Blob payload unreadable (corrupt data or detached instance);
                    # skip rather than failing the entire search.
                    logger.debug(
                        "Skipping blob %s: payload unreadable", row.content_hash,
                        exc_info=True,
                    )
                    continue

                if content is not None and content not in blob_text:
                    continue
                if compiled_re is not None and not compiled_re.search(blob_text):
                    continue

            info = self._commit_engine._row_to_info(row)
            results.append(info)

            if len(results) >= limit:
                break

        return self._enrich_with_priorities(results)

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
        """Search commits and return the first match, or ``None``.

        Accepts the same filters as :meth:`find`.  Equivalent to
        ``find(..., limit=1)[0]`` but returns ``None`` instead of
        raising on empty results.

        Returns:
            The first matching :class:`CommitInfo`, or ``None``.
        """
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

    def skipped(self, limit: int = 100) -> list[CommitInfo]:
        """Return commits with effective priority SKIP.

        These commits are excluded from :meth:`compile` output.

        Args:
            limit: Maximum number of commits to scan. Default 100.

        Returns:
            List of :class:`CommitInfo` with ``effective_priority == "skip"``,
            in reverse chronological order.
        """
        entries = self.log(limit=limit)
        return [e for e in entries if e.effective_priority == Priority.SKIP.value]

    def pinned(self, limit: int = 100) -> list[CommitInfo]:
        """Return commits with effective priority PINNED.

        These commits are always included in :meth:`compile` output and
        preserved during compression.

        Args:
            limit: Maximum number of commits to scan. Default 100.

        Returns:
            List of :class:`CommitInfo` with ``effective_priority == "pinned"``,
            in reverse chronological order.
        """
        entries = self.log(limit=limit)
        return [e for e in entries if e.effective_priority == Priority.PINNED.value]

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

    def health(self) -> HealthReport:
        """Run health checks on this tract's DAG.

        Validates blob integrity, parent references, branch HEAD validity,
        and identifies unreachable (orphaned) commits.

        Returns:
            :class:`HealthReport` with validation results and any warnings.
        """
        from tract.operations.health import check_health

        return check_health(
            self._tract_id,
            self._commit_repo,
            self._blob_repo,
            self._ref_repo,
            self._parent_repo,
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

    def compare(
        self,
        branch_a: str | None = None,
        branch_b: str | None = None,
        *,
        commit_a: str | None = None,
        commit_b: str | None = None,
    ) -> "DiffResult":
        """Compare compiled contexts between two branches or commits without switching HEAD.

        Unlike :meth:`diff` which compares sequential commits on the current branch,
        ``compare()`` works across branches.  Useful for A/B variant comparison,
        inspecting divergent context windows, or auditing branch differences before
        a merge.

        Exactly one of ``branch_a``/``commit_a`` may be provided for side A,
        and exactly one of ``branch_b``/``commit_b`` for side B.

        Args:
            branch_a: First branch name.  Resolved to the branch tip.
                Defaults to the current branch when *commit_a* is also None.
            branch_b: Second branch name.  Resolved to the branch tip.
            commit_a: Specific commit hash (or prefix / ref) for side A.
                Mutually exclusive with *branch_a*.
            commit_b: Specific commit hash (or prefix / ref) for side B.
                Mutually exclusive with *branch_b*.

        Returns:
            :class:`DiffResult` comparing the two compiled contexts.

        Raises:
            ValueError: If both branch and commit are given for the same side,
                or if no target is specified for side B.
            TraceError: If the current branch has no commits (when defaulting side A).
            CommitNotFoundError: If a reference cannot be resolved.
        """
        from tract.operations.diff import compute_diff

        # --- Validate mutual exclusivity ---
        if branch_a is not None and commit_a is not None:
            raise ValueError("Cannot specify both branch_a and commit_a; use one or the other.")
        if branch_b is not None and commit_b is not None:
            raise ValueError("Cannot specify both branch_b and commit_b; use one or the other.")

        # --- Resolve side A ---
        if commit_a is not None:
            hash_a = self.resolve_commit(commit_a)
        elif branch_a is not None:
            hash_a = self.resolve_commit(branch_a)
        else:
            # Default to current HEAD
            current_head = self.head
            if current_head is None:
                raise TraceError("No commits on current branch to use as side A")
            hash_a = current_head

        # --- Resolve side B ---
        if commit_b is not None:
            hash_b = self.resolve_commit(commit_b)
        elif branch_b is not None:
            hash_b = self.resolve_commit(branch_b)
        else:
            raise ValueError("Must specify branch_b or commit_b for the comparison target.")

        # --- Compile both sides without switching HEAD ---
        compiled_a = self._compile_at(hash_a)
        compiled_b = self._compile_at(hash_b)

        return compute_diff(
            commit_a_hash=hash_a,
            commit_b_hash=hash_b,
            messages_a=compiled_a.messages,
            messages_b=compiled_b.messages,
            configs_a=compiled_a.generation_configs,
            configs_b=compiled_b.generation_configs,
        )

    def edit_history(self, commit_hash: str) -> list[CommitInfo]:
        """Get the full edit chain for a commit.

        Returns a chronological list starting with the original commit,
        followed by each EDIT in the order they were created.

        Args:
            commit_hash: A commit hash, branch name, or hash prefix.
                Can be the original commit or any of its edits.

        Returns:
            List of :class:`CommitInfo` in chronological order:
            ``[original, edit1, edit2, ...]``.  If the commit has never
            been edited, returns a single-element list.

        Raises:
            CommitNotFoundError: If the commit cannot be resolved.
        """
        resolved = self.resolve_commit(commit_hash)
        row = self._commit_repo.get(resolved)
        if row is None:
            raise CommitNotFoundError(resolved)

        # If the resolved commit is itself an edit, follow to the original
        original_hash = row.edit_target if row.edit_target else resolved

        rows = self._commit_repo.get_edits_for(original_hash, self._tract_id)
        if not rows:
            raise CommitNotFoundError(resolved)

        return [self._commit_engine._row_to_info(r) for r in rows]

    def restore(
        self,
        commit_hash: str,
        version: int = 0,
        *,
        message: str | None = None,
    ) -> CommitInfo:
        """Restore a previous version of a commit by creating a new EDIT.

        Looks up the edit history for the given commit, picks the version
        at index ``version``, and creates a new EDIT commit with that
        version's content.

        Args:
            commit_hash: A commit hash, branch name, or hash prefix.
                Can be the original commit or any of its edits.
            version: Zero-based index into the edit history
                (0 = original, 1 = first edit, etc.).
            message: Optional commit message.  Defaults to
                ``"restore to version {version}"``.

        Returns:
            :class:`CommitInfo` for the new EDIT commit.

        Raises:
            CommitNotFoundError: If the commit cannot be resolved.
            IndexError: If ``version`` is out of range.
        """
        history = self.edit_history(commit_hash)
        if version < 0 or version >= len(history):
            raise IndexError(
                f"Version {version} out of range (0..{len(history) - 1})"
            )

        source = history[version]
        original = history[0]

        if message is None:
            message = f"restore to version {version}"

        # Get the content blob to reconstruct the content model
        blob_row = self._blob_repo.get(source.content_hash)
        if blob_row is None:
            raise CommitNotFoundError(source.commit_hash)

        # Reconstruct the content model from the blob
        import json
        content_data = json.loads(blob_row.payload_json)
        content = validate_content(
            content_data, custom_registry=self._custom_type_registry
        )

        return self.commit(
            content,
            operation=CommitOperation.EDIT,
            edit_target=original.commit_hash,
            message=message,
            generation_config=source.generation_config.to_dict()
            if source.generation_config
            else None,
        )

    def query_by_config(
        self,
        field_or_config: str | LLMConfig | None = None,
        operator: Operator | None = None,
        value: Any = None,
        *,
        conditions: list[tuple[str, Operator, Any]] | None = None,
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
        self._check_open()
        from tract.operations.navigation import reset as _reset

        resolved = self.resolve_commit(target)
        result = _reset(resolved, mode, self._tract_id, self._ref_repo)  # type: ignore[arg-type]  # resolve_commit narrows str
        self._commit_session()
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
        self._check_open()
        from tract.operations.navigation import checkout as _checkout

        commit_hash, _is_detached = _checkout(
            target, self._tract_id, self._commit_repo, self._ref_repo
        )
        self._commit_session()
        if self._config_index is not None:
            self._config_index.invalidate()
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
        self._check_open()
        from tract.operations.branch import create_branch

        result = create_branch(
            name,
            self._tract_id,
            self._ref_repo,
            self._commit_repo,
            source=source,
            switch=switch,
        )
        self._commit_session()
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
        self._check_open()
        # Validate that target is a branch
        branch_hash = self._ref_repo.get_branch(self._tract_id, target)
        if branch_hash is None:
            raise BranchNotFoundError(target)

        from tract.operations.navigation import checkout as _checkout

        commit_hash, _is_detached = _checkout(
            target, self._tract_id, self._commit_repo, self._ref_repo
        )
        self._commit_session()
        if self._config_index is not None:
            self._config_index.invalidate()
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
        self._check_open()
        from tract.operations.branch import delete_branch

        delete_branch(
            name,
            self._tract_id,
            self._ref_repo,
            self._commit_repo,
            self._parent_repo,
            force=force,
        )
        self._commit_session()

    # ------------------------------------------------------------------
    # Snapshot system
    # ------------------------------------------------------------------

    def snapshot(self, label: str = "", *, metadata: dict | None = None) -> str:
        """Create a named snapshot (restore point) at the current HEAD.

        Snapshots are implemented as specially-tagged commits with metadata.
        They record the current state for later restoration.

        Args:
            label: Human-readable snapshot label (e.g., "before-merge",
                "pre-compress").
            metadata: Optional extra metadata to store with the snapshot.

        Returns:
            The snapshot tag name (e.g., ``"snapshot:before-merge:abc123"``).

        Raises:
            TraceError: If there is no HEAD (empty tract).
        """
        import time

        current_head = self.head
        if current_head is None:
            raise TraceError("Cannot create snapshot: no commits yet")

        head_short = current_head[:7]
        timestamp = int(time.time())
        tag_name = (
            f"snapshot:{label}:{head_short}"
            if label
            else f"snapshot:{timestamp}:{head_short}"
        )

        # Gather lightweight state (avoid expensive compile via status())
        branch_name = self._ref_repo.get_current_branch(self._tract_id)

        # Build snapshot metadata
        snap_meta: dict = {
            "snapshot": True,
            "label": label,
            "head": current_head,
            "branch": branch_name,
            "timestamp": timestamp,
            **(metadata or {}),
        }

        # Register the tag so strict-mode allows it
        self.register_tag(tag_name, description=f"Snapshot: {label or 'unnamed'}")

        # Store as a metadata commit with the snapshot tag
        from tract.models.content import MetadataContent

        self.commit(
            MetadataContent(kind="snapshot", data=snap_meta),
            message=f"Snapshot: {label or 'unnamed'}",
            metadata=snap_meta,
            tags=[tag_name],
        )

        return tag_name

    def list_snapshots(self) -> list[dict]:
        """List all snapshots for this tract.

        Returns:
            List of snapshot metadata dicts, newest first.  Each dict has
            keys: ``tag``, ``label``, ``head``, ``branch``, ``timestamp``,
            ``hash``.
        """
        snapshots: list[dict] = []
        for entry in self.log(limit=500):
            meta = entry.metadata or {}
            if meta.get("snapshot"):
                snapshots.append({
                    "tag": next(
                        (t for t in entry.tags if t.startswith("snapshot:")), ""
                    ),
                    "label": meta.get("label", ""),
                    "head": meta.get("head", ""),
                    "branch": meta.get("branch", ""),
                    "timestamp": meta.get("timestamp", 0),
                    "hash": entry.commit_hash,
                })
        return snapshots

    def restore_snapshot(
        self,
        tag_or_label: str,
        *,
        create_branch: bool = True,
    ) -> str:
        """Restore to a previously created snapshot point.

        Args:
            tag_or_label: Snapshot tag name or label substring to match.
            create_branch: If ``True`` (default), create a recovery branch
                at the snapshot point (safe -- no history loss).  If
                ``False``, reset HEAD directly.

        Returns:
            The commit hash restored to.

        Raises:
            ValueError: If no matching snapshot is found.
        """
        snapshots = self.list_snapshots()
        match: dict | None = None
        for snap in snapshots:
            if snap["tag"] == tag_or_label or tag_or_label in snap.get("label", ""):
                match = snap
                break

        if match is None:
            raise ValueError(f"Snapshot not found: {tag_or_label}")

        target_head: str = match["head"]

        if create_branch:
            branch_name = f"restore/{match['label'] or 'snapshot'}"
            self.branch(branch_name, source=target_head)
            self.switch(branch_name)
        else:
            self.reset(target_head)

        return target_head

    def export_state(self, *, include_blobs: bool = True) -> dict:
        """Export the current branch's DAG as a portable JSON-serializable dict.

        Creates a snapshot of all commits reachable from HEAD with their
        content, metadata, annotations, and branch info. The result can be
        saved to a file and loaded into a different tract via
        :meth:`load_state`.

        Note:
            The exported dict contains full commit details, but
            :meth:`load_state` only replays content payloads — it does not
            reconstruct the original DAG. See :meth:`load_state` for the
            list of what is and is not preserved on import.

        Args:
            include_blobs: If True (default), include full content payloads.
                If False, include only commit metadata (smaller but not
                restorable).

        Returns:
            A dict with keys: version, tract_id, branch, head, commits,
            branches, exported_at.
        """
        from tract.operations.ancestry import walk_ancestry

        head = self.head
        if head is None:
            return {
                "version": 1,
                "tract_id": self._tract_id,
                "branch": self.current_branch,
                "head": None,
                "commits": [],
                "branches": {},
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }

        # Walk full ancestry from HEAD
        commits_data = []
        ancestry = walk_ancestry(
            self._commit_repo, self._blob_repo, head,
            parent_repo=self._parent_repo,
        )

        for commit_row in ancestry:
            entry: dict = {
                "hash": commit_row.commit_hash,
                "content_type": commit_row.content_type,
                "operation": commit_row.operation.value if hasattr(commit_row.operation, "value") else str(commit_row.operation),
                "message": commit_row.message,
                "metadata": commit_row.metadata_json,
                "created_at": commit_row.created_at.isoformat() if commit_row.created_at else None,
            }

            # Parent hashes -- get_parents returns list[str]
            if self._parent_repo:
                parents = self._parent_repo.get_parents(commit_row.commit_hash)
                entry["parents"] = parents
            else:
                entry["parents"] = [commit_row.parent_hash] if commit_row.parent_hash else []

            # Blob content
            if include_blobs:
                blob = self._blob_repo.get(commit_row.content_hash)
                if blob:
                    entry["content_hash"] = commit_row.content_hash
                    entry["payload"] = blob.payload_json
                else:
                    entry["content_hash"] = commit_row.content_hash
                    entry["payload"] = None
            else:
                entry["content_hash"] = commit_row.content_hash

            # Annotations
            ann = self._annotation_repo.get_latest(commit_row.commit_hash)
            if ann:
                entry["priority"] = ann.priority.value if hasattr(ann.priority, "value") else str(ann.priority)

            commits_data.append(entry)

        # Branch info
        branches = {}
        for branch_info in self.list_branches():
            branches[branch_info.name] = branch_info.commit_hash

        return {
            "version": 1,
            "tract_id": self._tract_id,
            "branch": self.current_branch,
            "head": head,
            "commits": commits_data,
            "branches": branches,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_state(self, state: dict) -> int:
        """Load commits from an exported state dict into this tract.

        Replays the exported commits as new APPEND commits on the current
        branch. Does NOT overwrite existing commits.

        This is a **content replay** tool, not a structural backup/restore.

        Preserved on import:
            - Content payloads (the actual data in each commit)
            - ``content_type``
            - ``metadata``
            - ``message``
            - ``priority`` annotations (non-normal values)

        Not preserved on import:
            - DAG structure and parent links (commits are re-appended linearly)
            - Branches (all commits land on the current branch)
            - Operation types (EDIT operations become APPENDs)
            - Original timestamps (commits get new ``created_at`` values)
            - Tags
            - ``edit_target`` relationships
            - Original commit hashes

        Args:
            state: A dict previously returned by :meth:`export_state`.

        Returns:
            Number of commits loaded.

        Raises:
            ValueError: If the state dict is invalid or version unsupported.
        """
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Invalid or unsupported export state (expected version=1)")

        commits = state.get("commits", [])
        if not commits:
            return 0

        loaded = 0
        for entry in commits:
            payload = entry.get("payload")
            if payload is None:
                continue  # skip commits without content (include_blobs=False)

            content_type = entry.get("content_type", "")
            message = entry.get("message")
            metadata_json = entry.get("metadata")

            # Parse metadata
            metadata = None
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                except (ValueError, TypeError):
                    metadata = None

            # Reconstruct content from payload
            try:
                payload_dict = json.loads(payload) if isinstance(payload, str) else payload
            except (ValueError, TypeError):
                continue

            # Ensure content_type is present in payload dict
            if "content_type" not in payload_dict:
                payload_dict["content_type"] = content_type

            # Use validate_content to reconstruct the proper content model
            try:
                content = validate_content(
                    payload_dict,
                    custom_registry=self._custom_type_registry,
                )
            except Exception:
                # Content validation failed (unrecognized type, schema mismatch,
                # or corrupt payload); skip this entry rather than aborting import.
                logger.debug(
                    "load_state: skipping entry with content_type=%r: validation failed",
                    content_type, exc_info=True,
                )
                continue

            info = self.commit(
                content,
                message=message or f"imported: {content_type}",
                metadata=metadata,
            )

            # Restore priority annotation if present
            priority_val = entry.get("priority")
            if priority_val and priority_val != "normal":
                try:
                    priority = Priority(priority_val)
                    self.annotate(info.commit_hash, priority, reason="imported")
                except (ValueError, KeyError):
                    pass

            loaded += 1

        return loaded

    @property
    def llm_client(self) -> LLMClient | None:
        """The configured LLM client, or ``None`` if not configured.

        Use this for advanced agent patterns that need raw LLM calls
        without committing to conversation history (e.g. custom
        multi-turn agent loops, evaluation pipelines).

        For most use cases, prefer :meth:`chat`, :meth:`generate`,
        or :meth:`run`.

        Returns:
            The LLM client instance, or ``None``.
        """
        return self._llm_client

    @property
    def default_config(self) -> LLMConfig | None:
        """The default LLM configuration, or ``None`` if not set."""
        return self._default_config

    @property
    def retry_config(self) -> RetryConfig | None:
        """The retry configuration for LLM calls, or ``None``."""
        return self._retry_config

    @property
    def commit_reasoning(self) -> bool:
        """Whether reasoning traces are committed during agent loops."""
        return self._commit_reasoning

    @property
    def tool_summarization_config(self) -> ToolSummarizationConfig | None:
        """Tool summarization configuration, or ``None`` if disabled."""
        return self._tool_summarization_config

    def configure_llm(
        self,
        client: LLMClient,
        *,
        resolver: ResolverCallable | None = None,
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
        self._log_config_change("llm_client", source="api")

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
                Valid names: ``"chat"``, ``"merge"``, ``"compress"``.

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
            self._log_config_change(
                "operation_config",
                config_json=self._serialize_operation_configs(),
                source="api",
            )
            return
        # Keyword path: validate and construct OperationConfigs
        for name, cfg in operation_configs.items():
            if not isinstance(cfg, LLMConfig):
                raise TypeError(
                    f"Expected LLMConfig for '{name}', "
                    f"got {type(cfg).__name__}"
                )
            if name not in _VALID_OPERATION_NAMES:
                raise ValueError(
                    f"Unknown operation '{name}'. "
                    f"Valid operations: {', '.join(sorted(_VALID_OPERATION_NAMES))}"
                )
        # Merge with existing: only replace fields that are provided
        self._operation_configs = replace(self._operation_configs, **operation_configs)
        # Log config change
        self._log_config_change(
            "operation_config",
            config_json=self._serialize_operation_configs(),
            source="api",
        )

    @property
    def operation_configs(self) -> OperationConfigs:
        """Current per-operation LLM configurations (read-only, frozen)."""
        return self._operation_configs

    def configure_clients(
        self,
        _clients: OperationClients | None = None,
        /,
        **operation_clients: LLMClient,
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
                Valid names: ``"chat"``, ``"merge"``, ``"compress"``.

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
            self._log_config_change("operation_client", source="api")
            return
        for name in operation_clients:
            if name not in _VALID_OPERATION_NAMES:
                raise ValueError(
                    f"Unknown operation '{name}'. "
                    f"Valid operations: {', '.join(sorted(_VALID_OPERATION_NAMES))}"
                )
        self._operation_clients = replace(self._operation_clients, **operation_clients)
        # Log config change
        self._log_config_change("operation_client", source="api")

    @property
    def operation_clients(self) -> OperationClients:
        """Current per-operation LLM client overrides (read-only, frozen)."""
        return self._operation_clients

    def _log_config_change(
        self,
        change_type: str,
        *,
        change_key: str | None = None,
        config_json: str | None = None,
        previous_json: str | None = None,
        source: str | None = None,
    ) -> None:
        """Log a configuration change to the audit trail.

        No-op when persistence repo is not available.
        """
        repo = self._persistence_repo
        if repo is None:
            return
        from tract.storage.schema import ConfigChangeRow

        entry = ConfigChangeRow(
            tract_id=self._tract_id,
            change_type=change_type,
            change_key=change_key,
            config_json=config_json,
            previous_json=previous_json,
            source=source,
            created_at=datetime.now(timezone.utc),
        )
        repo.save_config_change(entry)
        self._commit_session()

    def _serialize_operation_configs(self) -> str | None:
        """Serialize current operation configs to JSON string."""
        result = {}
        for f in dc_fields(self._operation_configs):
            val = getattr(self._operation_configs, f.name)
            if val is not None:
                result[f.name] = val.to_dict()
        return json.dumps(result) if result else None

    def config_history(
        self,
        *,
        change_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get the configuration change audit trail.

        Returns a list of dicts, each with keys: change_type, change_key,
        config_json, previous_json, source, created_at.
        Ordered by most recent first.

        Args:
            change_type: Filter by change type (e.g. "operation_config",
                "llm_client", "operation_client", "prompts").
            limit: Maximum number of entries to return.
        """
        repo = self._persistence_repo
        if repo is None:
            return []
        rows = repo.get_config_changes(
            self._tract_id, change_type=change_type, limit=limit,
        )
        return [
            {
                "change_type": row.change_type,
                "change_key": row.change_key,
                "config_json": row.config_json,
                "previous_json": row.previous_json,
                "source": row.source,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    def configure_prompts(
        self,
        _prompts: OperationPrompts | None = None,
        /,
        **prompt_overrides: str,
    ) -> None:
        """Set per-operation prompt overrides.

        Accepts either an OperationPrompts instance or keyword arguments.

        Args:
            _prompts: OperationPrompts instance with typed fields.
            **prompt_overrides: Operation name -> prompt string mappings.
                Valid names: ``"compress"``, ``"merge"``,
                ``"message"``, ``"commit_message"``.

        Raises:
            TypeError: If both positional and keyword arguments provided.
            ValueError: If an unknown operation name is given.
        """
        if _prompts is not None and prompt_overrides:
            raise TypeError(
                "Cannot mix positional OperationPrompts with keyword arguments"
            )
        if _prompts is not None:
            if not isinstance(_prompts, OperationPrompts):
                raise TypeError(
                    f"Expected OperationPrompts, got {type(_prompts).__name__}"
                )
            self._operation_prompts = _prompts
            self._log_config_change(
                "prompts",
                config_json=self._serialize_prompts(),
                source="api",
            )
            return
        for name, val in prompt_overrides.items():
            if not isinstance(val, str):
                raise TypeError(
                    f"Expected str for '{name}', got {type(val).__name__}"
                )
            if name not in _VALID_PROMPT_NAMES:
                raise ValueError(
                    f"Unknown operation '{name}'. "
                    f"Valid operations: {', '.join(sorted(_VALID_PROMPT_NAMES))}"
                )
        self._operation_prompts = replace(self._operation_prompts, **prompt_overrides)
        self._log_config_change(
            "prompts",
            config_json=self._serialize_prompts(),
            source="api",
        )

    @property
    def operation_prompts(self) -> OperationPrompts:
        """Current per-operation prompt overrides (read-only, frozen)."""
        return self._operation_prompts

    def _serialize_prompts(self) -> str | None:
        """Serialize current operation prompts to JSON string."""
        result = {}
        for f in dc_fields(self._operation_prompts):
            val = getattr(self._operation_prompts, f.name)
            if val is not None:
                result[f.name] = val
        return json.dumps(result) if result else None

    def _resolve_llm_client(self, operation: str) -> LLMClient:
        """Resolve the LLM client for a given operation.

        Two-level lookup: per-operation client > tract-level default.

        Args:
            operation: Operation name (``"chat"``, ``"merge"``, etc.).

        Returns:
            The resolved LLM client.

        Raises:
            RuntimeError: If no client is configured at any level.
        """
        client = getattr(self._operation_clients, operation, None)
        if client is not None:
            return client
        if self._llm_client is not None:
            return self._llm_client
        raise RuntimeError(
            "No LLM client configured. Pass api_key= to Tract.open() "
            "or call configure_llm(client)."
        )

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
        return self._llm_client is not None

    def _resolve_resolver(
        self, resolver: ResolverCallable | str | None, operation: str = "merge",
    ) -> ResolverCallable | None:
        """Resolve a resolver argument to a callable.

        Handles three cases:
        1. ``resolver="llm"`` — build an :class:`OpenAIResolver` from the
           configured LLM client for *operation*.
        2. ``resolver=None`` — fall back to ``_default_resolver`` if set.
        3. Anything else (callable) — return as-is.

        Returns:
            A resolver callable or *None*.
        """
        if resolver == "llm":
            if not self._has_llm_client(operation):
                raise RuntimeError(
                    "resolver='llm' requires an LLM client.  "
                    "Pass api_key= to Tract.open() or call configure_llm()."
                )
            from tract.llm.resolver import OpenAIResolver

            llm_cfg = self._resolve_llm_config(operation) or {}
            return OpenAIResolver(
                self._resolve_llm_client(operation),
                model=llm_cfg.get("model"),
                temperature=llm_cfg.get("temperature", 0.3),
                max_tokens=llm_cfg.get("max_tokens", 2048),
            )
        if resolver is None:
            return getattr(self, "_default_resolver", None)
        return resolver

    def _auto_message(self, content_type: str, text: str) -> str:
        """Generate a commit message, using LLM summarization when available.

        Falls back to truncation when:
        - ``auto_message`` was not enabled on ``Tract.open()``
        - No LLM client is available (per-operation or default)
        - Currently inside a batch (``_in_batch``)
        - The LLM call fails for any reason

        Args:
            content_type: The content type discriminator.
            text: The text content to summarize.

        Returns:
            A concise one-sentence commit message, or a truncated preview.
        """
        if (
            not self._auto_message_enabled
            or self._in_batch
            or not self._has_llm_client("message")
        ):
            return _fallback_message(content_type, text)

        try:
            from tract.prompts.commit_message import (
                COMMIT_MESSAGE_SYSTEM,
                build_commit_message_prompt,
            )

            client = self._resolve_llm_client("message")
            llm_kwargs = self._resolve_llm_config(
                "message", temperature=0.0, max_tokens=200,
            )
            messages = [
                {"role": "system", "content": COMMIT_MESSAGE_SYSTEM},
                {"role": "user", "content": build_commit_message_prompt(content_type, text)},
            ]
            response = client.chat(messages, **llm_kwargs)
            summary = self._extract_content(response, client=client).strip()
            # Strip <think>...</think> tags from models that emit reasoning.
            # Also strip unclosed <think> blocks (model hit max_tokens mid-thought).
            import re as _re
            summary = _re.sub(r"<think>.*?</think>\s*", "", summary, flags=_re.DOTALL).strip()
            summary = _re.sub(r"<think>.*", "", summary, flags=_re.DOTALL).strip()
            if not summary:
                return _fallback_message(content_type, text)
            # Cap at 100 chars for safety
            if len(summary) > 100:
                summary = summary[:97] + "..."
            return summary
        except Exception:
            return _fallback_message(content_type, text)

    def merge(
        self,
        source_branch: str,
        *,
        resolver: ResolverCallable | str | None = None,
        strategy: str | MergeStrategy = "auto",
        no_ff: bool = False,
        auto_commit: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        delete_branch: bool | None = None,
        message: str | None = None,
    ) -> MergeResult:
        """Merge a source branch into the current branch.

        Args:
            source_branch: Name of the branch to merge.
            resolver: Optional conflict resolver (ResolverCallable).
                Falls back to self._default_resolver if configured.
            strategy: Merge strategy: ``"auto"`` (default), ``"ours"``,
                ``"theirs"``, or ``"semantic"``.  With ``"ours"``, conflicts
                are resolved by keeping the current branch's version.  With
                ``"theirs"``, the source branch's version wins.
            no_ff: If True, always create a merge commit (no fast-forward).
            auto_commit: If True, auto-commit even with resolved conflicts.
            model: Override model for the default resolver.
            temperature: Override temperature for the default resolver.
            max_tokens: Override max_tokens for the default resolver.
            llm_config: Full LLMConfig override for this call.
            delete_branch: If True, delete the source branch after merge.
                If None (default), uses ``config.delete_branch_on_merge``.
            message: Optional merge commit message. If not provided, a
                default message is generated.

        Returns:
            :class:`MergeResult`.
        """
        self._check_open()
        from tract.operations.merge import merge_branches

        # Resolve delete_branch from config default
        if delete_branch is None:
            delete_branch = self.config.delete_branch_on_merge

        # Determine resolver (handles "llm" string, None -> default, callables)
        effective_resolver = self._resolve_resolver(resolver, "merge")

        # If using default resolver AND per-call config overrides exist,
        # create a tailored resolver with those overrides.
        if effective_resolver is getattr(self, "_default_resolver", None):
            merge_config = self._resolve_llm_config(
                "merge", model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
            )
            if merge_config and self._has_llm_client("merge"):
                from tract.llm.resolver import OpenAIResolver

                effective_resolver = OpenAIResolver(
                    self._resolve_llm_client("merge"),
                    model=merge_config.get("model"),
                    temperature=merge_config.get("temperature", 0.3),
                    max_tokens=merge_config.get("max_tokens", 2048),
                )

        # Pre-merge middleware (can block)
        self._run_middleware("pre_merge")

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

        self._commit_session()

        # Auto-commit conflict merges with full resolutions
        if (
            result.merge_type == "conflict"
            and auto_commit
            and len(result.resolutions) >= len(result.conflicts)
        ):
            result = self.commit_merge(result, message=message)

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
            self._commit_session()

        self._cache.clear()
        if self._config_index is not None:
            self._config_index.invalidate()
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

        self._commit_session()

        result.committed = True
        result.merge_commit_hash = merge_info.commit_hash

        # Clear compile cache
        self._cache.clear()
        if self._config_index is not None:
            self._config_index.invalidate()

        return result

    def import_commit(
        self,
        commit_hash: str,
        *,
        resolver: ResolverCallable | str | None = None,
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
        from tract.operations.rebase import import_commit as _import_commit

        # Resolve commit hash (supports prefixes and branch names)
        resolved = self.resolve_commit(commit_hash)

        # Determine resolver (handles "llm" string, None -> default, callables)
        effective_resolver = self._resolve_resolver(resolver, "merge")

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

        self._commit_session()

        # Clear compile cache (import changes HEAD)
        self._cache.clear()

        return result

    def rebase(
        self,
        target_branch: str,
        *,
        resolver: ResolverCallable | str | None = None,
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
            :class:`RebaseResult`.

        Raises:
            RebaseError: On merge commits in range, resolver abort, etc.
            SemanticSafetyError: If safety warnings and no resolver.
        """
        self._check_open()
        from tract.models.merge import RebaseResult
        from tract.operations.rebase import execute_rebase, plan_rebase

        # Determine resolver (handles "llm" string, None -> default, callables)
        effective_resolver = self._resolve_resolver(resolver, "merge")

        # Plan phase: determine what to replay and check safety
        plan = plan_rebase(
            tract_id=self._tract_id,
            target_branch=target_branch,
            commit_repo=self._commit_repo,
            ref_repo=self._ref_repo,
            parent_repo=self._parent_repo,
            resolver=effective_resolver,
        )

        if plan is None:
            # Nothing to rebase
            current_tip = self._ref_repo.get_head(self._tract_id)
            return RebaseResult(new_head=current_tip or "")

        commits_to_replay, target_tip, warnings, current_branch, current_tip = plan

        result = execute_rebase(
            tract_id=self._tract_id,
            commits_to_replay=commits_to_replay,
            target_tip=target_tip,
            current_branch=current_branch,
            current_tip=current_tip,
            commit_repo=self._commit_repo,
            ref_repo=self._ref_repo,
            parent_repo=self._parent_repo,
            blob_repo=self._blob_repo,
            commit_engine=self._commit_engine,
            event_repo=self._event_repo,
            warnings=warnings,
        )
        self._commit_session()
        self._cache.clear()
        if self._config_index is not None:
            self._config_index.invalidate()
        return result

    def compress(
        self,
        *,
        commits: list[str] | None = None,
        from_commit: str | None = None,
        to_commit: str | None = None,
        target_tokens: int | None = None,
        preserve: list[str] | None = None,
        content: str | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        two_stage: bool | None = None,
        triggered_by: str | None = None,
        strategy: str = "default",
        window_size: int = 5,
    ) -> CompressResult:
        """Compress commit chains into summaries.

        Supports two content modes:
        - **Manual** (``content`` provided): Uses your text as the summary.
        - **LLM** (``configure_llm()`` called): Uses LLM for summarization.

        PINNED commits survive verbatim. SKIP commits are excluded.
        Original commits remain in DB (non-destructive).

        Args:
            commits: Explicit list of commit hashes to compress.
            from_commit: Start of range (inclusive).
            to_commit: End of range (inclusive).
            target_tokens: Target token count for summaries.
            preserve: Hashes to treat as temporarily PINNED.
            content: Manual summary text (bypasses LLM).
            instructions: Extra guidance appended to the **user message** of the
                summarization LLM call (the default prompt is preserved). This
                is added as "Additional instructions: ..." at the end of the
                task prompt. Use this to steer what the summary focuses on.
                Stored in provenance for auditability.
            system_prompt: Completely replaces the **system message** of the
                summarization LLM call (``DEFAULT_SUMMARIZE_SYSTEM``). This
                controls the LLM's persona and behavioral guidelines. When
                ``None``, the built-in neutral default is used. Built-in
                variants for common use cases::

                    from tract.prompts.summarize import (
                        DEFAULT_SUMMARIZE_SYSTEM,        # neutral (default)
                        CONVERSATION_SUMMARIZE_SYSTEM,   # full-conversation recap
                        TOOL_SUMMARIZE_SYSTEM,           # tool-call sequences
                    )

                Stored in provenance for auditability.
            model: Override model for LLM summarization.
            temperature: Override temperature for LLM summarization.
            max_tokens: Override max_tokens for LLM summarization.
            llm_config: Full LLMConfig override for this call.
            two_stage: When True and LLM is available, generate guidance first
                (what should the summary focus on?) then generate summaries using
                that guidance. When None/False, uses one-shot summarization.
            strategy: Compression strategy. ``"default"`` uses partition-around-pinned
                on the specified range. ``"sliding_window"`` keeps the last
                ``window_size`` commits in full detail and compresses everything
                older (PINNED commits always survive).
            window_size: For ``strategy="sliding_window"``: number of most-recent
                commits to keep in full detail. Defaults to 5.

        Returns:
            :class:`CompressResult`.

        Raises:
            DetachedHeadError: If HEAD is detached.
            CompressionError: On various error conditions.
            LLMConfigError: If explicit LLM params given without client.
        """
        self._check_open()
        from tract.operations.compression import (
            _classify_by_priority,
            _commit_compression,
            _partition_around_pinned,
            _reconstruct_content,
            _resolve_commit_range,
            compress_range,
            sliding_window_compress,
        )

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

        # Use operation prompt as fallback for system_prompt
        effective_system_prompt = system_prompt
        if effective_system_prompt is None and self._operation_prompts.compress is not None:
            effective_system_prompt = self._operation_prompts.compress

        # Pre-compress middleware (can block)
        self._run_middleware("pre_compress")

        # --- Sliding window strategy ---
        if strategy == "sliding_window":
            return self._compress_sliding_window(
                window_size=window_size,
                target_tokens=target_tokens,
                preserve=preserve,
                content=content,
                instructions=instructions,
                system_prompt=effective_system_prompt,
                llm_client=llm_client,
                llm_kwargs=llm_kwargs,
                two_stage=two_stage,
                sliding_window_compress_fn=sliding_window_compress,
                _classify_by_priority_fn=_classify_by_priority,
                _commit_compression_fn=_commit_compression,
                _partition_around_pinned_fn=_partition_around_pinned,
                _reconstruct_content_fn=_reconstruct_content,
            )

        # --- Default strategy (partition-around-pinned) ---

        # Step 1: Generate summaries via compress_range
        range_result = compress_range(
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
            llm_client=llm_client,
            llm_kwargs=llm_kwargs,
            generation_config=llm_kwargs if llm_kwargs else None,
            content=content,
            instructions=instructions,
            system_prompt=effective_system_prompt,
            type_registry=self._custom_type_registry,
            two_stage=two_stage or False,
        )

        # Step 2: Re-resolve range data for the commit phase
        head_hash = self._ref_repo.get_head(self._tract_id)
        branch_name = self._ref_repo.get_current_branch(self._tract_id)
        range_commits = _resolve_commit_range(
            self._commit_repo, self._ref_repo, self._annotation_repo,
            self._tract_id, head_hash,
            commits=commits, from_commit=from_commit, to_commit=to_commit,
        )
        pinned_commits, _important, normal_commits, skip_commits = (
            _classify_by_priority(range_commits, self._annotation_repo, preserve=preserve)
        )
        normal_commits = normal_commits + _important
        pinned_hashes = {r.commit_hash for r in pinned_commits}
        skip_hashes = {r.commit_hash for r in skip_commits}
        groups = _partition_around_pinned(range_commits, pinned_hashes, skip_hashes)
        original_tokens = sum(c.token_count for c in normal_commits)

        # Step 3: Commit compression
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
                summaries=range_result.summary_commits,
                range_commits=range_commits,
                pinned_commits=pinned_commits,
                normal_commits=normal_commits,
                pinned_hashes=pinned_hashes,
                skip_hashes=skip_hashes,
                groups=groups,
                original_tokens=original_tokens,
                target_tokens=target_tokens,
                instructions=instructions,
                system_prompt=effective_system_prompt,
                branch_name=branch_name,
                type_registry=self._custom_type_registry,
                expected_head=head_hash,
                generation_config=range_result.generation_config,
            )
        except Exception:
            nested.rollback()
            raise

        self._commit_session()
        self._cache.clear()

        # Attach resolved config to result for display
        if llm_kwargs:
            import dataclasses as _dc
            result = _dc.replace(result, config=LLMConfig.from_dict(llm_kwargs))

        return result

    def _compress_sliding_window(
        self,
        *,
        window_size: int,
        target_tokens: int | None,
        preserve: list[str] | None,
        content: str | None,
        instructions: str | None,
        system_prompt: str | None,
        llm_client,
        llm_kwargs: dict,
        two_stage: bool | None,
        sliding_window_compress_fn,
        _classify_by_priority_fn,
        _commit_compression_fn,
        _partition_around_pinned_fn,
        _reconstruct_content_fn,
    ) -> CompressResult:
        """Internal helper for sliding-window compression strategy.

        Keeps the most recent ``window_size`` commits in full detail and
        compresses everything older into summaries.  PINNED commits outside
        the window are preserved verbatim.
        """
        from tract.exceptions import CompressionError

        # Step 1: Generate summaries for pre-window commits
        range_result = sliding_window_compress_fn(
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            annotation_repo=self._annotation_repo,
            ref_repo=self._ref_repo,
            commit_engine=self._commit_engine,
            token_counter=self._token_counter,
            event_repo=self._event_repo,
            parent_repo=self._parent_repo,
            window_size=window_size,
            target_tokens=target_tokens,
            preserve=preserve,
            llm_client=llm_client,
            llm_kwargs=llm_kwargs,
            generation_config=llm_kwargs if llm_kwargs else None,
            content=content,
            instructions=instructions,
            system_prompt=system_prompt,
            type_registry=self._custom_type_registry,
            two_stage=two_stage or False,
        )

        if range_result is None:
            raise CompressionError(
                "Nothing to compress -- all commits are within the sliding "
                "window or are pinned/skipped"
            )

        # Step 2: Re-resolve the pre-window commits for the commit phase
        head_hash = self._ref_repo.get_head(self._tract_id)
        branch_name = self._ref_repo.get_current_branch(self._tract_id)

        # Walk ancestry to split into window / pre-window
        all_ancestors = list(self._commit_repo.get_ancestors(head_hash))
        window_commits = all_ancestors[:window_size]  # newest first
        pre_window_commits = all_ancestors[window_size:]  # older commits
        pre_window_oldest_first = list(reversed(pre_window_commits))

        # Classify pre-window commits
        pinned_commits, _important, normal_commits, skip_commits = (
            _classify_by_priority_fn(
                pre_window_oldest_first, self._annotation_repo, preserve=preserve
            )
        )
        normal_commits = normal_commits + _important
        pinned_hashes = {r.commit_hash for r in pinned_commits}
        skip_hashes = {r.commit_hash for r in skip_commits}
        groups = _partition_around_pinned_fn(
            pre_window_oldest_first, pinned_hashes, skip_hashes
        )
        original_tokens = sum(c.token_count for c in normal_commits)

        # Step 3: Commit the compressed pre-window chain, then replay window
        nested = self._session.begin_nested()
        try:
            result = _commit_compression_fn(
                tract_id=self._tract_id,
                commit_repo=self._commit_repo,
                blob_repo=self._blob_repo,
                ref_repo=self._ref_repo,
                commit_engine=self._commit_engine,
                token_counter=self._token_counter,
                event_repo=self._event_repo,
                summaries=range_result.summary_commits,
                range_commits=pre_window_oldest_first,
                pinned_commits=pinned_commits,
                normal_commits=normal_commits,
                pinned_hashes=pinned_hashes,
                skip_hashes=skip_hashes,
                groups=groups,
                original_tokens=original_tokens,
                target_tokens=target_tokens,
                instructions=instructions,
                system_prompt=system_prompt,
                branch_name=branch_name,
                type_registry=self._custom_type_registry,
                expected_head=head_hash,
                generation_config=range_result.generation_config,
            )

            # Step 4: Replay window commits on top of the compressed chain
            # Window commits are in newest-first order; reverse to oldest-first
            for row in reversed(window_commits):
                content_model = _reconstruct_content_fn(
                    row, self._blob_repo, self._custom_type_registry
                )
                self._commit_engine.create_commit(
                    content=content_model,
                    operation=row.operation,
                    message=row.message or "Replayed window commit",
                    metadata=row.metadata_json,
                    generation_config=row.generation_config_json,
                )

            # Update result with final HEAD (after window replay)
            import dataclasses as _dc

            new_head = self._ref_repo.get_head(self._tract_id) or ""
            result = _dc.replace(result, new_head=new_head)

        except Exception:
            nested.rollback()
            raise

        self._commit_session()
        self._cache.clear()

        if llm_kwargs:
            import dataclasses as _dc
            result = _dc.replace(result, config=LLMConfig.from_dict(llm_kwargs))

        return result

    def compress_tool_calls(
        self,
        commits: list[str] | None = None,
        *,
        name: str | None = None,
        target_tokens: int | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        triggered_by: str | None = None,
    ) -> "ToolCompactResult":
        """Compact tool-call sequences using EDIT commits.

        Sends the full tool-calling sequence to the LLM for holistic
        context, then applies per-result summaries as EDIT commits.
        This preserves commit structure, tool roles, metadata
        (``tool_call_id``, ``name``), and keeps tool turns queryable
        via :meth:`find_tool_turns`.

        For bulk compression that collapses commits into a single
        summary (losing structure), use :meth:`compress` directly.

        Args:
            commits: Optional commit hashes to scope which turns to
                compact.  When ``None`` (default), uses
                :meth:`find_tool_turns` to auto-detect all tool-call
                turns on the current branch.  When explicit, only turns
                whose hashes overlap with this list are compacted.
            name: Filter to compact only turns involving this tool name.
                Passed to :meth:`find_tool_turns`.
            target_tokens: Target token count per compacted result.
            instructions: Extra guidance appended to the compaction
                prompt.
            system_prompt: Override the compaction system prompt.
            model: Override model for LLM compaction.
            temperature: Override temperature for LLM compaction.
            max_tokens: Override max_tokens for LLM compaction.
            llm_config: Full LLMConfig override for this call.
            triggered_by: Optional provenance string.

        Returns:
            :class:`ToolCompactResult` with edit commit details.

        Raises:
            CompressionError: If no tool turns found or LLM returns
                malformed response.
        """
        
        from tract.exceptions import CompressionError
        from tract.models.compression import ToolCompactResult
        from tract.operations.compression import build_role_label
        from tract.prompts.summarize import (
            TOOL_COMPACT_SYSTEM,
            build_tool_compact_prompt,
        )

        # 1. Find tool turns
        turns = self.find_tool_turns(name=name)

        # Scope to explicit commits if provided
        if commits is not None:
            commit_set = set(commits)
            turns = [
                turn for turn in turns
                if any(h in commit_set for h in turn.all_hashes)
            ]

        if not turns:
            raise CompressionError("No tool turns found to compact")

        # 2. Collect tool results to compact and build sequence text
        results_to_compact: list[CommitInfo] = []
        parts: list[str] = []

        for turn in turns:
            # Add assistant tool-calling message for context
            call_meta = turn.call.metadata or {}
            call_text = self.get_content(turn.call) or ""
            parts.append(f"{build_role_label('assistant', call_meta)}: {call_text}")

            # Add each tool result (these will be compacted)
            for r in turn.results:
                r_meta = r.metadata or {}
                r_text = self.get_content(r) or ""
                parts.append(f"{build_role_label('tool', r_meta)}: {r_text}")
                results_to_compact.append(r)

        if not results_to_compact:
            raise CompressionError("No tool results found to compact")

        sequence_text = "\n".join(parts)

        # 3. Build prompt and call LLM
        prompt = build_tool_compact_prompt(
            sequence_text,
            result_count=len(results_to_compact),
            target_tokens=target_tokens,
            instructions=instructions,
        )
        sys_prompt = (
            system_prompt if system_prompt is not None else TOOL_COMPACT_SYSTEM
        )

        # Resolve LLM config and client
        llm_kwargs: dict = {}
        if any(v is not None for v in (model, temperature, max_tokens, llm_config)):
            resolved = self._resolve_llm_config(
                "compress", model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
            )
            if resolved:
                llm_kwargs = resolved

        llm = self._resolve_llm_client("compress")
        response = llm.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            **llm_kwargs,
        )

        # 4. Parse per-result summaries from LLM response
        raw_content = response["choices"][0]["message"]["content"]
        try:
            summaries = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CompressionError(
                f"LLM returned invalid JSON for tool compaction: {exc}\n"
                f"Response: {raw_content[:200]}"
            ) from exc

        if not isinstance(summaries, list) or len(summaries) != len(results_to_compact):
            raise CompressionError(
                f"Expected {len(results_to_compact)} summaries, "
                f"got {len(summaries) if isinstance(summaries, list) else type(summaries).__name__}"
            )

        # 5. Apply each summary as an EDIT commit
        original_tokens = 0
        compacted_tokens = 0
        edit_commits: list[str] = []
        source_commits: list[str] = []

        for result_ci, summary in zip(results_to_compact, summaries):
            r_meta: CommitMetadata = result_ci.metadata or {}  # type: ignore[assignment]  # {} is valid CommitMetadata (total=False)
            original_tokens += result_ci.token_count
            source_commits.append(result_ci.commit_hash)

            edited = self.tool_result(
                tool_call_id=r_meta.get("tool_call_id", ""),
                name=r_meta.get("name", ""),
                content=str(summary),
                edit=result_ci.commit_hash,
            )
            compacted_tokens += edited.token_count
            edit_commits.append(edited.commit_hash)

        # 6. Return result
        all_tool_names = sorted({n for turn in turns for n in turn.tool_names})

        # Build effective config for display
        effective_config = LLMConfig.from_dict(
            self._resolve_llm_config(
                "compress", model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
            )
        )

        return ToolCompactResult(
            edit_commits=tuple(edit_commits),
            source_commits=tuple(source_commits),
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            tool_names=tuple(all_tool_names),
            turn_count=len(turns),
            config=effective_config,
        )

    def gc(
        self,
        *,
        orphan_retention_days: int = 7,
        archive_retention_days: int | None = None,
        branch: str | None = None,
    ) -> GCResult:
        """Garbage-collect unreachable commits.

        Removes commits not reachable from any branch tip, subject to
        configurable retention periods.

        Args:
            orphan_retention_days: Days before orphaned commits become
                eligible for removal. Default 7.
            archive_retention_days: If set, days before archived (compression
                source) commits become eligible for removal. None (default)
                means archives are never removed.
            branch: If set, only check this branch for reachability.
                WARNING: commits reachable from other branches may be removed.

        Returns:
            :class:`GCResult`.

        Raises:
            CompressionError: If compression repository is not available.
        """
        self._check_open()
        from tract.exceptions import CompressionError
        from tract.operations.compression import execute_gc, plan_gc

        if self._event_repo is None:
            raise CompressionError("Compression repository not available")

        if self._parent_repo is None:
            raise CompressionError("Parent repository not available")

        # Pre-GC middleware (can block)
        self._run_middleware("pre_gc")

        # Plan phase: determine which commits to remove
        commits_to_remove, tokens_to_free = plan_gc(
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            ref_repo=self._ref_repo,
            parent_repo=self._parent_repo,
            event_repo=self._event_repo,
            orphan_retention_days=orphan_retention_days,
            archive_retention_days=archive_retention_days,
            branch=branch,
        )

        # Execute phase: remove commits
        result = execute_gc(
            tract_id=self._tract_id,
            commits_to_remove=commits_to_remove,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            event_repo=self._event_repo,
        )
        self._cache.clear()
        self._commit_session()
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

        current_head = self.head
        target_hash = head_hash or current_head
        if target_hash is None:
            raise TraceError("Cannot record usage: no commits exist")

        # Validate explicit head_hash matches current HEAD
        if head_hash is not None and head_hash != current_head:
            raise TraceError(
                f"Cannot record usage: head_hash {head_hash} "
                f"does not match current HEAD {current_head}"
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
            # Persist as compile record for cross-session durability
            if self._compile_record_repo is not None:
                self._save_compile_record(target_hash, context_tokens, updated.commit_count, token_source, updated.commit_hashes)
            return self._cache.to_compiled(updated)

        # Fallback (custom compiler, no snapshot): return minimal result
        context_tokens = usage.prompt_tokens + usage.completion_tokens
        token_source = f"api:{usage.prompt_tokens}+{usage.completion_tokens}"
        # Persist as compile record even without snapshot
        if self._compile_record_repo is not None:
            self._save_compile_record(target_hash, context_tokens, 0, token_source)
        return CompiledContext(
            messages=[],
            token_count=context_tokens,
            commit_count=0,
            token_source=token_source,
        )

    def _save_compile_record(
        self,
        head_hash: str,
        token_count: int,
        commit_count: int,
        token_source: str,
        commit_hashes: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Persist a compile record to storage."""
        if self._compile_record_repo is None:
            return  # compile records not enabled; silently skip
        record_id = uuid.uuid4().hex
        self._compile_record_repo.save_record(
            record_id=record_id,
            tract_id=self._tract_id,
            head_hash=head_hash,
            token_count=token_count,
            commit_count=commit_count,
            token_source=token_source,
            params_json=None,
            created_at=datetime.now(timezone.utc),
        )
        for pos, ch in enumerate(commit_hashes):
            self._compile_record_repo.add_effective(record_id, ch, pos)
        self._commit_session()

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

    def _commit_session(self) -> None:
        """Commit the underlying session, unless inside a :meth:`batch`.

        All ``self._session.commit()`` calls in the class should go
        through this method so that batch mode can defer the commit
        without monkey-patching the session object.
        """
        if not self._in_batch:
            self._session.commit()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Context manager for atomic multi-commit batches.

        Defers all session commits until the batch exits successfully.
        On exception, all pending changes are rolled back.

        Example::

            with t.batch():
                t.commit(InstructionContent(text="System prompt"))
                t.commit(DialogueContent(role="user", text="Hi"))
        """
        # Invalidate compile cache on batch entry and set _in_batch flag
        # so _commit_session() defers and commit() skips cache updates.
        self._cache.clear()
        self._in_batch = True
        try:
            yield
            # Success: single commit for the entire batch
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        finally:
            self._in_batch = False


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

    def _resolve_tools(
        self,
        *,
        profile: ProfileName | ToolProfile | str = "self",
        tool_names: list[ToolName | str] | None = None,
        overrides: dict[ToolName | str, str] | None = None,
    ) -> list:
        """Shared tool resolution: profile filtering, name filtering, overrides.

        Returns list of ToolDefinition objects.
        """
        from tract.toolkit.definitions import get_all_tools
        from tract.toolkit.models import ToolProfile as _ToolProfile
        from tract.toolkit.profiles import get_profile

        # Resolve profile
        if isinstance(profile, str):
            resolved_profile = get_profile(profile)
        elif isinstance(profile, _ToolProfile):
            resolved_profile = profile
        else:
            raise TypeError(
                f"profile must be a string or ToolProfile, got {type(profile).__name__}"
            )

        # Special case: compact profile generates domain-grouped tools
        if resolved_profile.name == "compact":
            from tract.toolkit.compact import ACTION_TO_DOMAIN, get_compact_tools

            compact_tools = get_compact_tools(self)
            if tool_names is not None:
                allowed = set(tool_names)
                # Translate action-level names (e.g. "commit", "status") to
                # the compact domain tools that contain them (e.g.
                # "tract_context").  This lets callers use the same tool_names
                # regardless of whether the profile is "full" or "compact".
                expanded = set(allowed)  # keep any direct compact names
                for name in allowed:
                    domain = ACTION_TO_DOMAIN.get(name)
                    if domain is not None:
                        expanded.add(f"tract_{domain}")
                compact_tools = [t for t in compact_tools if t.name in expanded]
            if overrides:
                compact_tools = [
                    replace(t, description=overrides[t.name])
                    if t.name in overrides else t
                    for t in compact_tools
                ]
            return compact_tools

        all_tools = get_all_tools(self)

        # Apply profile filtering
        filtered = resolved_profile.filter_tools(all_tools)

        # Include dynamic operation tools (not in static profile configs)
        filtered_names = {t.name for t in filtered}
        for tool in all_tools:
            if tool.name.startswith("fire_") and tool.name not in filtered_names:
                filtered.append(tool)

        # Filter to specific tool names if requested
        if tool_names is not None:
            allowed = set(tool_names)
            filtered = [t for t in filtered if t.name in allowed]

        # Apply description overrides
        if overrides:
            new_filtered = []
            for tool in filtered:
                if tool.name in overrides:
                    tool = replace(tool, description=overrides[tool.name])
                new_filtered.append(tool)
            filtered = new_filtered

        # Append custom tools registered via @t.tool
        if self._custom_tools:
            existing_names = {t.name for t in filtered}
            for ct in self._custom_tools.values():
                if ct.name not in existing_names:
                    # Apply tool_names filter if active
                    if tool_names is not None and ct.name not in set(tool_names):
                        continue
                    filtered.append(ct)

        return filtered

    def as_tools(
        self,
        *,
        profile: ProfileName | ToolProfile | str | object = _PROFILE_SENTINEL,
        tool_names: list[ToolName | str] | None = None,
        overrides: dict[ToolName | str, str] | None = None,
        format: Literal["openai", "anthropic"] = "openai",
    ) -> list[dict]:
        """Get tool definitions for this tract in LLM-consumable format.

        Combines tool definitions, profile filtering, optional description
        overrides, and format conversion in one call.

        Args:
            profile: A profile name (``"compact"``, ``"self"``, ``"supervisor"``,
                ``"full"``) or a :class:`~tract.toolkit.models.ToolProfile`
                instance.  Falls back to ``tool_profile`` from :meth:`open`,
                then ``"compact"``.
            tool_names: Optional list of tool names to include. When provided,
                only tools whose names are in this list are returned (applied
                after profile filtering).
            overrides: Optional dict mapping tool names to replacement
                descriptions.  Applied on top of the profile's descriptions.
            format: Output format -- ``"openai"`` (default) or ``"anthropic"``.

        Returns:
            List of tool definition dicts in the requested format.
        """
        effective_profile = (
            self._tool_profile or "compact"
        ) if profile is self._PROFILE_SENTINEL else profile
        filtered = self._resolve_tools(
            profile=effective_profile, tool_names=tool_names, overrides=overrides,
        )
        if format == "openai":
            return [tool.to_openai() for tool in filtered]
        elif format == "anthropic":
            return [tool.to_anthropic() for tool in filtered]
        else:
            raise ValueError(
                f"Unknown format '{format}'. Supported: 'openai', 'anthropic'."
            )

    def as_callable_tools(
        self,
        *,
        profile: ProfileName | ToolProfile | str | object = _PROFILE_SENTINEL,
        tool_names: list[ToolName | str] | None = None,
        overrides: dict[ToolName | str, str] | None = None,
    ) -> list:
        """Get tools as typed Python callables for framework integration.

        Returns tract tools as functions with proper ``__name__``, ``__doc__``,
        ``__signature__``, and type annotations.  Works with any framework
        that introspects callables: Agno, LangChain, CrewAI, LangGraph, etc.

        Args:
            profile: A profile name (``"compact"``, ``"self"``, ``"supervisor"``,
                ``"full"``) or a :class:`~tract.toolkit.models.ToolProfile`
                instance.  Falls back to ``tool_profile`` from :meth:`open`,
                then ``"compact"``.
            tool_names: Optional list of tool names to include.
            overrides: Optional dict mapping tool names to replacement
                descriptions.  Applied on top of the profile's descriptions.

        Returns:
            List of typed Python callables, one per tool.
        """
        from tract.toolkit.callables import tools_to_callables

        effective_profile = (
            self._tool_profile or "compact"
        ) if profile is self._PROFILE_SENTINEL else profile
        filtered = self._resolve_tools(
            profile=effective_profile, tool_names=tool_names, overrides=overrides,
        )
        return tools_to_callables(filtered)

    def switch_profile(self, profile: ProfileName | ToolProfile | str) -> None:
        """Switch the active tool profile.

        Changes which tools are available for the current session.
        Clears any per-tool overrides.

        Args:
            profile: Profile name (``"self"``, ``"supervisor"``, ``"full"``) or
                a ToolProfile instance.
        """
        if self._tool_executor is None:
            from tract.toolkit.executor import ToolExecutor
            self._tool_executor = ToolExecutor(self)
        self._tool_executor.set_profile(profile)

    def unlock_tool(self, tool_name: ToolName | str) -> None:
        """Force-enable a tool regardless of current profile.

        Args:
            tool_name: Name of the tool to unlock.
        """
        if self._tool_executor is None:
            from tract.toolkit.executor import ToolExecutor
            self._tool_executor = ToolExecutor(self)
        self._tool_executor.unlock_tool(tool_name)

    def lock_tool(self, tool_name: ToolName | str) -> None:
        """Force-disable a tool regardless of current profile.

        Args:
            tool_name: Name of the tool to lock.
        """
        if self._tool_executor is None:
            from tract.toolkit.executor import ToolExecutor
            self._tool_executor = ToolExecutor(self)
        self._tool_executor.lock_tool(tool_name)

    def tool(
        self,
        fn: Any | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a custom tool from a typed Python function.

        Works as a decorator (with or without arguments)::

            @t.tool
            def search(query: str) -> str:
                \"\"\"Search the database.\"\"\"
                ...

            @t.tool(name="calc", description="Math evaluator")
            def calculator(expression: str) -> str:
                ...

        Registered tools are automatically included in :meth:`run` and
        :meth:`as_tools` alongside tract's built-in tools.

        Args:
            fn: The function to register (when used as ``@t.tool``
                without parentheses).
            name: Override the tool name (defaults to ``fn.__name__``).
            description: Override the description (defaults to the first
                line of the docstring).

        Returns:
            The original function (unmodified), or a decorator if called
            with keyword arguments.
        """
        from tract.toolkit.callables import callable_to_tool

        def _register(func: Any) -> Any:
            tool_def = callable_to_tool(func, name=name, description=description)
            self._custom_tools[tool_def.name] = tool_def
            return func

        if fn is not None:
            # Used as @t.tool (no parentheses)
            return _register(fn)
        # Used as @t.tool(...) (with parentheses)
        return _register

    def remove_tool(self, tool_name: str) -> None:
        """Unregister a custom tool previously added via :meth:`tool`.

        Args:
            tool_name: Name of the custom tool to remove.

        Raises:
            KeyError: If no custom tool with that name is registered.
        """
        if tool_name not in self._custom_tools:
            available = ", ".join(sorted(self._custom_tools.keys())) or "(none)"
            raise KeyError(
                f"No custom tool '{tool_name}'. Registered: {available}"
            )
        del self._custom_tools[tool_name]

    @property
    def custom_tools(self) -> dict[str, Any]:
        """Read-only view of registered custom tools (name -> ToolDefinition)."""
        return dict(self._custom_tools)

    # ------------------------------------------------------------------
    # File-based persistence (.tract/ directory)
    # ------------------------------------------------------------------

    @property
    def tract_dir(self) -> "Path | None":
        """Path to .tract/ directory, or None for in-memory databases."""
        from pathlib import Path

        if self._db_path == ":memory:":
            return None
        db = Path(self._db_path)
        return db.parent / ".tract"

    @property
    def quarantined(self) -> list[str]:
        """List of modules that failed to load on startup."""
        return list(self._quarantined)

    def _ensure_tract_dir(self, subdir: str | None = None) -> "Path":
        """Create .tract/ (and optional subdir) lazily. Returns the dir path.

        Raises RuntimeError for in-memory databases.
        """
        td = self.tract_dir
        if td is None:
            raise RuntimeError(
                "File-based persistence is not available for in-memory databases."
            )
        if subdir:
            target = td / subdir
        else:
            target = td
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _load_persisted_state(self) -> None:
        """Load persisted configs from DB.

        Called during Tract.open() after all repos are initialized.
        """
        import logging

        logger = logging.getLogger(__name__)
        repo = self._persistence_repo
        if repo is None:
            return

        # Load operation configs from DB
        for config_row in repo.get_operation_configs(self._tract_id):
            try:
                json.loads(config_row.config_json)  # validate JSON
            except Exception:
                logger.warning(
                    "Failed to load config '%s': skipping.",
                    config_row.config_key,
                    exc_info=True,
                )
                self._quarantined.append(f"config:{config_row.config_key}")

    def save_workflow(
        self,
        name: str,
        code: str,
        *,
        description: str = "",
    ) -> "Path":
        """Write a workflow to .tract/workflows/{name}.py.

        Args:
            name: Workflow name (used as filename).
            code: Python source code.
            description: Human-readable description.

        Returns:
            Path to the written file.

        Raises:
            SyntaxError: If code has syntax errors (validated before writing).
            RuntimeError: If database is in-memory.
        """
        # Validate syntax
        compile(code, f"{name}.py", "exec")

        # Write file
        workflows_dir = self._ensure_tract_dir("workflows")
        file_path = workflows_dir / f"{name}.py"
        file_path.write_text(code, encoding="utf-8")

        return file_path

    def close(self) -> None:
        """Close the session and dispose the engine."""
        if self._closed:
            return
        self._closed = True
        # Close internally-created LLM client (not externally-provided ones)
        if self._owns_llm_client and self._llm_client is not None:
            try:
                self._llm_client.close()
            except Exception:
                logger.debug("Failed to close LLM client", exc_info=True)
        try:
            self._session.close()
        except Exception:
            logger.debug("Failed to close SQLAlchemy session", exc_info=True)
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception:
                logger.debug("Failed to dispose engine", exc_info=True)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Tract:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

    async def aclose(self) -> None:
        """Close the tract, releasing async LLM resources first.

        If the LLM client supports ``aclose()``, it is awaited for clean
        async teardown.  All remaining (sync) resources are then released
        via :meth:`close`.
        """
        if self._closed:
            return
        # Async-close the LLM client if it supports it
        if self._owns_llm_client and self._llm_client is not None:
            _aclose = getattr(self._llm_client, "aclose", None)
            if _aclose is not None:
                try:
                    await _aclose()
                except Exception:
                    logger.debug("Failed to async-close LLM client", exc_info=True)
                # Prevent sync close() from double-closing the client
                self._owns_llm_client = False
        # Delegate remaining teardown to sync close()
        self.close()

    async def __aenter__(self) -> Tract:
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._closed:
            return f"Tract(tract_id='{self._tract_id}', closed=True)"
        return f"Tract(tract_id='{self._tract_id}', head='{self.head}')"
