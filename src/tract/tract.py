"""Tract -- the public SDK entry point.

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
    SqliteBehavioralSpecRepository,
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

    from tract.autonomous import AutoBranchResult, AutoRebaseResult, AutoSplitResult
    from tract.intelligence import CherryPickResult, DedupResult
    from tract.llm.protocols import LLMClient, ResolverCallable
    from tract.loop import LoopResult
    from tract.middleware import MiddlewareEvent
    from tract.profiles import WorkflowProfile
    from tract.routing import Route, RoutingResult, RoutingTable, SemanticRouter
    from tract.templates import DirectiveTemplate
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
    from tract.managers import LLMManager, CompressionManager, PersistenceManager

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
        # Treat Unix-rooted paths (e.g. /etc/passwd) as absolute on all
        # platforms — Path.is_absolute() on Windows requires a drive letter.
        is_abs = p.is_absolute() or str(path).startswith("/")
        if not is_abs and prompt_dir is not None:
            # Relative path with prompt_dir: sandbox to prompt_dir
            base = _Path(prompt_dir).resolve()
            resolved = (base / p).resolve()
            # Security: prevent traversal outside prompt_dir
            if not resolved.is_relative_to(base):
                raise ValueError(
                    f"Path traversal detected: {path!r} escapes the prompt "
                    f"directory {prompt_dir!r}."
                )
            if not resolved.is_file():
                raise ValueError(f"File not found: {resolved}")
            return resolved.read_text(encoding="utf-8")
        # Absolute path or no prompt_dir: use path directly
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
# NOTE: Canonical validation constants live in managers/config.py.
# These module-level aliases are kept for backward compatibility only.
_VALID_OPERATION_NAMES: frozenset[str] = frozenset({"chat", "merge", "compress", "message"})
_VALID_PROMPT_NAMES: frozenset[str] = frozenset({
    "compress", "merge", "message", "commit_message",
    "gate", "maintain", "maintain_peek", "cherry_pick", "dedup",
    "split", "rebase", "branch", "route", "tool_compact", "peek",
})


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
    text = text or ""
    preview = text.strip().replace("\n", " ")
    if not preview:
        return content_type
    if len(preview) > _MAX_AUTO_MSG_LEN:
        preview = preview[: _MAX_AUTO_MSG_LEN - 3] + "..."
    return preview


def _detect_provider(
    base_url: str | None, model: str | None,
) -> Literal["openai", "anthropic", "claude_code"]:
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


class _Runtime:
    """Runtime sub-object -- LLM operations and tool management.

    Accessed via ``t.runtime``.  Provides LLM chat/generate and tool
    management in a single namespace, replacing the old ``t.llm`` and
    ``t.toolkit`` sub-objects.
    """

    def __init__(self, tract: Tract) -> None:
        self._tract = tract

    @property
    def tools(self):
        """Tool management (profile switching, as_tools, custom tools)."""
        return self._tract._toolkit_mgr

    def chat(self, *args, **kwargs):
        """Send a user message and get an LLM response.

        Delegates to :class:`~tract.managers.LLMManager`.
        """
        return self._tract._llm_mgr.chat(*args, **kwargs)

    async def achat(self, *args, **kwargs):
        """Async version of :meth:`chat`."""
        return await self._tract._llm_mgr.achat(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Compile context, call LLM, commit response.

        Delegates to :class:`~tract.managers.LLMManager`.
        """
        return self._tract._llm_mgr.generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        """Async version of :meth:`generate`."""
        return await self._tract._llm_mgr.agenerate(*args, **kwargs)


class Tract:
    """Primary entry point for Tract -- git-like version control for LLM context.

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
        self._session_owner: object | None = None  # Session back-reference (set by Session)
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
        self._default_resolver: ResolverCallable | None = None  # type: ignore[assignment]

        # Tool defaults (set via open() or set_tool_profile/set_tool_result_format)
        self._tool_profile: str | ToolProfile | None = None
        self._tool_result_format: Literal["minimal", "json", "verbose"] = "minimal"

        # Config index (per-key resolution from DAG ancestry)
        self._config_index: ConfigIndex | None = None

        # (Middleware state lives in _middleware_mgr after open())

        # Persistence state
        self._db_path: str = ":memory:"
        self._persistence_repo: SqlitePersistenceRepository | None = None
        self._behavioral_spec_repo: SqliteBehavioralSpecRepository | None = None
        self._quarantined: list[str] = []

        # (Workflow profile state lives in _templates_mgr after open())

        # Per-instance template and profile registries (seeded from defaults)
        from tract.templates import default_template_registry
        from tract.profiles import default_profile_registry
        self._template_registry: dict = default_template_registry()
        self._profile_registry: dict = default_profile_registry()

        # Policy engine
        from tract.policy import PolicyEngine
        self._policy_engine = PolicyEngine()

        # Custom tools registered via @t.tool decorator
        self._custom_tools: dict[str, Any] = {}  # name -> ToolDefinition

        # Prompt file directory (auto-discovered or explicit)
        self._prompt_dir: str | Path | None = None

        # (Routing table lives in _routing_mgr after open())

        # Deferred managers (created in open() / from_components())
        self._llm_mgr: LLMManager | None = None  # type: ignore[assignment]
        self._compression_mgr: CompressionManager | None = None  # type: ignore[assignment]
        self._persistence_mgr: PersistenceManager | None = None  # type: ignore[assignment]

        # Managers are created in open()/from_components() after all repos are set

    # ------------------------------------------------------------------
    # Sub-object manager creation
    # ------------------------------------------------------------------

    def _create_managers(self) -> None:
        """Initialize all sub-object managers with dependency injection."""
        from tract.managers.state import LLMState
        from tract.managers import (
            TagManager, BranchManager, AnnotationManager, MiddlewareManager,
            ToolManager, ConfigManager, SearchManager, TemplateManager,
            SpawnManager, RoutingManager, IntelligenceManager, ToolkitManager,
        )

        # Shared LLM state
        self._llm_state = LLMState(
            llm_client=self._llm_client,
            default_config=self._default_config,
            operation_configs=self._operation_configs,
            operation_prompts=self._operation_prompts,
            operation_clients=self._operation_clients,
            retry_config=self._retry_config,
            default_resolver=self._default_resolver,
            commit_reasoning=self._commit_reasoning,
            auto_message_enabled=self._auto_message_enabled,
            tool_summarization_config=self._tool_summarization_config,
            owns_llm_client=self._owns_llm_client,
        )

        # Leaf managers (no outgoing dependencies)
        self._tags_mgr = TagManager(
            tract_id=self._tract_id,
            get_tag_annotation_repo=lambda: self._tag_annotation_repo,
            get_tag_registry_repo=lambda: self._tag_registry_repo,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            annotation_repo=self._annotation_repo,
            parent_repo=self._parent_repo,
            get_strict_tags=lambda: self._strict_tags,
            check_open=self._check_open,
            commit_session=self._commit_session,
            get_ancestors=self._get_merge_aware_ancestors,
            row_to_info=self._commit_engine._row_to_info,
            get_head=lambda: self.head,
        )

        self._branches_mgr = BranchManager(
            tract_id=self._tract_id,
            ref_repo=self._ref_repo,
            commit_repo=self._commit_repo,
            parent_repo=self._parent_repo,
            cache=self._cache,
            check_open=self._check_open,
            commit_session=self._commit_session,
            get_config_index=lambda: getattr(self._config_mgr, '_config_index', None),
        )

        self._annotations_mgr = AnnotationManager(
            tract_id=self._tract_id,
            annotation_repo=self._annotation_repo,
            commit_repo=self._commit_repo,
            commit_engine=self._commit_engine,
            cache=self._cache,
            check_open=self._check_open,
            commit_session=self._commit_session,
            get_head=lambda: self.head,
            log_fn=lambda **kw: self._search_mgr.log(**kw),
        )

        self._middleware_mgr = MiddlewareManager(
            check_open=self._check_open,
            persist_behavioral_spec=lambda *a, **kw: self._persistence_mgr.persist_behavioral_spec(*a, **kw),
            remove_behavioral_spec=lambda *a, **kw: self._persistence_mgr.remove_behavioral_spec(*a, **kw),
            get_current_branch=lambda: self.current_branch,
            get_head=lambda: self.head,
            tract_ref=lambda: self,
            policy_engine=self._policy_engine,
        )

        self._tools_mgr = ToolManager(
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            ref_repo=self._ref_repo,
            parent_repo=self._parent_repo,
            annotation_repo=self._annotation_repo,
            tool_schema_repo=self._tool_schema_repo,
            check_open=self._check_open,
            commit_session=self._commit_session,
            get_head=lambda: self.head,
            log_fn=lambda **kw: self._search_mgr.log(**kw),
            annotate_fn=lambda *a, **kw: self._annotations_mgr.set(*a, **kw),
            row_to_info=self._commit_engine._row_to_info,
        )

        # Config manager (writes LLMState)
        self._config_mgr = ConfigManager(
            tract_id=self._tract_id,
            commit_engine=self._commit_engine,
            ref_repo=self._ref_repo,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            event_repo=self._event_repo,
            persistence_repo=self._persistence_repo,
            config=self._config,
            llm_state=self._llm_state,
            parent_repo=self._parent_repo,
            check_open=self._check_open,
            commit_session=self._commit_session,
            commit_fn=lambda *a, **kw: self.commit(*a, **kw),
            get_head=lambda: self.head,
        )

        # Search manager (read-only + callbacks)
        self._search_mgr = SearchManager(
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            ref_repo=self._ref_repo,
            annotation_repo=self._annotation_repo,
            parent_repo=self._parent_repo,
            event_repo=self._event_repo,
            commit_engine=self._commit_engine,
            token_counter=self._token_counter,
            compiler=self._compiler,
            config=self._config,
            custom_type_registry=self._custom_type_registry,
            check_open=self._check_open,
            enrich=lambda entries: self._annotations_mgr._enrich_with_priorities(entries),
            get_head=lambda: self.head,
            get_ancestors=self._get_merge_aware_ancestors,
            row_to_info=self._commit_engine._row_to_info,
            compile_fn=lambda **kw: self.compile(**kw),
            compile_at_fn=lambda h: self._compile_at(h),
            resolve_commit_fn=lambda r: self._branches_mgr.resolve(r),
            get_config_fn=lambda k, **kw: self._config_mgr.get(k, **kw),
            commit_fn=lambda *a, **kw: self.commit(*a, **kw),
            tag_annotation_repo=self._tag_annotation_repo,
            tract_ref=self,
        )

        # Template manager (shares Tract's registries)
        self._templates_mgr = TemplateManager(
            check_open=self._check_open,
            directive_fn=lambda *a, **kw: self.directive(*a, **kw),
            configure_fn=lambda **kw: self._config_mgr.set(**kw),
            template_registry=self._template_registry,
            profile_registry=self._profile_registry,
        )

        # Spawn manager
        self._spawn_mgr = SpawnManager(
            tract_id=self._tract_id,
            spawn_repo=self._spawn_repo,
            check_open=self._check_open,
            session_owner=self._session_owner,
        )

        # Routing manager
        self._routing_mgr = RoutingManager(
            tract_id=self._tract_id,
            ref_repo=self._ref_repo,
            commit_repo=self._commit_repo,
            check_open=self._check_open,
            commit_session=self._commit_session,
            list_branches_fn=lambda: self._branches_mgr.list(),
            checkout_fn=lambda t: self._branches_mgr.checkout(t),
            branch_fn=lambda n: self._branches_mgr.create(n),
            apply_stage_fn=lambda s: self._templates_mgr.apply_stage(s),
            load_profile_fn=lambda n: self._templates_mgr.load_profile(n),
            tract_ref=lambda: self,
        )

        # Intelligence manager
        self._intelligence_mgr = IntelligenceManager(
            tract_id=self._tract_id,
            check_open=self._check_open,
            tract_ref=self,
        )

        # Toolkit manager (needs back-ref to Tract)
        self._toolkit_mgr = ToolkitManager(
            tract_ref=self,
            check_open=self._check_open,
        )

    def _create_deferred_managers(self) -> None:
        """Create managers that need fully-configured LLM state."""
        from tract.managers import LLMManager, CompressionManager, PersistenceManager

        # LLM state is already on self._llm_state (created in _create_managers,
        # updated by open() config setup).  No sync needed.

        self._llm_mgr = LLMManager(
            tract_id=self._tract_id,
            ref_repo=self._ref_repo,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            annotation_repo=self._annotation_repo,
            parent_repo=self._parent_repo,
            compile_record_repo=self._compile_record_repo,
            config=self._config,
            token_counter=self._token_counter,
            llm_state=self._llm_state,
            check_open=self._check_open,
            commit_fn=lambda *a, **kw: self.commit(*a, **kw),
            system_fn=lambda *a, **kw: self.system(*a, **kw),
            user_fn=lambda *a, **kw: self.user(*a, **kw),
            assistant_fn=lambda *a, **kw: self.assistant(*a, **kw),
            reasoning_fn=lambda *a, **kw: self.reasoning(*a, **kw),
            tool_result_fn=lambda *a, **kw: self.tool_result(*a, **kw),
            compile_fn=lambda **kw: self.compile(**kw),
            annotate_fn=lambda *a, **kw: self._annotations_mgr.set(*a, **kw),
            run_middleware=lambda *a, **kw: self._middleware_mgr._run(*a, **kw),
            record_usage=lambda *a, **kw: self._compression_mgr.record_usage(*a, **kw),
            get_tools=lambda: self._tools_mgr.get(),
            commit_session=self._commit_session,
            resolve_llm_config=lambda *a, **kw: self._config_mgr._resolve_llm_config(*a, **kw),
            resolve_llm_client=lambda *a, **kw: self._config_mgr._resolve_llm_client(*a, **kw),
            has_llm_client=lambda *a, **kw: self._config_mgr._has_llm_client(*a, **kw),
            resolve_commit_fn=lambda r: self._branches_mgr.resolve(r),
            get_head=lambda: self.head,
            as_tools_fn=lambda **kw: self._toolkit_mgr.as_tools(**kw),
            save_compile_record_fn=lambda *a, **kw: self._save_compile_record(*a, **kw),
            get_in_batch=lambda: self._in_batch,
            get_tool_profile=lambda: self._tool_profile,
            get_custom_tools=lambda: self._custom_tools,
            get_tract=lambda: self,
        )

        self._compression_mgr = CompressionManager(
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            ref_repo=self._ref_repo,
            annotation_repo=self._annotation_repo,
            parent_repo=self._parent_repo,
            event_repo=self._event_repo,
            compile_record_repo=self._compile_record_repo,
            token_counter=self._token_counter,
            commit_engine=self._commit_engine,
            compiler=self._compiler,
            config=self._config,
            cache=self._cache,
            llm_state=self._llm_state,
            check_open=self._check_open,
            commit_fn=lambda *a, **kw: self.commit(*a, **kw),
            compile_fn=lambda **kw: self.compile(**kw),
            run_middleware=lambda *a, **kw: self._middleware_mgr._run(*a, **kw),
            commit_session=self._commit_session,
            resolve_llm_config=lambda *a, **kw: self._config_mgr._resolve_llm_config(*a, **kw),
            resolve_llm_client=lambda *a, **kw: self._config_mgr._resolve_llm_client(*a, **kw),
            has_llm_client=lambda *a, **kw: self._config_mgr._has_llm_client(*a, **kw),
            annotate_fn=lambda *a, **kw: self._annotations_mgr.set(*a, **kw),
            get_head=lambda: self.head,
            get_ancestors=self._get_merge_aware_ancestors,
            row_to_info=self._commit_engine._row_to_info,
            get_session=lambda: self._session,
            get_custom_type_registry=lambda: self._custom_type_registry,
            extract_content_fn=lambda *a, **kw: self._llm_mgr._extract_content(*a, **kw),
            normalize_usage_dict_fn=lambda *a, **kw: self._normalize_usage_dict(*a, **kw),
            save_compile_record_fn=lambda *a, **kw: self._save_compile_record(*a, **kw),
            tool_result_fn=lambda *a, **kw: self.tool_result(*a, **kw),
            get_content_fn=lambda *a, **kw: self._search_mgr.get_content(*a, **kw),
            find_tool_turns_fn=lambda **kw: self._tools_mgr.find_turns(**kw),
        )

        self._persistence_mgr = PersistenceManager(
            tract_id=self._tract_id,
            commit_repo=self._commit_repo,
            blob_repo=self._blob_repo,
            ref_repo=self._ref_repo,
            annotation_repo=self._annotation_repo,
            parent_repo=self._parent_repo,
            event_repo=self._event_repo,
            compile_record_repo=self._compile_record_repo,
            persistence_repo=self._persistence_repo,
            behavioral_spec_repo=self._behavioral_spec_repo,
            config=self._config,
            custom_type_registry=self._custom_type_registry,
            db_path=self._db_path,
            check_open=self._check_open,
            commit_fn=lambda *a, **kw: self.commit(*a, **kw),
            log_fn=lambda **kw: self._search_mgr.log(**kw),
            branch_fn=lambda *a, **kw: self._branches_mgr.create(*a, **kw),
            switch_fn=lambda *a, **kw: self._branches_mgr.switch(*a, **kw),
            reset_fn=lambda *a, **kw: self._branches_mgr.reset(*a, **kw),
            register_tag_fn=lambda *a, **kw: self._tags_mgr.register(*a, **kw),
            annotate_fn=lambda *a, **kw: self._annotations_mgr.set(*a, **kw),
            list_branches_fn=lambda: self._branches_mgr.list(),
            commit_session=self._commit_session,
            get_head=lambda: self.head,
            row_to_info=self._commit_engine._row_to_info,
        )

        # Runtime sub-object (wraps LLM + toolkit)
        self._runtime = _Runtime(self)

    # ------------------------------------------------------------------
    # Sub-object accessors (kept: config, middleware, policies, runtime)
    # ------------------------------------------------------------------

    @property
    def config(self):
        """Configuration management sub-object."""
        return self._config_mgr

    @property
    def middleware(self):
        """Middleware management sub-object."""
        return self._middleware_mgr

    @property
    def policies(self):
        """Policy engine for managing context management policies."""
        return self._policy_engine

    @property
    def runtime(self):
        """Runtime sub-object -- LLM and tool operations."""
        return self._runtime

    # ------------------------------------------------------------------
    # Branch operations
    # ------------------------------------------------------------------

    def branch(self, name: str, *, source: str | None = None, switch: bool = True) -> str:
        """Create a new branch.

        Args:
            name: Branch name.
            source: Commit hash to branch from.  Defaults to HEAD.
            switch: If True (default), switch HEAD to the new branch.

        Returns:
            The commit hash the new branch points to.
        """
        self._check_open()
        return self._branches_mgr.create(name, source=source, switch=switch)

    def checkout(self, target: str) -> str:
        """Checkout a commit or branch.

        Branch name attaches HEAD; commit hash detaches HEAD (read-only).
        ``"-"`` returns to previous position via PREV_HEAD.

        Args:
            target: A branch name, commit hash, hash prefix, or ``"-"``.

        Returns:
            The resolved commit hash at the new HEAD position.
        """
        self._check_open()
        return self._branches_mgr.checkout(target)

    def switch(self, target: str) -> str:
        """Switch to a branch (branch-only, unlike checkout).

        Args:
            target: A branch name.

        Returns:
            The commit hash at the target branch HEAD.

        Raises:
            BranchNotFoundError: If target is not a valid branch name.
        """
        self._check_open()
        return self._branches_mgr.switch(target)

    def list_branches(self) -> list[BranchInfo]:
        """List all branches with current branch indicator.

        Returns:
            List of :class:`BranchInfo` with ``is_current=True`` for
            the active branch.
        """
        self._check_open()
        return self._branches_mgr.list()

    def delete_branch(self, name: str, *, force: bool = False) -> None:
        """Delete a branch.

        Args:
            name: Branch name to delete.
            force: If True, delete even if branch has unmerged commits.
        """
        self._check_open()
        return self._branches_mgr.delete(name, force=force)

    def reset(self, target: str, *, mode: str = "soft") -> str:
        """Reset HEAD to a target commit.

        Args:
            target: A commit hash, branch name, or hash prefix.
            mode: ``"soft"`` (default) or ``"hard"``.

        Returns:
            The resolved target commit hash (new HEAD).
        """
        self._check_open()
        return self._branches_mgr.reset(target, mode=mode)

    def resolve(self, ref_or_prefix: str) -> str:
        """Resolve a commit reference to a full commit hash.

        Resolution order: full hash, branch name, hash prefix (min 4 chars).

        Args:
            ref_or_prefix: A commit hash, branch name, or hash prefix.

        Returns:
            The full commit hash.
        """
        self._check_open()
        return self._branches_mgr.resolve(ref_or_prefix)

    # ------------------------------------------------------------------
    # Tag operations
    # ------------------------------------------------------------------

    def register_tag(self, name: str, description: str | None = None) -> None:
        """Register a new tag name.

        Args:
            name: Tag name.
            description: Optional description of the tag.
        """
        self._check_open()
        return self._tags_mgr.register(name, description=description)

    def tag(self, target_hash: str, tag_name: str) -> None:
        """Add a mutable tag annotation to a commit.

        Args:
            target_hash: Hash of the commit to tag.
            tag_name: Tag name to add.
        """
        self._check_open()
        return self._tags_mgr.add(target_hash, tag_name)

    def untag(self, target_hash: str, tag_name: str) -> bool:
        """Remove a mutable tag annotation from a commit.

        Args:
            target_hash: Hash of the commit to untag.
            tag_name: Tag name to remove.

        Returns:
            True if the tag was removed, False if it didn't exist.
        """
        self._check_open()
        return self._tags_mgr.remove(target_hash, tag_name)

    def list_tags(self) -> list[dict]:
        """List all registered tags with descriptions and usage counts.

        Returns:
            List of dicts with ``name``, ``description``, ``auto_created``,
            and ``count`` keys.
        """
        self._check_open()
        return self._tags_mgr.list()

    def get_tags(self, target_hash: str) -> list[str]:
        """Get all tags for a commit (immutable + mutable combined).

        Args:
            target_hash: Hash of the commit.

        Returns:
            Deduplicated list of tag names.
        """
        self._check_open()
        return self._tags_mgr.get(target_hash)

    # ------------------------------------------------------------------
    # Annotation operations
    # ------------------------------------------------------------------

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
        """Set priority annotation on a commit.

        Args:
            target_hash: Hash of the commit to annotate.
            priority: Priority level (SKIP, NORMAL, IMPORTANT, PINNED).
            reason: Optional reason for the annotation.
            retain: Fuzzy retention instructions (for IMPORTANT).
            retain_match: Deterministic retention patterns.
            retain_match_mode: ``"substring"`` (default) or ``"regex"``.

        Returns:
            :class:`PriorityAnnotation` model.
        """
        self._check_open()
        return self._annotations_mgr.set(
            target_hash, priority,
            reason=reason, retain=retain,
            retain_match=retain_match,
            retain_match_mode=retain_match_mode,
        )

    def get_annotation(self, target_hash: str) -> list[PriorityAnnotation]:
        """Get the full annotation history for a commit.

        Args:
            target_hash: Hash of the commit.

        Returns:
            List of :class:`PriorityAnnotation` in chronological order.
        """
        self._check_open()
        return self._annotations_mgr.get(target_hash)

    # ------------------------------------------------------------------
    # Search / query operations
    # ------------------------------------------------------------------

    def find(self, **kwargs) -> list[CommitInfo]:
        """Search commits by content, tags, content type, or metadata.

        Walks ancestry and returns commits matching all provided criteria.
        Supports: ``content``, ``pattern``, ``tag``, ``content_type``,
        ``metadata_key``, ``metadata_value``, ``branch``, ``limit``.
        """
        self._check_open()
        return self._search_mgr.find(**kwargs)

    def find_one(self, **kwargs) -> CommitInfo | None:
        """Search commits and return the first match, or None.

        Accepts the same keyword arguments as :meth:`find`.
        """
        self._check_open()
        return self._search_mgr.find_one(**kwargs)

    def log(self, limit: int = 20, **kwargs) -> list[CommitInfo]:
        """Walk commit history from HEAD backward.

        Args:
            limit: Maximum number of commits to return.
            **kwargs: Additional filters (``op_filter``, ``tags``, ``tag_match``).

        Returns:
            List of :class:`CommitInfo` in reverse chronological order.
        """
        self._check_open()
        return self._search_mgr.log(limit=limit, **kwargs)

    def status(self) -> StatusInfo:
        """Get current tract status (token count, commit count, branch, etc.).

        Returns:
            :class:`StatusInfo` with head, branch, tokens, and recent commits.
        """
        self._check_open()
        return self._search_mgr.status()

    def diff(self, commit_a: str | None = None, commit_b: str | None = None) -> DiffResult:
        """Compare two commits and return structured diff.

        Args:
            commit_a: First commit (default: parent of commit_b).
            commit_b: Second commit (default: HEAD).

        Returns:
            :class:`DiffResult` with message-level diffs.
        """
        self._check_open()
        return self._search_mgr.diff(commit_a, commit_b)

    def compare(
        self,
        branch_a: str | None = None,
        branch_b: str | None = None,
        *,
        commit_a: str | None = None,
        commit_b: str | None = None,
    ) -> DiffResult:
        """Compare compiled contexts between two branches or commits.

        Args:
            branch_a: First branch (default: current).
            branch_b: Second branch.
            commit_a: Explicit first commit (mutually exclusive with branch_a).
            commit_b: Explicit second commit (mutually exclusive with branch_b).

        Returns:
            :class:`DiffResult`.
        """
        self._check_open()
        return self._search_mgr.compare(
            branch_a, branch_b, commit_a=commit_a, commit_b=commit_b,
        )

    def health(self) -> HealthReport:
        """Run health checks on this tract's DAG.

        Returns:
            :class:`HealthReport` with integrity check results.
        """
        self._check_open()
        return self._search_mgr.health()

    def get_content(self, commit_or_hash: CommitInfo | str) -> str | dict | None:
        """Load the content for a commit.

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.

        Returns:
            The content text (str), full content dict, or None.
        """
        self._check_open()
        return self._search_mgr.get_content(commit_or_hash)

    def get_commit(self, commit_hash: str) -> CommitInfo | None:
        """Fetch a commit by its hash.

        Returns:
            :class:`CommitInfo` if found, None otherwise.
        """
        self._check_open()
        return self._search_mgr.get_commit(commit_hash)

    def get_metadata(self, commit_or_hash: CommitInfo | str) -> dict | None:
        """Load the metadata dict for a commit.

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.

        Returns:
            The metadata dict, or None.
        """
        self._check_open()
        return self._search_mgr.get_metadata(commit_or_hash)

    def show(self, commit_or_hash: CommitInfo | str) -> None:
        """Pretty-print a commit with its full content (like git show).

        Args:
            commit_or_hash: A :class:`CommitInfo` or a commit hash string.
        """
        self._check_open()
        return self._search_mgr.show(commit_or_hash)

    def edit_history(self, commit_hash: str) -> list[CommitInfo]:
        """Get the full edit chain for a commit.

        Args:
            commit_hash: Hash of any version in the edit chain.

        Returns:
            List of :class:`CommitInfo` in chronological order.
        """
        self._check_open()
        return self._search_mgr.edit_history(commit_hash)

    def restore(self, commit_hash: str, version: int = 0, *, message: str | None = None) -> CommitInfo:
        """Restore a previous version of a commit by creating a new EDIT.

        Args:
            commit_hash: Hash of any version in the edit chain.
            version: Index into edit history (0 = original).
            message: Optional commit message.

        Returns:
            :class:`CommitInfo` for the restore EDIT commit.
        """
        self._check_open()
        return self._search_mgr.restore(commit_hash, version, message=message)

    def query_by_config(self, field_or_config=None, operator=None, value=None, **kwargs) -> list[CommitInfo]:
        """Query commits by generation config values.

        Accepts (field, operator, value), conditions=[...], or an LLMConfig.
        """
        self._check_open()
        return self._search_mgr.query_by_config(field_or_config, operator, value, **kwargs)

    # ------------------------------------------------------------------
    # Compression operations
    # ------------------------------------------------------------------

    def compress(self, **kwargs):
        """Compress commit chains into summaries.

        Supports manual (``content=`` provided) and LLM-powered compression.
        PINNED commits survive verbatim. SKIP commits are excluded.

        See :class:`~tract.managers.CompressionManager` for full signature.
        """
        self._check_open()
        return self._compression_mgr.compress(**kwargs)

    async def acompress(self, **kwargs):
        """Async version of :meth:`compress`."""
        self._check_open()
        return await self._compression_mgr.acompress(**kwargs)

    def gc(self, *, orphan_retention_days: int = 7, archive_retention_days: int | None = None, branch: str | None = None):
        """Garbage-collect unreachable commits.

        Args:
            orphan_retention_days: Retention for orphaned commits (default 7).
            archive_retention_days: Retention for archived commits.
            branch: Limit GC to a specific branch.

        Returns:
            :class:`GCResult` describing what was collected.
        """
        self._check_open()
        return self._compression_mgr.gc(
            orphan_retention_days=orphan_retention_days,
            archive_retention_days=archive_retention_days,
            branch=branch,
        )

    def record_usage(self, usage, *, head_hash: str | None = None):
        """Record API-reported token usage, updating cached compilation.

        Args:
            usage: :class:`TokenUsage` or provider-specific dict.
            head_hash: Commit hash to associate with (default: current HEAD).

        Returns:
            Updated :class:`CompiledContext` with calibrated token count.
        """
        self._check_open()
        return self._compression_mgr.record_usage(usage, head_hash=head_hash)

    # ------------------------------------------------------------------
    # Spawn operations
    # ------------------------------------------------------------------

    def spawn_parent(self):
        """Get the spawn info for this tract's parent.

        Returns:
            SpawnInfo if this tract was spawned, None for root tracts.
        """
        self._check_open()
        return self._spawn_mgr.parent()

    def spawn_children(self):
        """Get spawn info for all child tracts.

        Returns:
            List of SpawnInfo for each child, in chronological order.
        """
        self._check_open()
        return self._spawn_mgr.children()

    # ------------------------------------------------------------------
    # Persistence operations
    # ------------------------------------------------------------------

    def snapshot(self, label: str = "", *, metadata: dict | None = None) -> str:
        """Create a named snapshot (restore point) at the current HEAD.

        Args:
            label: Human-readable snapshot label.
            metadata: Optional extra metadata.

        Returns:
            The snapshot tag name.
        """
        self._check_open()
        return self._persistence_mgr.snapshot(label, metadata=metadata)

    def list_snapshots(self) -> list[dict]:
        """List all snapshots.

        Returns:
            List of snapshot dicts with ``tag``, ``label``, ``commit_hash``, etc.
        """
        self._check_open()
        return self._persistence_mgr.list_snapshots()

    def export_state(self, *, include_blobs: bool = True) -> dict:
        """Export tract state as a portable dict.

        Args:
            include_blobs: If True (default), include blob content.

        Returns:
            Dict suitable for JSON serialization.
        """
        self._check_open()
        return self._persistence_mgr.export_state(include_blobs=include_blobs)

    def load_state(self, state: dict) -> int:
        """Import tract state from a previously exported dict.

        Args:
            state: Dict from :meth:`export_state`.

        Returns:
            Number of commits imported.
        """
        self._check_open()
        return self._persistence_mgr.load_state(state)

    def compile_records(self, limit: int = 100) -> list:
        """Get compile records for this tract, newest first.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of compile record objects.
        """
        self._check_open()
        return self._persistence_mgr.compile_records(limit)

    # ------------------------------------------------------------------
    # Template / profile operations
    # ------------------------------------------------------------------

    def load_profile(self, name: str) -> None:
        """Load a workflow profile (config + directives + stages).

        Args:
            name: Profile name (e.g. ``"coding"``, ``"research"``, ``"ecommerce"``).
        """
        self._check_open()
        return self._templates_mgr.load_profile(name)

    def apply_stage(self, stage: str) -> None:
        """Apply a workflow stage from the loaded profile.

        Args:
            stage: Stage name (e.g. ``"design"``, ``"implementation"``).
        """
        self._check_open()
        return self._templates_mgr.apply_stage(stage)

    def apply_template(self, name: str, **params) -> CommitInfo:
        """Apply a directive template.

        Args:
            name: Template name (e.g. ``"focus"``, ``"persona"``).
            **params: Template parameters.

        Returns:
            :class:`CommitInfo` for the directive commit.
        """
        self._check_open()
        return self._templates_mgr.apply(name, **params)

    # ------------------------------------------------------------------
    # Routing operations
    # ------------------------------------------------------------------

    # route() and aroute() are already defined as direct methods below

    # ------------------------------------------------------------------
    # Internal check
    # ------------------------------------------------------------------

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
        provider: Literal["openai", "anthropic", "claude_code"] | None = None,
        tool_profile: str | ToolProfile | None = None,
        tool_result_format: Literal["minimal", "json", "verbose"] | None = None,
        retry: RetryConfig | None = None,
        prompt_dir: str | Path | None = None,
    ) -> Tract:
        """Open (or create) a Tract repository.

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

        # Persistence repository + file-based state
        persistence_repo = SqlitePersistenceRepository(session)
        tract._persistence_repo = persistence_repo
        behavioral_spec_repo = SqliteBehavioralSpecRepository(session)
        tract._behavioral_spec_repo = behavioral_spec_repo
        tract._db_path = path

        # Create managers now that all repos are set
        tract._create_managers()

        # Seed base tags (idempotent)
        tract._tags_mgr._seed_base()

        # Validate: model= and default_config= are mutually exclusive
        if model is not None and default_config is not None:
            raise ValueError(
                "Cannot specify both model= and default_config=. "
                "Use default_config=LLMConfig(model=...) for full control."
            )

        # Normalize empty api_key to None
        if api_key is not None and not api_key.strip():
            api_key = None

        # Validate: api_key= and llm_client= are mutually exclusive
        if api_key is not None and llm_client is not None:
            raise ValueError(
                "Cannot specify both api_key= and llm_client=. "
                "Use api_key= for built-in OpenAI client, or "
                "llm_client= for a custom client."
            )

        # Auto-configure LLM via Claude Code CLI (no API key needed)
        if provider == "claude_code" or (
            api_key is None and llm_client is None and provider is None
            and model is not None
            and any(alias in model for alias in ("sonnet", "opus", "haiku", "claude"))
        ):
            if api_key is None and llm_client is None:
                from tract.llm.claude_code import ClaudeCodeClient

                client = ClaudeCodeClient(
                    model=model or "sonnet",
                )
                tract.config.configure_llm(client)
                tract._llm_state.owns_llm_client = True
                if model is not None:
                    tract._llm_state.default_config = LLMConfig(model=model)

        elif api_key is not None:
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
            tract.config.configure_llm(client)
            tract._llm_state.owns_llm_client = True
            if model is not None:
                tract._llm_state.default_config = LLMConfig(model=model)

        # Configure externally-provided LLM client (caller owns lifecycle)
        elif llm_client is not None:
            tract.config.configure_llm(llm_client)
            tract._llm_state.owns_llm_client = False

        # Apply default_config if provided (without api_key)
        if default_config is not None and tract._llm_state.default_config is None:
            tract._llm_state.default_config = default_config

        # Apply per-operation configs (new typed path)
        if operations is not None:
            tract._llm_state.operation_configs = operations
        # Apply per-operation configs (legacy dict path)
        elif operation_configs is not None:
            tract.config.configure_operations(**operation_configs)

        # Reasoning commit config
        tract._llm_state.commit_reasoning = commit_reasoning

        # Auto-message config
        if auto_message is not False:
            tract._llm_state.auto_message_enabled = True
            if isinstance(auto_message, str):
                tract._llm_state.operation_configs = replace(
                    tract._llm_state.operation_configs,
                    message=LLMConfig(model=auto_message, temperature=0.0),
                )
            elif isinstance(auto_message, LLMConfig):
                tract._llm_state.operation_configs = replace(
                    tract._llm_state.operation_configs, message=auto_message,
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
            tract._llm_state.retry_config = retry

        # Prompt directory: explicit > auto-discover .tract/prompts/
        if prompt_dir is not None:
            tract._prompt_dir = prompt_dir
        else:
            from pathlib import Path as _Path
            auto = _Path(".tract") / "prompts"
            if auto.is_dir():
                tract._prompt_dir = auto

        # Create deferred managers that need full LLM state
        tract._create_deferred_managers()

        # Load persisted state (needs persistence_mgr from deferred managers)
        tract._persistence_mgr._load_persisted_state(
            profile_registry=tract._profile_registry,
            template_registry=tract._template_registry,
        )

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
        # Create managers (repos are set from __init__ for from_components)
        tract._create_managers()
        if llm_client is not None:
            tract.config.configure_llm(llm_client)
            tract._llm_state.owns_llm_client = False
        # Create deferred managers that need full LLM state
        tract._create_deferred_managers()
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
        mgr = getattr(self, '_config_mgr', None)
        if mgr is not None:
            return mgr.config_index
        # Fallback for pre-open access
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
        if priority is not None:
            self._annotations_mgr.set(info.commit_hash, actual_priority)
        return info


    # ------------------------------------------------------------------
    # Workflow profiles
    # ------------------------------------------------------------------


    # Backward-compatible alias


    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------


    def route(
        self,
        query: str,
        *,
        router: SemanticRouter | None = None,
        apply: bool = False,
    ) -> RoutingResult:
        """Route a query to the best matching branch, stage, or workflow.

        Uses a :class:`~tract.routing.SemanticRouter` for LLM-powered
        routing, falling back to fuzzy matching from the default routing
        table.

        Args:
            query: The user query or intent string.
            router: An optional :class:`~tract.routing.SemanticRouter`.
                If ``None``, uses the default routing table with fuzzy matching only.
            apply: If ``True``, automatically apply the route (switch branch,
                apply stage, etc.).

        Returns:
            A :class:`~tract.routing.RoutingResult`.

        Example::

            result = t.routing.route("time to start implementing")
            print(result.route.target, result.route.confidence)
        """
        return self._routing_mgr.route(query=query, router=router, apply=apply)

    async def aroute(
        self,
        query: str,
        *,
        router: SemanticRouter | None = None,
        apply: bool = False,
    ) -> RoutingResult:
        """See :attr:`routing` manager :meth:`aroute`."""
        return await self._routing_mgr.aroute(query=query, router=router, apply=apply)

    # ------------------------------------------------------------------
    # Context intelligence (cherry-pick & deduplication)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Autonomous operations
    # ------------------------------------------------------------------


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
        self._check_open()
        self._middleware_mgr._run("pre_transition", target=target)

        payload = None
        if handoff == "full":
            payload = str(self.compile().to_dicts())
        elif handoff == "summary":
            k = self._config_mgr.get("handoff_summary_k") or 3
            payload = str(self.compile(strategy="adaptive", strategy_k=k).to_dicts())
        elif handoff != "none":
            payload = handoff

        existing = {b.name for b in self._branches_mgr.list()}
        if target not in existing:
            from tract.operations.branch import create_branch

            create_branch(target, self._tract_id, self._ref_repo, self._commit_repo)
            self._commit_session()
        self._branches_mgr.switch(target)

        result = None
        if payload:
            result = self.system(f"Context handoff:\n{payload}", message=f"handoff to {target}")

        self._middleware_mgr._run("post_transition", target=target)
        return result

    @property
    def spawn_repo(self) -> SqliteSpawnPointerRepository | None:
        """Expose spawn repo for internal use by Session."""
        return self._spawn_repo

    # ------------------------------------------------------------------
    # Spawn relationship helpers
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Subagent communication
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Tool tracking
    # ------------------------------------------------------------------


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
        tools = self._tools_mgr._gather_for_compile()
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
            auto_tags = self._tags_mgr._classify(
                _ctype, role=_role, operation=operation, metadata=metadata,
            )

        # Merge explicit tags with auto-classified tags (deduplicated)
        explicit_tags = list(tags) if tags else []
        all_tags = list(dict.fromkeys(explicit_tags + auto_tags))  # preserves order, deduplicates

        # Validate tags against registry in strict mode
        if all_tags:
            self._tags_mgr._validate(all_tags)

        # Pre-commit middleware (can block)
        self._middleware_mgr._run("pre_commit", pending=content)

        # Config enforcement: max_commit_tokens
        max_commit_tokens = self._config_mgr.get("max_commit_tokens")
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
        effective_tools = tools if tools is not None else self._tools_mgr.get()
        if effective_tools is not None and self._tool_schema_repo is not None:
            self._tools_mgr._store_and_link(info.commit_hash, effective_tools)

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
        self._middleware_mgr._run("post_commit", commit=info)

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
        self._check_open()
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

        content: InstructionContent | DialogueContent
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
            self._annotations_mgr.set(
                info.commit_hash, priority,
                retain=retain, retain_match=retain_match,
            )
        if improve:
            if not self._config_mgr._has_llm_client("improve"):
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
        """See :attr:`llm` manager :meth:`_improve_commit`."""
        return self._llm_mgr._improve_commit(original_info=original_info, text=text, role=role)

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
        self._check_open()
        from tract.models.content import ReasoningContent

        return self.commit(
            ReasoningContent(text=text, format=format),
            message=message or self._llm_mgr._auto_message("reasoning", text),
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
        self._check_open()
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


    # ------------------------------------------------------------------
    # Tool query API
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Conversation layer (chat/generate)
    # ------------------------------------------------------------------


    # Shared sentinels — must be the same objects as in managers for identity checks
    from tract.managers.state import TOOLS_SENTINEL as _TOOLS_SENTINEL
    from tract.managers.state import PROFILE_SENTINEL as _PROFILE_SENTINEL


    # ------------------------------------------------------------------
    # Async LLM methods
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Compile record accessors
    # ------------------------------------------------------------------


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
        recent_ratio: float | None = ...,
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
        recent_ratio: float | None = ...,
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
        recent_ratio: float | None = None,
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
            recent_ratio: If set, compute ``strategy_k`` as a ratio of the
                total effective commits. Must be between 0.0 and 1.0.
                Overrides ``strategy_k`` when both are provided. Only used
                when ``strategy`` is ``"adaptive"``.

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
        self._middleware_mgr._run("pre_compile")

        # Strategy kwargs to forward to the compiler
        _strategy_kw = dict(strategy=strategy, strategy_k=strategy_k, recent_ratio=recent_ratio)

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
            or recent_ratio is not None
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
        new_priorities = (
            [result.priorities[i] for i in final_order]
            if result.priorities
            else []
        )

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
            priorities=new_priorities,
        )


    # ------------------------------------------------------------------
    # Evaluate (one-shot Judgment)
    # ------------------------------------------------------------------

    def evaluate(self, judgment, *, llm_client=None):
        """Evaluate a one-shot Judgment against this tract's context state.

        Args:
            judgment: A Judgment instance specifying what to evaluate.
            llm_client: Optional LLM client override.

        Returns:
            JudgmentResult with the parsed output.
        """
        self._check_open()
        return judgment.evaluate(self, llm_client=llm_client)

    async def aevaluate(self, judgment, *, llm_client=None):
        """Async version of evaluate()."""
        self._check_open()
        return await judgment.aevaluate(self, llm_client=llm_client)

    # ------------------------------------------------------------------
    # Tag system
    # ------------------------------------------------------------------


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
        tags = self._tags_mgr._classify(
            content_type, role=role, operation=operation, metadata=metadata,
        )
        message = self._llm_mgr._auto_message(content_type, text)
        return message, tags


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

        Uses a single recursive CTE query instead of per-commit BFS.
        """
        if self._parent_repo is None:
            return list(
                self._commit_repo.get_ancestors(
                    start_hash, limit=limit, op_filter=op_filter,
                )
            )

        # Fetch a larger window when op_filter will discard rows
        fetch_limit = None if op_filter is not None else limit
        all_rows = list(
            self._commit_repo.get_ancestors_with_merges(start_hash, limit=fetch_limit)
        )

        # Apply op_filter in Python (content_type filtering etc.)
        if op_filter is not None:
            all_rows = [r for r in all_rows if r.operation == op_filter]
            if limit is not None:
                all_rows = all_rows[:limit]

        return all_rows


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


    # ------------------------------------------------------------------
    # Snapshot system
    # ------------------------------------------------------------------


    @property
    def llm_client(self) -> LLMClient | None:
        """The configured LLM client, or ``None``."""
        return self._llm_state.llm_client

    @property
    def default_config(self) -> LLMConfig | None:
        """The default LLM configuration, or ``None``."""
        return self._llm_state.default_config

    @property
    def retry_config(self) -> RetryConfig | None:
        """The retry configuration for LLM calls, or ``None``."""
        return self._llm_state.retry_config

    @property
    def commit_reasoning(self) -> bool:
        """Whether reasoning traces are committed during agent loops."""
        return self._llm_state.commit_reasoning

    @property
    def tool_summarization_config(self) -> ToolSummarizationConfig | None:
        """Tool summarization configuration, or ``None`` if disabled."""
        return self._llm_state.tool_summarization_config

    @property
    def operation_configs(self) -> OperationConfigs:
        """Current per-operation LLM configurations (read-only, frozen)."""
        return self._llm_state.operation_configs

    @property
    def operation_clients(self) -> OperationClients:
        """Current per-operation LLM client overrides (read-only, frozen)."""
        return self._llm_state.operation_clients

    @property
    def operation_prompts(self) -> OperationPrompts:
        """Current per-operation prompt overrides (read-only, frozen)."""
        return self._llm_state.operation_prompts


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
            delete_branch = self._config.delete_branch_on_merge

        # Determine resolver (handles "llm" string, None -> default, callables)
        effective_resolver = self._config_mgr._resolve_resolver(resolver, "merge")

        # If using default resolver AND per-call config overrides exist,
        # create a tailored resolver with those overrides.
        if effective_resolver is self._default_resolver:
            merge_config = self._config_mgr._resolve_llm_config(
                "merge", model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
            )
            if merge_config and self._config_mgr._has_llm_client("merge"):
                from tract.llm.resolver import OpenAIResolver

                effective_resolver = OpenAIResolver(
                    self._config_mgr._resolve_llm_client("merge"),
                    model=merge_config.get("model"),
                    temperature=merge_config.get("temperature", 0.3),
                    max_tokens=merge_config.get("max_tokens", 2048),
                )

        # Pre-merge middleware (can block)
        self._middleware_mgr._run("pre_merge")

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
        if hasattr(self, '_config_mgr') and self._config_mgr is not None:
            self._config_mgr.invalidate_cache()
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
        if hasattr(self, '_config_mgr') and self._config_mgr is not None:
            self._config_mgr.invalidate_cache()

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
        self._check_open()
        from tract.operations.rebase import import_commit as _import_commit

        # Resolve commit hash (supports prefixes and branch names)
        resolved = self._branches_mgr.resolve(commit_hash)

        # Determine resolver (handles "llm" string, None -> default, callables)
        effective_resolver = self._config_mgr._resolve_resolver(resolver, "merge")

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
        effective_resolver = self._config_mgr._resolve_resolver(resolver, "merge")

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
        if hasattr(self, '_config_mgr') and self._config_mgr is not None:
            self._config_mgr.invalidate_cache()
        return result


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
        was_in_batch = self._in_batch
        self._in_batch = True
        try:
            yield
            # Success: single commit for the entire batch
            if not was_in_batch:
                self._session.commit()
        except Exception:
            if not was_in_batch:
                self._session.rollback()
            raise
        finally:
            self._in_batch = was_in_batch


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


    # ------------------------------------------------------------------
    # File-based persistence (.tract/ directory)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Behavioral spec persistence (gates, maintainers, profiles, templates)
    # ------------------------------------------------------------------


    def close(self) -> None:
        """Close the session and dispose the engine."""
        if self._closed:
            return
        self._closed = True
        # Close internally-created LLM client (not externally-provided ones)
        owns = self._llm_state.owns_llm_client
        client = self._llm_state.llm_client
        if owns and client is not None:
            try:
                client.close()
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
        owns = self._llm_state.owns_llm_client
        client = self._llm_state.llm_client
        if owns and client is not None:
            _aclose = getattr(client, "aclose", None)
            if _aclose is not None:
                try:
                    await _aclose()
                except Exception:
                    logger.debug("Failed to async-close LLM client", exc_info=True)
                # Prevent sync close() from double-closing the client
                self._llm_state.owns_llm_client = False
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
