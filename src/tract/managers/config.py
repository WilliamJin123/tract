"""Configuration manager extracted from ConfigMixin."""

from __future__ import annotations

import json
import logging
from dataclasses import fields as dc_fields, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tract.models.config import LLMConfig, OperationClients, OperationConfigs, OperationPrompts

if TYPE_CHECKING:
    from typing import Any, Callable

    from tract.llm.protocols import LLMClient, ResolverCallable
    from tract.managers.state import LLMState
    from tract.models.commit import CommitInfo
    from tract.models.config import TractConfig, ToolSummarizationConfig
    from tract.operations.config_index import ConfigIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid operation names for configure_operations / configure_clients
# ---------------------------------------------------------------------------
_VALID_OPERATION_NAMES: frozenset[str] = frozenset({"chat", "merge", "compress", "message"})
_VALID_PROMPT_NAMES: frozenset[str] = frozenset({"compress", "merge", "message", "commit_message"})


class ConfigManager:
    """Configuration: configure, get_config, configure_llm, configure_operations, etc."""

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
        "compile_recent_ratio": (float, int),
        "handoff_summary_k": (int,),
    }

    def __init__(
        self,
        tract_id: str,
        commit_engine,  # CommitEngine
        ref_repo,
        commit_repo,
        blob_repo,
        event_repo,
        persistence_repo,  # for config_history / _log_config_change
        config: TractConfig,
        llm_state: LLMState,  # shared mutable bag
        parent_repo=None,
        check_open: Callable | None = None,  # Callable
        commit_session: Callable | None = None,  # Callable
        commit_fn: Callable | None = None,  # Callable - Tract.commit
        get_head: Callable | None = None,  # Callable -> str|None
    ) -> None:
        self._tract_id = tract_id
        self._commit_engine = commit_engine
        self._ref_repo = ref_repo
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._event_repo = event_repo
        self._persistence_repo = persistence_repo
        self._config = config
        self._llm_state = llm_state
        self._parent_repo = parent_repo
        self._check_open_fn = check_open or (lambda: None)
        self._commit_session_fn = commit_session or (lambda: None)
        self._commit_fn = commit_fn
        self._get_head = get_head or (lambda: self._ref_repo.get_head(self._tract_id))

        # Lazy, stale-aware config index cache
        self._config_index: ConfigIndex | None = None

    # ------------------------------------------------------------------
    # Properties delegating to LLMState
    # ------------------------------------------------------------------

    @property
    def llm_client(self) -> LLMClient | None:
        """The configured LLM client, or ``None`` if not configured."""
        return self._llm_state.llm_client

    @property
    def default_config(self) -> LLMConfig | None:
        """The default LLM configuration, or ``None`` if not set."""
        return self._llm_state.default_config

    @property
    def retry_config(self):
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

    # ------------------------------------------------------------------
    # Config index (lazy, stale-aware)
    # ------------------------------------------------------------------

    @property
    def config_index(self) -> ConfigIndex:
        """Get the current config index (built/cached from DAG ancestry)."""
        from tract.operations.config_index import ConfigIndex as _ConfigIndex

        if self._config_index is None or self._config_index.is_stale:
            head = self._get_head()
            if head is None:
                return _ConfigIndex()
            self._config_index = _ConfigIndex.build(
                self._commit_repo, self._blob_repo, head,
                parent_repo=self._parent_repo,
            )
        return self._config_index

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Resolve a config value from DAG.

        Uses DAG precedence: closest to HEAD wins.
        """
        return self.config_index.get(key, default=default)

    def get_all(self) -> dict[str, Any]:
        """Resolve all config key-value pairs from DAG."""
        return self.config_index.get_all()

    def set(self, **settings: Any) -> CommitInfo:
        """Commit config to DAG. Well-known keys are type-checked.

        Raises ValueError if a well-known key has the wrong type.
        Unknown keys pass through without validation.
        None values are valid (unset semantics).
        """
        self._check_open_fn()
        from tract.models.content import ConfigContent

        for key, value in settings.items():
            if value is not None and key in self._WELL_KNOWN_CONFIG_TYPES:
                expected = self._WELL_KNOWN_CONFIG_TYPES[key]
                if not isinstance(value, expected):
                    raise ValueError(
                        f"Config '{key}' expects {expected}, got {type(value).__name__}"
                    )
        content = ConfigContent(settings=settings)
        info = self._commit_fn(content, message=f"configure: {', '.join(settings)}")
        if self._config_index is not None:
            self._config_index.invalidate()
        return info

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
        self._check_open_fn()
        # Close the old client if we own it (prevents resource leak on swap)
        if self._llm_state.owns_llm_client and self._llm_state.llm_client is not None:
            try:
                self._llm_state.llm_client.close()
            except Exception:
                pass
        self._llm_state.llm_client = client
        self._llm_state.owns_llm_client = False  # External client -- caller owns lifecycle
        if resolver is not None:
            self._llm_state.default_resolver = resolver
        else:
            from tract.llm.resolver import OpenAIResolver

            self._llm_state.default_resolver = OpenAIResolver(client)
        self._log_change("llm_client", source="api")

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
        self._check_open_fn()
        if _configs is not None and operation_configs:
            raise TypeError(
                "Cannot mix positional OperationConfigs with keyword arguments"
            )
        if _configs is not None:
            if not isinstance(_configs, OperationConfigs):
                raise TypeError(
                    f"Expected OperationConfigs, got {type(_configs).__name__}"
                )
            self._llm_state.operation_configs = _configs
            self._log_change(
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
        self._llm_state.operation_configs = replace(
            self._llm_state.operation_configs, **operation_configs,
        )
        # Log config change
        self._log_change(
            "operation_config",
            config_json=self._serialize_operation_configs(),
            source="api",
        )

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
        self._check_open_fn()
        if _clients is not None and operation_clients:
            raise TypeError(
                "Cannot mix positional OperationClients with keyword arguments"
            )
        if _clients is not None:
            if not isinstance(_clients, OperationClients):
                raise TypeError(
                    f"Expected OperationClients, got {type(_clients).__name__}"
                )
            self._llm_state.operation_clients = _clients
            self._log_change("operation_client", source="api")
            return
        for name in operation_clients:
            if name not in _VALID_OPERATION_NAMES:
                raise ValueError(
                    f"Unknown operation '{name}'. "
                    f"Valid operations: {', '.join(sorted(_VALID_OPERATION_NAMES))}"
                )
        self._llm_state.operation_clients = replace(
            self._llm_state.operation_clients, **operation_clients,
        )
        # Log config change
        self._log_change("operation_client", source="api")

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
        self._check_open_fn()
        if _prompts is not None and prompt_overrides:
            raise TypeError(
                "Cannot mix positional OperationPrompts with keyword arguments"
            )
        if _prompts is not None:
            if not isinstance(_prompts, OperationPrompts):
                raise TypeError(
                    f"Expected OperationPrompts, got {type(_prompts).__name__}"
                )
            self._llm_state.operation_prompts = _prompts
            self._log_change(
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
        self._llm_state.operation_prompts = replace(
            self._llm_state.operation_prompts, **prompt_overrides,
        )
        self._log_change(
            "prompts",
            config_json=self._serialize_prompts(),
            source="api",
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
            instructions: Per-tool summarization instructions.
            auto_threshold: Token count threshold for auto-summarization.
            default_instructions: Fallback instructions for tools not listed.
            include_context: If True, compile context for summarization LLM.
            system_prompt: Override the default summarization system prompt.
        """
        from tract.models.config import ToolSummarizationConfig

        self._llm_state.tool_summarization_config = ToolSummarizationConfig(
            instructions=instructions or {},
            auto_threshold=auto_threshold,
            default_instructions=default_instructions,
            include_context=include_context,
            system_prompt=system_prompt,
        )

    def history(
        self,
        *,
        change_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get the configuration change audit trail."""
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

    # ------------------------------------------------------------------
    # LLM resolution helpers
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
        """
        op_config = getattr(self._llm_state.operation_configs, operation, None)
        default = self._llm_state.default_config

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

    def _resolve_llm_client(self, operation: str) -> LLMClient:
        """Resolve the LLM client for a given operation.

        Two-level lookup: per-operation client > tract-level default.
        """
        client = getattr(self._llm_state.operation_clients, operation, None)
        if client is not None:
            return client
        if self._llm_state.llm_client is not None:
            return self._llm_state.llm_client
        raise RuntimeError(
            "No LLM client configured. Pass api_key= to Tract.open() "
            "or call configure_llm(client)."
        )

    def _has_llm_client(self, operation: str | None = None) -> bool:
        """Check if an LLM client is available."""
        if operation is not None:
            op_client = getattr(self._llm_state.operation_clients, operation, None)
            if op_client is not None:
                return True
        return self._llm_state.llm_client is not None

    def _resolve_resolver(
        self, resolver: ResolverCallable | str | None, operation: str = "merge",
    ) -> ResolverCallable | None:
        """Resolve a resolver argument to a callable."""
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
            return self._llm_state.default_resolver
        return resolver

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_change(
        self,
        change_type: str,
        *,
        change_key: str | None = None,
        config_json: str | None = None,
        previous_json: str | None = None,
        source: str | None = None,
    ) -> None:
        """Log a configuration change to the audit trail."""
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
        self._commit_session_fn()

    def _serialize_operation_configs(self) -> str | None:
        """Serialize current operation configs to JSON string."""
        result = {}
        for f in dc_fields(self._llm_state.operation_configs):
            val = getattr(self._llm_state.operation_configs, f.name)
            if val is not None:
                result[f.name] = val.to_dict()
        return json.dumps(result) if result else None

    def _serialize_prompts(self) -> str | None:
        """Serialize current operation prompts to JSON string."""
        result = {}
        for f in dc_fields(self._llm_state.operation_prompts):
            val = getattr(self._llm_state.operation_prompts, f.name)
            if val is not None:
                result[f.name] = val
        return json.dumps(result) if result else None
