"""LLM manager for Tract.

Extracted from tract.py — handles generate, chat, run, revise and their
async counterparts, plus internal helpers like _generate_once, _generate_pre,
_build_generation_config, _extract_content, _extract_usage, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from tract.llm.protocols import LLMClient
    from tract.loop import LoopResult
    from tract.managers.state import LLMState
    from tract.models.commit import CommitInfo
    from tract.models.config import LLMConfig, RetryConfig
    from tract.protocols import ChatResponse, CompiledContext, TokenUsage
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitRepository,
        CompileRecordRepository,
        ParentRepository,
        RefRepository,
    )
    from tract.toolkit.models import ToolProfile

logger = logging.getLogger(__name__)


class LLMManager:
    """LLM operations: chat, generate, run, revise and their async counterparts."""

    # Shared sentinels — must match Tract's for delegation identity checks
    from tract.managers.state import TOOLS_SENTINEL as _TOOLS_SENTINEL
    from tract.managers.state import PROFILE_SENTINEL as _PROFILE_SENTINEL

    def __init__(
        self,
        tract_id: str,
        ref_repo: RefRepository,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        annotation_repo: AnnotationRepository,
        parent_repo: ParentRepository,
        compile_record_repo: CompileRecordRepository | None,
        config: Any,  # TractConfig
        token_counter: Any,
        llm_state: LLMState,  # shared with Config/Compression
        check_open: Callable[[], None],
        commit_fn: Callable,  # Tract.commit
        system_fn: Callable,  # Tract.system
        user_fn: Callable,  # Tract.user
        assistant_fn: Callable,  # Tract.assistant
        reasoning_fn: Callable,  # Tract.reasoning
        tool_result_fn: Callable,  # Tract.tool_result
        compile_fn: Callable,  # Tract.compile
        annotate_fn: Callable,  # AnnotationManager.set
        run_middleware: Callable,  # MiddlewareManager._run
        record_usage: Callable,  # CompressionManager.record_usage
        get_tools: Callable,  # ToolManager.get
        commit_session: Callable,
        resolve_llm_config: Callable,  # ConfigManager._resolve_llm_config
        resolve_llm_client: Callable,  # ConfigManager._resolve_llm_client
        has_llm_client: Callable,  # ConfigManager._has_llm_client
        resolve_commit_fn: Callable,  # Tract.resolve_commit
        get_head: Callable[[], str | None],
        as_tools_fn: Callable,  # Tract.as_tools
        save_compile_record_fn: Callable,  # Tract._save_compile_record
        get_in_batch: Callable[[], bool],  # lambda: self._in_batch
        get_tool_profile: Callable,  # lambda: self._tool_profile
        get_custom_tools: Callable,  # lambda: self._custom_tools
        get_tract: Callable,  # lambda: self (Tract instance) — for run_loop
    ) -> None:
        self._tract_id = tract_id
        self._ref_repo = ref_repo
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._annotation_repo = annotation_repo
        self._parent_repo = parent_repo
        self._compile_record_repo = compile_record_repo
        self._config = config
        self._token_counter = token_counter
        self._llm_state = llm_state
        self._check_open = check_open
        self._commit_fn = commit_fn
        self._system_fn = system_fn
        self._user_fn = user_fn
        self._assistant_fn = assistant_fn
        self._reasoning_fn = reasoning_fn
        self._tool_result_fn = tool_result_fn
        self._compile_fn = compile_fn
        self._annotate_fn = annotate_fn
        self._run_middleware = run_middleware
        self._record_usage = record_usage
        self._get_tools = get_tools
        self._commit_session = commit_session
        self._resolve_llm_config = resolve_llm_config
        self._resolve_llm_client = resolve_llm_client
        self._has_llm_client = has_llm_client
        self._resolve_commit_fn = resolve_commit_fn
        self._get_head = get_head
        self._as_tools_fn = as_tools_fn
        self._save_compile_record_fn = save_compile_record_fn
        self._get_in_batch = get_in_batch
        self._get_tool_profile = get_tool_profile
        self._get_custom_tools = get_custom_tools
        self._get_tract = get_tract

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_generation_config(self, response: dict, *, resolved: dict) -> dict:
        """Build a generation_config dict from the LLM response and resolved config.

        Response model is authoritative (overrides resolved model).
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
        c = client if client is not None else self._llm_state.llm_client
        if c is not None and hasattr(c, "extract_content"):
            return c.extract_content(response)
        # Default: OpenAI format
        try:
            return response["choices"][0]["message"]["content"] or ""
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
        c = client if client is not None else self._llm_state.llm_client
        if c is not None and hasattr(c, "extract_usage"):
            return c.extract_usage(response)
        # Default: OpenAI format
        return response.get("usage")

    def _normalize_usage_dict(self, usage_dict: dict) -> TokenUsage:
        """Normalise provider-specific usage dicts to :class:`TokenUsage`.

        Supports OpenAI (``prompt_tokens``) and Anthropic (``input_tokens``)
        formats.
        """
        from tract.exceptions import ContentValidationError
        from tract.protocols import TokenUsage

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

    def _generate_pre(self) -> None:
        """Shared guards for generate/agenerate."""
        from tract.exceptions import TraceError

        self._check_open()
        if not self._has_llm_client("chat"):
            from tract.llm.errors import LLMConfigError
            raise LLMConfigError(
                "No LLM client configured. Pass api_key= or llm_client= "
                "to Tract.open(), or call configure_llm(client)."
            )
        if self._get_in_batch():
            raise TraceError("chat()/generate() cannot be used inside batch()")

    def _generate_validate_loop(
        self,
        response: ChatResponse,
        validator: Callable,
        attempt: int,
        intermediate_hashes: list[str],
        hide_retries: bool,
        max_retries: int,
        retry_prompt: str | None,
    ) -> tuple[ChatResponse | None, str | None]:
        """Process one validation attempt. Returns (final_response, diagnosis).

        If final_response is not None, the loop should return it.
        If None, diagnosis is returned for steering.
        """
        import dataclasses as _dc

        from tract.models.annotations import Priority

        ok, diagnosis = validator(response.text)
        if ok:
            if hide_retries and intermediate_hashes:
                for h in intermediate_hashes:
                    self._annotate_fn(h, Priority.SKIP,
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

            return response, None

        failed_hash = self._get_head()
        if failed_hash:
            intermediate_hashes.append(failed_hash)

        if attempt < max_retries:
            steering = retry_prompt or "Your previous response did not pass validation. Please try again."
            if diagnosis:
                steering = f"{steering}\n\nDiagnosis: {diagnosis}"
            steering_info = self._user_fn(steering)
            intermediate_hashes.append(steering_info.commit_hash)

        return None, diagnosis

    def _generate_once_pre(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> tuple[Any, list[dict], dict]:
        """Pre-LLM logic for _generate_once: compile, resolve, middleware.

        Returns (chat_client, messages, llm_kwargs).
        """
        compiled = self._compile_fn()
        messages = compiled.to_dicts()

        if self._compile_record_repo is not None:
            self._save_compile_record_fn(
                self._get_head() or "",
                compiled.token_count,
                compiled.commit_count,
                compiled.token_source,
                compiled.commit_hashes,
            )

        chat_client = self._resolve_llm_client("chat")
        llm_kwargs = self._resolve_llm_config(
            "chat", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config, **kwargs,
        )
        if compiled.tools:
            llm_kwargs["tools"] = compiled.tools

        self._run_middleware(
            "pre_generate",
            pending={"messages": messages, "config": llm_kwargs},
        )

        return chat_client, messages, llm_kwargs

    def _generate_once_post(
        self,
        response: Any,
        chat_client: Any,
        llm_kwargs: dict,
        *,
        message: str | None = None,
        metadata: dict | None = None,
        reasoning: bool = True,
    ) -> ChatResponse:
        """Post-LLM logic for _generate_once: extract, commit, usage."""
        from tract.models.config import LLMConfig
        from tract.protocols import ChatResponse, ToolCall as _ToolCall

        text = self._extract_content(response, client=chat_client)
        usage_dict = self._extract_usage(response, client=chat_client)

        self._run_middleware(
            "post_generate",
            pending={
                "response": text or "",
                "tokens_used": (
                    usage_dict.get("total_tokens", 0) if usage_dict else 0
                ),
            },
        )

        reasoning_text: str | None = None
        reasoning_format: str = "parsed"
        if hasattr(chat_client, "extract_reasoning"):
            reasoning_result = chat_client.extract_reasoning(response)
            if reasoning_result is not None:
                reasoning_text, reasoning_format = reasoning_result
                if reasoning_format == "think_tags":
                    import re as _re
                    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

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

        gen_config = self._build_generation_config(response, resolved=llm_kwargs)

        reasoning_commit_info: CommitInfo | None = None
        if reasoning_text and self._llm_state.commit_reasoning and reasoning:
            reasoning_commit_info = self._reasoning_fn(
                reasoning_text,
                format=reasoning_format,
            )

        commit_meta = metadata
        if tool_calls:
            commit_meta = {**(metadata or {}), "tool_calls": [tc.to_dict() for tc in tool_calls]}
        commit_info = self._assistant_fn(
            text, message=message, metadata=commit_meta, generation_config=gen_config
        )

        usage = None
        if usage_dict:
            usage = self._normalize_usage_dict(usage_dict)
            self._record_usage(usage)

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
        """Single generate attempt (no retry). Internal helper."""
        from tract.tract import _retry_with_backoff

        chat_client, messages, llm_kwargs = self._generate_once_pre(
            model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config, **kwargs,
        )

        effective_retry = retry or self._llm_state.retry_config
        response = _retry_with_backoff(
            chat_client.chat, effective_retry, messages, **llm_kwargs,
        )

        return self._generate_once_post(
            response, chat_client, llm_kwargs,
            message=message, metadata=metadata, reasoning=reasoning,
        )

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
        from tract.tract import _aretry_with_backoff

        chat_client, messages, llm_kwargs = self._generate_once_pre(
            model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config, **kwargs,
        )

        effective_retry = retry or self._llm_state.retry_config

        async def _do_llm_call() -> Any:
            return await acall_llm(chat_client, messages, **llm_kwargs)

        response = await _aretry_with_backoff(_do_llm_call, effective_retry)

        return self._generate_once_post(
            response, chat_client, llm_kwargs,
            message=message, metadata=metadata, reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

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
        self._generate_pre()

        if validator is None:
            return self._generate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
                reasoning=reasoning, retry=retry, **kwargs,
            )

        intermediate_hashes: list[str] = []
        last_diagnosis: str | None = None

        for attempt in range(max_retries + 1):
            response = self._generate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
                reasoning=reasoning, retry=retry, **kwargs,
            )

            final, diagnosis = self._generate_validate_loop(
                response, validator, attempt, intermediate_hashes,
                hide_retries, max_retries, retry_prompt,
            )
            if final is not None:
                return final
            last_diagnosis = diagnosis

        from tract.exceptions import RetryExhaustedError
        raise RetryExhaustedError(
            attempts=max_retries + 1,
            last_diagnosis=last_diagnosis or "validation failed",
            last_result=response.text,
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
        self._user_fn(text, message=message, name=name, metadata=metadata)
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

        tract = self._get_tract()

        # Resolve tools
        if tools is self._TOOLS_SENTINEL:
            effective_profile = (
                self._get_tool_profile() or "compact"
            ) if profile is self._PROFILE_SENTINEL else profile
            resolved_tools = self._as_tools_fn(
                profile=effective_profile,
                tool_names=tool_names,
                format="openai",
            )
        else:
            resolved_tools = tools  # type: ignore[assignment]  # user-supplied tools passthrough

        # Merge custom tool handlers from @t.tool into tool_handlers
        custom_tools = self._get_custom_tools()
        if custom_tools:
            merged_handlers = {
                name: td.handler for name, td in custom_tools.items()
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
            tract,
            task=task,
            config=config,
            llm_client=llm_client,
            tools=resolved_tools,
            tool_handlers=tool_handlers,
            on_step=on_step,
            on_token=on_token,
            on_tool_result=on_tool_result,
        )

    def _revise_post(
        self,
        response: ChatResponse,
        commit_hash: str,
        prompt: str,
        message: str | None,
    ) -> ChatResponse:
        """Shared post-chat logic for revise/arevise: resolve, edit, skip."""
        import dataclasses as _dc

        from tract.models.annotations import Priority

        resolved = self._resolve_commit_fn(commit_hash)
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

        shorthand = {"system": self._system_fn, "user": self._user_fn, "assistant": self._assistant_fn}
        edit_fn = shorthand.get(role, self._assistant_fn)
        edit_info = edit_fn(
            response.text,
            edit=resolved,
            message=message or f"revise: {prompt[:60]}",
        )

        if response.commit_info.parent_hash is not None:
            self._annotate_fn(response.commit_info.parent_hash, Priority.SKIP)
        self._annotate_fn(response.commit_info.commit_hash, Priority.SKIP)
        if response.reasoning_commit is not None:
            self._annotate_fn(response.reasoning_commit.commit_hash, Priority.SKIP)

        return _dc.replace(
            response,
            commit_info=edit_info,
            prompt=prompt,
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
        response = self.chat(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config=llm_config,
            reasoning=reasoning,
            **kwargs,
        )
        return self._revise_post(response, commit_hash, prompt, message)

    # ------------------------------------------------------------------
    # Auto-message generation
    # ------------------------------------------------------------------

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
        from tract.tract import _fallback_message

        if (
            not self._llm_state.auto_message_enabled
            or self._get_in_batch()
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
        from tract.models.commit import CommitOperation
        from tract.models.content import DialogueContent, InstructionContent

        if role == "system":
            content = InstructionContent(text=improved_text)
        else:
            content = DialogueContent(role=role, text=improved_text)

        edit_info = self._commit_fn(
            content,
            operation=CommitOperation.EDIT,
            edit_target=original_info.commit_hash,
            message=f"improve: {role} message",
        )
        return edit_info

    # ------------------------------------------------------------------
    # Async LLM methods
    # ------------------------------------------------------------------

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
        self._generate_pre()

        if validator is None:
            return await self._agenerate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
                reasoning=reasoning, retry=retry, **kwargs,
            )

        intermediate_hashes: list[str] = []
        last_diagnosis: str | None = None

        for attempt in range(max_retries + 1):
            response = await self._agenerate_once(
                model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
                message=message, metadata=metadata,
                reasoning=reasoning, retry=retry, **kwargs,
            )

            final, diagnosis = self._generate_validate_loop(
                response, validator, attempt, intermediate_hashes,
                hide_retries, max_retries, retry_prompt,
            )
            if final is not None:
                return final
            last_diagnosis = diagnosis

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

        self._user_fn(text, message=message, name=name, metadata=metadata)
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

        tract = self._get_tract()

        # Resolve tools (same logic as sync run)
        if tools is self._TOOLS_SENTINEL:
            effective_profile = (
                self._get_tool_profile() or "compact"
            ) if profile is self._PROFILE_SENTINEL else profile
            resolved_tools = self._as_tools_fn(
                profile=effective_profile,
                tool_names=tool_names,
                format="openai",
            )
        else:
            resolved_tools = tools  # type: ignore[assignment]

        # Merge custom tool handlers from @t.tool into tool_handlers
        custom_tools = self._get_custom_tools()
        if custom_tools:
            merged_handlers = {
                name: td.handler for name, td in custom_tools.items()
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
            tract,
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
        response = await self.achat(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_config=llm_config,
            reasoning=reasoning,
            **kwargs,
        )
        return self._revise_post(response, commit_hash, prompt, message)
