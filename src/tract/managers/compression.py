"""Compression manager for Tract.

Extracted from tract.py — handles compress, acompress, compress_tool_calls,
acompress_tool_calls, gc, record_usage, and internal helpers like
_compress_pre, _compress_finalize, _compress_sliding_window,
_compress_tool_calls_pre, _compress_tool_calls_post.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tract.engine.cache import CacheManager
    from tract.engine.commit import CommitEngine
    from tract.engine.compiler import DefaultContextCompiler
    from tract.llm.protocols import LLMClient
    from tract.managers.state import LLMState
    from tract.models.commit import CommitInfo
    from tract.models.compression import CompressResult, GCResult, ToolCompactResult
    from tract.models.config import LLMConfig
    from tract.protocols import CompiledContext, TokenUsage
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitRepository,
        CompileRecordRepository,
        OperationEventRepository,
        ParentRepository,
        RefRepository,
    )

logger = logging.getLogger(__name__)


class CompressionManager:
    """Compression operations: compress, gc, record_usage, and their async variants."""

    def __init__(
        self,
        tract_id: str,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        ref_repo: RefRepository,
        annotation_repo: AnnotationRepository,
        parent_repo: ParentRepository | None,
        event_repo: OperationEventRepository | None,
        compile_record_repo: CompileRecordRepository | None,
        token_counter: Any,
        commit_engine: CommitEngine,
        compiler: DefaultContextCompiler,
        config: Any,  # TractConfig
        cache: CacheManager,
        llm_state: LLMState,  # shared with LLM/Config
        check_open: Callable[[], None],
        commit_fn: Callable,  # Tract.commit
        compile_fn: Callable,  # Tract.compile
        run_middleware: Callable,  # MiddlewareManager._run
        commit_session: Callable,
        resolve_llm_config: Callable,  # ConfigManager._resolve_llm_config
        resolve_llm_client: Callable,  # ConfigManager._resolve_llm_client
        has_llm_client: Callable,  # ConfigManager._has_llm_client
        annotate_fn: Callable,  # AnnotationManager.set
        get_head: Callable[[], str | None],
        get_ancestors: Callable,
        row_to_info: Callable,
        get_session: Callable,  # lambda: self._session
        get_custom_type_registry: Callable,  # lambda: self._custom_type_registry
        extract_content_fn: Callable,  # LLMManager._extract_content
        normalize_usage_dict_fn: Callable,  # Tract._normalize_usage_dict or LLMManager._normalize_usage_dict
        save_compile_record_fn: Callable,  # Tract._save_compile_record
        tool_result_fn: Callable,  # Tract.tool_result
        get_content_fn: Callable,  # Tract.get_content
        find_tool_turns_fn: Callable,  # Tract.find_tool_turns
    ) -> None:
        self._tract_id = tract_id
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._ref_repo = ref_repo
        self._annotation_repo = annotation_repo
        self._parent_repo = parent_repo
        self._event_repo = event_repo
        self._compile_record_repo = compile_record_repo
        self._token_counter = token_counter
        self._commit_engine = commit_engine
        self._compiler = compiler
        self._config = config
        self._cache = cache
        self._llm_state = llm_state
        self._check_open = check_open
        self._commit_fn = commit_fn
        self._compile_fn = compile_fn
        self._run_middleware = run_middleware
        self._commit_session = commit_session
        self._resolve_llm_config = resolve_llm_config
        self._resolve_llm_client = resolve_llm_client
        self._has_llm_client = has_llm_client
        self._annotate_fn = annotate_fn
        self._get_head = get_head
        self._get_ancestors = get_ancestors
        self._row_to_info = row_to_info
        self._get_session = get_session
        self._get_custom_type_registry = get_custom_type_registry
        self._extract_content_fn = extract_content_fn
        self._normalize_usage_dict_fn = normalize_usage_dict_fn
        self._save_compile_record_fn = save_compile_record_fn
        self._tool_result_fn = tool_result_fn
        self._get_content_fn = get_content_fn
        self._find_tool_turns_fn = find_tool_turns_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_pre(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        content: str | None = None,
        system_prompt: str | None = None,
    ) -> tuple[Any, dict, str | None]:
        """Shared pre-LLM guards and setup for compress/acompress.

        Returns (llm_client, llm_kwargs, effective_system_prompt).
        """
        from tract.exceptions import DetachedHeadError

        self._check_open()

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
                "LLM parameters provided (model, temperature, max_tokens, or "
                "llm_config) but no LLM client is configured. Call "
                "configure_llm() or pass api_key to Tract.open(), or provide "
                "content= for manual compression."
            )

        llm_kwargs = self._resolve_llm_config(
            "compress", model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
        ) if llm_client is not None else {}

        effective_system_prompt = system_prompt
        if effective_system_prompt is None and self._llm_state.operation_prompts.compress is not None:
            effective_system_prompt = self._llm_state.operation_prompts.compress

        self._run_middleware("pre_compress")

        return llm_client, llm_kwargs, effective_system_prompt

    def _compress_finalize(
        self,
        range_result: Any,
        *,
        commits: list[str] | None = None,
        from_commit: str | None = None,
        to_commit: str | None = None,
        target_tokens: int | None = None,
        preserve: list[str] | None = None,
        instructions: str | None = None,
        effective_system_prompt: str | None = None,
        llm_kwargs: dict,
    ) -> CompressResult:
        """Shared post-LLM finalization for compress/acompress default strategy."""
        from tract.models.config import LLMConfig
        from tract.operations.compression import (
            _classify_by_priority,
            _commit_compression,
            _partition_around_pinned,
            _resolve_commit_range,
        )

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

        session = self._get_session()
        nested = session.begin_nested()
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
                type_registry=self._get_custom_type_registry(),
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

    def _compress_sliding_window(
        self,
        *,
        window_size: int,
        target_tokens: int | None,
        preserve: list[str] | None,
        content: str | None,
        instructions: str | None,
        system_prompt: str | None,
        llm_client: Any,
        llm_kwargs: dict,
        two_stage: bool | None,
        sliding_window_compress_fn: Callable,
        _classify_by_priority_fn: Callable,
        _commit_compression_fn: Callable,
        _partition_around_pinned_fn: Callable,
        _reconstruct_content_fn: Callable,
    ) -> CompressResult:
        """Internal helper for sliding-window compression strategy.

        Keeps the most recent ``window_size`` commits in full detail and
        compresses everything older into summaries.  PINNED commits outside
        the window are preserved verbatim.
        """
        from tract.exceptions import CompressionError
        from tract.models.config import LLMConfig

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
            type_registry=self._get_custom_type_registry(),
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
        session = self._get_session()
        nested = session.begin_nested()
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
                type_registry=self._get_custom_type_registry(),
                expected_head=head_hash,
                generation_config=range_result.generation_config,
            )

            # Step 4: Replay window commits on top of the compressed chain
            # Window commits are in newest-first order; reverse to oldest-first
            for row in reversed(window_commits):
                content_model = _reconstruct_content_fn(
                    row, self._blob_repo, self._get_custom_type_registry()
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

    def _compress_tool_calls_pre(
        self,
        commits: list[str] | None = None,
        *,
        name: str | None = None,
        target_tokens: int | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        include_context: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
    ) -> tuple[Any, list[dict], dict, list, list]:
        """Shared pre-LLM logic for compress_tool_calls/acompress_tool_calls.

        Returns (llm_client, messages, llm_kwargs, results_to_compact, turns).
        """
        self._check_open()
        from tract.exceptions import CompressionError
        from tract.operations.compression import build_role_label
        from tract.prompts.summarize import (
            TOOL_COMPACT_CONTEXT_SYSTEM,
            TOOL_COMPACT_SYSTEM,
            build_tool_compact_prompt,
        )

        turns = self._find_tool_turns_fn(name=name)

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
            call_text = self._get_content_fn(turn.call) or ""
            parts.append(f"{build_role_label('assistant', call_meta)}: {call_text}")

            for r in turn.results:
                r_meta = r.metadata or {}
                r_text = self._get_content_fn(r) or ""
                parts.append(f"{build_role_label('tool', r_meta)}: {r_text}")
                results_to_compact.append(r)

        if not results_to_compact:
            raise CompressionError("No tool results found to compact")

        sequence_text = "\n".join(parts)

        context_text: str | None = None
        if include_context:
            try:
                compiled = self._compile_fn()
                context_text = "\n".join(
                    f"{m.role}: {m.content}" for m in compiled.messages
                )
            except Exception:
                context_text = None

        prompt = build_tool_compact_prompt(
            sequence_text,
            result_count=len(results_to_compact),
            target_tokens=target_tokens,
            instructions=instructions,
            context_text=context_text,
        )
        if system_prompt is not None:
            sys_prompt = system_prompt
        elif include_context and context_text is not None:
            sys_prompt = TOOL_COMPACT_CONTEXT_SYSTEM
        else:
            sys_prompt = TOOL_COMPACT_SYSTEM

        llm_kwargs: dict = {}
        if any(v is not None for v in (model, temperature, max_tokens, llm_config)):
            resolved = self._resolve_llm_config(
                "compress", model=model, temperature=temperature,
                max_tokens=max_tokens, llm_config=llm_config,
            )
            if resolved:
                llm_kwargs = resolved

        llm = self._resolve_llm_client("compress")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        return llm, messages, llm_kwargs, results_to_compact, turns

    def _compress_tool_calls_post(
        self,
        response: Any,
        llm: Any,
        results_to_compact: list,
        turns: list,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
    ) -> ToolCompactResult:
        """Shared post-LLM logic for compress_tool_calls/acompress_tool_calls."""
        from tract.exceptions import CompressionError
        from tract.models.compression import ToolCompactResult
        from tract.models.config import LLMConfig

        raw_content = self._extract_content_fn(response, client=llm)
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

            edited = self._tool_result_fn(
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
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            tool_names=tuple(all_tool_names),
            turn_count=len(turns),
            config=effective_config,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

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
        from tract.operations.compression import (
            _classify_by_priority,
            _commit_compression,
            _partition_around_pinned,
            _reconstruct_content,
            compress_range,
            sliding_window_compress,
        )

        llm_client, llm_kwargs, effective_system_prompt = self._compress_pre(
            model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
            content=content, system_prompt=system_prompt,
        )

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
            type_registry=self._get_custom_type_registry(),
            two_stage=two_stage or False,
        )

        return self._compress_finalize(
            range_result,
            commits=commits, from_commit=from_commit, to_commit=to_commit,
            target_tokens=target_tokens, preserve=preserve,
            instructions=instructions,
            effective_system_prompt=effective_system_prompt,
            llm_kwargs=llm_kwargs,
        )

    def compress_tool_calls(
        self,
        commits: list[str] | None = None,
        *,
        name: str | None = None,
        target_tokens: int | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        include_context: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        triggered_by: str | None = None,
    ) -> ToolCompactResult:
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
            include_context: If ``True``, compile the current context
                and include it in the compaction prompt so the LLM can
                judge relevance. Defaults to ``False``.
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
        llm, messages, llm_kwargs, results_to_compact, turns = self._compress_tool_calls_pre(
            commits, name=name, target_tokens=target_tokens,
            instructions=instructions, system_prompt=system_prompt,
            include_context=include_context, model=model,
            temperature=temperature, max_tokens=max_tokens, llm_config=llm_config,
        )

        response = llm.chat(messages, **llm_kwargs)

        return self._compress_tool_calls_post(
            response, llm, results_to_compact, turns,
            model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
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
        from tract.exceptions import ContentValidationError, TraceError
        from tract.protocols import CompiledContext, TokenUsage

        self._check_open()
        # Normalize input
        if isinstance(usage, dict):
            usage = self._normalize_usage_dict_fn(usage)
        elif not isinstance(usage, TokenUsage):
            raise ContentValidationError(
                f"Expected TokenUsage or dict, got {type(usage).__name__}"
            )

        current_head = self._get_head()
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
            self._compile_fn()
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
                self._save_compile_record_fn(target_hash, context_tokens, updated.commit_count, token_source, updated.commit_hashes)
            return self._cache.to_compiled(updated)

        # Fallback (custom compiler, no snapshot): return minimal result
        context_tokens = usage.prompt_tokens + usage.completion_tokens
        token_source = f"api:{usage.prompt_tokens}+{usage.completion_tokens}"
        # Persist as compile record even without snapshot
        if self._compile_record_repo is not None:
            self._save_compile_record_fn(target_hash, context_tokens, 0, token_source)
        return CompiledContext(
            messages=[],
            token_count=context_tokens,
            commit_count=0,
            token_source=token_source,
        )

    # ------------------------------------------------------------------
    # Async methods
    # ------------------------------------------------------------------

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
            acompress_range,
            sliding_window_compress,
        )

        llm_client, llm_kwargs, effective_system_prompt = self._compress_pre(
            model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
            content=content, system_prompt=system_prompt,
        )

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

        # Async LLM summarization
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
            type_registry=self._get_custom_type_registry(),
            two_stage=two_stage or False,
        )

        return self._compress_finalize(
            range_result,
            commits=commits, from_commit=from_commit, to_commit=to_commit,
            target_tokens=target_tokens, preserve=preserve,
            instructions=instructions,
            effective_system_prompt=effective_system_prompt,
            llm_kwargs=llm_kwargs,
        )

    async def acompress_tool_calls(
        self,
        commits: list[str] | None = None,
        *,
        name: str | None = None,
        target_tokens: int | None = None,
        instructions: str | None = None,
        system_prompt: str | None = None,
        include_context: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        llm_config: LLMConfig | None = None,
        triggered_by: str | None = None,
    ) -> ToolCompactResult:
        """Async version of :meth:`compress_tool_calls`."""
        from tract.llm.protocols import acall_llm

        llm, messages, llm_kwargs, results_to_compact, turns = self._compress_tool_calls_pre(
            commits, name=name, target_tokens=target_tokens,
            instructions=instructions, system_prompt=system_prompt,
            include_context=include_context, model=model,
            temperature=temperature, max_tokens=max_tokens, llm_config=llm_config,
        )

        response = await acall_llm(llm, messages, **llm_kwargs)

        return self._compress_tool_calls_post(
            response, llm, results_to_compact, turns,
            model=model, temperature=temperature,
            max_tokens=max_tokens, llm_config=llm_config,
        )
