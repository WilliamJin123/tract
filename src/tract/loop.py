"""Default agent loop for Tract.

A minimal compile -> LLM -> tools -> repeat loop. Ships with tract like
the default LLM client -- easily replaced by LangChain, Agno, CrewAI, etc.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tract.exceptions import BlockedError
from tract.models.config import RetryConfig

if TYPE_CHECKING:
    from tract.llm.protocols import LLMClient
    from tract.protocols import CompiledContext, TokenUsage
    from tract.tract import CompileStrategy, Tract

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepMetrics:
    """Metrics for a single loop step."""

    step: int
    duration_ms: float  # wall-clock time for this step
    llm_duration_ms: float  # time spent in LLM call
    tool_count: int  # number of tool calls this step
    tool_names: tuple[str, ...]  # names of tools called
    context_tokens: int  # compiled context token count at start of step
    compressed: bool  # whether auto-compress fired this step


@dataclass(frozen=True)
class LoopResult:
    """Result of a loop execution."""

    status: Literal["completed", "blocked", "max_steps", "error"]
    reason: str | None
    steps: int
    tool_calls: int
    final_response: str | None = None
    compiled: CompiledContext | None = None
    usage: TokenUsage | None = None
    step_usages: tuple[TokenUsage, ...] = ()
    config: Any = None
    step_metrics: tuple[StepMetrics, ...] = ()

    @property
    def total_prompt_tokens(self) -> int:
        """Sum of prompt tokens across all steps."""
        return sum(u.prompt_tokens for u in self.step_usages)

    @property
    def total_completion_tokens(self) -> int:
        """Sum of completion tokens across all steps."""
        return sum(u.completion_tokens for u in self.step_usages)

    @property
    def total_tokens(self) -> int:
        """Sum of all tokens across all steps."""
        return sum(u.total_tokens for u in self.step_usages)

    @property
    def budget_exhausted(self) -> bool:
        """Whether the loop stopped due to token budget exhaustion."""
        return (
            self.status == "completed"
            and self.reason is not None
            and "budget" in self.reason.lower()
        )

    @property
    def total_duration_ms(self) -> float:
        """Total wall-clock time across all steps."""
        return sum(m.duration_ms for m in self.step_metrics)

    @property
    def total_llm_duration_ms(self) -> float:
        """Total time spent in LLM calls."""
        return sum(m.llm_duration_ms for m in self.step_metrics)

    @property
    def compressions_triggered(self) -> int:
        """Number of times auto-compression was triggered."""
        return sum(1 for m in self.step_metrics if m.compressed)

    def pprint(
        self,
        *,
        style: Literal["compact", "chat"] = "compact",
    ) -> None:
        """Pretty-print this loop result using rich formatting.

        Args:
            style: ``"compact"`` (default) shows one-line-per-commit context
                summary. ``"chat"`` shows full message content with panels.
        """
        from tract.formatting import pprint_loop_result

        pprint_loop_result(self, style=style)


@dataclass
class LoopConfig:
    """Configuration for the default loop."""

    max_steps: int = 50
    system_prompt: str | None = None
    strategy: CompileStrategy = "full"
    strategy_k: int = 5
    stop_on_no_tool_call: bool = True
    stream: bool = False
    max_tokens: int | None = None
    step_budget: int | None = None
    """Max total tokens across all steps; loop stops gracefully when exceeded."""
    auto_compress_threshold: float | None = None
    """Ratio of max_tokens (0.0-1.0) that triggers auto-compression when exceeded."""
    tool_validator: Callable[[str, dict], tuple[bool, str | None]] | None = None
    """Validate tool calls before execution: (tool_name, args) -> (ok, error_msg)."""
    transparent_meta_tools: bool = True
    """When True, tract's built-in tool call/result messages are kept in an
    ephemeral buffer (visible to the LLM for one turn) instead of committed
    to the DAG.  This keeps compiled context clean — only the *content*
    produced by tools (e.g. the blob from ``commit()``) persists."""
    presentation: Any = True
    """Layer 2 presentation for tool results sent to the LLM.
    True (default): enabled with default config.
    False/None: disabled (raw output sent to LLM).
    PresentationConfig instance: enabled with custom config."""


def run_loop(
    tract: Tract,
    *,
    task: str | None = None,
    config: LoopConfig | None = None,
    llm_client: LLMClient | None = None,
    tools: list[dict] | None = None,
    tool_handlers: dict[str, Callable[..., Any]] | None = None,
    on_step: Callable[[int, Any], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    on_tool_result: Callable[[str, str, str], None] | None = None,
) -> LoopResult:
    """Run the default agent loop.

    Loop:
    1. Compile context (respecting active config)
    2. Send to LLM with tools
    3. If LLM returns tool calls, execute them
    4. Repeat until: block, max_steps, no tool call, or error

    Args:
        tract: The Tract instance to operate on.
        task: Optional task description. Committed as a user message
            at the start if provided.
        config: Loop configuration. Defaults to LoopConfig().
        llm_client: LLM client to use. Falls back to tract's configured client.
        tools: Tool definitions (OpenAI format). Falls back to tract.runtime.tools.as_tools().
        tool_handlers: Optional mapping of custom tool names to callables.
            When the LLM calls a tool whose name is in this dict, the
            corresponding function is called with the tool arguments as
            keyword arguments.  Tools not in this dict are dispatched to
            tract's built-in :class:`ToolExecutor`.
        on_step: Optional step callback for logging/monitoring.
        on_token: Optional callback for streaming text deltas.  When
            provided and the client supports ``stream()``, the loop
            uses streaming and calls ``on_token(text_chunk)`` for each
            text delta.  The full response is accumulated and processed
            normally after the stream completes.
        on_tool_result: Optional callback fired after each tool execution.
            Signature: ``(tool_name, output, status) -> None``.  Called
            immediately after the tool result is committed.

    Returns:
        LoopResult with status and metadata.
    """
    from tract.toolkit.executor import ToolExecutor

    from tract.tract import _retry_with_backoff

    cfg = config or LoopConfig()
    client = llm_client or tract.llm_client
    if client is None:
        raise ValueError(
            "No LLM client available. Pass llm_client= or configure on Tract.open()."
        )

    if tools is None:
        tools = tract.runtime.tools.as_tools(format="openai")

    # Grab effective LLM config for the result
    effective_config = tract.default_config

    # Commit task as initial user message
    if task:
        tract.user(task)

    steps = 0
    total_tool_calls = 0
    last_response: str | None = None
    last_compiled = None
    step_usages: list[TokenUsage] = []
    step_metrics_list: list[StepMetrics] = []
    executor = ToolExecutor(tract)
    ephemeral_messages: list[dict[str, Any]] = []

    def _make_loop_result(
        status: Literal["completed", "blocked", "max_steps", "error"],
        reason: str | None,
        *,
        usage: TokenUsage | None = None,
    ) -> LoopResult:
        return LoopResult(
            status, reason, steps, total_tool_calls, last_response,
            compiled=last_compiled, usage=usage,
            step_usages=tuple(step_usages),
            config=effective_config,
            step_metrics=tuple(step_metrics_list),
        )

    # Build Layer 2 presenter if presentation is enabled
    presenter = None
    if cfg.presentation is not None and cfg.presentation is not False:
        from tract.toolkit.presentation import ToolPresenter, PresentationConfig

        if isinstance(cfg.presentation, PresentationConfig):
            presenter = ToolPresenter(tract, cfg.presentation)
        else:
            presenter = ToolPresenter(tract)

    # Resolve config once (unlikely to change mid-loop)
    strategy: CompileStrategy = tract.config.get("compile_strategy") or cfg.strategy
    strategy_k: int = tract.config.get("compile_strategy_k") or cfg.strategy_k

    for step in range(cfg.max_steps):
        steps = step + 1
        step_start = time.monotonic()
        context_tokens = 0
        compressed_this_step = False

        # 1. Compile
        try:
            last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
        except BlockedError as e:
            return _make_loop_result("blocked", str(e))
        except Exception as e:
            return _make_loop_result("error", f"Compile failed: {e}")

        context_tokens = last_compiled.token_count if last_compiled else 0

        # Auto-compress if context is too large
        if cfg.auto_compress_threshold is not None and cfg.max_tokens is not None:
            token_count = last_compiled.token_count
            threshold = int(cfg.max_tokens * cfg.auto_compress_threshold)
            if token_count > threshold:
                logger.debug(
                    "Auto-compressing: %d tokens > %d threshold (%.0f%% of %d)",
                    token_count, threshold,
                    cfg.auto_compress_threshold * 100, cfg.max_tokens,
                )
                try:
                    tract.compress(strategy="sliding_window", window_size=cfg.strategy_k)
                    last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
                    compressed_this_step = True
                except Exception as e:
                    logger.warning("Auto-compress failed, continuing with large context: %s", e, exc_info=True)

        # Build messages
        messages = last_compiled.to_dicts()
        # Append ephemeral meta-tool messages from previous step, then clear.
        if ephemeral_messages:
            messages.extend(ephemeral_messages)
            ephemeral_messages = []
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.system_prompt})

        # 2. Call LLM (streaming or sync)
        use_streaming = (
            (on_token is not None or cfg.stream)
            and hasattr(client, "stream")
        )
        llm_kwargs: dict[str, Any] = {}
        if cfg.max_tokens is not None:
            llm_kwargs["max_tokens"] = cfg.max_tokens

        # Pre-generate middleware (can block)
        try:
            tract.middleware._run(
                "pre_generate",
                pending={"messages": messages, "config": llm_kwargs},
            )
        except BlockedError as e:
            return _make_loop_result("blocked", str(e))

        retry_cfg: RetryConfig | None = tract.retry_config
        llm_start = time.monotonic()
        try:
            if use_streaming:
                response = _retry_with_backoff(
                    _stream_to_response, retry_cfg,
                    client, messages, tools, on_token, **llm_kwargs,
                )
            else:
                response = _retry_with_backoff(
                    client.chat, retry_cfg,
                    messages=messages, tools=tools, **llm_kwargs,
                )
        except Exception as e:
            return _make_loop_result("error", f"LLM call failed: {e}")
        llm_duration = time.monotonic() - llm_start

        content = _extract_content(response, client)
        tool_call_list = _extract_tool_calls(response, client)

        # Post-generate middleware (informational)
        _post_gen_usage = _extract_usage(response, client)
        tract.middleware._run(
            "post_generate",
            pending={
                "response": content or "",
                "tokens_used": (
                    _post_gen_usage.get("total_tokens", 0)
                    if _post_gen_usage else 0
                ),
            },
        )

        # Extract and commit reasoning traces (e.g. <think> tags from Qwen)
        try:
            content = _handle_reasoning(response, client, tract, content)
        except Exception:
            logger.warning("Failed to extract reasoning; continuing with original content.", exc_info=True)

        last_response = content

        # Check if ALL tool calls target tract built-in (meta) tools.
        # When transparent_meta_tools is on, meta-tool overhead goes to
        # an ephemeral buffer instead of the DAG.
        all_meta = (
            cfg.transparent_meta_tools
            and bool(tool_call_list)
            and all(_is_meta_tool(tc["name"], tool_handlers) for tc in tool_call_list)
        )

        # Commit assistant response
        if tool_call_list:
            if all_meta:
                # Meta-only: send tool_calls to ephemeral buffer.
                # The tool execution itself writes content to the DAG.
                ephemeral_messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [_build_ephemeral_tool_call(tc) for tc in tool_call_list],
                })
            else:
                # Domain or mixed: commit to DAG (current behavior)
                tc_meta = [
                    {"id": tc["id"], "name": tc["name"],
                     "arguments": tc.get("arguments", {}), "type": "function"}
                    for tc in tool_call_list
                ]
                tc_msg = ", ".join(tc["name"] for tc in tool_call_list)
                tract.assistant(
                    content or "",
                    message=f"call {tc_msg}" if not content else None,
                    metadata={"tool_calls": tc_meta},
                )
        elif content:
            tract.assistant(content)

        # Extract and record usage from the LLM response
        step_usage = _extract_and_record_usage(response, client, tract)
        if step_usage is not None:
            step_usages.append(step_usage)

        # Check step budget
        if cfg.step_budget is not None:
            total_used = sum(u.total_tokens for u in step_usages)
            if total_used >= cfg.step_budget:
                step_end = time.monotonic()
                step_metrics_list.append(StepMetrics(
                    step=steps,
                    duration_ms=(step_end - step_start) * 1000,
                    llm_duration_ms=llm_duration * 1000,
                    tool_count=0,
                    tool_names=(),
                    context_tokens=context_tokens,
                    compressed=compressed_this_step,
                ))
                try:
                    last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
                except Exception as e:
                    logger.warning("Re-compile after budget exhaustion failed, preserving prior context: %s", e, exc_info=True)
                return _make_loop_result(
                    "completed",
                    f"Token budget exhausted ({total_used}/{cfg.step_budget})",
                    usage=step_usage,
                )

        # Callback
        if on_step:
            on_step(steps, response)

        # 3. If no tool calls, check if we should stop
        if not tool_call_list:
            if cfg.stop_on_no_tool_call:
                step_end = time.monotonic()
                step_metrics_list.append(StepMetrics(
                    step=steps,
                    duration_ms=(step_end - step_start) * 1000,
                    llm_duration_ms=llm_duration * 1000,
                    tool_count=0,
                    tool_names=(),
                    context_tokens=context_tokens,
                    compressed=compressed_this_step,
                ))
                # Re-compile to capture the final state (includes assistant commit)
                try:
                    last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
                except Exception as e:
                    logger.warning("Re-compile after LLM completion failed, preserving prior context: %s", e, exc_info=True)
                return _make_loop_result(
                    "completed",
                    "LLM finished (no tool calls)",
                    usage=step_usage,
                )
            # Not stopping — record metrics for this step and continue
            step_end = time.monotonic()
            step_metrics_list.append(StepMetrics(
                step=steps,
                duration_ms=(step_end - step_start) * 1000,
                llm_duration_ms=llm_duration * 1000,
                tool_count=0,
                tool_names=(),
                context_tokens=context_tokens,
                compressed=compressed_this_step,
            ))
            continue

        # 4. Execute tool calls
        for tc in tool_call_list:
            total_tool_calls += 1
            tc_name = tc["name"]
            tc_id = tc.get("id", "")
            tc_args = tc.get("arguments", {})
            result_meta = {"tool_call_id": tc_id, "name": tc_name}
            use_ephemeral = all_meta and _is_meta_tool(tc_name, tool_handlers)

            # Validate tool arguments if validator configured
            if cfg.tool_validator is not None:
                valid, err_msg = cfg.tool_validator(tc_name, tc_args)
                if not valid:
                    error_output = f"Tool validation failed: {err_msg or 'invalid arguments'}"
                    if use_ephemeral:
                        _append_ephemeral_tool_result(ephemeral_messages, tc_id, error_output)
                    else:
                        _commit_tool_result(tract, tc_name, error_output, "error", result_meta)
                    if on_tool_result:
                        on_tool_result(tc_name, error_output, "error")
                    continue

            # Pre-tool-execute middleware (can block to skip this tool)
            try:
                tract.middleware._run(
                    "pre_tool_execute",
                    pending={"tool_name": tc_name, "arguments": tc_args},
                )
            except BlockedError:
                blocked_msg = "Tool execution blocked by middleware"
                if use_ephemeral:
                    _append_ephemeral_tool_result(ephemeral_messages, tc_id, blocked_msg)
                else:
                    _commit_tool_result(tract, tc_name, blocked_msg, "error", result_meta)
                if on_tool_result:
                    on_tool_result(tc_name, blocked_msg, "error")
                continue

            # Custom handler takes priority over built-in executor
            if tool_handlers and tc_name in tool_handlers:
                try:
                    output = tool_handlers[tc_name](**tc_args)
                    _commit_tool_result(tract, tc_name, str(output), "success", result_meta)
                    if on_tool_result:
                        on_tool_result(tc_name, str(output), "success")
                    # Post-tool-execute middleware (informational)
                    tract.middleware._run(
                        "post_tool_execute",
                        pending={"tool_name": tc_name, "result": str(output), "success": True},
                    )
                except Exception as exc:
                    _commit_tool_result(
                        tract, tc_name,
                        f"{type(exc).__name__}: {exc}", "error", result_meta,
                    )
                    if on_tool_result:
                        on_tool_result(tc_name, f"{type(exc).__name__}: {exc}", "error")
                    # Post-tool-execute middleware (informational)
                    tract.middleware._run(
                        "post_tool_execute",
                        pending={
                            "tool_name": tc_name,
                            "result": f"{type(exc).__name__}: {exc}",
                            "success": False,
                        },
                    )
            else:
                result = executor.execute(tc_name, tc_args)
                output_text = result.output if result.success else result.error
                if presenter:
                    output_text = presenter.present_result(result)
                if use_ephemeral:
                    _append_ephemeral_tool_result(ephemeral_messages, tc_id, output_text)
                else:
                    status: Literal["success", "error"] = "success" if result.success else "error"
                    _commit_tool_result(tract, tc_name, output_text, status, result_meta)
                if on_tool_result:
                    on_tool_result(tc_name, output_text, "success" if result.success else "error")
                # Post-tool-execute middleware (informational)
                tract.middleware._run(
                    "post_tool_execute",
                    pending={
                        "tool_name": tc_name,
                        "result": output_text,
                        "success": result.success,
                    },
                )

        # Record step metrics after tool execution
        tool_names_this_step = tuple(tc["name"] for tc in tool_call_list) if tool_call_list else ()
        step_end = time.monotonic()
        step_metrics_list.append(StepMetrics(
            step=steps,
            duration_ms=(step_end - step_start) * 1000,
            llm_duration_ms=llm_duration * 1000,
            tool_count=len(tool_call_list) if tool_call_list else 0,
            tool_names=tool_names_this_step,
            context_tokens=context_tokens,
            compressed=compressed_this_step,
        ))

    return _make_loop_result("max_steps", f"Reached max steps ({cfg.max_steps})")


# ---------------------------------------------------------------------------
# Reasoning extraction
# ---------------------------------------------------------------------------


def _handle_reasoning(
    response: Any, client: Any, tract: Tract, content: str | None,
) -> str | None:
    """Extract reasoning from LLM response, commit it, and return clean content.

    Supports all formats detected by ``LLMClient.extract_reasoning()``:
    Anthropic thinking blocks, OpenAI reasoning_content, ``<think>`` tags, etc.

    Returns the content with reasoning stripped out.
    """
    import re

    if not hasattr(client, "extract_reasoning"):
        # Fallback: strip <think> tags directly if present
        if content and "<think>" in content:
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                reasoning_text = think_match.group(1).strip()
                if reasoning_text and tract.commit_reasoning:
                    tract.reasoning(reasoning_text, format="think_tags")
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                # Also strip unclosed <think> (model hit max_tokens mid-thought)
                content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()
        return content

    reasoning_result = client.extract_reasoning(response)
    if reasoning_result is None:
        return content

    reasoning_text, reasoning_format = reasoning_result

    # Commit reasoning as a separate commit (SKIP by default)
    if reasoning_text and tract.commit_reasoning:
        tract.reasoning(reasoning_text, format=reasoning_format)

    # Strip <think> tags from content when that was the extraction format
    if reasoning_format == "think_tags" and content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()

    return content


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _commit_tool_result(
    tract: Tract,
    tool_name: str,
    output: str,
    status: Literal["success", "error"],
    metadata: dict,
) -> None:
    """Commit a tool result/error to the tract.

    When a :class:`~tract.models.config.ToolSummarizationConfig` is
    configured on the tract and the result exceeds the ``auto_threshold``
    token count, the output is summarized via the LLM before committing.
    The original token count is stored in metadata as
    ``summarized_from_length``.
    """
    from tract.models.content import ToolIOContent

    # --- Auto-summarize large successful results -------------------------
    if status == "success":
        output, metadata = _maybe_summarize_tool_result(
            tract, tool_name, output, metadata,
        )

    payload_key = "result" if status == "success" else "error"
    msg_prefix = "tool result" if status == "success" else "tool error"
    tract.commit(
        ToolIOContent(
            tool_name=tool_name,
            direction="result",
            payload={payload_key: output},
            status=status,
        ),
        message=f"{msg_prefix}: {tool_name}",
        metadata=metadata,
    )


def _maybe_summarize_tool_result(
    tract: Tract,
    tool_name: str,
    output: str,
    metadata: dict,
) -> tuple[str, dict]:
    """Summarize a tool result if ToolSummarizationConfig requires it.

    Returns the (possibly summarized) output and updated metadata.
    If summarization is not configured or the result is under threshold,
    the original output and metadata are returned unchanged.
    """
    config = tract.tool_summarization_config
    if config is None:
        return output, metadata

    # Check threshold
    if config.auto_threshold is None:
        return output, metadata

    # Count tokens in the output
    token_count = tract._token_counter.count_text(output)
    if token_count <= config.auto_threshold:
        return output, metadata

    # Determine summarization instructions
    instructions = config.instructions.get(tool_name, config.default_instructions)

    # Need an LLM client to summarize
    if not tract._config_mgr._has_llm_client("compress"):
        logger.debug(
            "Tool result for %s exceeds threshold (%d > %d) but no LLM client "
            "available for summarization",
            tool_name, token_count, config.auto_threshold,
        )
        return output, metadata

    try:
        summarized = _summarize_tool_output(
            tract, tool_name, output, instructions,
            context=config.context,
            system_prompt=config.system_prompt,
        )
        # Update metadata with original token count
        metadata = {**metadata, "summarized_from_length": token_count}
        logger.debug(
            "Summarized tool result for %s: %d -> %d tokens",
            tool_name, token_count,
            tract._token_counter.count_text(summarized),
        )
        return summarized, metadata
    except Exception:
        logger.debug(
            "Failed to summarize tool result for %s, using original",
            tool_name, exc_info=True,
        )
        return output, metadata


def _summarize_tool_output(
    tract: Tract,
    tool_name: str,
    output: str,
    instructions: str | None,
    *,
    context: Any = None,
    system_prompt: str | None = None,
) -> str:
    """Call the LLM to summarize a single tool result.

    Args:
        tract: The Tract instance (used for LLM client and context).
        tool_name: Name of the tool whose result is being summarized.
        output: The raw tool output to summarize.
        instructions: Per-tool or default summarization instructions.
        context: Optional ContextView or truthy value controlling what DAG
            context to include. String values are used directly.
        system_prompt: Override the default summarization system prompt.

    Returns:
        The summarized output string.
    """
    from tract.prompts.summarize import (
        TOOL_CONTEXT_SUMMARIZE_SYSTEM,
        TOOL_SUMMARIZE_SYSTEM,
        build_summarize_prompt,
    )

    # Resolve system prompt
    if system_prompt is not None:
        sys_prompt = system_prompt
    elif context is not None:
        sys_prompt = TOOL_CONTEXT_SUMMARIZE_SYSTEM
    else:
        sys_prompt = TOOL_SUMMARIZE_SYSTEM

    # Build context text if requested
    context_text: str | None = None
    if context is not None:
        if isinstance(context, str):
            context_text = context
        else:
            try:
                compiled = tract.compile()
                context_text = "\n".join(
                    f"{m.role}: {m.content}" for m in compiled.messages
                )
            except Exception:
                context_text = None

    # Build the user prompt
    user_prompt = build_summarize_prompt(
        f"Tool: {tool_name}\nResult:\n{output}",
        instructions=instructions,
        context_text=context_text,
    )

    llm = tract.config._resolve_llm_client("compress")
    response = llm.chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Extract content from response
    return _extract_content(response, llm) or output


def _extract_usage(response: Any, client: LLMClient | None = None) -> dict | None:
    """Extract usage dict from LLM response (provider-agnostic)."""
    if client is not None and hasattr(client, "extract_usage"):
        try:
            return client.extract_usage(response)
        except Exception:
            pass

    # OpenAI object format
    if hasattr(response, "usage") and response.usage is not None:
        u = response.usage
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }

    # Dict format
    if isinstance(response, dict) and "usage" in response:
        return response["usage"]

    return None


def _extract_and_record_usage(
    response: Any, client: LLMClient | None, tract: Tract,
) -> TokenUsage | None:
    """Extract usage from LLM response and record it on the tract."""
    usage_dict = _extract_usage(response, client)
    if not usage_dict:
        return None
    try:
        usage = tract._normalize_usage_dict(usage_dict)
        tract.record_usage(usage)
        return usage
    except Exception:
        logger.debug("Failed to record usage", exc_info=True)
        return None


def _extract_content(response: Any, client: LLMClient | None = None) -> str | None:
    """Extract text content from LLM response (provider-agnostic)."""
    if client is not None and hasattr(client, "extract_content"):
        try:
            return client.extract_content(response)
        except Exception:
            pass

    # OpenAI object format
    if hasattr(response, "choices"):
        msg = response.choices[0].message
        return msg.content

    # Dict format (OpenAI-style)
    if isinstance(response, dict):
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return response.get("content")

    # String
    if isinstance(response, str):
        return response

    return str(response)


def _extract_tool_calls(response: Any, client: LLMClient | None = None) -> list[dict]:
    """Extract tool calls from LLM response.

    Uses the client's ``extract_tool_calls()`` method when available,
    falling back to generic OpenAI-format parsing.
    """
    # Prefer client's extraction method (duck-typed)
    if client is not None and hasattr(client, "extract_tool_calls"):
        try:
            result = client.extract_tool_calls(response)
            if result is not None:
                return result
        except Exception:
            pass

    # OpenAI object format
    if hasattr(response, "choices"):
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                raw_args = tc.function.arguments
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except (json.JSONDecodeError, ValueError):
                        parsed = {"_raw": raw_args}
                else:
                    parsed = raw_args
                calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": parsed,
                })
            return calls

    # Dict format (OpenAI-style)
    if isinstance(response, dict):
        try:
            msg = response["choices"][0]["message"]
            tcs = msg.get("tool_calls", [])
            if tcs:
                result = []
                for tc in tcs:
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {"_raw": args}
                    result.append(
                        {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": args,
                        }
                    )
                return result
        except (KeyError, IndexError, TypeError):
            pass
        # Flat dict format
        if "tool_calls" in response:
            return response["tool_calls"]

    return []


def _stream_to_response(
    client: Any,
    messages: list[dict],
    tools: list[dict] | None,
    on_token: Callable[[str], None] | None,
    **extra_kwargs: Any,
) -> dict:
    """Run a streaming LLM call, yield text to on_token, return full response.

    Calls ``client.stream()`` and accumulates the result into a single
    response dict (same format as ``client.chat()`` would return).
    """
    from tract.llm.anthropic_client import MessageDone, TextDelta, ThinkingDelta

    kwargs: dict = {**extra_kwargs}
    if tools:
        kwargs["tools"] = tools

    response: dict = {}
    for event in client.stream(messages, **kwargs):
        if isinstance(event, TextDelta):
            if on_token is not None:
                on_token(event.text)
        elif isinstance(event, ThinkingDelta):
            pass  # thinking is captured in the final response
        elif isinstance(event, MessageDone):
            response = event.response

    if not response:
        raise ValueError("Stream completed without a MessageDone event")

    return response


# ---------------------------------------------------------------------------
# Meta-tool transparency helpers
# ---------------------------------------------------------------------------


def _is_meta_tool(
    tool_name: str,
    tool_handlers: dict[str, Callable[..., Any]] | None,
) -> bool:
    """Return True if *tool_name* is a tract built-in (not a user-provided handler)."""
    return not (tool_handlers and tool_name in tool_handlers)


def _build_ephemeral_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-format tool_call entry for ephemeral messages."""
    args = tc.get("arguments", {})
    return {
        "id": tc.get("id", ""),
        "type": "function",
        "function": {
            "name": tc["name"],
            "arguments": json.dumps(args) if not isinstance(args, str) else args,
        },
    }


def _append_ephemeral_tool_result(
    ephemeral: list[dict[str, Any]],
    tc_id: str,
    output: str,
) -> None:
    """Append a tool result to the ephemeral buffer."""
    ephemeral.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "content": output,
    })


# ---------------------------------------------------------------------------
# Async loop
# ---------------------------------------------------------------------------


async def arun_loop(
    tract: Tract,
    *,
    task: str | None = None,
    config: LoopConfig | None = None,
    llm_client: LLMClient | None = None,
    tools: list[dict] | None = None,
    tool_handlers: dict[str, Callable[..., Any]] | None = None,
    on_step: Callable[[int, Any], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    on_tool_result: Callable[[str, str, str], None] | None = None,
) -> LoopResult:
    """Async version of :func:`run_loop`.

    LLM calls are awaited. Tool execution uses ``asyncio.to_thread`` for
    sync handlers. Custom async tool handlers (coroutine functions) are
    awaited directly.

    Args:
        Same as :func:`run_loop`.

    Returns:
        LoopResult with status and metadata.
    """
    import asyncio
    import inspect

    from tract.llm.protocols import acall_llm
    from tract.toolkit.executor import ToolExecutor
    from tract.tract import _aretry_with_backoff

    cfg = config or LoopConfig()
    client = llm_client or tract.llm_client
    if client is None:
        raise ValueError(
            "No LLM client available. Pass llm_client= or configure on Tract.open()."
        )

    if tools is None:
        tools = tract.runtime.tools.as_tools(format="openai")

    effective_config = tract.default_config

    # Commit task as initial user message (sync — local operation)
    if task:
        tract.user(task)

    steps = 0
    total_tool_calls = 0
    last_response: str | None = None
    last_compiled = None
    step_usages: list[TokenUsage] = []
    step_metrics_list: list[StepMetrics] = []
    executor = ToolExecutor(tract)
    ephemeral_messages: list[dict[str, Any]] = []

    def _make_loop_result(
        status: Literal["completed", "blocked", "max_steps", "error"],
        reason: str | None,
        *,
        usage: TokenUsage | None = None,
    ) -> LoopResult:
        return LoopResult(
            status, reason, steps, total_tool_calls, last_response,
            compiled=last_compiled, usage=usage,
            step_usages=tuple(step_usages),
            config=effective_config,
            step_metrics=tuple(step_metrics_list),
        )

    # Build Layer 2 presenter if presentation is enabled
    presenter = None
    if cfg.presentation is not None and cfg.presentation is not False:
        from tract.toolkit.presentation import ToolPresenter, PresentationConfig

        if isinstance(cfg.presentation, PresentationConfig):
            presenter = ToolPresenter(tract, cfg.presentation)
        else:
            presenter = ToolPresenter(tract)

    strategy: CompileStrategy = tract.config.get("compile_strategy") or cfg.strategy
    strategy_k: int = tract.config.get("compile_strategy_k") or cfg.strategy_k

    for step in range(cfg.max_steps):
        steps = step + 1
        step_start = time.monotonic()
        context_tokens = 0
        compressed_this_step = False

        # 1. Compile (sync — local SQLite operation)
        try:
            last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
        except BlockedError as e:
            return _make_loop_result("blocked", str(e))
        except Exception as e:
            return _make_loop_result("error", f"Compile failed: {e}")

        context_tokens = last_compiled.token_count if last_compiled else 0

        # Auto-compress if context is too large
        if cfg.auto_compress_threshold is not None and cfg.max_tokens is not None:
            token_count = last_compiled.token_count
            threshold = int(cfg.max_tokens * cfg.auto_compress_threshold)
            if token_count > threshold:
                logger.debug(
                    "Auto-compressing: %d tokens > %d threshold (%.0f%% of %d)",
                    token_count, threshold,
                    cfg.auto_compress_threshold * 100, cfg.max_tokens,
                )
                try:
                    tract.compress(strategy="sliding_window", window_size=cfg.strategy_k)
                    last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
                    compressed_this_step = True
                except Exception as e:
                    logger.warning("Auto-compress failed, continuing with large context: %s", e, exc_info=True)

        # Build messages
        messages = last_compiled.to_dicts()
        if ephemeral_messages:
            messages.extend(ephemeral_messages)
            ephemeral_messages = []
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.system_prompt})

        # 2. Call LLM (async)
        use_streaming = (
            (on_token is not None or cfg.stream)
            and hasattr(client, "astream")
        )
        llm_kwargs: dict[str, Any] = {}
        if cfg.max_tokens is not None:
            llm_kwargs["max_tokens"] = cfg.max_tokens

        # Pre-generate middleware (can block)
        try:
            tract.middleware._run(
                "pre_generate",
                pending={"messages": messages, "config": llm_kwargs},
            )
        except BlockedError as e:
            return _make_loop_result("blocked", str(e))

        retry_cfg: RetryConfig | None = tract.retry_config
        llm_start = time.monotonic()
        try:
            if use_streaming:
                async def _do_stream() -> dict:
                    return await _astream_to_response(
                        client, messages, tools, on_token, **llm_kwargs,
                    )
                response = await _aretry_with_backoff(_do_stream, retry_cfg)
            else:
                async def _do_llm() -> Any:
                    return await acall_llm(
                        client, messages, tools=tools, **llm_kwargs,
                    )
                response = await _aretry_with_backoff(_do_llm, retry_cfg)
        except Exception as e:
            return _make_loop_result("error", f"LLM call failed: {e}")
        llm_duration = time.monotonic() - llm_start

        content = _extract_content(response, client)
        tool_call_list = _extract_tool_calls(response, client)

        # Post-generate middleware (informational)
        _post_gen_usage = _extract_usage(response, client)
        tract.middleware._run(
            "post_generate",
            pending={
                "response": content or "",
                "tokens_used": (
                    _post_gen_usage.get("total_tokens", 0)
                    if _post_gen_usage else 0
                ),
            },
        )

        # Extract and commit reasoning traces (sync)
        try:
            content = _handle_reasoning(response, client, tract, content)
        except Exception:
            logger.warning("Failed to extract reasoning; continuing with original content.", exc_info=True)

        last_response = content

        # Check if ALL tool calls target tract built-in (meta) tools.
        all_meta = (
            cfg.transparent_meta_tools
            and bool(tool_call_list)
            and all(_is_meta_tool(tc["name"], tool_handlers) for tc in tool_call_list)
        )

        # Commit assistant response (sync — local)
        if tool_call_list:
            if all_meta:
                ephemeral_messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [_build_ephemeral_tool_call(tc) for tc in tool_call_list],
                })
            else:
                tc_meta = [
                    {"id": tc["id"], "name": tc["name"],
                     "arguments": tc.get("arguments", {}), "type": "function"}
                    for tc in tool_call_list
                ]
                tc_msg = ", ".join(tc["name"] for tc in tool_call_list)
                tract.assistant(
                    content or "",
                    message=f"call {tc_msg}" if not content else None,
                    metadata={"tool_calls": tc_meta},
                )
        elif content:
            tract.assistant(content)

        step_usage = _extract_and_record_usage(response, client, tract)
        if step_usage is not None:
            step_usages.append(step_usage)

        # Check step budget
        if cfg.step_budget is not None:
            total_used = sum(u.total_tokens for u in step_usages)
            if total_used >= cfg.step_budget:
                step_end = time.monotonic()
                step_metrics_list.append(StepMetrics(
                    step=steps,
                    duration_ms=(step_end - step_start) * 1000,
                    llm_duration_ms=llm_duration * 1000,
                    tool_count=0,
                    tool_names=(),
                    context_tokens=context_tokens,
                    compressed=compressed_this_step,
                ))
                try:
                    last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
                except Exception as e:
                    logger.warning("Re-compile after budget exhaustion failed, preserving prior context: %s", e, exc_info=True)
                return _make_loop_result(
                    "completed",
                    f"Token budget exhausted ({total_used}/{cfg.step_budget})",
                    usage=step_usage,
                )

        if on_step:
            on_step(steps, response)

        # 3. If no tool calls, check if we should stop
        if not tool_call_list:
            if cfg.stop_on_no_tool_call:
                step_end = time.monotonic()
                step_metrics_list.append(StepMetrics(
                    step=steps,
                    duration_ms=(step_end - step_start) * 1000,
                    llm_duration_ms=llm_duration * 1000,
                    tool_count=0,
                    tool_names=(),
                    context_tokens=context_tokens,
                    compressed=compressed_this_step,
                ))
                try:
                    last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
                except Exception as e:
                    logger.warning("Re-compile after LLM completion failed, preserving prior context: %s", e, exc_info=True)
                return _make_loop_result(
                    "completed",
                    "LLM finished (no tool calls)",
                    usage=step_usage,
                )
            # Not stopping — record metrics for this step and continue
            step_end = time.monotonic()
            step_metrics_list.append(StepMetrics(
                step=steps,
                duration_ms=(step_end - step_start) * 1000,
                llm_duration_ms=llm_duration * 1000,
                tool_count=0,
                tool_names=(),
                context_tokens=context_tokens,
                compressed=compressed_this_step,
            ))
            continue

        # 4. Execute tool calls
        for tc in tool_call_list:
            total_tool_calls += 1
            tc_name = tc["name"]
            tc_id = tc.get("id", "")
            tc_args = tc.get("arguments", {})
            result_meta = {"tool_call_id": tc_id, "name": tc_name}
            use_ephemeral = all_meta and _is_meta_tool(tc_name, tool_handlers)

            # Validate tool arguments if validator configured
            if cfg.tool_validator is not None:
                valid, err_msg = cfg.tool_validator(tc_name, tc_args)
                if not valid:
                    error_output = f"Tool validation failed: {err_msg or 'invalid arguments'}"
                    if use_ephemeral:
                        _append_ephemeral_tool_result(ephemeral_messages, tc_id, error_output)
                    else:
                        _commit_tool_result(tract, tc_name, error_output, "error", result_meta)
                    if on_tool_result:
                        on_tool_result(tc_name, error_output, "error")
                    continue

            # Pre-tool-execute middleware (can block to skip this tool)
            try:
                tract.middleware._run(
                    "pre_tool_execute",
                    pending={"tool_name": tc_name, "arguments": tc_args},
                )
            except BlockedError:
                blocked_msg = "Tool execution blocked by middleware"
                if use_ephemeral:
                    _append_ephemeral_tool_result(ephemeral_messages, tc_id, blocked_msg)
                else:
                    _commit_tool_result(tract, tc_name, blocked_msg, "error", result_meta)
                if on_tool_result:
                    on_tool_result(tc_name, blocked_msg, "error")
                continue

            if tool_handlers and tc_name in tool_handlers:
                try:
                    handler = tool_handlers[tc_name]
                    if inspect.iscoroutinefunction(handler):
                        output = await handler(**tc_args)
                    else:
                        output = await asyncio.to_thread(handler, **tc_args)
                    _commit_tool_result(tract, tc_name, str(output), "success", result_meta)
                    if on_tool_result:
                        on_tool_result(tc_name, str(output), "success")
                    tract.middleware._run(
                        "post_tool_execute",
                        pending={"tool_name": tc_name, "result": str(output), "success": True},
                    )
                except Exception as exc:
                    _commit_tool_result(
                        tract, tc_name,
                        f"{type(exc).__name__}: {exc}", "error", result_meta,
                    )
                    if on_tool_result:
                        on_tool_result(tc_name, f"{type(exc).__name__}: {exc}", "error")
                    tract.middleware._run(
                        "post_tool_execute",
                        pending={
                            "tool_name": tc_name,
                            "result": f"{type(exc).__name__}: {exc}",
                            "success": False,
                        },
                    )
            else:
                result = await asyncio.to_thread(executor.execute, tc_name, tc_args)
                output_text = result.output if result.success else result.error
                if presenter:
                    output_text = presenter.present_result(result)
                if use_ephemeral:
                    _append_ephemeral_tool_result(ephemeral_messages, tc_id, output_text)
                else:
                    status: Literal["success", "error"] = "success" if result.success else "error"
                    _commit_tool_result(tract, tc_name, output_text, status, result_meta)
                if on_tool_result:
                    on_tool_result(tc_name, output_text, "success" if result.success else "error")
                tract.middleware._run(
                    "post_tool_execute",
                    pending={
                        "tool_name": tc_name,
                        "result": output_text,
                        "success": result.success,
                    },
                )

        # Record step metrics after tool execution
        tool_names_this_step = tuple(tc["name"] for tc in tool_call_list) if tool_call_list else ()
        step_end = time.monotonic()
        step_metrics_list.append(StepMetrics(
            step=steps,
            duration_ms=(step_end - step_start) * 1000,
            llm_duration_ms=llm_duration * 1000,
            tool_count=len(tool_call_list) if tool_call_list else 0,
            tool_names=tool_names_this_step,
            context_tokens=context_tokens,
            compressed=compressed_this_step,
        ))

    return _make_loop_result("max_steps", f"Reached max steps ({cfg.max_steps})")


async def _astream_to_response(
    client: Any,
    messages: list[dict],
    tools: list[dict] | None,
    on_token: Callable[[str], None] | None,
    **extra_kwargs: Any,
) -> dict:
    """Async version of :func:`_stream_to_response`.

    Uses ``client.astream()`` (async generator) instead of ``client.stream()``.
    """
    from tract.llm.anthropic_client import MessageDone, TextDelta, ThinkingDelta

    kwargs: dict = {**extra_kwargs}
    if tools:
        kwargs["tools"] = tools

    response: dict = {}
    async for event in client.astream(messages, **kwargs):
        if isinstance(event, TextDelta):
            if on_token is not None:
                on_token(event.text)
        elif isinstance(event, ThinkingDelta):
            pass
        elif isinstance(event, MessageDone):
            response = event.response

    if not response:
        raise ValueError("Stream completed without a MessageDone event")

    return response
