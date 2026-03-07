"""Default agent loop for Tract.

A minimal compile -> LLM -> tools -> repeat loop. Ships with tract like
the default LLM client -- easily replaced by LangChain, Agno, CrewAI, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tract.exceptions import BlockedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tract.llm.protocols import LLMClient
    from tract.protocols import CompiledContext, TokenUsage
    from tract.tract import CompileStrategy, Tract

logger = logging.getLogger(__name__)


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
        tools: Tool definitions (OpenAI format). Falls back to tract.as_tools().
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

    Returns:
        LoopResult with status and metadata.
    """
    from tract.toolkit.executor import ToolExecutor

    cfg = config or LoopConfig()
    client = llm_client or getattr(tract, "_llm_client", None)
    if client is None:
        raise ValueError(
            "No LLM client available. Pass llm_client= or configure on Tract.open()."
        )

    if tools is None:
        tools = tract.as_tools(format="openai")

    # Grab effective LLM config for the result
    effective_config = getattr(tract, "_default_config", None)

    # Commit task as initial user message
    if task:
        tract.user(task)

    steps = 0
    total_tool_calls = 0
    last_response: str | None = None
    last_compiled = None
    step_usages: list[TokenUsage] = []
    executor = ToolExecutor(tract)

    # Resolve config once (unlikely to change mid-loop)
    strategy: CompileStrategy = tract.get_config("compile_strategy") or cfg.strategy
    strategy_k: int = tract.get_config("compile_strategy_k") or cfg.strategy_k

    for step in range(cfg.max_steps):
        steps = step + 1

        # 1. Compile
        try:
            last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
        except BlockedError as e:
            return LoopResult(
                "blocked", str(e), steps, total_tool_calls, last_response,
                compiled=last_compiled, step_usages=tuple(step_usages),
                config=effective_config,
            )
        except Exception as e:
            return LoopResult(
                "error", f"Compile failed: {e}", steps, total_tool_calls,
                compiled=last_compiled, step_usages=tuple(step_usages),
                config=effective_config,
            )

        # Build messages
        messages = last_compiled.to_dicts()
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.system_prompt})

        # 2. Call LLM (streaming or sync)
        use_streaming = (
            (on_token is not None or cfg.stream)
            and hasattr(client, "stream")
        )
        try:
            if use_streaming:
                response = _stream_to_response(
                    client, messages, tools, on_token,
                )
            else:
                response = client.chat(messages=messages, tools=tools)
        except Exception as e:
            return LoopResult(
                "error", f"LLM call failed: {e}", steps, total_tool_calls,
                compiled=last_compiled, step_usages=tuple(step_usages),
                config=effective_config,
            )

        content = _extract_content(response, client)
        tool_call_list = _extract_tool_calls(response, client)

        # Extract and commit reasoning traces (e.g. <think> tags from Qwen)
        content = _handle_reasoning(response, client, tract, content)

        last_response = content

        # Commit assistant response (with tool_calls metadata if present)
        if tool_call_list:
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

        # Callback
        if on_step:
            on_step(steps, response)

        # 3. If no tool calls, check if we should stop
        if not tool_call_list:
            if cfg.stop_on_no_tool_call:
                # Re-compile to capture the final state (includes assistant commit)
                try:
                    last_compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
                except Exception:
                    pass  # keep the pre-commit compiled if re-compile fails
                return LoopResult(
                    "completed",
                    "LLM finished (no tool calls)",
                    steps,
                    total_tool_calls,
                    last_response,
                    compiled=last_compiled,
                    usage=step_usage,
                    step_usages=tuple(step_usages),
                    config=effective_config,
                )
            continue

        # 4. Execute tool calls
        for tc in tool_call_list:
            total_tool_calls += 1
            tc_name = tc["name"]
            tc_id = tc.get("id", "")
            tc_args = tc.get("arguments", {})
            result_meta = {"tool_call_id": tc_id, "name": tc_name}

            # Custom handler takes priority over built-in executor
            if tool_handlers and tc_name in tool_handlers:
                try:
                    output = tool_handlers[tc_name](**tc_args)
                    _commit_tool_result(tract, tc_name, str(output), "success", result_meta)
                except Exception as exc:
                    _commit_tool_result(
                        tract, tc_name,
                        f"{type(exc).__name__}: {exc}", "error", result_meta,
                    )
            else:
                result = executor.execute(tc_name, tc_args)
                if result.success:
                    _commit_tool_result(tract, tc_name, result.output, "success", result_meta)
                else:
                    _commit_tool_result(tract, tc_name, result.error, "error", result_meta)

    return LoopResult(
        "max_steps",
        f"Reached max steps ({cfg.max_steps})",
        steps,
        total_tool_calls,
        last_response,
        compiled=last_compiled,
        step_usages=tuple(step_usages),
        config=effective_config,
    )


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
                if reasoning_text and getattr(tract, "_commit_reasoning", True):
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
    if reasoning_text and getattr(tract, "_commit_reasoning", True):
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
    """Commit a tool result/error to the tract."""
    from tract.models.content import ToolIOContent

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


def _extract_usage(response: Any, client: LLMClient | None = None) -> dict | None:
    """Extract usage dict from LLM response (provider-agnostic)."""
    if client is not None and hasattr(client, "extract_usage"):
        try:
            return client.extract_usage(response)
        except (ValueError, KeyError, TypeError):
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
        except (ValueError, KeyError, TypeError):
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
        except (ValueError, KeyError, TypeError):
            pass

    # OpenAI object format
    if hasattr(response, "choices"):
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    ),
                }
                for tc in msg.tool_calls
            ]

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
                        args = json.loads(args)
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
) -> dict:
    """Run a streaming LLM call, yield text to on_token, return full response.

    Calls ``client.stream()`` and accumulates the result into a single
    response dict (same format as ``client.chat()`` would return).
    """
    from tract.llm.anthropic_client import MessageDone, TextDelta, ThinkingDelta

    kwargs: dict = {}
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
