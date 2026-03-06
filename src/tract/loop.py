"""Default agent loop for Tract.

A minimal compile -> LLM -> tools -> repeat loop. Ships with tract like
the default LLM client -- easily replaced by LangChain, Agno, CrewAI, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tract.exceptions import BlockedByRuleError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tract.llm.protocols import LLMClient
    from tract.tract import Tract

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopResult:
    """Result of a loop execution."""

    status: Literal["completed", "blocked", "max_steps", "error"]
    reason: str | None
    steps: int
    tool_calls: int
    final_response: str | None = None

    def pprint(self) -> None:
        """Pretty-print this loop result using rich formatting."""
        from tract.formatting import pprint_loop_result

        pprint_loop_result(self)


@dataclass
class LoopConfig:
    """Configuration for the default loop."""

    max_steps: int = 50
    system_prompt: str | None = None
    strategy: str = "full"
    strategy_k: int = 5
    stop_on_no_tool_call: bool = True


def run_loop(
    tract: Tract,
    *,
    task: str | None = None,
    config: LoopConfig | None = None,
    llm_client: LLMClient | None = None,
    tools: list[dict] | None = None,
    on_step: Callable | None = None,
) -> LoopResult:
    """Run the default agent loop.

    Loop:
    1. Compile context (respecting active rule configs)
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
        on_step: Optional step callback for logging/monitoring.

    Returns:
        LoopResult with status and metadata.
    """
    cfg = config or LoopConfig()
    client = llm_client or getattr(tract, "_llm_client", None)
    if client is None:
        raise ValueError(
            "No LLM client available. Pass llm_client= or configure on Tract.open()."
        )

    if tools is None:
        tools = tract.as_tools(format="openai")

    # Commit task as initial user message
    if task:
        tract.user(task)

    steps = 0
    total_tool_calls = 0
    last_response: str | None = None

    for step in range(cfg.max_steps):
        steps = step + 1

        # 1. Compile (active rules override LoopConfig defaults)
        try:
            strategy = tract.get_config("compile_strategy") or cfg.strategy
            strategy_k = tract.get_config("compile_strategy_k") or cfg.strategy_k
            compiled = tract.compile(strategy=strategy, strategy_k=strategy_k)
        except BlockedByRuleError as e:
            return LoopResult(
                "blocked", str(e), steps, total_tool_calls, last_response
            )
        except Exception as e:
            return LoopResult(
                "error", f"Compile failed: {e}", steps, total_tool_calls
            )

        # Build messages
        messages = compiled.to_dicts()
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.system_prompt})

        # 2. Call LLM
        try:
            response = client.chat(messages=messages, tools=tools)
        except Exception as e:
            return LoopResult(
                "error", f"LLM call failed: {e}", steps, total_tool_calls
            )

        content = _extract_content(response, client)
        tool_call_list = _extract_tool_calls(response)
        last_response = content

        # Commit assistant response
        if content:
            tract.assistant(content)

        # Callback
        if on_step:
            on_step(steps, response)

        # 3. If no tool calls, check if we should stop
        if not tool_call_list:
            if cfg.stop_on_no_tool_call:
                return LoopResult(
                    "completed",
                    "LLM finished (no tool calls)",
                    steps,
                    total_tool_calls,
                    last_response,
                )
            continue

        # 4. Execute tool calls
        from tract.models.content import ToolIOContent
        from tract.toolkit.executor import ToolExecutor

        executor = ToolExecutor(tract)

        for tc in tool_call_list:
            total_tool_calls += 1
            result = executor.execute(tc["name"], tc.get("arguments", {}))
            if result.success:
                tract.commit(
                    ToolIOContent(
                        tool_name=tc["name"],
                        direction="result",
                        payload={"result": result.output},
                        status="success",
                    ),
                    message=f"tool result: {tc['name']}",
                )
            else:
                tract.commit(
                    ToolIOContent(
                        tool_name=tc["name"],
                        direction="result",
                        payload={"error": result.error},
                        status="error",
                    ),
                    message=f"tool error: {tc['name']}",
                )

    return LoopResult(
        "max_steps",
        f"Reached max steps ({cfg.max_steps})",
        steps,
        total_tool_calls,
        last_response,
    )


def _extract_content(response: Any, client: Any = None) -> str | None:
    """Extract text content from LLM response (provider-agnostic)."""
    # Use client's extract_content if available
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


def _extract_tool_calls(response: Any) -> list[dict]:
    """Extract tool calls from LLM response."""
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
