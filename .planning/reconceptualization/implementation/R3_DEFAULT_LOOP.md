# Phase R3: Default Loop + Toolkit Rewire

## Goal

Build the default dumb loop (`src/tract/loop.py`) and rewire the toolkit to
work without the orchestrator. After this phase, an agent can run a compile ->
LLM -> tools -> repeat loop using tract's built-in loop.

**Depends on:** R2 (event processing wired into operations, transition works)

## Architecture

```
src/tract/loop.py           # Default dumb loop (public API)
src/tract/toolkit/           # Simplified: remove orchestrator coupling
    definitions.py           # Keep tool definitions (minor updates)
    executor.py              # Simplified executor
    models.py                # Keep ToolDefinition, simplify profiles
    profiles.py              # Simplify or remove orchestrator profiles
    callables.py             # Keep (framework integration)
```

## Task Breakdown

### Task 3.1: Default Loop (`loop.py`)

```python
"""Default agent loop for Tract.

A minimal compile -> LLM -> tools -> repeat loop. Ships with tract like
the default LLM client -- easily replaced by LangChain, Agno, CrewAI, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tract.llm.protocols import LLMClient
    from tract.tract import Tract


@dataclass(frozen=True)
class LoopResult:
    """Result of a loop execution."""
    status: str              # "completed" | "blocked" | "max_steps" | "error"
    reason: str | None       # human-readable reason for stopping
    steps: int               # number of loop iterations executed
    tool_calls: int          # total tool calls executed
    final_response: str | None = None  # last LLM response text


@dataclass
class LoopConfig:
    """Configuration for the default loop."""
    max_steps: int = 50
    system_prompt: str | None = None  # prepended to compiled context
    strategy: str = "full"            # compile strategy
    strategy_k: int = 5               # K for adaptive strategy
    stop_on_no_tool_call: bool = True  # stop if LLM doesn't call tools


def run_loop(
    tract: Tract,
    *,
    task: str | None = None,
    config: LoopConfig | None = None,
    llm_client: LLMClient | None = None,
    tools: list[dict] | None = None,
    on_step: Callable | None = None,  # callback(step_num, response) for logging
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
    client = llm_client or getattr(tract, '_llm_client', None)
    if client is None:
        raise ValueError("No LLM client available. Pass llm_client= or configure on Tract.open().")

    if tools is None:
        tools = tract.as_tools(format="openai")

    # Commit task as initial user message
    if task:
        from tract.models.content import DialogueContent
        tract.commit(DialogueContent(role="user", text=task), message=task[:72])

    steps = 0
    total_tool_calls = 0
    last_response = None

    for step in range(cfg.max_steps):
        steps = step + 1

        # 1. Compile
        try:
            compiled = tract.compile(strategy=cfg.strategy, strategy_k=cfg.strategy_k)
        except Exception as e:
            return LoopResult("error", f"Compile failed: {e}", steps, total_tool_calls)

        # Build messages
        messages = compiled.to_dicts()
        if cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": cfg.system_prompt})

        # 2. Call LLM
        try:
            response = client.chat(messages=messages, tools=tools)
        except Exception as e:
            return LoopResult("error", f"LLM call failed: {e}", steps, total_tool_calls)

        content = _extract_content(response)
        tool_calls = _extract_tool_calls(response)
        last_response = content

        # Commit assistant response
        if content:
            from tract.models.content import DialogueContent
            tract.commit(DialogueContent(role="assistant", text=content),
                        message=content[:72])

        # Callback
        if on_step:
            on_step(steps, response)

        # 3. If no tool calls, check if we should stop
        if not tool_calls:
            if cfg.stop_on_no_tool_call:
                return LoopResult("completed", "LLM finished (no tool calls)",
                                steps, total_tool_calls, last_response)
            continue

        # 4. Execute tool calls
        from tract.toolkit.executor import ToolExecutor
        executor = ToolExecutor(tract)

        for tc in tool_calls:
            total_tool_calls += 1
            try:
                result = executor.execute(tc["name"], tc.get("arguments", {}))
                # Commit tool result
                from tract.models.content import ToolIOContent
                tract.commit(
                    ToolIOContent(
                        tool_name=tc["name"],
                        direction="result",
                        payload={"result": str(result)},
                        status="success",
                    ),
                    message=f"tool result: {tc['name']}",
                )
            except Exception as e:
                # Commit error result
                from tract.models.content import ToolIOContent
                tract.commit(
                    ToolIOContent(
                        tool_name=tc["name"],
                        direction="result",
                        payload={"error": str(e)},
                        status="error",
                    ),
                    message=f"tool error: {tc['name']}",
                )

        # 5. Check for blocks from rule engine
        # (block actions during commit events would have raised)

    return LoopResult("max_steps", f"Reached max steps ({cfg.max_steps})",
                     steps, total_tool_calls, last_response)


def _extract_content(response: Any) -> str | None:
    """Extract text content from LLM response (provider-agnostic)."""
    # OpenAI format
    if hasattr(response, 'choices'):
        msg = response.choices[0].message
        return msg.content
    # Dict format
    if isinstance(response, dict):
        return response.get("content")
    # String
    if isinstance(response, str):
        return response
    return str(response)


def _extract_tool_calls(response: Any) -> list[dict]:
    """Extract tool calls from LLM response."""
    import json
    # OpenAI format
    if hasattr(response, 'choices'):
        msg = response.choices[0].message
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            return [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]
    # Dict format
    if isinstance(response, dict) and "tool_calls" in response:
        return response["tool_calls"]
    return []
```

### Task 3.2: Simplify Toolkit

**`toolkit/executor.py` modifications:**
- Remove all orchestrator-related imports and references
- Remove PendingToolResult hook integration
- Simplify to direct dispatch: name -> handler -> result
- Keep error handling

**`toolkit/profiles.py` modifications:**
- Keep SELF_PROFILE, SUPERVISOR_PROFILE, FULL_PROFILE
- Remove orchestrator-specific tool configs (if any)
- These profiles are still useful for controlling tool exposure

**`toolkit/definitions.py` modifications:**
- Remove any tools that only existed for orchestrator use
- Keep all substrate operation tools (commit, compile, log, branch, merge, etc.)
- Add `create_rule` tool definition:

```python
ToolDefinition(
    name="create_rule",
    description="Create a rule on the current branch",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Rule name"},
            "trigger": {"type": "string", "description": "Trigger event"},
            "condition": {"type": "object", "description": "Condition dict (optional)"},
            "action": {"type": "object", "description": "Action dict"},
        },
        "required": ["name", "trigger", "action"],
    },
    handler=lambda t, **kwargs: t.rule(**kwargs),
)
```

- Add `create_metadata` tool definition
- Add `get_config` tool definition
- Add `transition` tool definition

### Task 3.3: Wire Loop into Tract (convenience)

**Add to `tract.py`:**

```python
def run(
    self,
    task: str | None = None,
    *,
    max_steps: int = 50,
    system_prompt: str | None = None,
    on_step: callable | None = None,
) -> LoopResult:
    """Run the default agent loop on this tract.

    Convenience wrapper around loop.run_loop().

    Args:
        task: Task description (committed as user message).
        max_steps: Maximum loop iterations.
        system_prompt: System prompt prepended to context.
        on_step: Step callback.

    Returns:
        LoopResult with status and metadata.
    """
    from tract.loop import LoopConfig, run_loop
    config = LoopConfig(max_steps=max_steps, system_prompt=system_prompt)
    return run_loop(self, task=task, config=config, on_step=on_step)
```

### Task 3.4: Remove Dead Orchestrator References

Audit and remove any remaining references to:
- `self._orchestrating`
- `self._orchestrator`
- `self._agent_loop`
- `_check_orchestrator_triggers`
- Any orchestrator imports

These should have been removed in R0 but verify completeness.

---

## Test Plan

### `tests/test_loop.py` (~25 tests)

- `test_loop_basic` -- single step, LLM returns text, no tools
- `test_loop_with_tools` -- LLM calls a tool, result committed
- `test_loop_multi_step` -- multiple tool calls across steps
- `test_loop_max_steps` -- stops at max_steps
- `test_loop_stops_on_no_tool_call` -- stop_on_no_tool_call=True
- `test_loop_continues_without_tools` -- stop_on_no_tool_call=False
- `test_loop_with_task` -- task committed as user message
- `test_loop_without_task` -- no initial commit
- `test_loop_system_prompt` -- system prompt prepended
- `test_loop_compile_strategy` -- strategy passed through
- `test_loop_error_on_compile` -- returns error status
- `test_loop_error_on_llm` -- returns error status
- `test_loop_tool_error` -- error committed, loop continues
- `test_loop_on_step_callback` -- callback called each step
- `test_loop_result_fields` -- all LoopResult fields populated
- `test_loop_no_client_raises` -- ValueError if no LLM client
- `test_loop_custom_tools` -- tools= parameter
- `test_loop_through_facade` -- t.run(task="...") works
- `test_loop_with_rules` -- rules fire during loop execution
- `test_loop_blocked_by_rule` -- block action stops operation
- `test_loop_config_from_rules` -- compile strategy from active rule
- `test_extract_content_openai` -- OpenAI response format
- `test_extract_content_dict` -- dict response format
- `test_extract_tool_calls_openai` -- OpenAI tool call format
- `test_extract_tool_calls_dict` -- dict tool call format

### `tests/test_toolkit_simplified.py` (~10 tests)

- `test_executor_direct_dispatch` -- no hook routing
- `test_create_rule_tool` -- tool creates rule commit
- `test_create_metadata_tool` -- tool creates metadata commit
- `test_get_config_tool` -- tool resolves config
- `test_transition_tool` -- tool triggers transition
- `test_as_tools_no_orchestrator` -- no orchestrator-only tools
- `test_profiles_still_work` -- SELF/SUPERVISOR/FULL profiles
- `test_callable_tools` -- as_callable_tools() works

---

## Acceptance Criteria

1. `run_loop(t, task="...")` executes compile->LLM->tools->repeat
2. Loop stops cleanly on: no tool calls, max_steps, block, error
3. LoopResult contains status, reason, steps, tool_calls
4. `t.run(task="...")` convenience method works
5. Toolkit executor works without hooks/orchestrator
6. New tool definitions: create_rule, create_metadata, get_config, transition
7. All ~35 new tests pass
8. All surviving R0+R1+R2 tests still pass
