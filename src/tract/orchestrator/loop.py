"""Core orchestrator agent loop for context management.

Provides the Orchestrator class that runs a tool-calling loop:
assess context, send tools + assessment to LLM, execute tool calls,
repeat until LLM stops or max_steps reached.

The orchestrator no longer has its own proposal system. Hookable
operations (compress, gc, merge, rebase, trigger) are gated by
Tract's unified hook system. Non-hookable tool calls are gated
by the ``on_tool_call`` callback in collaborative mode.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

from tract.exceptions import OrchestratorError
from tract.orchestrator.config import AutonomyLevel, OrchestratorState
from tract.orchestrator.models import (
    OrchestratorResult,
    StepResult,
    ToolCall,
    ToolCallDecision,
    ToolCallReview,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tract.orchestrator.config import OrchestratorConfig
    from tract.toolkit.executor import ToolExecutor
    from tract.toolkit.models import ToolResult
    from tract.tract import Tract

logger = logging.getLogger(__name__)

# Autonomy level ordering for ceiling computation
_AUTONOMY_ORDER: dict[AutonomyLevel, int] = {
    AutonomyLevel.MANUAL: 0,
    AutonomyLevel.COLLABORATIVE: 1,
    AutonomyLevel.AUTONOMOUS: 2,
}


class Orchestrator:
    """Context management orchestrator that runs a tool-calling loop.

    The orchestrator assesses context health, sends tools and assessment
    to an LLM, executes tool calls (respecting autonomy constraints),
    and repeats until the LLM stops calling tools or max_steps is reached.

    Hookable operations (compress, gc, merge, rebase, trigger) are gated
    automatically by Tract's hook system -- the orchestrator does not
    need to intercept them separately.

    For non-hookable tool calls in collaborative mode, the
    ``on_tool_call`` callback on OrchestratorConfig provides review.

    Usage::

        from tract.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(tract, config=OrchestratorConfig(), llm_callable=my_llm)
        result = orch.run()
        print(f"Completed {len(result.steps)} steps")
    """

    def __init__(
        self,
        tract: Tract,
        config: OrchestratorConfig | None = None,
        llm_callable: Callable | None = None,
    ) -> None:
        from tract.orchestrator.config import OrchestratorConfig as _OrchestratorConfig
        from tract.toolkit.executor import ToolExecutor as _ToolExecutor

        self._tract = tract
        self._config = config or _OrchestratorConfig()
        self._executor = _ToolExecutor(tract)
        self._llm = llm_callable
        self._state = OrchestratorState.IDLE
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._orchestrating = False
        self._trigger_autonomy: AutonomyLevel | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> OrchestratorState:
        """Return the current orchestrator state."""
        return self._state

    def run(
        self,
        *,
        trigger_autonomy: AutonomyLevel | None = None,
    ) -> OrchestratorResult:
        """Execute the orchestrator agent loop.

        1. Build context assessment
        2. Get tools for the configured profile
        3. Loop: call LLM -> extract tool calls -> execute -> repeat
        4. Return result with all steps

        Args:
            trigger_autonomy: Optional autonomy override from the trigger
                that invoked this run. When set, effective autonomy is
                ``min(ceiling, trigger_autonomy)`` for every tool call.

        Returns:
            OrchestratorResult with all steps and final state.

        Raises:
            OrchestratorError: If no LLM is configured.
        """
        self._trigger_autonomy = trigger_autonomy
        self._state = OrchestratorState.RUNNING
        self._orchestrating = True
        self._tract._set_orchestrating(True)

        steps: list[StepResult] = []
        assessment_text = ""
        step_counter = 0

        try:
            # Build context assessment
            from tract.orchestrator.assessment import build_context_assessment

            assessment_text = build_context_assessment(
                self._tract,
                task_context=self._config.task_context,
            )

            # Get tools for the configured profile
            tools = self._tract.as_tools(profile=self._config.profile)

            # Build initial messages
            from tract.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

            system_prompt = self._config.system_prompt or ORCHESTRATOR_SYSTEM_PROMPT
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": assessment_text},
            ]

            # Main loop
            for _loop_idx in range(self._config.max_steps):
                # Check stop/pause
                if self._stop_event.is_set():
                    self._state = OrchestratorState.STOPPED
                    break
                if self._pause_event.is_set():
                    self._state = OrchestratorState.PAUSING
                    break

                # Call LLM
                response = self._call_llm(messages, tools)

                # Extract tool calls
                tool_calls = self._extract_tool_calls(response)

                # If no tool calls, LLM is done
                if not tool_calls:
                    break

                # Execute each tool call
                call_results: list[tuple[ToolCall, ToolResult]] = []
                should_break = False

                for tc in tool_calls:
                    step_counter += 1
                    step_result = self._execute_tool_call(tc, step_counter)
                    steps.append(step_result)

                    # Track result for formatting
                    from tract.toolkit.models import ToolResult as _ToolResult

                    if step_result.success:
                        tr = _ToolResult(
                            tool_name=tc.name,
                            success=True,
                            output=step_result.result_output,
                        )
                    else:
                        tr = _ToolResult(
                            tool_name=tc.name,
                            success=False,
                            error=step_result.result_error,
                        )
                    call_results.append((tc, tr))

                    # Call on_step callback if set
                    if self._config.on_step is not None:
                        try:
                            self._config.on_step(step_result)
                        except Exception:
                            logger.debug(
                                "on_step callback error", exc_info=True
                            )

                    # Check stop/pause between tool calls
                    if self._stop_event.is_set():
                        self._state = OrchestratorState.STOPPED
                        should_break = True
                        break
                    if self._pause_event.is_set():
                        self._state = OrchestratorState.PAUSING
                        should_break = True
                        break

                # Format tool results and append to conversation
                formatted = self._format_tool_results(
                    response, [tc for tc, _ in call_results], [tr for _, tr in call_results]
                )
                messages.extend(formatted)

                if should_break:
                    break

                # Check stop/pause after all tool calls
                if self._stop_event.is_set():
                    self._state = OrchestratorState.STOPPED
                    break
                if self._pause_event.is_set():
                    self._state = OrchestratorState.PAUSING
                    break

        finally:
            self._orchestrating = False
            self._tract._set_orchestrating(False)
            if self._state == OrchestratorState.RUNNING:
                self._state = OrchestratorState.IDLE

        return OrchestratorResult(
            steps=steps,
            state=self._state,
            assessment=assessment_text,
            total_tool_calls=len(steps),
        )

    def stop(self) -> None:
        """Signal the orchestrator to stop immediately.

        The loop will exit before the next tool call.
        """
        self._stop_event.set()
        self._state = OrchestratorState.STOPPED

    def pause(self) -> None:
        """Signal the orchestrator to pause gracefully.

        The loop will exit before the next tool call.
        """
        self._pause_event.set()
        self._state = OrchestratorState.PAUSING

    def reset(self) -> None:
        """Reset the orchestrator for reuse.

        Clears stop/pause events and returns to IDLE state.
        """
        self._stop_event.clear()
        self._pause_event.clear()
        self._state = OrchestratorState.IDLE

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
    ) -> dict:
        """Call the LLM with messages and tools.

        Dispatches to the configured llm_callable if set, otherwise
        falls back to the tract's built-in LLM client.

        Args:
            messages: Conversation messages.
            tools: Tool definitions in OpenAI format.

        Returns:
            Raw LLM response dict.

        Raises:
            OrchestratorError: If no LLM is configured.
        """
        if self._llm is not None:
            return self._llm(messages=messages, tools=tools)

        # Fall back to tract's LLM client
        client = getattr(self._tract, "_llm_client", None)
        if client is None:
            raise OrchestratorError(
                "No LLM client configured. Call tract.configure_llm() "
                "or provide llm_callable to Orchestrator."
            )

        # OpenAIClient.chat() accepts tools via **kwargs passthrough
        kwargs: dict = {"tools": tools}
        if self._config.model:
            kwargs["model"] = self._config.model
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            kwargs["max_tokens"] = self._config.max_tokens
        if self._config.extra_llm_kwargs:
            kwargs.update(self._config.extra_llm_kwargs)
        return client.chat(messages, **kwargs)

    def _extract_tool_calls(self, response: dict) -> list[ToolCall]:
        """Parse OpenAI-format response to extract tool calls.

        Navigates response["choices"][0]["message"]["tool_calls"] and
        parses each into a ToolCall dataclass.

        Args:
            response: Raw LLM response dict.

        Returns:
            List of ToolCall instances. Empty if no tool calls.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return []
            message = choices[0].get("message", {})
            raw_calls = message.get("tool_calls", [])
            if not raw_calls:
                return []

            result: list[ToolCall] = []
            for raw in raw_calls:
                call_id = raw.get("id", f"call_{uuid.uuid4().hex[:8]}")
                func = raw.get("function", {})
                name = func.get("name", "")
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                    logger.warning(
                        "Malformed JSON in tool call arguments for %s", name
                    )
                result.append(ToolCall(id=call_id, name=name, arguments=arguments))
            return result
        except (KeyError, IndexError, TypeError) as exc:
            logger.debug("Failed to extract tool calls: %s", exc)
            return []

    def _format_tool_results(
        self,
        response: dict,
        tool_calls: list[ToolCall],
        results: list[ToolResult],
    ) -> list[dict]:
        """Format tool execution results for the LLM conversation.

        Returns a list of dicts to append to the conversation:
        1. First: the assistant message from the response (preserves tool_calls)
        2. Then: one tool result message per tool call

        This follows the OpenAI tool-calling conversation format.

        Args:
            response: The original LLM response dict.
            tool_calls: The parsed tool calls.
            results: The execution results.

        Returns:
            List of message dicts to extend the conversation with.
        """
        formatted: list[dict] = []

        # Extract assistant message (preserves tool_calls array intact)
        try:
            assistant_msg = response["choices"][0]["message"]
            formatted.append(copy.deepcopy(assistant_msg))
        except (KeyError, IndexError, TypeError):
            # Fallback: construct minimal assistant message
            formatted.append({
                "role": "assistant",
                "content": "",
            })

        # Add tool result messages
        for tc, tr in zip(tool_calls, results):
            if tr.success:
                content = tr.output
            else:
                content = f"Error: {tr.error}"
            formatted.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

        return formatted

    def _execute_tool_call(self, tc: ToolCall, step_num: int) -> StepResult:
        """Execute a single tool call respecting autonomy constraints.

        Determines effective autonomy and routes accordingly:
        - MANUAL: skip all tool calls
        - COLLABORATIVE: invoke on_tool_call callback for review
        - AUTONOMOUS: execute directly

        Args:
            tc: The tool call to execute.
            step_num: The step number for the result.

        Returns:
            StepResult recording what happened.
        """
        effective = self._effective_autonomy(self._trigger_autonomy)

        # MANUAL: skip all tool calls
        if effective == AutonomyLevel.MANUAL:
            return StepResult(
                step=step_num,
                tool_call=tc,
                result_output="",
                result_error="Skipped (manual mode)",
                success=False,
                review_decision="skipped",
            )

        # COLLABORATIVE: review via on_tool_call callback
        if effective == AutonomyLevel.COLLABORATIVE:
            return self._handle_collaborative(tc, step_num)

        # AUTONOMOUS: execute directly
        return self._execute_directly(tc, step_num)

    def _handle_collaborative(self, tc: ToolCall, step_num: int) -> StepResult:
        """Handle a tool call in collaborative mode.

        Invokes the ``on_tool_call`` callback to get a review decision.
        Executes, skips, or modifies based on the callback response.

        Args:
            tc: The tool call to execute.
            step_num: The step number.

        Returns:
            StepResult with review_decision information.
        """
        callback = self._config.on_tool_call
        if callback is None:
            # No callback: cannot get approval, skip
            return StepResult(
                step=step_num,
                tool_call=tc,
                result_output="",
                result_error="Skipped (collaborative mode, no callback)",
                success=False,
                review_decision=ToolCallDecision.REJECTED.value,
            )

        try:
            review = callback(tc)
        except Exception as exc:
            logger.debug("on_tool_call callback error: %s", exc)
            return StepResult(
                step=step_num,
                tool_call=tc,
                result_output="",
                result_error=f"Callback error: {exc}",
                success=False,
                review_decision=ToolCallDecision.REJECTED.value,
            )

        if review.decision == ToolCallDecision.APPROVED:
            result = self._executor.execute(tc.name, tc.arguments)
            return StepResult(
                step=step_num,
                tool_call=tc,
                result_output=result.output if result.success else "",
                result_error=result.error if not result.success else "",
                success=result.success,
                review_decision=ToolCallDecision.APPROVED.value,
            )

        if review.decision == ToolCallDecision.MODIFIED:
            modified_tc = review.modified_action or tc
            result = self._executor.execute(modified_tc.name, modified_tc.arguments)
            return StepResult(
                step=step_num,
                tool_call=modified_tc,
                result_output=result.output if result.success else "",
                result_error=result.error if not result.success else "",
                success=result.success,
                review_decision=ToolCallDecision.MODIFIED.value,
            )

        # REJECTED
        return StepResult(
            step=step_num,
            tool_call=tc,
            result_output="",
            result_error=f"Rejected: {review.reason}",
            success=False,
            review_decision=ToolCallDecision.REJECTED.value,
        )

    def _execute_directly(self, tc: ToolCall, step_num: int) -> StepResult:
        """Execute a tool call directly (autonomous mode).

        Args:
            tc: The tool call to execute.
            step_num: The step number.

        Returns:
            StepResult recording the execution.
        """
        result = self._executor.execute(tc.name, tc.arguments)
        return StepResult(
            step=step_num,
            tool_call=tc,
            result_output=result.output if result.success else "",
            result_error=result.error if not result.success else "",
            success=result.success,
        )

    def _effective_autonomy(
        self, trigger_autonomy: AutonomyLevel | None = None
    ) -> AutonomyLevel:
        """Compute effective autonomy as min(ceiling, trigger_autonomy).

        The hierarchy is MANUAL < COLLABORATIVE < AUTONOMOUS.
        If trigger_autonomy is None, returns the ceiling.

        Args:
            trigger_autonomy: The autonomy level requested by trigger.

        Returns:
            The effective autonomy level (the lower of the two).
        """
        ceiling = self._config.autonomy_ceiling
        if trigger_autonomy is None:
            return ceiling

        ceiling_order = _AUTONOMY_ORDER[ceiling]
        trigger_order = _AUTONOMY_ORDER[trigger_autonomy]
        min_order = min(ceiling_order, trigger_order)

        # Return the level with the lower ordinal
        for level, order in _AUTONOMY_ORDER.items():
            if order == min_order:
                return level

        return ceiling  # Fallback (should not reach here)
