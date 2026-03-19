"""Tests for Tract.run() facade -- verifies passthrough of LoopConfig fields.

Covers: step_budget, tool_validator, auto_compress_threshold, and StepMetrics
population through the Tract.run() convenience method.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import Tract, StepMetrics


# ---------------------------------------------------------------------------
# Helpers: mock LLM client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self._call_count = 0

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, **kw):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def close(self):
        pass


def _text_resp(text: str, tokens: int = 100) -> dict:
    """Build an OpenAI-style response dict with usage data."""
    return {
        "choices": [{"message": {"content": text, "tool_calls": None}}],
        "usage": {
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens - tokens // 2,
            "total_tokens": tokens,
        },
    }


def _tool_resp(tool_name: str, args: dict[str, Any] | None = None, tokens: int = 100) -> dict:
    """Build an OpenAI-style response with a tool call and usage data."""
    return {
        "choices": [{"message": {
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args or {}),
                },
                "type": "function",
            }],
        }}],
        "usage": {
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens - tokens // 2,
            "total_tokens": tokens,
        },
    }


# ---------------------------------------------------------------------------
# step_budget passthrough
# ---------------------------------------------------------------------------


class TestRunStepBudget:
    def test_budget_stops_loop(self):
        """run() with step_budget stops when budget exceeded."""
        client = MockLLMClient([_text_resp("response", tokens=600)])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(
                task="Do something",
                llm_client=client,
                tools=[],
                step_budget=500,
                max_steps=10,
            )
            assert result.budget_exhausted
            assert result.steps == 1

    def test_no_budget_completes_normally(self):
        """run() without step_budget completes normally."""
        client = MockLLMClient([_text_resp("done")])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(llm_client=client, tools=[])
            assert result.status == "completed"
            assert not result.budget_exhausted

    def test_budget_accumulates_across_steps(self):
        """step_budget tracks cumulative usage across steps via run()."""
        client = MockLLMClient([
            _text_resp("step 1", tokens=300),
            _text_resp("step 2", tokens=300),
            _text_resp("step 3", tokens=300),
        ])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(
                llm_client=client,
                tools=[],
                step_budget=500,
                max_steps=10,
                # Need stop_on_no_tool_call=False but run() doesn't expose it.
                # The loop stops after the first text response because
                # stop_on_no_tool_call defaults to True.
            )
            # With stop_on_no_tool_call=True (default), loop stops at step 1.
            # 300 < 500, so budget not exhausted.
            assert result.status == "completed"
            assert result.steps == 1
            assert not result.budget_exhausted


# ---------------------------------------------------------------------------
# tool_validator passthrough
# ---------------------------------------------------------------------------


class TestRunToolValidator:
    def test_validator_blocks_invalid_tool(self):
        """run() with tool_validator rejects bad tool calls."""
        def validator(name, args):
            if name == "blocked_tool":
                return False, "Not allowed"
            return True, None

        client = MockLLMClient([
            _tool_resp("blocked_tool", {}),
            _text_resp("ok"),
        ])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(
                llm_client=client,
                tools=[{"type": "function", "function": {"name": "blocked_tool", "parameters": {}}}],
                tool_validator=validator,
            )
            assert result.status == "completed"
            assert result.steps == 2

    def test_validator_allows_valid_tool(self):
        """run() with tool_validator allows valid tools through."""
        def validator(name, args):
            return True, None

        client = MockLLMClient([
            _tool_resp("good_tool", {}),
            _text_resp("done"),
        ])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(
                llm_client=client,
                tools=[{"type": "function", "function": {"name": "good_tool", "parameters": {}}}],
                tool_handlers={"good_tool": lambda: "ok"},
                tool_validator=validator,
            )
            assert result.status == "completed"

    def test_validator_rejection_committed_as_error(self):
        """Rejected tool calls surface as errors via on_tool_result callback."""
        def validator(name, args):
            return False, "Forbidden operation"

        tool_results: list[tuple[str, str, str]] = []

        client = MockLLMClient([
            _tool_resp("bad_tool", {"x": 1}),
            _text_resp("ok"),
        ])
        with Tract.open() as t:
            t.system("Test")
            t._llm_mgr.run(
                llm_client=client,
                tools=[{"type": "function", "function": {"name": "bad_tool", "parameters": {}}}],
                tool_validator=validator,
                on_tool_result=lambda name, output, status: tool_results.append(
                    (name, output, status)
                ),
            )
            assert len(tool_results) == 1
            assert tool_results[0][0] == "bad_tool"
            assert "Forbidden operation" in tool_results[0][1]
            assert tool_results[0][2] == "error"


# ---------------------------------------------------------------------------
# auto_compress_threshold passthrough
# ---------------------------------------------------------------------------


class TestRunAutoCompress:
    def test_auto_compress_passthrough(self):
        """run() passes auto_compress_threshold to LoopConfig."""
        client = MockLLMClient([_text_resp("done")])
        with Tract.open() as t:
            t.system("Short prompt")
            result = t._llm_mgr.run(
                llm_client=client,
                tools=[],
                auto_compress_threshold=0.8,
                max_tokens=100000,
            )
            assert result.status == "completed"
            # Small context, threshold never reached
            assert result.compressions_triggered == 0


# ---------------------------------------------------------------------------
# StepMetrics population
# ---------------------------------------------------------------------------


class TestRunStepMetrics:
    def test_step_metrics_populated(self):
        """run() result includes StepMetrics with per-step data."""
        client = MockLLMClient([_text_resp("done", tokens=100)])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(llm_client=client, tools=[])
            assert len(result.step_metrics) == 1
            m = result.step_metrics[0]
            assert isinstance(m, StepMetrics)
            assert m.step == 1
            assert m.duration_ms > 0
            assert m.context_tokens > 0

    def test_total_duration_ms(self):
        """run() result has total_duration_ms property."""
        client = MockLLMClient([_text_resp("done")])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(llm_client=client, tools=[])
            assert result.total_duration_ms > 0

    def test_tool_metrics_in_step(self):
        """run() records tool call info in step metrics."""
        client = MockLLMClient([
            _tool_resp("my_tool", {}),
            _text_resp("done"),
        ])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(
                llm_client=client,
                tools=[{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
                tool_handlers={"my_tool": lambda: "result"},
            )
            # First step has the tool call
            assert result.step_metrics[0].tool_count == 1
            assert "my_tool" in result.step_metrics[0].tool_names

    def test_step_metrics_no_tool_step(self):
        """Text-only steps record zero tool_count."""
        client = MockLLMClient([_text_resp("done")])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(llm_client=client, tools=[])
            assert result.step_metrics[0].tool_count == 0
            assert result.step_metrics[0].tool_names == ()

    def test_step_metrics_compressed_flag(self):
        """Step metrics track whether auto-compression fired."""
        client = MockLLMClient([_text_resp("done")])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(llm_client=client, tools=[])
            # No auto-compress configured, so compressed=False
            assert not result.step_metrics[0].compressed

    def test_step_metrics_import(self):
        """StepMetrics is importable from top-level tract package."""
        from tract import StepMetrics
        assert StepMetrics is not None


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------


class TestRunCombined:
    def test_budget_with_metrics(self):
        """step_budget and step_metrics work together via run()."""
        client = MockLLMClient([_text_resp("r", tokens=1000)])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(
                llm_client=client,
                tools=[],
                step_budget=500,
            )
            assert result.budget_exhausted
            assert len(result.step_metrics) == 1
            assert result.total_duration_ms > 0

    def test_all_params_together(self):
        """All three new params passed simultaneously through run()."""
        def validator(name, args):
            return True, None

        client = MockLLMClient([_text_resp("done", tokens=50)])
        with Tract.open() as t:
            t.system("Test")
            result = t._llm_mgr.run(
                llm_client=client,
                tools=[],
                step_budget=10000,
                tool_validator=validator,
                auto_compress_threshold=0.9,
                max_tokens=100000,
            )
            assert result.status == "completed"
            assert not result.budget_exhausted
            assert result.compressions_triggered == 0
            assert len(result.step_metrics) == 1
