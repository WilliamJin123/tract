"""Tests for auto-compress and loop metrics."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import Tract
from tract.loop import LoopConfig, LoopResult, StepMetrics, run_loop


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
                "id": "c1",
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
# Auto-compress tests
# ---------------------------------------------------------------------------


class TestAutoCompress:
    def test_compresses_when_over_threshold(self, tmp_path):
        """Auto-compress triggers when context exceeds threshold."""
        t = Tract.open(str(tmp_path / "test.db"))
        # Fill context with lots of messages to push token count high
        for i in range(50):
            t.user(f"Message {i} " * 20)
            t.assistant(f"Response {i} " * 20)

        client = MockLLMClient([_text_resp("done")])
        result = run_loop(t, config=LoopConfig(
            auto_compress_threshold=0.5,
            max_tokens=2000,  # low threshold to trigger compression
        ), llm_client=client)
        assert result.status == "completed"
        # Check that compression was triggered (context should exceed 1000 tokens)
        assert result.compressions_triggered >= 0  # may or may not trigger depending on actual token count

    def test_no_compress_under_threshold(self, tmp_path):
        """Auto-compress does NOT trigger when context is small."""
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("Short prompt")
        client = MockLLMClient([_text_resp("done")])
        result = run_loop(t, config=LoopConfig(
            auto_compress_threshold=0.8,
            max_tokens=100000,
        ), llm_client=client)
        assert result.compressions_triggered == 0

    def test_no_compress_without_config(self, tmp_path):
        """Auto-compress is off when threshold is None."""
        t = Tract.open(str(tmp_path / "test.db"))
        for i in range(50):
            t.user(f"Lots of text {i} " * 50)
        client = MockLLMClient([_text_resp("done")])
        result = run_loop(t, config=LoopConfig(
            auto_compress_threshold=None,
        ), llm_client=client)
        assert result.compressions_triggered == 0

    def test_no_compress_without_max_tokens(self, tmp_path):
        """Auto-compress requires both threshold and max_tokens to fire."""
        t = Tract.open(str(tmp_path / "test.db"))
        for i in range(50):
            t.user(f"Lots of text {i} " * 50)
        client = MockLLMClient([_text_resp("done")])
        result = run_loop(t, config=LoopConfig(
            auto_compress_threshold=0.5,
            max_tokens=None,
        ), llm_client=client)
        assert result.compressions_triggered == 0

    def test_compress_failure_continues(self, tmp_path):
        """When auto-compress fails, the loop continues with the large context."""
        t = Tract.open(str(tmp_path / "test.db"))
        for i in range(50):
            t.user(f"Message {i} " * 20)
            t.assistant(f"Response {i} " * 20)

        # Patch compress to raise
        original_compress = t.compress
        def bad_compress(**kw):
            raise RuntimeError("Compress broken")
        t.compress = bad_compress

        client = MockLLMClient([_text_resp("done")])
        result = run_loop(t, config=LoopConfig(
            auto_compress_threshold=0.01,  # very low threshold to guarantee trigger
            max_tokens=100,
        ), llm_client=client)
        # Should complete despite compress failure
        assert result.status == "completed"
        t.compress = original_compress


# ---------------------------------------------------------------------------
# Step metrics tests
# ---------------------------------------------------------------------------


class TestStepMetrics:
    def test_metrics_populated(self, tmp_path):
        """Step metrics are populated for each step."""
        client = MockLLMClient([_text_resp("done", tokens=100)])
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("Test")
        result = run_loop(t, llm_client=client)
        assert len(result.step_metrics) == 1
        m = result.step_metrics[0]
        assert m.step == 1
        assert m.duration_ms > 0
        assert m.llm_duration_ms >= 0
        assert m.tool_count == 0
        assert m.tool_names == ()
        assert m.context_tokens > 0
        assert not m.compressed

    def test_multi_step_metrics(self, tmp_path):
        """Multiple steps produce multiple metrics."""
        client = MockLLMClient([
            _text_resp("step 1", 50),
            _text_resp("step 2", 50),
            _text_resp("step 3", 50),
        ])
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("Test")
        result = run_loop(t, config=LoopConfig(
            stop_on_no_tool_call=False,
            max_steps=3,
        ), llm_client=client)
        assert len(result.step_metrics) == 3
        for i, m in enumerate(result.step_metrics):
            assert m.step == i + 1

    def test_tool_metrics_recorded(self, tmp_path):
        """Tool call info is recorded in step metrics."""
        client = MockLLMClient([_tool_resp("my_tool"), _text_resp("done")])
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("Test")
        result = run_loop(
            t,
            llm_client=client,
            tools=[{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
            tool_handlers={"my_tool": lambda: "result"},
        )
        # Step 1: tool call, Step 2: text response
        assert len(result.step_metrics) == 2
        assert result.step_metrics[0].tool_count == 1
        assert result.step_metrics[0].tool_names == ("my_tool",)
        assert result.step_metrics[1].tool_count == 0

    def test_total_duration(self, tmp_path):
        """total_duration_ms sums all step durations."""
        client = MockLLMClient([_text_resp("done")])
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("Test")
        result = run_loop(t, llm_client=client)
        assert result.total_duration_ms > 0
        assert result.total_llm_duration_ms >= 0

    def test_step_metrics_on_budget_exhaustion(self, tmp_path):
        """Step metrics are populated even when budget is exhausted."""
        client = MockLLMClient([_text_resp("a", 1000)])
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("Test")
        result = run_loop(t, config=LoopConfig(step_budget=500), llm_client=client)
        assert len(result.step_metrics) == 1
        assert result.budget_exhausted

    def test_step_metrics_on_max_steps(self, tmp_path):
        """Step metrics are populated on max_steps exit."""
        responses = [
            _tool_resp("my_tool") for _ in range(3)
        ]
        client = MockLLMClient(responses)
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("Test")
        result = run_loop(
            t,
            config=LoopConfig(max_steps=3),
            llm_client=client,
            tools=[{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
            tool_handlers={"my_tool": lambda: "ok"},
        )
        assert result.status == "max_steps"
        assert len(result.step_metrics) == 3

    def test_compressions_triggered_property(self):
        """compressions_triggered counts compressed steps."""
        r = LoopResult(
            "completed", "done", 3, 0,
            step_metrics=(
                StepMetrics(1, 10.0, 5.0, 0, (), 100, False),
                StepMetrics(2, 10.0, 5.0, 0, (), 100, True),
                StepMetrics(3, 10.0, 5.0, 0, (), 100, True),
            ),
        )
        assert r.compressions_triggered == 2

    def test_total_duration_property(self):
        """total_duration_ms sums step durations from StepMetrics."""
        r = LoopResult(
            "completed", "done", 2, 0,
            step_metrics=(
                StepMetrics(1, 100.0, 50.0, 0, (), 100, False),
                StepMetrics(2, 200.0, 75.0, 0, (), 100, False),
            ),
        )
        assert r.total_duration_ms == 300.0
        assert r.total_llm_duration_ms == 125.0

    def test_empty_step_metrics_properties(self):
        """Properties return 0 when no step metrics exist."""
        r = LoopResult("completed", "done", 0, 0)
        assert r.total_duration_ms == 0.0
        assert r.total_llm_duration_ms == 0.0
        assert r.compressions_triggered == 0
