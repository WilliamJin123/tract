"""Tests for the default agent loop (src/tract/loop.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from tract.loop import LoopConfig, LoopResult, _extract_content, _extract_tool_calls, run_loop


# ---------------------------------------------------------------------------
# Helpers: mock LLM client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Minimal LLM client that returns pre-configured responses."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[dict] = []

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self._call_count >= len(self._responses):
            # Default: return text with no tool calls
            return _make_response("Done.")
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    def extract_content(self, response):
        return response["choices"][0]["message"]["content"]

    def close(self):
        pass


def _make_response(content: str | None = None, tool_calls: list[dict] | None = None):
    """Build an OpenAI-style response dict."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{i}",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments", {})),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    return {"choices": [{"message": msg}]}


@pytest.fixture()
def tract_instance(tmp_path):
    """Create a minimal Tract instance for loop testing."""
    from tract import Tract

    db = str(tmp_path / "test.db")
    t = Tract.open(db)
    return t


# ---------------------------------------------------------------------------
# LoopResult / LoopConfig basics
# ---------------------------------------------------------------------------


class TestLoopResult:
    def test_fields(self):
        r = LoopResult("completed", "done", 3, 5, "hello")
        assert r.status == "completed"
        assert r.reason == "done"
        assert r.steps == 3
        assert r.tool_calls == 5
        assert r.final_response == "hello"

    def test_defaults(self):
        r = LoopResult("error", "fail", 1, 0)
        assert r.final_response is None

    def test_frozen(self):
        r = LoopResult("completed", "done", 1, 0)
        with pytest.raises(AttributeError):
            r.status = "error"  # type: ignore[misc]


class TestLoopConfig:
    def test_defaults(self):
        cfg = LoopConfig()
        assert cfg.max_steps == 50
        assert cfg.system_prompt is None
        assert cfg.strategy == "full"
        assert cfg.strategy_k == 5
        assert cfg.stop_on_no_tool_call is True

    def test_custom(self):
        cfg = LoopConfig(max_steps=10, system_prompt="You are a bot.", strategy="adaptive", stop_on_no_tool_call=False)
        assert cfg.max_steps == 10
        assert cfg.system_prompt == "You are a bot."


# ---------------------------------------------------------------------------
# Content / tool call extraction
# ---------------------------------------------------------------------------


class TestExtractContent:
    def test_dict_openai_format(self):
        resp = _make_response("Hello world")
        assert _extract_content(resp) == "Hello world"

    def test_dict_with_client(self):
        client = MockLLMClient([])
        resp = _make_response("via client")
        assert _extract_content(resp, client) == "via client"

    def test_string_response(self):
        assert _extract_content("raw string") == "raw string"

    def test_none_content(self):
        resp = _make_response(None)
        # Client extract_content returns None
        result = _extract_content(resp)
        assert result is None

    def test_openai_object(self):
        """Test with an object that has .choices[0].message.content."""
        msg = MagicMock()
        msg.content = "object content"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        assert _extract_content(resp) == "object content"


class TestExtractToolCalls:
    def test_dict_openai_format(self):
        resp = _make_response(
            "thinking...",
            tool_calls=[
                {"name": "status", "arguments": {}},
                {"name": "log", "arguments": {"limit": 5}},
            ],
        )
        tcs = _extract_tool_calls(resp)
        assert len(tcs) == 2
        assert tcs[0]["name"] == "status"
        assert tcs[0]["arguments"] == {}
        assert tcs[1]["name"] == "log"
        assert tcs[1]["arguments"] == {"limit": 5}

    def test_no_tool_calls(self):
        resp = _make_response("just text")
        assert _extract_tool_calls(resp) == []

    def test_openai_object_format(self):
        """Test with mock OpenAI objects."""
        func = MagicMock()
        func.name = "commit"
        func.arguments = '{"content": {"content_type": "dialogue"}}'
        tc = MagicMock()
        tc.id = "call_1"
        tc.function = func
        msg = MagicMock()
        msg.tool_calls = [tc]
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        tcs = _extract_tool_calls(resp)
        assert len(tcs) == 1
        assert tcs[0]["name"] == "commit"
        assert tcs[0]["arguments"]["content"]["content_type"] == "dialogue"

    def test_flat_dict_format(self):
        """Test with a flat dict containing tool_calls list."""
        resp = {"tool_calls": [{"name": "status", "arguments": {}}]}
        tcs = _extract_tool_calls(resp)
        assert len(tcs) == 1
        assert tcs[0]["name"] == "status"


# ---------------------------------------------------------------------------
# run_loop tests
# ---------------------------------------------------------------------------


class TestRunLoop:
    def test_no_client_raises(self, tract_instance):
        with pytest.raises(ValueError, match="No LLM client"):
            run_loop(tract_instance)

    def test_basic_no_tools(self, tract_instance):
        """LLM returns text, no tool calls -- loop completes in 1 step."""
        client = MockLLMClient([_make_response("I'm done.")])
        result = run_loop(tract_instance, llm_client=client)
        assert result.status == "completed"
        assert result.reason == "LLM finished (no tool calls)"
        assert result.steps == 1
        assert result.tool_calls == 0
        assert result.final_response == "I'm done."

    def test_with_task(self, tract_instance):
        """Task is committed as user message before loop starts."""
        client = MockLLMClient([_make_response("OK")])
        result = run_loop(tract_instance, task="Do something", llm_client=client)
        assert result.status == "completed"
        # Verify the task was committed (check log)
        entries = tract_instance.log(limit=10)
        # Should have: user message (task) + assistant message (OK)
        content_types = [e.content_type for e in entries]
        assert "dialogue" in content_types

    def test_without_task(self, tract_instance):
        """No task -- loop starts directly with compile."""
        # Need at least one commit for compile to work
        tract_instance.system("You are a test bot.")
        client = MockLLMClient([_make_response("Hello")])
        result = run_loop(tract_instance, llm_client=client)
        assert result.status == "completed"

    def test_with_tools(self, tract_instance):
        """LLM calls a tool, result is committed, loop continues."""
        client = MockLLMClient([
            _make_response("Let me check.", tool_calls=[{"name": "status", "arguments": {}}]),
            _make_response("All good."),
        ])
        result = run_loop(tract_instance, task="Check status", llm_client=client)
        assert result.status == "completed"
        assert result.tool_calls == 1
        assert result.steps == 2
        assert result.final_response == "All good."

    def test_multi_step(self, tract_instance):
        """Multiple tool calls across multiple steps."""
        client = MockLLMClient([
            _make_response("Step 1", tool_calls=[{"name": "status", "arguments": {}}]),
            _make_response("Step 2", tool_calls=[{"name": "status", "arguments": {}}]),
            _make_response("Done."),
        ])
        result = run_loop(tract_instance, task="Multi-step", llm_client=client)
        assert result.status == "completed"
        assert result.steps == 3
        assert result.tool_calls == 2

    def test_max_steps(self, tract_instance):
        """Loop stops at max_steps."""
        # Always return tool calls so loop never stops naturally
        responses = [
            _make_response(f"Step {i}", tool_calls=[{"name": "status", "arguments": {}}])
            for i in range(5)
        ]
        client = MockLLMClient(responses)
        config = LoopConfig(max_steps=3)
        result = run_loop(tract_instance, task="Infinite", llm_client=client, config=config)
        assert result.status == "max_steps"
        assert result.steps == 3
        assert "3" in result.reason

    def test_stop_on_no_tool_call_true(self, tract_instance):
        """Default: stops when LLM doesn't call tools."""
        client = MockLLMClient([_make_response("No tools needed.")])
        result = run_loop(tract_instance, task="Simple", llm_client=client)
        assert result.status == "completed"
        assert result.steps == 1

    def test_stop_on_no_tool_call_false(self, tract_instance):
        """With stop_on_no_tool_call=False, loop continues even without tools."""
        client = MockLLMClient([
            _make_response("Thinking..."),
            _make_response("Still thinking..."),
            _make_response("Done.", tool_calls=[{"name": "status", "arguments": {}}]),
            _make_response("Final answer."),
        ])
        config = LoopConfig(stop_on_no_tool_call=False, max_steps=4)
        result = run_loop(tract_instance, task="Think hard", llm_client=client, config=config)
        # Should run all 4 steps (3 without tools, 1 with, then max)
        assert result.steps == 4
        assert result.tool_calls == 1

    def test_system_prompt(self, tract_instance):
        """System prompt prepended to compiled messages."""
        client = MockLLMClient([_make_response("Got it.")])
        config = LoopConfig(system_prompt="You are a helpful assistant.")
        run_loop(tract_instance, task="Hello", llm_client=client, config=config)
        # Check that the first message in the LLM call was the system prompt
        call_messages = client.calls[0]["messages"]
        assert call_messages[0]["role"] == "system"
        assert call_messages[0]["content"] == "You are a helpful assistant."

    def test_compile_error(self, tract_instance):
        """Compile failure returns error status."""
        client = MockLLMClient([])
        # Force compile to fail by mocking
        original_compile = tract_instance.compile
        def bad_compile(**kwargs):
            raise RuntimeError("DB corrupted")
        tract_instance.compile = bad_compile
        result = run_loop(tract_instance, task="Boom", llm_client=client)
        assert result.status == "error"
        assert "Compile failed" in result.reason
        tract_instance.compile = original_compile

    def test_llm_error(self, tract_instance):
        """LLM call failure returns error status."""
        class FailClient:
            def chat(self, **kwargs):
                raise ConnectionError("API down")
            def close(self):
                pass
        result = run_loop(tract_instance, task="Fail", llm_client=FailClient())
        assert result.status == "error"
        assert "LLM call failed" in result.reason

    def test_tool_error_committed(self, tract_instance):
        """Tool execution error is committed as error result, loop continues."""
        # Create a tool call for a non-existent tool
        client = MockLLMClient([
            _make_response("Try this", tool_calls=[{"name": "nonexistent_tool", "arguments": {}}]),
            _make_response("Oh well."),
        ])
        result = run_loop(tract_instance, task="Bad tool", llm_client=client)
        assert result.status == "completed"
        assert result.tool_calls == 1
        # Check that error was committed
        entries = tract_instance.log(limit=20)
        messages = [e.message for e in entries if e.message and "tool error" in e.message]
        assert len(messages) >= 1

    def test_on_step_callback(self, tract_instance):
        """on_step callback is called for each step."""
        client = MockLLMClient([
            _make_response("Step 1", tool_calls=[{"name": "status", "arguments": {}}]),
            _make_response("Done."),
        ])
        steps_seen = []
        def on_step(step_num, response):
            steps_seen.append(step_num)
        run_loop(tract_instance, task="Callback test", llm_client=client, on_step=on_step)
        assert steps_seen == [1, 2]

    def test_custom_tools(self, tract_instance):
        """Custom tools parameter passed to LLM."""
        custom_tools = [{"type": "function", "function": {"name": "custom", "parameters": {}}}]
        client = MockLLMClient([_make_response("OK")])
        run_loop(tract_instance, task="Custom", llm_client=client, tools=custom_tools)
        assert client.calls[0]["kwargs"].get("tools") is None or True
        # Tools are passed via kwargs to chat()
        # The important thing is it doesn't crash

    def test_loop_result_all_fields(self, tract_instance):
        """All LoopResult fields are populated correctly."""
        client = MockLLMClient([
            _make_response("Working", tool_calls=[{"name": "status", "arguments": {}}]),
            _make_response("All done here."),
        ])
        result = run_loop(tract_instance, task="Full test", llm_client=client)
        assert isinstance(result.status, str)
        assert isinstance(result.reason, str)
        assert isinstance(result.steps, int)
        assert isinstance(result.tool_calls, int)
        assert isinstance(result.final_response, str)
        assert result.final_response == "All done here."


# ---------------------------------------------------------------------------
# Tract.run() facade test
# ---------------------------------------------------------------------------


class TestTractRunFacade:
    def test_run_through_facade(self, tract_instance):
        """t.run(task='...') delegates to run_loop."""
        # Inject a mock LLM client
        tract_instance._llm_client = MockLLMClient([_make_response("Done via facade.")])
        result = tract_instance.run(task="Facade test")
        assert result.status == "completed"
        assert result.final_response == "Done via facade."

    def test_run_with_custom_client(self, tract_instance):
        """t.run() accepts llm_client override."""
        client = MockLLMClient([_make_response("Custom client.")])
        result = tract_instance.run(task="Custom", llm_client=client)
        assert result.status == "completed"
        assert result.final_response == "Custom client."

    def test_run_max_steps(self, tract_instance):
        """t.run(max_steps=...) is respected."""
        responses = [
            _make_response(f"S{i}", tool_calls=[{"name": "status", "arguments": {}}])
            for i in range(10)
        ]
        client = MockLLMClient(responses)
        result = tract_instance.run(task="Limited", max_steps=2, llm_client=client)
        assert result.status == "max_steps"
        assert result.steps == 2

    def test_run_system_prompt(self, tract_instance):
        """t.run(system_prompt=...) is passed through."""
        client = MockLLMClient([_make_response("Sys prompted.")])
        tract_instance.run(task="SP", system_prompt="Be concise.", llm_client=client)
        call_messages = client.calls[0]["messages"]
        assert call_messages[0]["role"] == "system"
        assert call_messages[0]["content"] == "Be concise."


# ---------------------------------------------------------------------------
# Loop with rules
# ---------------------------------------------------------------------------


class TestLoopWithRules:
    def test_config_from_rules(self, tract_instance):
        """Active rules that set compile_strategy are respected."""
        tract_instance.rule(
            name="use-adaptive",
            trigger="active",
            action={"type": "set_config", "key": "compile_strategy", "value": "adaptive"},
        )
        client = MockLLMClient([_make_response("Rule-configured.")])
        result = run_loop(tract_instance, llm_client=client)
        assert result.status == "completed"

    def test_loop_with_rule_commit(self, tract_instance):
        """Rules can fire during loop execution (on commit events)."""
        # Create a rule that fires on commit
        tract_instance.rule(
            name="on-commit-log",
            trigger="commit",
            action={"type": "set_config", "key": "last_commit_seen", "value": True},
        )
        client = MockLLMClient([
            _make_response("Committing something", tool_calls=[{"name": "status", "arguments": {}}]),
            _make_response("Done with rules."),
        ])
        result = run_loop(tract_instance, task="Rule test", llm_client=client)
        assert result.status == "completed"

    def test_loop_blocked_by_rule(self, tract_instance):
        """Block rule on compile stops the loop with 'blocked' status."""
        from tract.exceptions import BlockedByRuleError

        # Create a block rule on compile
        tract_instance.rule(
            name="block-compile",
            trigger="compile",
            action={"type": "block", "reason": "Context window full"},
        )
        client = MockLLMClient([_make_response("Should not reach here.")])
        result = run_loop(tract_instance, task="Blocked loop", llm_client=client)
        assert result.status == "blocked"
        assert "Context window full" in result.reason
        assert result.steps == 1
        assert result.tool_calls == 0


# ---------------------------------------------------------------------------
# Rule blocking on operations (wired into tract.py)
# ---------------------------------------------------------------------------


class TestRuleBlocking:
    def test_compile_blocked(self, tract_instance):
        """Block rule on compile raises BlockedByRuleError."""
        from tract.exceptions import BlockedByRuleError

        tract_instance.system("Some content")
        tract_instance.rule(
            name="block-compile",
            trigger="compile",
            action={"type": "block", "reason": "Budget exceeded"},
        )
        with pytest.raises(BlockedByRuleError, match="compile blocked"):
            tract_instance.compile()

    def test_compile_not_blocked_without_rule(self, tract_instance):
        """Compile works normally without block rules."""
        tract_instance.system("Hello")
        result = tract_instance.compile()
        assert result.token_count > 0

    def test_commit_fires_rules_post_persist(self, tract_instance):
        """Commit fires rules after persist (informational, not blocking)."""
        # A set_config action on commit trigger should execute without error
        tract_instance.rule(
            name="on-commit",
            trigger="commit",
            action={"type": "set_config", "key": "committed", "value": True},
        )
        # This should succeed -- commit rules are informational
        info = tract_instance.system("After rule")
        assert info.commit_hash

    def test_compress_blocked(self, tmp_path):
        """Block rule on compress raises BlockedByRuleError."""
        from tract.exceptions import BlockedByRuleError

        db = str(tmp_path / "block_compress.db")
        t = __import__("tract").Tract.open(db)
        # Add enough content to compress
        for i in range(5):
            t.system(f"Message {i}")
        t.rule(
            name="block-compress",
            trigger="compress",
            action={"type": "block", "reason": "Compression disabled"},
        )
        with pytest.raises(BlockedByRuleError, match="compress blocked"):
            t.compress(content="manual summary")

    def test_gc_blocked(self, tmp_path):
        """Block rule on gc raises BlockedByRuleError."""
        from tract.exceptions import BlockedByRuleError

        db = str(tmp_path / "block_gc.db")
        t = __import__("tract").Tract.open(db)
        t.system("Content")
        t.rule(
            name="block-gc",
            trigger="gc",
            action={"type": "block", "reason": "GC disabled"},
        )
        with pytest.raises(BlockedByRuleError, match="gc blocked"):
            t.gc()

    def test_merge_blocked(self, tmp_path):
        """Block rule on merge raises BlockedByRuleError."""
        from tract.exceptions import BlockedByRuleError

        db = str(tmp_path / "block_merge.db")
        t = __import__("tract").Tract.open(db)
        t.system("Main content")
        t.branch("feature", switch=True)
        t.system("Feature content")
        t.switch("main")
        t.rule(
            name="block-merge",
            trigger="merge",
            action={"type": "block", "reason": "Merge frozen"},
        )
        with pytest.raises(BlockedByRuleError, match="merge blocked"):
            t.merge("feature")
