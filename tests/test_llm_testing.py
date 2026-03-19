"""Tests for tract.llm.testing -- first-party LLM test utilities.

Covers MockLLMClient, ReplayLLMClient, and FunctionLLMClient:
protocol conformance, response handling, call tracking, close(),
and integration with Tract.open(llm_client=...).
"""

from __future__ import annotations

import pytest

from tract import Tract
from tract.llm.protocols import LLMClient
from tract.llm.testing import (
    FunctionLLMClient,
    MockLLMClient,
    ReplayLLMClient,
    _make_response,
)


# -------------------------------------------------------------------
# _make_response helper
# -------------------------------------------------------------------

class TestMakeResponse:
    def test_structure(self):
        resp = _make_response("hello", "test-model")
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert resp["choices"][0]["message"]["content"] == "hello"
        assert resp["model"] == "test-model"
        assert resp["usage"]["prompt_tokens"] == 0
        assert resp["usage"]["completion_tokens"] == 0
        assert resp["usage"]["total_tokens"] == 0


# -------------------------------------------------------------------
# MockLLMClient
# -------------------------------------------------------------------

class TestMockLLMClient:
    def test_satisfies_protocol(self):
        mock = MockLLMClient(["hi"])
        assert isinstance(mock, LLMClient)

    def test_cycles_responses(self):
        mock = MockLLMClient(["A", "B"])
        r1 = mock.chat([])
        r2 = mock.chat([])
        r3 = mock.chat([])
        assert mock.extract_content(r1) == "A"
        assert mock.extract_content(r2) == "B"
        assert mock.extract_content(r3) == "A"  # wraps around

    def test_records_calls(self):
        mock = MockLLMClient(["resp"])
        msgs = [{"role": "user", "content": "question"}]
        mock.chat(msgs, model="custom", temperature=0.5)
        assert len(mock.calls) == 1
        assert mock.call_count == 1
        assert mock.calls[0]["messages"] == msgs
        assert mock.calls[0]["model"] == "custom"
        assert mock.calls[0]["temperature"] == 0.5

    def test_call_count_increments(self):
        mock = MockLLMClient(["a", "b", "c"])
        for _ in range(5):
            mock.chat([])
        assert mock.call_count == 5
        assert len(mock.calls) == 5

    def test_close_sets_closed(self):
        mock = MockLLMClient(["hi"])
        assert mock.closed is False
        mock.close()
        assert mock.closed is True

    def test_extract_content(self):
        mock = MockLLMClient(["hello world"])
        resp = mock.chat([])
        assert mock.extract_content(resp) == "hello world"

    def test_extract_usage(self):
        mock = MockLLMClient(["hi"])
        resp = mock.chat([])
        usage = mock.extract_usage(resp)
        assert usage is not None
        assert "total_tokens" in usage

    def test_custom_model_name(self):
        mock = MockLLMClient(["hi"], model="my-model")
        resp = mock.chat([])
        assert resp["model"] == "my-model"

    def test_model_kwarg_overrides(self):
        mock = MockLLMClient(["hi"], model="default")
        resp = mock.chat([], model="override")
        assert resp["model"] == "override"

    def test_with_tract_open_chat(self):
        mock = MockLLMClient(["I can help with that."])
        with Tract.open(llm_client=mock) as t:
            t.system("You are a test assistant.")
            r = t.runtime.chat("Hello")
            assert r.text == "I can help with that."
            assert mock.call_count == 1


# -------------------------------------------------------------------
# ReplayLLMClient
# -------------------------------------------------------------------

class TestReplayLLMClient:
    def test_satisfies_protocol(self):
        replay = ReplayLLMClient(["hi"])
        assert isinstance(replay, LLMClient)

    def test_plays_sequentially(self):
        replay = ReplayLLMClient(["first", "second", "third"])
        r1 = replay.chat([])
        r2 = replay.chat([])
        r3 = replay.chat([])
        assert replay.extract_content(r1) == "first"
        assert replay.extract_content(r2) == "second"
        assert replay.extract_content(r3) == "third"

    def test_raises_on_exhaustion(self):
        replay = ReplayLLMClient(["only-one"])
        replay.chat([])
        with pytest.raises(IndexError, match="ReplayLLMClient exhausted: 1 responses consumed"):
            replay.chat([])

    def test_exhaustion_message_includes_count(self):
        replay = ReplayLLMClient(["a", "b", "c"])
        for _ in range(3):
            replay.chat([])
        with pytest.raises(IndexError, match="3 responses consumed"):
            replay.chat([])

    def test_accepts_full_dict_responses(self):
        custom = {
            "choices": [{"message": {"role": "assistant", "content": "from dict"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "dict-model",
        }
        replay = ReplayLLMClient([custom])
        resp = replay.chat([])
        assert resp is custom  # returned as-is, not wrapped
        assert replay.extract_content(resp) == "from dict"

    def test_mixed_string_and_dict(self):
        custom_dict = {
            "choices": [{"message": {"role": "assistant", "content": "dict response"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": "x",
        }
        replay = ReplayLLMClient(["string response", custom_dict])
        r1 = replay.chat([])
        r2 = replay.chat([])
        assert replay.extract_content(r1) == "string response"
        assert replay.extract_content(r2) == "dict response"

    def test_records_calls(self):
        replay = ReplayLLMClient(["a", "b"])
        replay.chat([{"role": "user", "content": "q1"}])
        replay.chat([{"role": "user", "content": "q2"}])
        assert len(replay.calls) == 2
        assert replay.call_count == 2
        assert replay.calls[0]["messages"][0]["content"] == "q1"

    def test_records_calls_even_on_exhaustion(self):
        replay = ReplayLLMClient(["only"])
        replay.chat([])
        with pytest.raises(IndexError):
            replay.chat([{"role": "user", "content": "extra"}])
        # The exhausted call is still recorded
        assert replay.call_count == 2
        assert len(replay.calls) == 2

    def test_close_sets_closed(self):
        replay = ReplayLLMClient(["hi"])
        assert replay.closed is False
        replay.close()
        assert replay.closed is True

    def test_with_tract_open_chat(self):
        replay = ReplayLLMClient(["Replay answer."])
        with Tract.open(llm_client=replay) as t:
            t.system("Test.")
            r = t.runtime.chat("Question")
            assert r.text == "Replay answer."
            assert replay.call_count == 1


# -------------------------------------------------------------------
# FunctionLLMClient
# -------------------------------------------------------------------

class TestFunctionLLMClient:
    def test_satisfies_protocol(self):
        fn = FunctionLLMClient(lambda msgs, kw: "ok")
        assert isinstance(fn, LLMClient)

    def test_calls_function_with_correct_args(self):
        received_args = {}

        def capture(messages, kwargs):
            received_args["messages"] = messages
            received_args["kwargs"] = kwargs
            return "captured"

        fn = FunctionLLMClient(capture)
        msgs = [{"role": "user", "content": "hello"}]
        fn.chat(msgs, model="m", temperature=0.7, max_tokens=100)

        assert received_args["messages"] == msgs
        assert received_args["kwargs"]["model"] == "m"
        assert received_args["kwargs"]["temperature"] == 0.7
        assert received_args["kwargs"]["max_tokens"] == 100

    def test_string_return_wrapped(self):
        fn = FunctionLLMClient(lambda msgs, kw: "plain text")
        resp = fn.chat([])
        assert fn.extract_content(resp) == "plain text"
        assert resp["model"] == "function-model"
        assert "usage" in resp

    def test_dict_return_passthrough(self):
        custom = {
            "choices": [{"message": {"role": "assistant", "content": "custom"}}],
            "usage": {"prompt_tokens": 99, "completion_tokens": 1, "total_tokens": 100},
            "model": "custom",
        }

        fn = FunctionLLMClient(lambda msgs, kw: custom)
        resp = fn.chat([])
        assert resp is custom

    def test_conditional_responses(self):
        def responder(messages, kwargs):
            last = messages[-1]["content"] if messages else ""
            if "weather" in last:
                return "Sunny"
            return "I don't know"

        fn = FunctionLLMClient(responder)
        r1 = fn.chat([{"role": "user", "content": "weather today?"}])
        r2 = fn.chat([{"role": "user", "content": "something else"}])
        assert fn.extract_content(r1) == "Sunny"
        assert fn.extract_content(r2) == "I don't know"

    def test_records_calls(self):
        fn = FunctionLLMClient(lambda msgs, kw: "ok")
        fn.chat([{"role": "user", "content": "a"}])
        fn.chat([{"role": "user", "content": "b"}])
        assert len(fn.calls) == 2
        assert fn.call_count == 2

    def test_close_sets_closed(self):
        fn = FunctionLLMClient(lambda msgs, kw: "ok")
        assert fn.closed is False
        fn.close()
        assert fn.closed is True

    def test_custom_model_name(self):
        fn = FunctionLLMClient(lambda msgs, kw: "ok", model="my-fn-model")
        resp = fn.chat([])
        assert resp["model"] == "my-fn-model"

    def test_function_receives_extra_kwargs(self):
        received = {}

        def capture(messages, kwargs):
            received.update(kwargs)
            return "ok"

        fn = FunctionLLMClient(capture)
        fn.chat([], model="m", temperature=0.0, max_tokens=10, top_p=0.9)
        assert received["top_p"] == 0.9

    def test_with_tract_open_chat(self):
        fn = FunctionLLMClient(lambda msgs, kw: "Function answer.")
        with Tract.open(llm_client=fn) as t:
            t.system("Test.")
            r = t.runtime.chat("Question")
            assert r.text == "Function answer."
            assert fn.call_count == 1


# -------------------------------------------------------------------
# Cross-cutting: all three with Tract integration
# -------------------------------------------------------------------

class TestTractIntegration:
    """Verify all three clients work end-to-end with Tract.open()."""

    def test_mock_multi_turn(self):
        mock = MockLLMClient(["Reply 1", "Reply 2"])
        with Tract.open(llm_client=mock) as t:
            t.system("Test system.")
            r1 = t.runtime.chat("Turn 1")
            r2 = t.runtime.chat("Turn 2")
            assert r1.text == "Reply 1"
            assert r2.text == "Reply 2"
            # Verify conversation context grows
            assert len(mock.calls[1]["messages"]) > len(mock.calls[0]["messages"])

    def test_replay_exact_sequence(self):
        replay = ReplayLLMClient(["Answer A", "Answer B"])
        with Tract.open(llm_client=replay) as t:
            t.system("Test.")
            assert t.runtime.chat("Q1").text == "Answer A"
            assert t.runtime.chat("Q2").text == "Answer B"

    def test_function_inspects_context(self):
        """FunctionLLMClient can inspect the compiled messages tract sends."""
        seen_message_counts = []

        def inspector(messages, kwargs):
            seen_message_counts.append(len(messages))
            return "ok"

        fn = FunctionLLMClient(inspector)
        with Tract.open(llm_client=fn) as t:
            t.system("Sys.")
            t.runtime.chat("First")
            t.runtime.chat("Second")
            # Each successive call should have more messages in context
            assert seen_message_counts[1] > seen_message_counts[0]
