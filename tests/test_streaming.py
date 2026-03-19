"""Tests for streaming support in the loop and clients.

Uses mock clients to test the streaming integration without real API calls.
"""

import json
from unittest.mock import MagicMock

import pytest

from tract import Tract, LoopConfig, LoopResult
from tract.llm.anthropic_client import (
    MessageDone,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallStart,
    UsageEvent,
)
from tract.loop import _stream_to_response, run_loop


# -----------------------------------------------------------------------
# Mock streaming client
# -----------------------------------------------------------------------


class MockStreamingClient:
    """A mock LLM client that supports both chat() and stream()."""

    def __init__(self, responses=None, stream_events=None):
        self._responses = list(responses or [])
        self._stream_events = list(stream_events or [])
        self._call_count = 0

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs):
        if self._responses:
            resp = self._responses[self._call_count % len(self._responses)]
        else:
            resp = _make_response("Default response")
        self._call_count += 1
        return resp

    def stream(self, messages, *, model=None, temperature=None, max_tokens=None, **kwargs):
        """Yield pre-configured stream events."""
        if self._stream_events:
            events = self._stream_events[self._call_count % len(self._stream_events)]
        else:
            events = [
                TextDelta(text="Hello "),
                TextDelta(text="world!"),
                MessageDone(response=_make_response("Hello world!")),
            ]
        self._call_count += 1
        yield from events

    def close(self):
        pass


def _make_response(text, tool_calls=None):
    """Build a minimal OpenAI-format response dict."""
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# -----------------------------------------------------------------------
# _stream_to_response
# -----------------------------------------------------------------------


class TestStreamToResponse:
    def test_basic_text_stream(self):
        client = MockStreamingClient()
        chunks = []
        resp = _stream_to_response(
            client, [{"role": "user", "content": "Hi"}], None,
            on_token=lambda t: chunks.append(t),
        )
        assert resp["choices"][0]["message"]["content"] == "Hello world!"
        assert chunks == ["Hello ", "world!"]

    def test_stream_without_callback(self):
        client = MockStreamingClient()
        resp = _stream_to_response(
            client, [{"role": "user", "content": "Hi"}], None,
            on_token=None,
        )
        assert resp["choices"][0]["message"]["content"] == "Hello world!"

    def test_stream_with_thinking(self):
        events = [
            ThinkingDelta(text="Let me think..."),
            TextDelta(text="The answer"),
            MessageDone(response=_make_response("The answer")),
        ]
        client = MockStreamingClient(stream_events=[events])
        chunks = []
        resp = _stream_to_response(
            client, [], None,
            on_token=lambda t: chunks.append(t),
        )
        # ThinkingDelta should NOT be passed to on_token
        assert chunks == ["The answer"]
        assert resp["choices"][0]["message"]["content"] == "The answer"

    def test_stream_raises_on_no_done(self):
        """Raises if stream ends without MessageDone."""
        class BadClient:
            def stream(self, *a, **kw):
                yield TextDelta(text="partial")
            def close(self):
                pass

        with pytest.raises(ValueError, match="MessageDone"):
            _stream_to_response(BadClient(), [], None, on_token=None)


# -----------------------------------------------------------------------
# Loop integration
# -----------------------------------------------------------------------


class TestLoopStreaming:
    def test_on_token_triggers_streaming(self):
        """Passing on_token= to run_loop uses stream() instead of chat()."""
        t = Tract.open()
        t.system("Be helpful.")

        client = MockStreamingClient()
        chunks = []

        result = run_loop(
            t,
            task="Say hello",
            config=LoopConfig(max_steps=1, stop_on_no_tool_call=True),
            llm_client=client,
            tools=[],
            on_token=lambda text: chunks.append(text),
        )

        assert result.status == "completed"
        assert len(chunks) >= 1  # Got streaming chunks
        t.close()

    def test_stream_config_triggers_streaming(self):
        """LoopConfig(stream=True) uses streaming even without on_token."""
        t = Tract.open()
        t.system("Be helpful.")

        client = MockStreamingClient()

        result = run_loop(
            t,
            task="Say hello",
            config=LoopConfig(max_steps=1, stream=True),
            llm_client=client,
            tools=[],
        )

        assert result.status == "completed"
        t.close()

    def test_no_streaming_without_support(self):
        """If client has no stream(), falls back to chat() even with on_token."""
        t = Tract.open()
        t.system("Be helpful.")

        class SyncOnlyClient:
            def chat(self, messages, **kwargs):
                return _make_response("sync response")
            def close(self):
                pass

        chunks = []
        result = run_loop(
            t,
            task="Say hello",
            config=LoopConfig(max_steps=1),
            llm_client=SyncOnlyClient(),
            tools=[],
            on_token=lambda text: chunks.append(text),
        )

        assert result.status == "completed"
        assert chunks == []  # No streaming happened
        t.close()

    def test_streaming_with_tool_calls(self):
        """Streaming works when the LLM calls tools."""
        t = Tract.open()
        t.system("Be helpful.")

        # First call: tool call, second call: final response
        tool_call_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "status", "arguments": "{}"},
        }])
        tool_call_response["choices"][0]["finish_reason"] = "tool_calls"

        final_response = _make_response("Here's your status.")

        call_idx = {"n": 0}

        class StreamingToolClient:
            def stream(self, messages, **kwargs):
                idx = call_idx["n"]
                call_idx["n"] += 1
                if idx == 0:
                    yield ToolCallStart(index=0, id="call_1", name="status")
                    yield ToolCallDelta(index=0, partial_json='{}')
                    yield MessageDone(response=tool_call_response)
                else:
                    yield TextDelta(text="Here's your status.")
                    yield MessageDone(response=final_response)

            def close(self):
                pass

        chunks = []
        result = run_loop(
            t,
            task="Check status",
            config=LoopConfig(max_steps=3),
            llm_client=StreamingToolClient(),
            tools=t.runtime.tools.as_tools(format="openai", tool_names=["status"]),
            on_token=lambda text: chunks.append(text),
        )

        assert result.status == "completed"
        assert result.tool_calls >= 1
        assert "Here's your status." in chunks
        t.close()


# -----------------------------------------------------------------------
# Tract.run() streaming
# -----------------------------------------------------------------------


class TestTractRunStreaming:
    def test_run_with_on_token(self):
        """Tract.run() passes on_token through to the loop."""
        t = Tract.open()
        t.system("Be helpful.")
        t.config.configure_llm(MockStreamingClient())

        chunks = []
        result = t._llm_mgr.run(
            "Hello",
            max_steps=1,
            tools=[],
            on_token=lambda text: chunks.append(text),
        )

        assert result.status == "completed"
        assert len(chunks) >= 1
        t.close()

    def test_run_with_stream_flag(self):
        """Tract.run(stream=True) enables streaming."""
        t = Tract.open()
        t.system("Be helpful.")
        t.config.configure_llm(MockStreamingClient())

        result = t._llm_mgr.run(
            "Hello",
            max_steps=1,
            tools=[],
            stream=True,
        )

        assert result.status == "completed"
        t.close()


# -----------------------------------------------------------------------
# Stream event types
# -----------------------------------------------------------------------


class TestStreamEvents:
    def test_text_delta(self):
        e = TextDelta(text="hello")
        assert e.text == "hello"
        assert isinstance(e, TextDelta)

    def test_tool_call_start(self):
        e = ToolCallStart(index=0, id="call_1", name="search")
        assert e.index == 0
        assert e.id == "call_1"
        assert e.name == "search"

    def test_tool_call_delta(self):
        e = ToolCallDelta(index=0, partial_json='{"q":')
        assert e.index == 0
        assert e.partial_json == '{"q":'

    def test_thinking_delta(self):
        e = ThinkingDelta(text="thinking...")
        assert e.text == "thinking..."

    def test_usage_event(self):
        e = UsageEvent(usage={"prompt_tokens": 10})
        assert e.usage["prompt_tokens"] == 10

    def test_message_done(self):
        resp = _make_response("done")
        e = MessageDone(response=resp)
        assert e.response is resp

    def test_all_are_stream_events(self):
        from tract.llm.anthropic_client import StreamEvent
        assert issubclass(TextDelta, StreamEvent)
        assert issubclass(ToolCallStart, StreamEvent)
        assert issubclass(ThinkingDelta, StreamEvent)
        assert issubclass(MessageDone, StreamEvent)
