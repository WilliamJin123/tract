"""Tests for async LLM methods.

Verifies that async methods (achat, agenerate, arun, acompress, etc.)
produce the same results as their sync counterparts when given the
same mock LLM responses.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tract import Tract, LLMConfig


# ---------------------------------------------------------------------------
# Fixtures: mock LLM client with both sync and async
# ---------------------------------------------------------------------------


def _make_response(content: str = "Hello!", tool_calls=None):
    """Build a minimal OpenAI-format response dict."""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class MockLLMClient:
    """Mock client implementing both sync and async LLMClient protocols."""

    def __init__(self, responses=None):
        self._responses = responses or [_make_response()]
        self._call_count = 0

    def chat(self, messages, **kwargs):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    async def achat(self, messages, **kwargs):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def close(self):
        pass

    async def aclose(self):
        pass

    @staticmethod
    def extract_content(response):
        return response["choices"][0]["message"].get("content") or ""

    @staticmethod
    def extract_usage(response):
        return response.get("usage")

    @staticmethod
    def extract_tool_calls(response):
        msg = response["choices"][0]["message"]
        tcs = msg.get("tool_calls", [])
        if not tcs:
            return []
        result = []
        for tc in tcs:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            result.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": args,
            })
        return result


class SyncOnlyClient:
    """Mock client with only sync chat() -- no achat()."""

    def __init__(self, response=None):
        self._response = response or _make_response()

    def chat(self, messages, **kwargs):
        return self._response

    def close(self):
        pass

    @staticmethod
    def extract_content(response):
        return response["choices"][0]["message"].get("content") or ""

    @staticmethod
    def extract_usage(response):
        return response.get("usage")


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def sync_only_client():
    return SyncOnlyClient()


@pytest.fixture
def tract_with_client(mock_client):
    t = Tract.open()
    t.config.configure_llm(mock_client)
    return t


@pytest.fixture
def tract_with_sync_client(sync_only_client):
    t = Tract.open()
    t.config.configure_llm(sync_only_client)
    return t


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestAsyncLLMClientProtocol:
    """Test the AsyncLLMClient protocol and acall_llm helper."""

    @pytest.mark.asyncio
    async def test_acall_llm_uses_achat_when_available(self, mock_client):
        """acall_llm should prefer achat() over chat()."""
        from tract.llm.protocols import acall_llm

        result = await acall_llm(mock_client, [{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_acall_llm_falls_back_to_thread(self, sync_only_client):
        """acall_llm should wrap sync chat() in asyncio.to_thread."""
        from tract.llm.protocols import acall_llm

        result = await acall_llm(sync_only_client, [{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_async_protocol_check(self, mock_client):
        """MockLLMClient should satisfy AsyncLLMClient protocol."""
        from tract.llm.protocols import AsyncLLMClient
        assert isinstance(mock_client, AsyncLLMClient)

    def test_sync_only_not_async_protocol(self, sync_only_client):
        """SyncOnlyClient should NOT satisfy AsyncLLMClient protocol."""
        from tract.llm.protocols import AsyncLLMClient
        assert not isinstance(sync_only_client, AsyncLLMClient)


# ---------------------------------------------------------------------------
# Tract async method tests
# ---------------------------------------------------------------------------


class TestAchat:
    """Test Tract.achat()."""

    @pytest.mark.asyncio
    async def test_achat_basic(self, tract_with_client):
        t = tract_with_client
        t.system("You are helpful.")
        result = await t._llm_mgr.achat("Hello")
        assert result.text == "Hello!"
        assert result.prompt == "Hello"
        assert result.usage is not None
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_achat_matches_sync(self, mock_client):
        """achat() should produce the same result as chat()."""
        # Sync
        t1 = Tract.open()
        t1.config.configure_llm(MockLLMClient())
        t1.system("You are helpful.")
        sync_result = t1._llm_mgr.chat("Hello")

        # Async
        t2 = Tract.open()
        t2.config.configure_llm(MockLLMClient())
        t2.system("You are helpful.")
        async_result = await t2._llm_mgr.achat("Hello")

        assert sync_result.text == async_result.text
        assert sync_result.usage == async_result.usage

    @pytest.mark.asyncio
    async def test_achat_with_sync_only_client(self, tract_with_sync_client):
        """achat() should work with sync-only clients via to_thread fallback."""
        t = tract_with_sync_client
        t.system("You are helpful.")
        result = await t._llm_mgr.achat("Hello")
        assert result.text == "Hello!"


class TestAgenerate:
    """Test Tract.agenerate()."""

    @pytest.mark.asyncio
    async def test_agenerate_basic(self, tract_with_client):
        t = tract_with_client
        t.system("You are helpful.")
        t.user("Hello")
        result = await t._llm_mgr.agenerate()
        assert result.text == "Hello!"

    @pytest.mark.asyncio
    async def test_agenerate_with_validator(self):
        """agenerate with validator should retry on failure."""
        responses = [
            _make_response("bad answer"),
            _make_response("bad answer"),  # retry steering msg
            _make_response("good answer"),
        ]
        client = MockLLMClient(responses)
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Be helpful.")
        t.user("Question")

        def validator(text):
            if "good" in text:
                return (True, None)
            return (False, "not good enough")

        result = await t._llm_mgr.agenerate(validator=validator, max_retries=3)
        assert result.text == "good answer"


class TestArun:
    """Test Tract.arun()."""

    @pytest.mark.asyncio
    async def test_arun_basic(self):
        """arun() should complete when LLM returns no tool calls."""
        client = MockLLMClient([_make_response("Done!")])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You are a helpful agent.")
        result = await t._llm_mgr.arun(task="Do something")
        assert result.status == "completed"
        assert result.final_response == "Done!"
        assert result.steps == 1

    @pytest.mark.asyncio
    async def test_arun_with_tool_call(self):
        """arun() should execute tool calls via custom handler."""
        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "check", "arguments": "{}"},
        }])
        final_response = _make_response("All done.")
        client = MockLLMClient([tool_response, final_response])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You manage context.")

        def check_handler():
            return "ok"

        result = await t._llm_mgr.arun(
            task="Check status",
            tools=[{
                "type": "function",
                "function": {
                    "name": "check",
                    "description": "Check status",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            tool_handlers={"check": check_handler},
        )
        assert result.status == "completed"
        assert result.tool_calls >= 1

    @pytest.mark.asyncio
    async def test_arun_with_async_tool_handler(self):
        """arun() should await async tool handlers."""
        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "my_tool", "arguments": '{"x": 1}'},
        }])
        final_response = _make_response("Done.")
        client = MockLLMClient([tool_response, final_response])

        async def my_async_tool(x=0):
            await asyncio.sleep(0)  # Simulate async work
            return f"result: {x}"

        t = Tract.open()
        t.config.configure_llm(client)
        result = await t._llm_mgr.arun(
            task="Use tool",
            tools=[{
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
                },
            }],
            tool_handlers={"my_tool": my_async_tool},
        )
        assert result.status == "completed"


class TestArunNewParams:
    """Test arun() with step_budget, tool_validator, auto_compress_threshold."""

    @pytest.mark.asyncio
    async def test_arun_step_budget_accepts_param(self):
        """arun(step_budget=100) should not raise TypeError."""
        client = MockLLMClient([_make_response("Done!")])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You are helpful.")
        result = await t._llm_mgr.arun(task="Do something", step_budget=100)
        assert result.status == "completed"
        assert result.steps == 1

    @pytest.mark.asyncio
    async def test_arun_step_budget_limits_loop(self):
        """arun with step_budget should stop when budget is exhausted."""
        # Each response has usage.total_tokens = 15. A budget of 20 means the
        # loop should stop after a couple of steps (the tool-calling response
        # plus the budget check) rather than running to max_steps.
        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        }])
        client = MockLLMClient([tool_response])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You manage context.")

        result = await t._llm_mgr.arun(
            task="Keep going",
            max_steps=50,
            step_budget=20,
            tools=[{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            tool_handlers={"ping": lambda: "pong"},
        )
        # The loop should have stopped well before max_steps=50
        assert result.steps < 50
        assert result.budget_exhausted

    @pytest.mark.asyncio
    async def test_arun_tool_validator(self):
        """arun with tool_validator should reject invalid tool calls."""
        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "danger", "arguments": "{}"},
        }])
        final_response = _make_response("Ok, done.")
        client = MockLLMClient([tool_response, final_response])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You manage context.")

        def validator(tool_name, args):
            if tool_name == "danger":
                return (False, "tool blocked by validator")
            return (True, None)

        result = await t._llm_mgr.arun(
            task="Do something",
            tool_validator=validator,
            tools=[{
                "type": "function",
                "function": {
                    "name": "danger",
                    "description": "Dangerous tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            tool_handlers={"danger": lambda: "should not run"},
        )
        # Loop should complete (LLM sends final response on second call)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_arun_auto_compress_threshold_accepts_param(self):
        """arun(auto_compress_threshold=0.8) should not raise TypeError."""
        client = MockLLMClient([_make_response("Done!")])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You are helpful.")
        result = await t._llm_mgr.arun(
            task="Do something",
            auto_compress_threshold=0.8,
        )
        assert result.status == "completed"


class TestAcompress:
    """Test Tract.acompress()."""

    @pytest.mark.asyncio
    async def test_acompress_manual(self):
        """acompress with manual content should work (no LLM needed)."""
        t = Tract.open()
        t.system("You are helpful.")
        t.user("Message 1")
        t.assistant("Response 1")
        t.user("Message 2")
        t.assistant("Response 2")

        result = await t.acompress(content="Summary of conversation.")
        assert result.compressed_tokens > 0

    @pytest.mark.asyncio
    async def test_acompress_with_llm(self):
        """acompress with LLM should use async LLM call."""
        client = MockLLMClient([_make_response("Compressed summary.")])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You are helpful.")
        t.user("Message 1")
        t.assistant("Response 1")
        t.user("Message 2")
        t.assistant("Response 2")

        result = await t.acompress()
        assert result.compressed_tokens > 0


class TestArevise:
    """Test Tract.arevise()."""

    @pytest.mark.asyncio
    async def test_arevise_basic(self):
        """arevise should create an EDIT commit with revised content."""
        client = MockLLMClient([_make_response("Improved text.")])
        t = Tract.open()
        t.config.configure_llm(client)
        original = t.assistant("Original text.")
        result = await t._llm_mgr.arevise(original.commit_hash, "Make it better")
        assert result.text == "Improved text."
        # The commit should be an EDIT
        assert result.commit_info is not None


# ---------------------------------------------------------------------------
# Session async tests
# ---------------------------------------------------------------------------


class TestSessionAcollapse:
    """Test Session.acollapse()."""

    @pytest.mark.asyncio
    async def test_acollapse_manual(self):
        """acollapse with manual content should work."""
        from tract import Session

        with Session.open() as session:
            parent = session.create_tract(display_name="parent")
            child = session.spawn(parent, purpose="research")
            child.user("Research question")
            child.assistant("Research answer")

            result = await session.acollapse(
                child, into=parent, content="Research complete: answer found."
            )
            assert result.summary_text == "Research complete: answer found."
            assert result.purpose == "research"

    @pytest.mark.asyncio
    async def test_acollapse_with_llm(self):
        """acollapse with LLM should use async call."""
        from tract import Session

        client = MockLLMClient([_make_response("Summarized research.")])
        with Session.open() as session:
            parent = session.create_tract(display_name="parent")
            parent.config.configure_llm(client)
            child = session.spawn(parent, purpose="research")
            child.user("Research question")
            child.assistant("Research answer")

            result = await session.acollapse(
                child, into=parent, auto_commit=True
            )
            assert result.summary_text is not None
            assert len(result.summary_text) > 0


# ---------------------------------------------------------------------------
# Loop async tests
# ---------------------------------------------------------------------------


class TestArunLoop:
    """Test arun_loop directly."""

    @pytest.mark.asyncio
    async def test_arun_loop_basic(self):
        """arun_loop should complete on no tool calls."""
        from tract.loop import LoopConfig, arun_loop

        client = MockLLMClient()
        t = Tract.open()
        t.system("You are helpful.")
        t.user("Hello")

        result = await arun_loop(t, llm_client=client, config=LoopConfig(max_steps=5))
        assert result.status == "completed"
        assert result.steps == 1

    @pytest.mark.asyncio
    async def test_arun_loop_max_steps(self):
        """arun_loop should stop at max_steps."""
        # Response with tool calls (will loop forever) -- use custom handler
        # to avoid SQLite cross-thread issues with the built-in executor.
        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        }])
        client = MockLLMClient([tool_response])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("You manage context.")

        from tract.loop import LoopConfig, arun_loop

        result = await arun_loop(
            t, llm_client=client,
            config=LoopConfig(max_steps=3),
            tools=[{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            tool_handlers={"ping": lambda: "pong"},
        )
        assert result.status == "max_steps"
        assert result.steps == 3


# ---------------------------------------------------------------------------
# OpenAI client async tests (with mocked httpx)
# ---------------------------------------------------------------------------


class TestOpenAIClientAsync:
    """Test OpenAIClient.achat() with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_achat_returns_response(self):
        """achat should return the same format as chat."""
        import httpx
        from tract.llm.client import OpenAIClient

        response_data = _make_response("Async hello!")
        # httpx.Response needs a request to allow raise_for_status()
        mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_response = httpx.Response(200, json=response_data, request=mock_request)

        client = OpenAIClient.__new__(OpenAIClient)
        client._api_key = "test-key"
        client._base_url = "https://api.openai.com/v1"
        client._default_model = "gpt-4o-mini"
        client._timeout = 120.0
        client._max_retries = 1

        mock_async = AsyncMock(return_value=mock_response)
        mock_http_client = MagicMock()
        mock_http_client.post = mock_async
        client._async_client = mock_http_client

        result = await client.achat([{"role": "user", "content": "hello"}])
        assert result["choices"][0]["message"]["content"] == "Async hello!"

    @pytest.mark.asyncio
    async def test_aclose(self):
        """aclose should clean up the async client."""
        from tract.llm.client import OpenAIClient

        client = OpenAIClient.__new__(OpenAIClient)
        mock_async_client = MagicMock()
        mock_async_client.aclose = AsyncMock()
        client._async_client = mock_async_client

        await client.aclose()
        # aclose() sets _async_client to None, so check the saved reference
        mock_async_client.aclose.assert_awaited_once()
        assert client._async_client is None
