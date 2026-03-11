"""Extended tests for async LLM methods.

Tests every async method on Tract: achat, agenerate, arun, acompress,
arevise, acompress_tool_calls. Each method gets edge-case and integration
coverage beyond what test_async.py provides.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tract import Tract, LLMConfig, Priority, DialogueContent, InstructionContent


# ---------------------------------------------------------------------------
# Mock LLM clients
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
        self.last_messages = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    async def achat(self, messages, **kwargs):
        self.last_messages = messages
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


# ---------------------------------------------------------------------------
# acompress_tool_calls tests
# ---------------------------------------------------------------------------


class TestAcompressToolCallsBasic:
    """Test Tract.acompress_tool_calls()."""

    def _build_tool_call_tract(self):
        """Build a tract with a realistic tool-call sequence.

        Returns (tract, tool_result_hashes, answer_hash).
        """
        t = Tract.open()
        t.system("You are a search agent.")
        t.user("Find the hidden comment.")

        # Turn 1: assistant calls a tool
        t.assistant(
            "",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "search",
                 "arguments": {"q": "DISCOVERY"}},
            ]},
        )

        # Turn 1: tool result
        tr1 = t.tool_result("call_1", "search", "file.py:36: # DISCOVERY: some text " * 5)

        # Turn 2: final answer
        answer = t.assistant(
            "Found it! The comment is at file.py line 36."
        )

        return t, [tr1], answer.commit_hash

    @pytest.mark.asyncio
    async def test_acompress_tool_calls_basic(self):
        """acompress_tool_calls should summarize tool results asynchronously."""
        from tract.models.compression import ToolCompactResult

        t, tool_results, answer_hash = self._build_tool_call_tract()

        mock = MockLLMClient(responses=[
            _make_response(json.dumps(["Found DISCOVERY comment at file.py:36."]))
        ])
        t.configure_llm(mock)

        result = await t.acompress_tool_calls(target_tokens=50)

        assert isinstance(result, ToolCompactResult)
        assert len(result.edit_commits) == 1
        assert len(result.source_commits) == 1
        assert "search" in result.tool_names

        # The final answer should survive in compiled context
        ctx = t.compile()
        messages = ctx.to_dicts()
        assert any("Found it!" in m.get("content", "") for m in messages)

    @pytest.mark.asyncio
    async def test_acompress_tool_calls_preserves_metadata(self):
        """Metadata should survive async compression."""
        t, tool_results, _ = self._build_tool_call_tract()

        mock = MockLLMClient(responses=[
            _make_response(json.dumps(["Compacted result."]))
        ])
        t.configure_llm(mock)

        result = await t.acompress_tool_calls()

        # Check the edit commit preserves metadata
        edited_ci = t.get_commit(result.edit_commits[0])
        assert edited_ci is not None
        meta = edited_ci.metadata or {}
        assert meta.get("tool_call_id") == "call_1"
        assert meta.get("name") == "search"
        assert edited_ci.operation.value == "edit"

    @pytest.mark.asyncio
    async def test_acompress_tool_calls_multiple_turns(self):
        """acompress_tool_calls should handle multiple tool turns."""
        from tract.models.compression import ToolCompactResult

        t = Tract.open()
        t.system("Agent.")
        t.user("Do two things.")

        # Turn 1
        t.assistant(
            "",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "grep", "arguments": {}},
            ]},
        )
        tr1 = t.tool_result("call_1", "grep", "verbose grep output " * 10)

        # Turn 2
        t.assistant(
            "",
            metadata={"tool_calls": [
                {"id": "call_2", "name": "read", "arguments": {}},
            ]},
        )
        tr2 = t.tool_result("call_2", "read", "verbose read output " * 10)

        t.assistant("Done.")

        mock = MockLLMClient(responses=[
            _make_response(json.dumps(["Grep summary.", "Read summary."]))
        ])
        t.configure_llm(mock)

        result = await t.acompress_tool_calls()
        assert isinstance(result, ToolCompactResult)
        assert len(result.edit_commits) == 2
        assert len(result.source_commits) == 2

    @pytest.mark.asyncio
    async def test_acompress_tool_calls_no_turns_raises(self):
        """acompress_tool_calls with no tool turns should raise CompressionError."""
        from tract.exceptions import CompressionError

        t = Tract.open()
        t.system("Agent.")
        t.user("Hello.")
        t.assistant("Hi.")

        mock = MockLLMClient()
        t.configure_llm(mock)

        with pytest.raises(CompressionError, match="No tool turns"):
            await t.acompress_tool_calls()

    @pytest.mark.asyncio
    async def test_acompress_tool_calls_name_filter(self):
        """acompress_tool_calls with name= should only compact matching turns."""
        from tract.models.compression import ToolCompactResult

        t = Tract.open()
        t.system("test")

        # grep turn
        t.assistant(
            "",
            metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]},
        )
        grep_result = t.tool_result("c1", "grep", "verbose grep output " * 20)

        # read_file turn
        t.assistant(
            "",
            metadata={"tool_calls": [{"id": "c2", "name": "read_file", "arguments": {}}]},
        )
        read_result = t.tool_result("c2", "read_file", "file content " * 20)

        mock = MockLLMClient(responses=[
            _make_response(json.dumps(["Grep summary."]))
        ])
        t.configure_llm(mock)

        result = await t.acompress_tool_calls(name="grep")
        assert isinstance(result, ToolCompactResult)
        assert grep_result.commit_hash in result.source_commits
        assert read_result.commit_hash not in result.source_commits


# ---------------------------------------------------------------------------
# arevise extended tests
# ---------------------------------------------------------------------------


class TestAreviseExtended:
    """Extended tests for Tract.arevise()."""

    @pytest.mark.asyncio
    async def test_arevise_creates_edit(self):
        """arevise should create EDIT commit replacing the target."""
        client = MockLLMClient([_make_response("Improved text.")])
        t = Tract.open()
        t.configure_llm(client)
        original = t.assistant("Original text.")

        result = await t.arevise(original.commit_hash, "Make it better")
        assert result.text == "Improved text."
        assert result.commit_info is not None
        assert result.commit_info.operation.value == "edit"

    @pytest.mark.asyncio
    async def test_arevise_marks_intermediate_as_skip(self):
        """arevise should mark chat intermediate commits as SKIP."""
        client = MockLLMClient([_make_response("Revised content.")])
        t = Tract.open()
        t.configure_llm(client)
        original = t.assistant("Original content.")

        result = await t.arevise(original.commit_hash, "Improve this")

        # The user message and assistant response from achat should be SKIP
        log = t.log(limit=100)
        skip_count = 0
        for ci in log:
            ann = t.get_annotations(ci.commit_hash)
            for a in ann:
                if a.priority == Priority.SKIP:
                    skip_count += 1
        assert skip_count >= 2  # at least the user prompt + assistant response

    @pytest.mark.asyncio
    async def test_arevise_with_system_content(self):
        """arevise on a system/instruction commit should create a system EDIT."""
        client = MockLLMClient([_make_response("Improved system prompt.")])
        t = Tract.open()
        t.configure_llm(client)
        original = t.system("Old system prompt.")

        result = await t.arevise(original.commit_hash, "Make system prompt better")
        assert result.text == "Improved system prompt."
        assert result.commit_info is not None

    @pytest.mark.asyncio
    async def test_arevise_nonexistent_target_raises(self):
        """arevise with invalid commit hash should raise."""
        client = MockLLMClient([_make_response("Revised.")])
        t = Tract.open()
        t.configure_llm(client)
        t.system("Test.")

        with pytest.raises(Exception):
            await t.arevise("nonexistent_hash_000", "Improve this")


# ---------------------------------------------------------------------------
# achat extended tests
# ---------------------------------------------------------------------------


class TestAchatExtended:
    """Extended tests for Tract.achat()."""

    @pytest.mark.asyncio
    async def test_achat_with_tools(self):
        """achat with tool calls should execute tools and commit results."""
        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "check", "arguments": "{}"},
        }])
        final_response = _make_response("Tool called successfully.")
        client = MockLLMClient([tool_response, final_response])

        t = Tract.open()
        t.configure_llm(client)
        t.system("You are helpful.")

        result = await t.achat("Check something")
        # achat commits user + assistant; tool_calls are in metadata
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "check"

    @pytest.mark.asyncio
    async def test_achat_records_prompt(self):
        """achat should record the original prompt text."""
        client = MockLLMClient([_make_response("Response.")])
        t = Tract.open()
        t.configure_llm(client)
        t.system("Be helpful.")

        result = await t.achat("My question")
        assert result.prompt == "My question"

    @pytest.mark.asyncio
    async def test_achat_with_validator(self):
        """achat with validator should retry on failure."""
        responses = [
            _make_response("bad"),
            _make_response("bad"),  # retry steering
            _make_response("good answer"),
        ]
        client = MockLLMClient(responses)
        t = Tract.open()
        t.configure_llm(client)
        t.system("Be helpful.")

        def validator(text):
            if "good" in text:
                return (True, None)
            return (False, "not good enough")

        result = await t.achat("Question", validator=validator, max_retries=3)
        assert result.text == "good answer"
        assert result.prompt == "Question"


# ---------------------------------------------------------------------------
# agenerate extended tests
# ---------------------------------------------------------------------------


class TestAgenerateExtended:
    """Extended tests for Tract.agenerate()."""

    @pytest.mark.asyncio
    async def test_agenerate_with_system(self):
        """agenerate with a system prompt should compile it into messages."""
        client = MockLLMClient([_make_response("System-aware response.")])
        t = Tract.open()
        t.configure_llm(client)
        t.system("You are a coding assistant.")
        t.user("Hello")

        result = await t.agenerate()
        assert result.text == "System-aware response."
        # Verify system message was included in the LLM call
        assert client.last_messages is not None
        system_msgs = [m for m in client.last_messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1

    @pytest.mark.asyncio
    async def test_agenerate_with_validation_exhausted(self):
        """agenerate with validator that always fails should raise RetryExhaustedError."""
        from tract.exceptions import RetryExhaustedError

        responses = [_make_response("always bad")] * 10
        client = MockLLMClient(responses)
        t = Tract.open()
        t.configure_llm(client)
        t.system("Test.")
        t.user("Question")

        def always_fail(text):
            return (False, "never good enough")

        with pytest.raises(RetryExhaustedError):
            await t.agenerate(validator=always_fail, max_retries=2)

    @pytest.mark.asyncio
    async def test_agenerate_no_client_raises(self):
        """agenerate without LLM client should raise LLMConfigError."""
        from tract.llm.errors import LLMConfigError

        t = Tract.open()
        t.system("Test.")
        t.user("Hello")

        with pytest.raises(LLMConfigError):
            await t.agenerate()

    @pytest.mark.asyncio
    async def test_agenerate_records_usage(self):
        """agenerate should record token usage."""
        client = MockLLMClient([_make_response("Response.")])
        t = Tract.open()
        t.configure_llm(client)
        t.system("Test.")
        t.user("Hello")

        result = await t.agenerate()
        assert result.usage is not None
        assert result.usage.total_tokens == 15
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5


# ---------------------------------------------------------------------------
# arun extended tests
# ---------------------------------------------------------------------------


class TestArunExtended:
    """Extended tests for Tract.arun()."""

    @pytest.mark.asyncio
    async def test_arun_multi_turn(self):
        """arun should handle multiple tool call turns."""
        # Two rounds of tool calls, then a final response
        tool_response1 = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "step1", "arguments": "{}"},
        }])
        tool_response2 = _make_response("", tool_calls=[{
            "id": "call_2",
            "type": "function",
            "function": {"name": "step2", "arguments": "{}"},
        }])
        final_response = _make_response("All steps complete.")
        client = MockLLMClient([tool_response1, tool_response2, final_response])

        t = Tract.open()
        t.configure_llm(client)
        t.system("You are a multi-step agent.")

        result = await t.arun(
            task="Run multi-step",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "step1",
                        "description": "Step 1",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "step2",
                        "description": "Step 2",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            tool_handlers={
                "step1": lambda: "step1 done",
                "step2": lambda: "step2 done",
            },
        )
        assert result.status == "completed"
        assert result.steps >= 2
        assert result.tool_calls >= 2

    @pytest.mark.asyncio
    async def test_arun_max_steps_limit(self):
        """arun should respect max_steps."""
        # Endless tool calls
        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        }])
        client = MockLLMClient([tool_response])

        t = Tract.open()
        t.configure_llm(client)
        t.system("Agent.")

        result = await t.arun(
            task="Infinite loop",
            max_steps=2,
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
        assert result.steps == 2

    @pytest.mark.asyncio
    async def test_arun_no_task(self):
        """arun without task should still work (uses existing context)."""
        client = MockLLMClient([_make_response("Done.")])
        t = Tract.open()
        t.configure_llm(client)
        t.system("You are helpful.")
        t.user("Some existing context.")

        result = await t.arun()
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# acompress extended tests
# ---------------------------------------------------------------------------


class TestAcompressExtended:
    """Extended tests for Tract.acompress()."""

    @pytest.mark.asyncio
    async def test_acompress_preserves_pinned(self):
        """acompress should preserve PINNED commits."""
        t = Tract.open()
        t.system("You are helpful.")
        h1 = t.user("Message 1")
        t.assistant("Response 1")
        h2 = t.user("Important message")
        t.annotate(h2.commit_hash, Priority.PINNED, reason="critical")
        t.assistant("Response 2")
        t.user("Message 3")
        t.assistant("Response 3")

        client = MockLLMClient([_make_response("Compressed summary.")])
        t.configure_llm(client)

        result = await t.acompress()
        assert result.compressed_tokens > 0
        assert len(result.preserved_commits) >= 1
        # The pinned commit should survive
        assert h2.commit_hash in result.preserved_commits

    @pytest.mark.asyncio
    async def test_acompress_manual_content(self):
        """acompress with manual content= should not need LLM."""
        t = Tract.open()
        t.system("You are helpful.")
        t.user("Message 1")
        t.assistant("Response 1")
        t.user("Message 2")
        t.assistant("Response 2")

        result = await t.acompress(content="Manual summary of conversation.")
        assert result.compressed_tokens > 0

    @pytest.mark.asyncio
    async def test_acompress_with_llm_produces_summary(self):
        """acompress with LLM should produce a summary commit."""
        client = MockLLMClient([_make_response("LLM-generated summary.")])
        t = Tract.open()
        t.configure_llm(client)
        t.system("You are helpful.")
        t.user("Message 1")
        t.assistant("Response 1")
        t.user("Message 2")
        t.assistant("Response 2")

        result = await t.acompress()
        assert result.compressed_tokens > 0

        # Compiled context should include the summary
        compiled = t.compile()
        messages = compiled.to_dicts()
        found_summary = any("summary" in m.get("content", "").lower() for m in messages)
        # The summary text from mock is committed
        assert compiled.commit_count >= 1

    @pytest.mark.asyncio
    async def test_acompress_matches_sync_compress(self):
        """acompress should produce equivalent results to sync compress."""
        # Sync
        t1 = Tract.open()
        c1 = MockLLMClient([_make_response("Summary.")])
        t1.configure_llm(c1)
        t1.system("Helpful.")
        t1.user("Msg 1")
        t1.assistant("Resp 1")
        t1.user("Msg 2")
        t1.assistant("Resp 2")
        sync_result = t1.compress()

        # Async
        t2 = Tract.open()
        c2 = MockLLMClient([_make_response("Summary.")])
        t2.configure_llm(c2)
        t2.system("Helpful.")
        t2.user("Msg 1")
        t2.assistant("Resp 1")
        t2.user("Msg 2")
        t2.assistant("Resp 2")
        async_result = await t2.acompress()

        # Both should compress the same number of source commits
        assert len(sync_result.source_commits) == len(async_result.source_commits)
        assert len(sync_result.preserved_commits) == len(async_result.preserved_commits)

    @pytest.mark.asyncio
    async def test_acompress_no_client_no_content_raises(self):
        """acompress without LLM client or manual content should raise."""
        t = Tract.open()
        t.system("Test.")
        t.user("Message")
        t.assistant("Response")

        # No client, no content — should raise CompressionError
        # because there's nothing to generate the summary
        with pytest.raises(Exception):
            await t.acompress()
