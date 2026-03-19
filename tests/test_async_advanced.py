"""Advanced async test scenarios for tract.

Covers:
1. Concurrent async calls (asyncio.gather)
2. Cancellation handling mid-flight
3. Fallback from async to sync (acall_llm helper)
4. Error propagation in async methods
5. Async streaming with tool use
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
import pytest

from tract import Tract


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
        self.call_log: list[dict] = []

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.call_log.append({"method": "chat", "messages": messages, "kwargs": kwargs})
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    async def achat(self, messages, **kwargs):
        self.last_messages = messages
        self.call_log.append({"method": "achat", "messages": messages, "kwargs": kwargs})
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

    def __init__(self, response=None, delay: float = 0.0):
        self._response = response or _make_response()
        self._delay = delay
        self.call_count = 0

    def chat(self, messages, **kwargs):
        if self._delay > 0:
            time.sleep(self._delay)
        self.call_count += 1
        return self._response

    def close(self):
        pass

    @staticmethod
    def extract_content(response):
        return response["choices"][0]["message"].get("content") or ""

    @staticmethod
    def extract_usage(response):
        return response.get("usage")


class SlowAsyncClient:
    """Mock async client with configurable delay for cancellation tests."""

    def __init__(self, response=None, delay: float = 1.0):
        self._response = response or _make_response()
        self._delay = delay
        self.was_called = False
        self.call_count = 0

    async def achat(self, messages, **kwargs):
        self.was_called = True
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return self._response

    def chat(self, messages, **kwargs):
        self.was_called = True
        self.call_count += 1
        return self._response

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
        return [
            {
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")),
            }
            for tc in tcs
        ]


class ErrorClient:
    """Mock client that raises on specific call counts."""

    def __init__(self, error: Exception, fail_on: set[int] | None = None,
                 response=None):
        self._error = error
        self._fail_on = fail_on or {0}  # default: fail on first call
        self._response = response or _make_response()
        self._call_count = 0

    def chat(self, messages, **kwargs):
        current = self._call_count
        self._call_count += 1
        if current in self._fail_on:
            raise self._error
        return self._response

    async def achat(self, messages, **kwargs):
        current = self._call_count
        self._call_count += 1
        if current in self._fail_on:
            raise self._error
        return self._response

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
        return []


# ===========================================================================
# 1. Concurrent async calls
# ===========================================================================


class TestConcurrentAsyncCalls:
    """Test multiple async operations running concurrently via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_concurrent_agenerate_on_separate_tracts(self):
        """Multiple tracts calling agenerate concurrently should all succeed."""
        results = []
        for i in range(5):
            client = MockLLMClient([_make_response(f"Response {i}")])
            t = Tract.open()
            t.config.configure_llm(client)
            t.system("You are helpful.")
            t.user(f"Question {i}")
            results.append(t._llm_mgr.agenerate())

        responses = await asyncio.gather(*results)

        assert len(responses) == 5
        for i, resp in enumerate(responses):
            assert resp.text == f"Response {i}"
            assert resp.usage is not None
            assert resp.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_concurrent_achat_on_separate_tracts(self):
        """Multiple tracts calling achat concurrently should all succeed."""
        async def do_chat(idx: int):
            client = MockLLMClient([_make_response(f"Answer {idx}")])
            t = Tract.open()
            t.config.configure_llm(client)
            t.system("You are helpful.")
            return await t._llm_mgr.achat(f"Question {idx}")

        responses = await asyncio.gather(*[do_chat(i) for i in range(5)])

        assert len(responses) == 5
        for i, resp in enumerate(responses):
            assert resp.text == f"Answer {i}"
            assert resp.prompt == f"Question {i}"

    @pytest.mark.asyncio
    async def test_concurrent_acompress_on_separate_tracts(self):
        """Multiple tracts compressing concurrently should all succeed."""
        async def do_compress(idx: int):
            client = MockLLMClient([_make_response(f"Summary {idx}")])
            t = Tract.open()
            t.config.configure_llm(client)
            t.system("Helpful.")
            t.user(f"Msg {idx}")
            t.assistant(f"Resp {idx}")
            t.user(f"Msg {idx} part 2")
            t.assistant(f"Resp {idx} part 2")
            return await t.acompress()

        results = await asyncio.gather(*[do_compress(i) for i in range(3)])

        assert len(results) == 3
        for r in results:
            assert r.compressed_tokens > 0

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Different async operations (chat, generate, compress) concurrently."""
        # achat
        c1 = MockLLMClient([_make_response("Chat result")])
        t1 = Tract.open()
        t1.config.configure_llm(c1)
        t1.system("Helpful.")
        chat_coro = t1._llm_mgr.achat("Hello")

        # agenerate
        c2 = MockLLMClient([_make_response("Generate result")])
        t2 = Tract.open()
        t2.config.configure_llm(c2)
        t2.system("Helpful.")
        t2.user("Question")
        gen_coro = t2._llm_mgr.agenerate()

        # acompress
        c3 = MockLLMClient([_make_response("Compressed")])
        t3 = Tract.open()
        t3.config.configure_llm(c3)
        t3.system("Helpful.")
        t3.user("A")
        t3.assistant("B")
        t3.user("C")
        t3.assistant("D")
        compress_coro = t3.acompress()

        chat_r, gen_r, compress_r = await asyncio.gather(
            chat_coro, gen_coro, compress_coro
        )

        assert chat_r.text == "Chat result"
        assert gen_r.text == "Generate result"
        assert compress_r.compressed_tokens > 0

    @pytest.mark.asyncio
    async def test_concurrent_arun_on_separate_tracts(self):
        """Multiple tracts running arun concurrently should all complete."""
        async def do_run(idx: int):
            client = MockLLMClient([_make_response(f"Done {idx}")])
            t = Tract.open()
            t.config.configure_llm(client)
            t.system("Agent.")
            return await t._llm_mgr.arun(task=f"Task {idx}")

        results = await asyncio.gather(*[do_run(i) for i in range(3)])

        assert len(results) == 3
        for r in results:
            assert r.status == "completed"
            assert r.steps == 1

    @pytest.mark.asyncio
    async def test_gather_with_partial_failures(self):
        """asyncio.gather with return_exceptions should capture individual errors."""
        from tract.llm.errors import LLMConfigError

        # One good tract
        c1 = MockLLMClient([_make_response("OK")])
        t1 = Tract.open()
        t1.config.configure_llm(c1)
        t1.system("Good.")
        t1.user("Hello")

        # One tract with no client (will raise LLMConfigError)
        t2 = Tract.open()
        t2.system("Bad.")
        t2.user("Hello")

        results = await asyncio.gather(
            t1._llm_mgr.agenerate(),
            t2._llm_mgr.agenerate(),
            return_exceptions=True,
        )

        assert results[0].text == "OK"
        assert isinstance(results[1], LLMConfigError)


# ===========================================================================
# 2. Cancellation handling
# ===========================================================================


class TestCancellationHandling:
    """Test behavior when async operations are cancelled mid-flight."""

    @pytest.mark.asyncio
    async def test_cancelled_agenerate_raises_cancelled_error(self):
        """Cancelling agenerate mid-flight should raise CancelledError."""
        client = SlowAsyncClient(delay=5.0)
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Helpful.")
        t.user("Hello")

        task = asyncio.create_task(t._llm_mgr.agenerate())
        # Let the task start
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancelled_achat_raises_cancelled_error(self):
        """Cancelling achat mid-flight should raise CancelledError."""
        client = SlowAsyncClient(delay=5.0)
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Helpful.")

        task = asyncio.create_task(t._llm_mgr.achat("Hello"))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancelled_arun_raises_cancelled_error(self):
        """Cancelling arun mid-flight should raise CancelledError."""
        client = SlowAsyncClient(delay=5.0)
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Agent.")

        task = asyncio.create_task(t._llm_mgr.arun(task="Do something"))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancelled_acompress_raises_cancelled_error(self):
        """Cancelling acompress mid-flight should raise CancelledError."""
        client = SlowAsyncClient(delay=5.0)
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Helpful.")
        t.user("A")
        t.assistant("B")
        t.user("C")
        t.assistant("D")

        task = asyncio.create_task(t.acompress())
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_cancellation_in_gather_does_not_corrupt_other_tasks(self):
        """Cancelling one task in a gather should not affect independent tracts."""
        # Slow task (will be cancelled)
        slow_client = SlowAsyncClient(delay=5.0)
        t_slow = Tract.open()
        t_slow.config.configure_llm(slow_client)
        t_slow.system("Slow.")
        t_slow.user("Wait")

        # Fast task (should complete)
        fast_client = MockLLMClient([_make_response("Fast done")])
        t_fast = Tract.open()
        t_fast.config.configure_llm(fast_client)
        t_fast.system("Fast.")
        t_fast.user("Go")

        slow_task = asyncio.create_task(t_slow._llm_mgr.agenerate())
        fast_task = asyncio.create_task(t_fast._llm_mgr.agenerate())

        # Wait for the fast one
        fast_result = await fast_task
        assert fast_result.text == "Fast done"

        # Cancel the slow one
        slow_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow_task

    @pytest.mark.asyncio
    async def test_timeout_via_wait_for(self):
        """asyncio.wait_for should raise TimeoutError for slow operations."""
        client = SlowAsyncClient(delay=5.0)
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Helpful.")
        t.user("Hello")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(t._llm_mgr.agenerate(), timeout=0.1)


# ===========================================================================
# 3. Fallback from async to sync (acall_llm helper)
# ===========================================================================


class TestAcallLlmFallback:
    """Test the acall_llm helper that tries achat then falls back to asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_acall_llm_prefers_achat(self):
        """acall_llm should use achat when available."""
        from tract.llm.protocols import acall_llm

        client = MockLLMClient([_make_response("Async path")])
        messages = [{"role": "user", "content": "hi"}]

        result = await acall_llm(client, messages)

        assert result["choices"][0]["message"]["content"] == "Async path"
        # Verify achat was called (not chat)
        assert len(client.call_log) == 1
        assert client.call_log[0]["method"] == "achat"

    @pytest.mark.asyncio
    async def test_acall_llm_falls_back_to_thread_for_sync_only(self):
        """acall_llm should wrap sync chat in to_thread when no achat."""
        from tract.llm.protocols import acall_llm

        client = SyncOnlyClient(response=_make_response("Sync fallback"))
        messages = [{"role": "user", "content": "hi"}]

        result = await acall_llm(client, messages)

        assert result["choices"][0]["message"]["content"] == "Sync fallback"
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_acall_llm_passes_kwargs_to_achat(self):
        """acall_llm should forward kwargs to achat."""
        from tract.llm.protocols import acall_llm

        client = MockLLMClient([_make_response("With kwargs")])
        messages = [{"role": "user", "content": "hi"}]

        await acall_llm(client, messages, model="gpt-4", temperature=0.5)

        assert len(client.call_log) == 1
        assert client.call_log[0]["kwargs"]["model"] == "gpt-4"
        assert client.call_log[0]["kwargs"]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_acall_llm_passes_kwargs_to_sync_chat(self):
        """acall_llm should forward kwargs when falling back to sync chat."""
        from tract.llm.protocols import acall_llm

        class KwargsCapturingClient:
            """Captures kwargs passed to chat."""
            def __init__(self):
                self.captured_kwargs = {}

            def chat(self, messages, **kwargs):
                self.captured_kwargs = kwargs
                return _make_response("Captured")

            def close(self):
                pass

            @staticmethod
            def extract_content(response):
                return response["choices"][0]["message"].get("content") or ""

            @staticmethod
            def extract_usage(response):
                return response.get("usage")

        client = KwargsCapturingClient()
        messages = [{"role": "user", "content": "hi"}]

        result = await acall_llm(client, messages, model="gpt-3.5", temperature=0.9)

        assert result["choices"][0]["message"]["content"] == "Captured"
        assert client.captured_kwargs["model"] == "gpt-3.5"
        assert client.captured_kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_acall_llm_async_error_propagates(self):
        """Errors from achat should propagate through acall_llm."""
        from tract.llm.protocols import acall_llm
        from tract.llm.errors import LLMRateLimitError

        client = ErrorClient(LLMRateLimitError("Rate limited"))

        with pytest.raises(LLMRateLimitError, match="Rate limited"):
            await acall_llm(client, [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_acall_llm_sync_fallback_error_propagates(self):
        """Errors from sync chat via to_thread should propagate."""
        from tract.llm.protocols import acall_llm

        class SyncErrorClient:
            """Sync-only client that raises."""
            def chat(self, messages, **kwargs):
                raise ConnectionError("Network down")

            def close(self):
                pass

        client = SyncErrorClient()

        with pytest.raises(ConnectionError, match="Network down"):
            await acall_llm(client, [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_acall_llm_concurrent_sync_fallbacks(self):
        """Multiple sync-only clients should work concurrently via to_thread."""
        from tract.llm.protocols import acall_llm

        clients = [
            SyncOnlyClient(response=_make_response(f"Reply {i}"))
            for i in range(5)
        ]
        messages = [{"role": "user", "content": "hi"}]

        results = await asyncio.gather(
            *[acall_llm(c, messages) for c in clients]
        )

        for i, result in enumerate(results):
            assert result["choices"][0]["message"]["content"] == f"Reply {i}"


# ===========================================================================
# 4. Error propagation in async methods
# ===========================================================================


class TestAsyncErrorPropagation:
    """Async methods should propagate exceptions correctly, preserving type."""

    @pytest.mark.asyncio
    async def test_agenerate_no_client_raises_llm_config_error(self):
        """agenerate without LLM client should raise LLMConfigError."""
        from tract.llm.errors import LLMConfigError

        t = Tract.open()
        t.system("Test.")
        t.user("Hello")

        with pytest.raises(LLMConfigError):
            await t._llm_mgr.agenerate()

    @pytest.mark.asyncio
    async def test_achat_no_client_raises_llm_config_error(self):
        """achat without LLM client should raise LLMConfigError."""
        from tract.llm.errors import LLMConfigError

        t = Tract.open()
        t.system("Test.")

        with pytest.raises(LLMConfigError):
            await t._llm_mgr.achat("Hello")

    @pytest.mark.asyncio
    async def test_agenerate_llm_runtime_error_propagates(self):
        """Runtime errors from the LLM client should propagate through agenerate."""
        client = ErrorClient(RuntimeError("GPU on fire"))
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")
        t.user("Hello")

        with pytest.raises(RuntimeError, match="GPU on fire"):
            await t._llm_mgr.agenerate()

    @pytest.mark.asyncio
    async def test_achat_llm_connection_error_propagates(self):
        """Connection errors from LLM client should propagate through achat."""
        client = ErrorClient(ConnectionError("Timed out"))
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")

        with pytest.raises(ConnectionError, match="Timed out"):
            await t._llm_mgr.achat("Hello")

    @pytest.mark.asyncio
    async def test_agenerate_rate_limit_error_propagates(self):
        """LLMRateLimitError should propagate through agenerate."""
        from tract.llm.errors import LLMRateLimitError

        client = ErrorClient(LLMRateLimitError("429 rate limit"))
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")
        t.user("Hello")

        with pytest.raises(LLMRateLimitError, match="429 rate limit"):
            await t._llm_mgr.agenerate()

    @pytest.mark.asyncio
    async def test_agenerate_auth_error_propagates(self):
        """LLMAuthError should propagate through agenerate."""
        from tract.llm.errors import LLMAuthError

        client = ErrorClient(LLMAuthError("Bad API key"))
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")
        t.user("Hello")

        with pytest.raises(LLMAuthError, match="Bad API key"):
            await t._llm_mgr.agenerate()

    @pytest.mark.asyncio
    async def test_arun_llm_error_returns_error_status(self):
        """arun should return error status when LLM call fails."""
        client = ErrorClient(ConnectionError("Network fail"))
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Agent.")

        result = await t._llm_mgr.arun(task="Do work")

        assert result.status == "error"
        # Error detail goes into 'reason', not 'final_response'
        assert "Network fail" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_arevise_with_invalid_hash_raises(self):
        """arevise with nonexistent commit hash should raise."""
        from tract.exceptions import CommitNotFoundError

        client = MockLLMClient([_make_response("Revised.")])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")

        # resolve_commit raises CommitNotFoundError for unknown hashes
        with pytest.raises(CommitNotFoundError):
            await t._llm_mgr.arevise("deadbeef00000000", "Improve this")

    @pytest.mark.asyncio
    async def test_acompress_no_client_no_content_raises(self):
        """acompress without LLM or manual content should raise."""
        t = Tract.open()
        t.system("Test.")
        t.user("Msg")
        t.assistant("Resp")

        with pytest.raises(Exception):
            await t.acompress()

    @pytest.mark.asyncio
    async def test_acompress_on_detached_head_raises(self):
        """acompress on a detached HEAD should raise DetachedHeadError."""
        from tract.exceptions import DetachedHeadError

        client = MockLLMClient([_make_response("Summary.")])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")
        t.user("Msg")
        h = t.assistant("Resp")

        # Detach HEAD by checking out a specific commit
        t.checkout(h.commit_hash)

        with pytest.raises(DetachedHeadError):
            await t.acompress()

    @pytest.mark.asyncio
    async def test_retry_exhausted_error_in_agenerate(self):
        """agenerate with a validator that always rejects should raise RetryExhaustedError."""
        from tract.exceptions import RetryExhaustedError

        responses = [_make_response("always bad")] * 10
        client = MockLLMClient(responses)
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")
        t.user("Question")

        def always_reject(text):
            return (False, "not acceptable")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await t._llm_mgr.agenerate(validator=always_reject, max_retries=2)

        assert exc_info.value.attempts == 3
        assert "not acceptable" in exc_info.value.last_diagnosis

    @pytest.mark.asyncio
    async def test_error_type_preserved_through_acall_llm(self):
        """Custom exception types should be preserved through the async path."""
        from tract.llm.protocols import acall_llm

        class CustomLLMError(Exception):
            """Domain-specific error from a custom LLM client."""
            pass

        client = ErrorClient(CustomLLMError("Custom failure"))

        with pytest.raises(CustomLLMError, match="Custom failure"):
            await acall_llm(client, [{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_acompress_tool_calls_no_turns_raises(self):
        """acompress_tool_calls with no tool turns should raise CompressionError."""
        from tract.exceptions import CompressionError

        client = MockLLMClient()
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Agent.")
        t.user("Hello.")
        t.assistant("Hi.")

        with pytest.raises(CompressionError, match="No tool turns"):
            await t._compression_mgr.acompress_tool_calls()

    @pytest.mark.asyncio
    async def test_agenerate_blocked_error_propagates(self):
        """BlockedError from middleware should propagate through agenerate."""
        from tract.exceptions import BlockedError

        client = MockLLMClient([_make_response("OK")])
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Test.")
        t.user("Hello")

        # Register middleware that blocks pre_generate
        def blocker(event):
            from tract.exceptions import BlockedError
            raise BlockedError("pre_generate", "Blocked by policy")

        t.middleware.add("pre_generate", blocker)

        with pytest.raises(BlockedError, match="Blocked by policy"):
            await t._llm_mgr.agenerate()


# ===========================================================================
# 5. Async streaming with tools
# ===========================================================================


class TestAsyncStreamingWithTools:
    """Test async streaming paths and tool-use during streaming in arun_loop."""

    @pytest.mark.asyncio
    async def test_arun_loop_streaming_with_on_token(self):
        """arun_loop with stream=True and on_token should invoke streaming path."""
        from tract.llm.anthropic_client import MessageDone, TextDelta
        from tract.loop import LoopConfig, arun_loop

        tokens_received: list[str] = []
        final_response = _make_response("Streamed response.")

        class MockStreamingClient:
            """Client that supports astream and yields events."""

            def chat(self, messages, **kwargs):
                return final_response

            async def achat(self, messages, **kwargs):
                return final_response

            async def astream(self, messages, **kwargs):
                # Yield text deltas then the final message
                for word in ["Streamed ", "response", "."]:
                    yield TextDelta(word)
                yield MessageDone(final_response)

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
                return []

        client = MockStreamingClient()
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Helpful.")
        t.user("Hello")

        result = await arun_loop(
            t,
            llm_client=client,
            config=LoopConfig(max_steps=1, stream=True),
            on_token=lambda tok: tokens_received.append(tok),
        )

        assert result.status == "completed"
        assert len(tokens_received) == 3
        assert "".join(tokens_received) == "Streamed response."

    @pytest.mark.asyncio
    async def test_arun_loop_streaming_with_tool_calls(self):
        """arun_loop with stream should handle tool calls from streamed responses."""
        from tract.llm.anthropic_client import (
            MessageDone,
            TextDelta,
            ToolCallDelta,
            ToolCallStart,
        )
        from tract.loop import LoopConfig, arun_loop

        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        }])
        final_response = _make_response("Done after tool.")

        call_count = 0

        class MockStreamToolClient:
            """Streams tool call then final response."""

            def chat(self, messages, **kwargs):
                return final_response

            async def achat(self, messages, **kwargs):
                return final_response

            async def astream(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call: yield tool call
                    yield ToolCallStart(index=0, id="call_1", name="ping")
                    yield ToolCallDelta(index=0, partial_json="{}")
                    yield MessageDone(tool_response)
                else:
                    # Second call: yield final text
                    yield TextDelta("Done after tool.")
                    yield MessageDone(final_response)

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
                return [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": json.loads(
                            tc.get("function", {}).get("arguments", "{}")
                        ),
                    }
                    for tc in tcs
                ]

        client = MockStreamToolClient()
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Agent.")

        result = await arun_loop(
            t,
            llm_client=client,
            config=LoopConfig(max_steps=5, stream=True),
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

        assert result.status == "completed"
        assert result.steps >= 2
        assert result.tool_calls >= 1

    @pytest.mark.asyncio
    async def test_arun_with_async_tool_handler_during_stream(self):
        """Async tool handlers should be awaited properly during streamed loop."""
        from tract.llm.anthropic_client import MessageDone, TextDelta
        from tract.loop import LoopConfig, arun_loop

        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "async_op", "arguments": '{"val": 42}'},
        }])
        final_response = _make_response("Async tool done.")

        call_count = 0
        tool_called_with: dict = {}

        class StreamClient:
            """Streams tool call then final response."""

            async def astream(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    yield MessageDone(tool_response)
                else:
                    yield TextDelta("Async tool done.")
                    yield MessageDone(final_response)

            def chat(self, messages, **kwargs):
                return final_response

            async def achat(self, messages, **kwargs):
                return final_response

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
                return [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": json.loads(
                            tc.get("function", {}).get("arguments", "{}")
                        ),
                    }
                    for tc in tcs
                ]

        async def async_tool_handler(val=0):
            await asyncio.sleep(0)
            tool_called_with["val"] = val
            return f"processed {val}"

        client = StreamClient()
        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Agent.")

        result = await arun_loop(
            t,
            llm_client=client,
            config=LoopConfig(max_steps=5, stream=True),
            tools=[{
                "type": "function",
                "function": {
                    "name": "async_op",
                    "description": "Async operation",
                    "parameters": {
                        "type": "object",
                        "properties": {"val": {"type": "integer"}},
                    },
                },
            }],
            tool_handlers={"async_op": async_tool_handler},
        )

        assert result.status == "completed"
        assert tool_called_with.get("val") == 42

    @pytest.mark.asyncio
    async def test_arun_loop_non_streaming_with_tool_handlers(self):
        """Non-streaming arun_loop should still properly execute tool handlers."""
        from tract.loop import LoopConfig, arun_loop

        tool_response = _make_response("", tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "greet", "arguments": '{"name": "World"}'},
        }])
        final_response = _make_response("Greeted World.")
        client = MockLLMClient([tool_response, final_response])

        t = Tract.open()
        t.config.configure_llm(client)
        t.system("Agent.")

        handler_calls: list[dict] = []

        def greet_handler(name=""):
            handler_calls.append({"name": name})
            return f"Hello, {name}!"

        result = await arun_loop(
            t,
            llm_client=client,
            config=LoopConfig(max_steps=5),
            tools=[{
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            }],
            tool_handlers={"greet": greet_handler},
        )

        assert result.status == "completed"
        assert len(handler_calls) == 1
        assert handler_calls[0]["name"] == "World"


# ===========================================================================
# Additional edge cases
# ===========================================================================


class TestAsyncRetryWithBackoff:
    """Test the _aretry_with_backoff helper directly."""

    @pytest.mark.asyncio
    async def test_aretry_succeeds_on_first_try(self):
        """No retries needed when first call succeeds."""
        from tract.tract import _aretry_with_backoff
        from tract.models.config import RetryConfig

        call_count = 0

        async def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await _aretry_with_backoff(
            succeeds,
            RetryConfig(max_retries=3, initial_delay=0.01),
        )

        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_aretry_retries_on_transient_error(self):
        """Should retry on transient errors and eventually succeed."""
        from tract.tract import _aretry_with_backoff
        from tract.models.config import RetryConfig

        call_count = 0

        async def fails_twice_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Transient failure")
            return "recovered"

        result = await _aretry_with_backoff(
            fails_twice_then_succeeds,
            RetryConfig(max_retries=3, initial_delay=0.01, max_delay=0.02),
        )

        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_aretry_raises_after_exhaustion(self):
        """Should raise the last error after all retries exhausted."""
        from tract.tract import _aretry_with_backoff
        from tract.models.config import RetryConfig

        async def always_fails():
            raise ConnectionError("Always broken")

        with pytest.raises(ConnectionError, match="Always broken"):
            await _aretry_with_backoff(
                always_fails,
                RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.02),
            )

    @pytest.mark.asyncio
    async def test_aretry_does_not_retry_blocked_error(self):
        """BlockedError should not be retried (re-raised immediately)."""
        from tract.exceptions import BlockedError
        from tract.tract import _aretry_with_backoff
        from tract.models.config import RetryConfig

        call_count = 0

        async def raises_blocked():
            nonlocal call_count
            call_count += 1
            raise BlockedError("pre_generate", "Policy violation")

        with pytest.raises(BlockedError, match="Policy violation"):
            await _aretry_with_backoff(
                raises_blocked,
                RetryConfig(max_retries=5, initial_delay=0.01),
            )

        # Should have been called exactly once (no retries)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_aretry_does_not_retry_content_validation_error(self):
        """ContentValidationError should not be retried."""
        from tract.exceptions import ContentValidationError
        from tract.tract import _aretry_with_backoff
        from tract.models.config import RetryConfig

        call_count = 0

        async def raises_validation():
            nonlocal call_count
            call_count += 1
            raise ContentValidationError("Invalid content")

        with pytest.raises(ContentValidationError, match="Invalid content"):
            await _aretry_with_backoff(
                raises_validation,
                RetryConfig(max_retries=5, initial_delay=0.01),
            )

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_aretry_with_none_config_calls_once(self):
        """None retry config should call once without retries."""
        from tract.tract import _aretry_with_backoff

        call_count = 0

        async def succeeds():
            nonlocal call_count
            call_count += 1
            return "done"

        result = await _aretry_with_backoff(succeeds, None)

        assert result == "done"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_aretry_respects_retryable_errors_filter(self):
        """When retryable_errors is set, only those errors should be retried."""
        from tract.tract import _aretry_with_backoff
        from tract.models.config import RetryConfig

        call_count = 0

        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        # Only ConnectionError is retryable, so ValueError should not be retried
        with pytest.raises(ValueError, match="Not retryable"):
            await _aretry_with_backoff(
                raises_value_error,
                RetryConfig(
                    max_retries=3,
                    initial_delay=0.01,
                    retryable_errors=(ConnectionError,),
                ),
            )

        assert call_count == 1
