"""Tests for LLM call retry with exponential backoff."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from tract import RetryConfig, Tract
from tract.exceptions import BlockedError, ContentValidationError
from tract.tract import _aretry_with_backoff, _retry_with_backoff


# ---------------------------------------------------------------------------
# RetryConfig dataclass tests
# ---------------------------------------------------------------------------


class TestRetryConfig:
    """Tests for the RetryConfig frozen dataclass."""

    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.initial_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.backoff_factor == 2.0
        assert cfg.jitter is True
        assert cfg.retryable_errors == ()

    def test_custom_values(self):
        cfg = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_factor=3.0,
            jitter=False,
            retryable_errors=(ConnectionError, TimeoutError),
        )
        assert cfg.max_retries == 5
        assert cfg.initial_delay == 0.5
        assert cfg.max_delay == 30.0
        assert cfg.backoff_factor == 3.0
        assert cfg.jitter is False
        assert cfg.retryable_errors == (ConnectionError, TimeoutError)

    def test_frozen(self):
        cfg = RetryConfig()
        with pytest.raises(AttributeError):
            cfg.max_retries = 10  # type: ignore[misc]

    def test_import_from_tract(self):
        """RetryConfig is importable from the top-level package."""
        from tract import RetryConfig as RC
        assert RC is RetryConfig

    def test_import_from_models(self):
        """RetryConfig is importable from tract.models."""
        from tract.models import RetryConfig as RC
        assert RC is RetryConfig


# ---------------------------------------------------------------------------
# _retry_with_backoff (sync) tests
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Tests for the sync retry helper."""

    def test_no_config_calls_once(self):
        """When retry_config is None, function is called exactly once."""
        calls = []

        def func():
            calls.append(1)
            return "ok"

        result = _retry_with_backoff(func, None)
        assert result == "ok"
        assert len(calls) == 1

    def test_zero_retries_calls_once(self):
        """max_retries=0 means no retries."""
        calls = []

        def func():
            calls.append(1)
            return "ok"

        cfg = RetryConfig(max_retries=0)
        result = _retry_with_backoff(func, cfg)
        assert result == "ok"
        assert len(calls) == 1

    def test_success_on_first_try(self):
        """No retries needed when function succeeds."""
        cfg = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False)
        result = _retry_with_backoff(lambda: "success", cfg)
        assert result == "success"

    def test_retries_on_transient_error(self):
        """Retries transient errors and succeeds eventually."""
        calls = []

        def func():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("network down")
            return "recovered"

        cfg = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False)
        result = _retry_with_backoff(func, cfg)
        assert result == "recovered"
        assert len(calls) == 3

    def test_raises_after_exhausting_retries(self):
        """Raises the last error after all retries fail."""
        calls = []

        def func():
            calls.append(1)
            raise ConnectionError(f"fail #{len(calls)}")

        cfg = RetryConfig(max_retries=2, initial_delay=0.001, jitter=False)
        with pytest.raises(ConnectionError, match="fail #3"):
            _retry_with_backoff(func, cfg)
        assert len(calls) == 3  # initial + 2 retries

    def test_never_retries_content_validation_error(self):
        """ContentValidationError propagates immediately."""
        calls = []

        def func():
            calls.append(1)
            raise ContentValidationError("bad content")

        cfg = RetryConfig(max_retries=3, initial_delay=0.001)
        with pytest.raises(ContentValidationError):
            _retry_with_backoff(func, cfg)
        assert len(calls) == 1

    def test_never_retries_blocked_error(self):
        """BlockedError propagates immediately."""
        calls = []

        def func():
            calls.append(1)
            raise BlockedError("pre_commit", "blocked")

        cfg = RetryConfig(max_retries=3, initial_delay=0.001)
        with pytest.raises(BlockedError):
            _retry_with_backoff(func, cfg)
        assert len(calls) == 1

    def test_retryable_errors_filter(self):
        """Only retries specified error types."""
        calls = []

        def func():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("not retryable")
            return "ok"

        cfg = RetryConfig(
            max_retries=3,
            initial_delay=0.001,
            retryable_errors=(ConnectionError,),
        )
        with pytest.raises(ValueError, match="not retryable"):
            _retry_with_backoff(func, cfg)
        assert len(calls) == 1  # not retried

    def test_retryable_errors_retries_matching(self):
        """Retries when error matches retryable_errors."""
        calls = []

        def func():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("retry me")
            return "ok"

        cfg = RetryConfig(
            max_retries=3,
            initial_delay=0.001,
            jitter=False,
            retryable_errors=(ConnectionError,),
        )
        result = _retry_with_backoff(func, cfg)
        assert result == "ok"
        assert len(calls) == 2

    @patch("time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Verify exponential delay calculation (no jitter)."""
        calls = []

        def func():
            calls.append(1)
            if len(calls) <= 3:
                raise ConnectionError("fail")
            return "ok"

        cfg = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            jitter=False,
        )
        result = _retry_with_backoff(func, cfg)
        assert result == "ok"
        assert mock_sleep.call_count == 3
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays[0] == pytest.approx(1.0)   # 1.0 * 2^0
        assert delays[1] == pytest.approx(2.0)   # 1.0 * 2^1
        assert delays[2] == pytest.approx(4.0)   # 1.0 * 2^2

    @patch("time.sleep")
    def test_max_delay_cap(self, mock_sleep):
        """Delay is capped at max_delay."""
        calls = []

        def func():
            calls.append(1)
            if len(calls) <= 2:
                raise ConnectionError("fail")
            return "ok"

        cfg = RetryConfig(
            max_retries=3,
            initial_delay=10.0,
            max_delay=15.0,
            backoff_factor=2.0,
            jitter=False,
        )
        result = _retry_with_backoff(func, cfg)
        assert result == "ok"
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays[0] == pytest.approx(10.0)  # 10 * 2^0 = 10
        assert delays[1] == pytest.approx(15.0)  # min(20, 15) = 15

    def test_jitter_varies_delay(self):
        """With jitter enabled, delays are randomized (50-150% of calculated)."""
        delays_observed = []
        original_sleep = time.sleep

        def capture_sleep(d):
            delays_observed.append(d)
            # Don't actually sleep

        calls = []

        def func():
            calls.append(1)
            if len(calls) <= 3:
                raise ConnectionError("fail")
            return "ok"

        cfg = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            jitter=True,
        )
        with patch("time.sleep", side_effect=capture_sleep):
            result = _retry_with_backoff(func, cfg)

        assert result == "ok"
        assert len(delays_observed) == 3
        # With jitter, delay = base * (0.5 + random()) where random() in [0, 1)
        # So delay range is [base * 0.5, base * 1.5)
        assert 0.5 <= delays_observed[0] < 1.5   # base=1.0
        assert 1.0 <= delays_observed[1] < 3.0   # base=2.0
        assert 2.0 <= delays_observed[2] < 6.0   # base=4.0

    def test_passes_args_and_kwargs(self):
        """Arguments are forwarded correctly to the wrapped function."""
        received = {}

        def func(a, b, *, key=None):
            received["a"] = a
            received["b"] = b
            received["key"] = key
            return "ok"

        cfg = RetryConfig(max_retries=1, initial_delay=0.001)
        result = _retry_with_backoff(func, cfg, "x", "y", key="z")
        assert result == "ok"
        assert received == {"a": "x", "b": "y", "key": "z"}


# ---------------------------------------------------------------------------
# _aretry_with_backoff (async) tests
# ---------------------------------------------------------------------------


class TestAsyncRetryWithBackoff:
    """Tests for the async retry helper."""

    @pytest.mark.asyncio
    async def test_no_config_calls_once(self):
        calls = []

        async def func():
            calls.append(1)
            return "ok"

        result = await _aretry_with_backoff(func, None)
        assert result == "ok"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        calls = []

        async def func():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("network")
            return "recovered"

        cfg = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False)
        result = await _aretry_with_backoff(func, cfg)
        assert result == "recovered"
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self):
        calls = []

        async def func():
            calls.append(1)
            raise ConnectionError(f"fail #{len(calls)}")

        cfg = RetryConfig(max_retries=2, initial_delay=0.001, jitter=False)
        with pytest.raises(ConnectionError, match="fail #3"):
            await _aretry_with_backoff(func, cfg)
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_never_retries_content_validation_error(self):
        calls = []

        async def func():
            calls.append(1)
            raise ContentValidationError("bad")

        cfg = RetryConfig(max_retries=3, initial_delay=0.001)
        with pytest.raises(ContentValidationError):
            await _aretry_with_backoff(func, cfg)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_never_retries_blocked_error(self):
        calls = []

        async def func():
            calls.append(1)
            raise BlockedError("pre_commit", "blocked")

        cfg = RetryConfig(max_retries=3, initial_delay=0.001)
        with pytest.raises(BlockedError):
            await _aretry_with_backoff(func, cfg)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retryable_errors_filter(self):
        calls = []

        async def func():
            calls.append(1)
            raise ValueError("nope")

        cfg = RetryConfig(
            max_retries=3,
            initial_delay=0.001,
            retryable_errors=(ConnectionError,),
        )
        with pytest.raises(ValueError):
            await _aretry_with_backoff(func, cfg)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_uses_asyncio_sleep(self):
        """Confirm async retry uses asyncio.sleep, not time.sleep."""
        calls = []
        sleep_delays = []

        async def func():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("fail")
            return "ok"

        async def mock_sleep(delay):
            sleep_delays.append(delay)

        cfg = RetryConfig(max_retries=2, initial_delay=0.001, jitter=False)
        with patch("asyncio.sleep", side_effect=mock_sleep):
            result = await _aretry_with_backoff(func, cfg)

        assert result == "ok"
        assert len(sleep_delays) == 1  # one retry = one sleep
        assert sleep_delays[0] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Integration: retry wired into Tract.generate() / Tract.chat()
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM client that can fail N times before succeeding."""

    def __init__(self, fail_count=0, fail_error=None, responses=None):
        self._fail_count = fail_count
        self._fail_error = fail_error or ConnectionError("transient failure")
        self._responses = responses or ["Mock response"]
        self._call_count = 0
        self.calls: list = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise self._fail_error
        idx = min(self._call_count - self._fail_count - 1, len(self._responses) - 1)
        text = self._responses[idx]
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def close(self):
        pass


class TestRetryIntegrationGenerate:
    """Integration tests: retry wired into generate() and chat()."""

    def test_generate_with_retry_config_on_open(self):
        """Retry config set on Tract.open() retries transient failures."""
        client = MockLLMClient(fail_count=2)
        t = Tract.open(
            llm_client=client,
            retry=RetryConfig(max_retries=3, initial_delay=0.001, jitter=False),
        )
        t.system("You are helpful.")
        t.user("Hello")
        response = t._llm_mgr.generate()
        assert response.text == "Mock response"
        assert len(client.calls) == 3  # 2 failures + 1 success

    def test_generate_with_per_call_retry(self):
        """Per-call retry overrides tract-level."""
        client = MockLLMClient(fail_count=1)
        t = Tract.open(llm_client=client)
        t.system("You are helpful.")
        t.user("Hello")
        response = t._llm_mgr.generate(
            retry=RetryConfig(max_retries=2, initial_delay=0.001, jitter=False),
        )
        assert response.text == "Mock response"
        assert len(client.calls) == 2

    def test_generate_no_retry_raises_immediately(self):
        """Without retry config, errors propagate immediately."""
        client = MockLLMClient(fail_count=1)
        t = Tract.open(llm_client=client)
        t.system("You are helpful.")
        t.user("Hello")
        with pytest.raises(ConnectionError, match="transient failure"):
            t._llm_mgr.generate()
        assert len(client.calls) == 1

    def test_generate_retries_exhausted(self):
        """Raises after all retries exhausted."""
        client = MockLLMClient(fail_count=10)
        t = Tract.open(
            llm_client=client,
            retry=RetryConfig(max_retries=2, initial_delay=0.001, jitter=False),
        )
        t.system("You are helpful.")
        t.user("Hello")
        with pytest.raises(ConnectionError, match="transient failure"):
            t._llm_mgr.generate()
        assert len(client.calls) == 3

    def test_chat_with_retry(self):
        """Retry works through chat() -> generate() path."""
        client = MockLLMClient(fail_count=1)
        t = Tract.open(
            llm_client=client,
            retry=RetryConfig(max_retries=2, initial_delay=0.001, jitter=False),
        )
        t.system("You are helpful.")
        response = t._llm_mgr.chat("Hello")
        assert response.text == "Mock response"
        assert len(client.calls) == 2

    def test_chat_per_call_retry(self):
        """Per-call retry in chat()."""
        client = MockLLMClient(fail_count=1)
        t = Tract.open(llm_client=client)
        t.system("You are helpful.")
        response = t._llm_mgr.chat(
            "Hello",
            retry=RetryConfig(max_retries=2, initial_delay=0.001, jitter=False),
        )
        assert response.text == "Mock response"

    def test_content_validation_error_not_retried(self):
        """ContentValidationError bypasses retry."""
        client = MockLLMClient(
            fail_count=1,
            fail_error=ContentValidationError("bad"),
        )
        t = Tract.open(
            llm_client=client,
            retry=RetryConfig(max_retries=3, initial_delay=0.001),
        )
        t.system("System")
        t.user("Hello")
        with pytest.raises(ContentValidationError):
            t._llm_mgr.generate()
        assert len(client.calls) == 1

    def test_blocked_error_not_retried(self):
        """BlockedError bypasses retry."""
        client = MockLLMClient(
            fail_count=1,
            fail_error=BlockedError("pre_commit", "blocked"),
        )
        t = Tract.open(
            llm_client=client,
            retry=RetryConfig(max_retries=3, initial_delay=0.001),
        )
        t.system("System")
        t.user("Hello")
        with pytest.raises(BlockedError):
            t._llm_mgr.generate()
        assert len(client.calls) == 1

    def test_per_call_retry_overrides_tract_level(self):
        """Per-call retry config takes precedence over tract-level."""
        client = MockLLMClient(fail_count=2)
        t = Tract.open(
            llm_client=client,
            retry=RetryConfig(max_retries=1, initial_delay=0.001, jitter=False),
        )
        t.system("System")
        t.user("Hello")
        # Tract-level allows only 1 retry, but per-call allows 3
        response = t._llm_mgr.generate(
            retry=RetryConfig(max_retries=3, initial_delay=0.001, jitter=False),
        )
        assert response.text == "Mock response"
        assert len(client.calls) == 3


# ---------------------------------------------------------------------------
# Integration: retry wired into loop
# ---------------------------------------------------------------------------


class TestRetryIntegrationLoop:
    """Integration tests: retry wired into run_loop."""

    def test_loop_uses_tract_retry_config(self):
        """Loop reads tract._retry_config for LLM call retries."""
        from tract.loop import LoopConfig, run_loop

        client = MockLLMClient(fail_count=1)
        t = Tract.open(
            llm_client=client,
            retry=RetryConfig(max_retries=2, initial_delay=0.001, jitter=False),
        )
        t.system("You are helpful.")
        result = run_loop(
            t,
            task="Hello",
            config=LoopConfig(max_steps=1, stop_on_no_tool_call=True),
            llm_client=client,
            tools=[],
        )
        assert result.status == "completed"
        assert len(client.calls) == 2  # 1 failure + 1 success

    def test_loop_no_retry_returns_error(self):
        """Without retry config, loop catches the error gracefully."""
        from tract.loop import LoopConfig, run_loop

        client = MockLLMClient(fail_count=1)
        t = Tract.open(llm_client=client)
        t.system("You are helpful.")
        result = run_loop(
            t,
            task="Hello",
            config=LoopConfig(max_steps=1),
            llm_client=client,
            tools=[],
        )
        assert result.status == "error"
        assert "transient failure" in result.reason
