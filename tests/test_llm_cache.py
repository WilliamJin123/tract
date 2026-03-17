"""Tests for tract.llm.cache.CachingLLMClient."""

from __future__ import annotations

import asyncio
import time

import pytest

from tract.llm.cache import CachingLLMClient
from tract.llm.testing import MockLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(responses: list[str] | None = None, **kwargs) -> tuple[CachingLLMClient, MockLLMClient]:
    """Create a CachingLLMClient wrapping a MockLLMClient."""
    inner = MockLLMClient(responses or ["Hello!"])
    cached = CachingLLMClient(inner, cache_path=":memory:", **kwargs)
    return cached, inner


# ---------------------------------------------------------------------------
# Basic cache behaviour
# ---------------------------------------------------------------------------

class TestCachingBasic:
    def test_first_call_is_miss(self) -> None:
        client, inner = _make_client()
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert client.extract_content(response) == "Hello!"
        assert inner.call_count == 1
        assert client.misses == 1
        assert client.hits == 0

    def test_second_identical_call_is_hit(self) -> None:
        client, inner = _make_client()
        msg = [{"role": "user", "content": "Hi"}]
        r1 = client.chat(msg)
        r2 = client.chat(msg)
        assert r1 == r2
        assert inner.call_count == 1  # only one real call
        assert client.hits == 1
        assert client.misses == 1

    def test_different_messages_are_different_keys(self) -> None:
        client, inner = _make_client(["A", "B"])
        client.chat([{"role": "user", "content": "first"}])
        client.chat([{"role": "user", "content": "second"}])
        assert inner.call_count == 2
        assert client.misses == 2

    def test_different_model_is_different_key(self) -> None:
        client, inner = _make_client(["A", "B"])
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg, model="gpt-4o")
        client.chat(msg, model="gpt-4o-mini")
        assert inner.call_count == 2

    def test_same_model_is_cache_hit(self) -> None:
        client, inner = _make_client()
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg, model="gpt-4o")
        client.chat(msg, model="gpt-4o")
        assert inner.call_count == 1


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

class TestCachingTTL:
    def test_expired_entry_is_evicted(self) -> None:
        client, inner = _make_client(["A", "B"], ttl_seconds=0.01)
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg)
        assert inner.call_count == 1
        time.sleep(0.05)  # let TTL expire
        client.chat(msg)
        assert inner.call_count == 2  # cache miss after expiry

    def test_zero_ttl_means_no_expiry(self) -> None:
        client, inner = _make_client(ttl_seconds=0)
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg)
        client.chat(msg)
        assert inner.call_count == 1  # still cached


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

class TestCacheManagement:
    def test_size(self) -> None:
        client, _ = _make_client(["A", "B", "C"])
        assert client.size == 0
        client.chat([{"role": "user", "content": "1"}])
        assert client.size == 1
        client.chat([{"role": "user", "content": "2"}])
        assert client.size == 2

    def test_clear(self) -> None:
        client, _ = _make_client(["A", "B"])
        client.chat([{"role": "user", "content": "1"}])
        client.chat([{"role": "user", "content": "2"}])
        assert client.size == 2
        removed = client.clear()
        assert removed == 2
        assert client.size == 0

    def test_evict_expired(self) -> None:
        client, _ = _make_client(["A", "B"], ttl_seconds=0.01)
        client.chat([{"role": "user", "content": "1"}])
        client.chat([{"role": "user", "content": "2"}])
        time.sleep(0.05)
        evicted = client.evict_expired()
        assert evicted == 2
        assert client.size == 0

    def test_stats(self) -> None:
        client, _ = _make_client()
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg)
        client.chat(msg)
        client.chat(msg)
        stats = client.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 66.7


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------

class TestDelegation:
    def test_extract_content_delegates(self) -> None:
        client, _ = _make_client()
        response = client.chat([{"role": "user", "content": "Hi"}])
        assert client.extract_content(response) == "Hello!"

    def test_extract_usage_delegates(self) -> None:
        client, _ = _make_client()
        response = client.chat([{"role": "user", "content": "Hi"}])
        usage = client.extract_usage(response)
        assert usage is not None
        assert "total_tokens" in usage

    def test_close_closes_both(self) -> None:
        client, inner = _make_client()
        client.close()
        assert inner.closed


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

class TestCachingAsync:
    def test_achat_miss_then_hit(self) -> None:
        client, inner = _make_client()
        msg = [{"role": "user", "content": "Hi"}]

        # MockLLMClient doesn't have achat, so CachingLLMClient
        # falls back to asyncio.to_thread(chat)
        r1 = asyncio.run(client.achat(msg))
        r2 = asyncio.run(client.achat(msg))
        assert r1 == r2
        assert inner.call_count == 1
        assert client.hits == 1

    def test_aclose(self) -> None:
        client, inner = _make_client()
        asyncio.run(client.aclose())
        assert inner.closed


# ---------------------------------------------------------------------------
# cache_all flag
# ---------------------------------------------------------------------------

class TestCacheAllFlag:
    def test_cache_all_false_skips_nonzero_temperature(self) -> None:
        client, inner = _make_client(cache_all=False)
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg, temperature=0.7)
        client.chat(msg, temperature=0.7)
        assert inner.call_count == 2  # no caching

    def test_cache_all_false_still_caches_zero_temperature(self) -> None:
        client, inner = _make_client(cache_all=False)
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg, temperature=0.0)
        client.chat(msg, temperature=0.0)
        assert inner.call_count == 1

    def test_cache_all_true_caches_all_temperatures(self) -> None:
        client, inner = _make_client(cache_all=True)
        msg = [{"role": "user", "content": "Hi"}]
        client.chat(msg, temperature=0.9)
        client.chat(msg, temperature=0.9)
        assert inner.call_count == 1
