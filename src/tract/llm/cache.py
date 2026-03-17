"""Caching wrapper for LLM clients.

Wraps any :class:`~tract.llm.protocols.LLMClient` and caches responses
by hashing the request parameters (messages, model, temperature,
max_tokens, and extra kwargs).  Cache hits bypass the network entirely.

Storage is SQLite-backed for durability across sessions.  An in-memory
mode (``":memory:"``) is available for testing.

Example::

    from tract.llm.cache import CachingLLMClient
    from tract.llm.anthropic_client import AnthropicClient

    inner = AnthropicClient(api_key="sk-...")
    client = CachingLLMClient(inner, cache_path="~/.tract/llm_cache.db")

    # First call hits the API
    response = client.chat([{"role": "user", "content": "Hello"}])

    # Second identical call returns from cache (zero tokens, instant)
    response = client.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CachingLLMClient"]

_DEFAULT_CACHE_DIR = Path.home() / ".tract"
_DEFAULT_CACHE_PATH = _DEFAULT_CACHE_DIR / "llm_cache.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key   TEXT PRIMARY KEY,
    response    TEXT NOT NULL,
    model       TEXT,
    created_at  REAL NOT NULL,
    hit_count   INTEGER DEFAULT 0
)
"""


def _make_cache_key(
    messages: list[dict[str, str]],
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    kwargs: dict[str, Any],
) -> str:
    """Produce a deterministic SHA-256 hash of the request parameters."""
    # Build a canonical representation.  Sort dict keys for determinism.
    canonical = json.dumps(
        {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": {
                k: v for k, v in sorted(kwargs.items())
                if k not in ("stream",)  # exclude non-deterministic params
            },
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class CachingLLMClient:
    """LLM client wrapper that caches responses in SQLite.

    Satisfies the :class:`~tract.llm.protocols.LLMClient` protocol and
    delegates extraction methods to the wrapped client.

    Cache behaviour:

    * **Deterministic requests only** — temperature=0 requests are cached
      by default.  Set ``cache_all=True`` to cache non-zero temperature
      requests as well (useful for development/testing).
    * **TTL** — cached entries expire after ``ttl_seconds`` (default: 7 days).
      Set to ``0`` to disable expiry.
    * **Invalidation** — call :meth:`clear` to wipe the cache, or
      :meth:`evict` to remove a specific entry.
    """

    def __init__(
        self,
        client: Any,
        *,
        cache_path: str | Path | None = None,
        cache_all: bool = True,
        ttl_seconds: float = 7 * 24 * 3600,  # 7 days
    ) -> None:
        self._client = client
        self._cache_all = cache_all
        self._ttl = ttl_seconds

        # Resolve cache path
        if cache_path is None:
            db_path = _DEFAULT_CACHE_PATH
        elif str(cache_path) == ":memory:":
            db_path = Path(":memory:")
        else:
            db_path = Path(cache_path).expanduser()

        # Ensure parent directory exists (unless in-memory)
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

        # Stats
        self.hits: int = 0
        self.misses: int = 0

    # -- LLMClient protocol ---------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send messages, returning a cached response if available."""
        # Decide whether this request is cacheable
        cacheable = self._cache_all or temperature in (None, 0, 0.0)

        if cacheable:
            key = _make_cache_key(messages, model, temperature, max_tokens, kwargs)
            cached = self._get(key)
            if cached is not None:
                self.hits += 1
                logger.debug("Cache HIT [%s…]", key[:12])
                return cached

        # Cache miss — forward to the real client
        response = self._client.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if cacheable:
            self.misses += 1
            logger.debug("Cache MISS [%s…], storing", key[:12])
            self._put(key, response, model)

        return response

    def close(self) -> None:
        """Close the cache database and the wrapped client."""
        self._conn.close()
        self._client.close()

    def extract_content(self, response: dict) -> str:
        """Delegate to the wrapped client."""
        return self._client.extract_content(response)

    def extract_usage(self, response: dict) -> dict | None:
        """Delegate to the wrapped client."""
        return self._client.extract_usage(response)

    # -- Async support (delegate to inner client) ------------------------------

    async def achat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Async chat with caching."""
        cacheable = self._cache_all or temperature in (None, 0, 0.0)

        if cacheable:
            key = _make_cache_key(messages, model, temperature, max_tokens, kwargs)
            cached = self._get(key)
            if cached is not None:
                self.hits += 1
                return cached

        # Forward to async client
        if hasattr(self._client, "achat"):
            response = await self._client.achat(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            import asyncio
            response = await asyncio.to_thread(
                self._client.chat,
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        if cacheable:
            self.misses += 1
            self._put(key, response, model)

        return response

    async def aclose(self) -> None:
        """Async close."""
        self._conn.close()
        if hasattr(self._client, "aclose"):
            await self._client.aclose()
        else:
            self._client.close()

    # -- Cache management ------------------------------------------------------

    def clear(self) -> int:
        """Delete all cached entries.  Returns the number of entries removed."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM llm_cache")
        count = cursor.fetchone()[0]
        self._conn.execute("DELETE FROM llm_cache")
        self._conn.commit()
        logger.info("Cache cleared: %d entries removed", count)
        return count

    def evict_expired(self) -> int:
        """Remove entries older than the TTL.  Returns the number removed."""
        if self._ttl <= 0:
            return 0
        cutoff = time.time() - self._ttl
        cursor = self._conn.execute(
            "DELETE FROM llm_cache WHERE created_at < ?", (cutoff,)
        )
        self._conn.commit()
        removed = cursor.rowcount
        if removed:
            logger.info("Evicted %d expired cache entries", removed)
        return removed

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM llm_cache")
        return cursor.fetchone()[0]

    @property
    def stats(self) -> dict[str, int]:
        """Cache hit/miss statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "hit_rate": round(
                self.hits / max(self.hits + self.misses, 1) * 100, 1
            ),
        }

    # -- Internal --------------------------------------------------------------

    def _get(self, key: str) -> dict | None:
        """Fetch a cached response, respecting TTL."""
        cursor = self._conn.execute(
            "SELECT response, created_at FROM llm_cache WHERE cache_key = ?",
            (key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        response_json, created_at = row

        # Check TTL
        if self._ttl > 0 and (time.time() - created_at) > self._ttl:
            self._conn.execute(
                "DELETE FROM llm_cache WHERE cache_key = ?", (key,)
            )
            self._conn.commit()
            return None

        # Bump hit count
        self._conn.execute(
            "UPDATE llm_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (key,),
        )
        self._conn.commit()

        return json.loads(response_json)

    def _put(self, key: str, response: dict, model: str | None) -> None:
        """Store a response in the cache."""
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, response, model, created_at, hit_count) "
            "VALUES (?, ?, ?, ?, 0)",
            (key, json.dumps(response, ensure_ascii=False), model, time.time()),
        )
        self._conn.commit()
