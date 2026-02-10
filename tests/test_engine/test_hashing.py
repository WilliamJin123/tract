"""Tests for deterministic hashing utilities.

Tests canonical JSON serialization, content hashing, and commit hashing.
Includes property-based tests via Hypothesis.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trace_context.engine.hashing import canonical_json, commit_hash, content_hash


class TestCanonicalJson:
    """Tests for canonical_json serialization."""

    def test_sorted_keys(self) -> None:
        """Keys are sorted alphabetically in output."""
        data = {"z": 1, "a": 2, "m": 3}
        result = json.loads(canonical_json(data))
        assert list(result.keys()) == ["a", "m", "z"]

    def test_compact_separators(self) -> None:
        """No spaces after colons or commas."""
        data = {"a": 1, "b": 2}
        result = canonical_json(data).decode("utf-8")
        assert result == '{"a":1,"b":2}'

    def test_nested_dicts_sorted(self) -> None:
        """Nested dicts also have sorted keys."""
        data = {"outer": {"z": 1, "a": 2}}
        result = canonical_json(data).decode("utf-8")
        assert '"a":2' in result
        assert result.index('"a"') < result.index('"z"')

    def test_unicode_preserved(self) -> None:
        """Non-ASCII characters are preserved (ensure_ascii=False)."""
        data = {"text": "hello"}
        result = canonical_json(data).decode("utf-8")
        assert "hello" in result

    def test_returns_bytes(self) -> None:
        """Output is bytes, not str."""
        result = canonical_json({"a": 1})
        assert isinstance(result, bytes)

    def test_empty_dict(self) -> None:
        """Empty dict produces '{}'."""
        assert canonical_json({}) == b"{}"

    def test_list_preserved(self) -> None:
        """Lists maintain order."""
        data = [3, 1, 2]
        result = json.loads(canonical_json(data))
        assert result == [3, 1, 2]


class TestContentHash:
    """Tests for content_hash function."""

    def test_deterministic(self) -> None:
        """Same input produces same hash, always."""
        payload = {"content_type": "instruction", "text": "hello world"}
        h1 = content_hash(payload)
        h2 = content_hash(payload)
        assert h1 == h2

    def test_hex_digest_format(self) -> None:
        """Hash is a 64-character hex string (SHA-256)."""
        h = content_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hashes."""
        h1 = content_hash({"text": "hello"})
        h2 = content_hash({"text": "world"})
        assert h1 != h2

    def test_key_order_irrelevant(self) -> None:
        """Dict key order doesn't affect hash (canonical JSON sorts keys)."""
        h1 = content_hash({"b": 2, "a": 1})
        h2 = content_hash({"a": 1, "b": 2})
        assert h1 == h2

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_property_deterministic(self, data: dict) -> None:
        """Property test: content_hash is always deterministic."""
        assert content_hash(data) == content_hash(data)


class TestCommitHash:
    """Tests for commit_hash function."""

    def test_deterministic(self) -> None:
        """Same inputs produce same commit hash."""
        h1 = commit_hash("abc123", None, "instruction", "append", "2024-01-01T00:00:00Z")
        h2 = commit_hash("abc123", None, "instruction", "append", "2024-01-01T00:00:00Z")
        assert h1 == h2

    def test_parent_hash_affects_result(self) -> None:
        """Different parent_hash produces different commit hash."""
        h1 = commit_hash("abc123", None, "instruction", "append", "2024-01-01T00:00:00Z")
        h2 = commit_hash("abc123", "parent1", "instruction", "append", "2024-01-01T00:00:00Z")
        assert h1 != h2

    def test_same_content_different_position(self) -> None:
        """Same content at different positions (different parents) = different hash."""
        h1 = commit_hash("abc123", "parent_a", "dialogue", "append", "2024-01-01T00:00:00Z")
        h2 = commit_hash("abc123", "parent_b", "dialogue", "append", "2024-01-01T00:00:00Z")
        assert h1 != h2

    def test_reply_to_included_when_set(self) -> None:
        """reply_to changes the hash when provided."""
        h1 = commit_hash("abc123", None, "dialogue", "edit", "2024-01-01T00:00:00Z")
        h2 = commit_hash("abc123", None, "dialogue", "edit", "2024-01-01T00:00:00Z", reply_to="target1")
        assert h1 != h2

    def test_reply_to_none_excluded_from_data(self) -> None:
        """When reply_to is None, it is excluded from the hash data."""
        # This ensures that None reply_to doesn't add a "reply_to": null entry
        h1 = commit_hash("abc", None, "instruction", "append", "2024-01-01T00:00:00Z")
        h2 = commit_hash("abc", None, "instruction", "append", "2024-01-01T00:00:00Z", reply_to=None)
        assert h1 == h2

    def test_hex_digest_format(self) -> None:
        """Commit hash is a 64-character hex string."""
        h = commit_hash("abc", None, "instruction", "append", "2024-01-01T00:00:00Z")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
