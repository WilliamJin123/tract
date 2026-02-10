"""Tests for token counting implementations.

Tests TiktokenCounter (production) and NullTokenCounter (testing stub).
"""

from __future__ import annotations

import pytest

from trace_context.engine.tokens import NullTokenCounter, TiktokenCounter
from trace_context.protocols import TokenCounter


class TestTiktokenCounter:
    """Tests for the TiktokenCounter implementation."""

    def test_implements_protocol(self) -> None:
        """TiktokenCounter satisfies the TokenCounter protocol."""
        counter = TiktokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_count_text_positive_for_nonempty(self) -> None:
        """Non-empty text returns a positive token count."""
        counter = TiktokenCounter()
        count = counter.count_text("Hello, world!")
        assert count > 0

    def test_count_text_zero_for_empty(self) -> None:
        """Empty string returns 0 tokens."""
        counter = TiktokenCounter()
        assert counter.count_text("") == 0

    def test_count_text_deterministic(self) -> None:
        """Same text produces same count."""
        counter = TiktokenCounter()
        text = "The quick brown fox jumps over the lazy dog."
        assert counter.count_text(text) == counter.count_text(text)

    def test_longer_text_more_tokens(self) -> None:
        """Longer text generally produces more tokens."""
        counter = TiktokenCounter()
        short = counter.count_text("Hi")
        long = counter.count_text("Hello, this is a much longer piece of text that should have more tokens.")
        assert long > short

    def test_count_messages_includes_overhead(self) -> None:
        """Message counting includes per-message overhead and response primer."""
        counter = TiktokenCounter()
        messages = [{"role": "user", "content": "Hello"}]
        text_tokens = counter.count_text("Hello") + counter.count_text("user")
        message_tokens = counter.count_messages(messages)
        # message_tokens should be > text_tokens due to per-message overhead (3) + primer (3)
        assert message_tokens > text_tokens

    def test_count_messages_empty_list(self) -> None:
        """Empty message list returns 0."""
        counter = TiktokenCounter()
        assert counter.count_messages([]) == 0

    def test_count_messages_with_name(self) -> None:
        """Messages with name field include extra name overhead."""
        counter = TiktokenCounter()
        without_name = [{"role": "user", "content": "Hello"}]
        with_name = [{"role": "user", "content": "Hello", "name": "Bob"}]
        count_without = counter.count_messages(without_name)
        count_with = counter.count_messages(with_name)
        # Name adds tokens for the name text itself plus 1 extra token
        assert count_with > count_without

    def test_count_messages_multiple(self) -> None:
        """Multiple messages each get per-message overhead."""
        counter = TiktokenCounter()
        single = counter.count_messages([{"role": "user", "content": "Hi"}])
        double = counter.count_messages([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi"},
        ])
        # Double should be more than single (extra message + overhead)
        assert double > single

    def test_explicit_encoding_name(self) -> None:
        """Can specify encoding by name instead of model."""
        counter = TiktokenCounter(encoding_name="cl100k_base")
        assert counter.encoding_name == "cl100k_base"
        assert counter.count_text("hello") > 0

    def test_unknown_model_falls_back(self) -> None:
        """Unknown model falls back to o200k_base."""
        counter = TiktokenCounter(model="totally-unknown-model-xyz")
        assert counter.encoding_name == "o200k_base"
        assert counter.count_text("hello") > 0


class TestNullTokenCounter:
    """Tests for the NullTokenCounter stub."""

    def test_implements_protocol(self) -> None:
        """NullTokenCounter satisfies the TokenCounter protocol."""
        counter = NullTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_count_text_always_zero(self) -> None:
        """Always returns 0 for text counting."""
        counter = NullTokenCounter()
        assert counter.count_text("Hello world, this is a test.") == 0
        assert counter.count_text("") == 0

    def test_count_messages_always_zero(self) -> None:
        """Always returns 0 for message counting."""
        counter = NullTokenCounter()
        assert counter.count_messages([{"role": "user", "content": "Hi"}]) == 0
        assert counter.count_messages([]) == 0
