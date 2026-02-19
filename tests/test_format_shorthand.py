"""Tests for CompiledContext format methods and Tract commit shorthand."""

from __future__ import annotations

import pytest

from tract.protocols import CompiledContext, Message


# -----------------------------------------------------------------------
# to_dicts() tests
# -----------------------------------------------------------------------


class TestToDicts:
    """Tests for CompiledContext.to_dicts()."""

    def test_basic_messages(self):
        """to_dicts() returns list of dicts with role/content."""
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
            token_count=10,
            commit_count=3,
        )
        result = ctx.to_dicts()
        assert result == [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_with_name_field(self):
        """to_dicts() includes name when present."""
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="Hello", name="Alice"),
                Message(role="assistant", content="Hi"),
            ],
        )
        result = ctx.to_dicts()
        assert result[0] == {"role": "user", "content": "Hello", "name": "Alice"}
        assert result[1] == {"role": "assistant", "content": "Hi"}
        # name=None should NOT produce a "name" key
        assert "name" not in result[1]

    def test_empty_messages(self):
        """to_dicts() with empty messages returns empty list."""
        ctx = CompiledContext(messages=[])
        assert ctx.to_dicts() == []


# -----------------------------------------------------------------------
# to_openai() tests
# -----------------------------------------------------------------------


class TestToOpenAI:
    """Tests for CompiledContext.to_openai()."""

    def test_returns_same_as_to_dicts(self):
        """to_openai() returns same result as to_dicts()."""
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
        )
        assert ctx.to_openai() == ctx.to_dicts()


# -----------------------------------------------------------------------
# to_anthropic() tests
# -----------------------------------------------------------------------


class TestToAnthropic:
    """Tests for CompiledContext.to_anthropic()."""

    def test_extracts_system_messages(self):
        """to_anthropic() extracts system messages to separate key."""
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
        )
        result = ctx.to_anthropic()
        assert result["system"] == "Be helpful."
        assert result["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_multiple_system_messages(self):
        """to_anthropic() concatenates multiple system messages with newlines."""
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="system", content="Be concise."),
                Message(role="user", content="Hello"),
            ],
        )
        result = ctx.to_anthropic()
        assert result["system"] == "Be helpful.\n\nBe concise."
        assert len(result["messages"]) == 1

    def test_no_system_messages(self):
        """to_anthropic() returns None for system key when no system messages."""
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
        )
        result = ctx.to_anthropic()
        assert result["system"] is None
        assert len(result["messages"]) == 2

    def test_preserves_name_on_non_system(self):
        """to_anthropic() preserves name field on non-system messages."""
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="user", content="Hello", name="Alice"),
            ],
        )
        result = ctx.to_anthropic()
        assert result["messages"][0] == {
            "role": "user",
            "content": "Hello",
            "name": "Alice",
        }

    def test_tool_role_passes_through(self):
        """to_anthropic() passes non-standard roles like 'tool' through."""
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="user", content="Call tool"),
                Message(role="tool", content='{"result": 42}'),
            ],
        )
        result = ctx.to_anthropic()
        assert result["system"] == "Be helpful."
        assert result["messages"][1] == {
            "role": "tool",
            "content": '{"result": 42}',
        }


# -----------------------------------------------------------------------
# Integration: Tract -> commit -> compile -> to_dicts()
# -----------------------------------------------------------------------


class TestFormatIntegration:
    """End-to-end integration tests for format methods."""

    def test_commit_compile_to_dicts(self):
        """Full path: Tract.open -> commit -> compile -> to_dicts()."""
        from tract import Tract
        from tract.models.content import DialogueContent, InstructionContent

        t = Tract.open()
        try:
            t.commit(InstructionContent(text="Be helpful."))
            t.commit(DialogueContent(role="user", text="Hello"))
            t.commit(DialogueContent(role="assistant", text="Hi there!"))

            compiled = t.compile()
            dicts = compiled.to_dicts()

            assert len(dicts) == 3
            assert dicts[0]["role"] == "system"
            assert dicts[0]["content"] == "Be helpful."
            assert dicts[1]["role"] == "user"
            assert dicts[1]["content"] == "Hello"
            assert dicts[2]["role"] == "assistant"
            assert dicts[2]["content"] == "Hi there!"
        finally:
            t.close()
