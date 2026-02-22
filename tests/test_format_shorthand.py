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


# -----------------------------------------------------------------------
# Shorthand commit method tests
# -----------------------------------------------------------------------


class TestShorthandMethods:
    """Tests for Tract.system(), user(), assistant() shorthand methods."""

    def test_system_creates_instruction(self):
        """t.system() creates commit with InstructionContent, role=system."""
        from tract import Tract

        t = Tract.open()
        try:
            info = t.system("Be helpful.")
            compiled = t.compile()
            dicts = compiled.to_dicts()
            assert len(dicts) == 1
            assert dicts[0]["role"] == "system"
            assert dicts[0]["content"] == "Be helpful."
        finally:
            t.close()

    def test_user_creates_dialogue(self):
        """t.user() creates commit with DialogueContent(role=user)."""
        from tract import Tract

        t = Tract.open()
        try:
            info = t.user("Hello")
            compiled = t.compile()
            dicts = compiled.to_dicts()
            assert len(dicts) == 1
            assert dicts[0]["role"] == "user"
            assert dicts[0]["content"] == "Hello"
        finally:
            t.close()

    def test_assistant_creates_dialogue(self):
        """t.assistant() creates commit with DialogueContent(role=assistant)."""
        from tract import Tract

        t = Tract.open()
        try:
            info = t.assistant("Hi there!")
            compiled = t.compile()
            dicts = compiled.to_dicts()
            assert len(dicts) == 1
            assert dicts[0]["role"] == "assistant"
            assert dicts[0]["content"] == "Hi there!"
        finally:
            t.close()

    def test_user_with_name(self):
        """t.user() preserves name field through compile."""
        from tract import Tract

        t = Tract.open()
        try:
            t.user("Hello", name="Alice")
            compiled = t.compile()
            dicts = compiled.to_dicts()
            assert dicts[0]["name"] == "Alice"
        finally:
            t.close()

    def test_assistant_with_generation_config(self):
        """t.assistant() stores generation_config."""
        from tract import Tract

        t = Tract.open()
        try:
            info = t.assistant("resp", generation_config={"model": "gpt-4"})
            # Verify via get_commit
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            from tract import LLMConfig
            assert commit.generation_config == LLMConfig(model="gpt-4")
        finally:
            t.close()

    def test_shorthand_returns_commit_info(self):
        """Shorthand methods return CommitInfo."""
        from tract import Tract
        from tract.models.commit import CommitInfo

        t = Tract.open()
        try:
            info_s = t.system("prompt")
            info_u = t.user("hello")
            info_a = t.assistant("hi")
            assert isinstance(info_s, CommitInfo)
            assert isinstance(info_u, CommitInfo)
            assert isinstance(info_a, CommitInfo)
        finally:
            t.close()

    def test_full_shorthand_integration(self):
        """system -> user -> assistant -> compile -> to_dicts without imports."""
        from tract import Tract

        # No content model imports needed!
        t = Tract.open()
        try:
            t.system("You are a helpful assistant.")
            t.user("Hi!")
            t.assistant("Hello! How can I help?")

            dicts = t.compile().to_dicts()
            assert len(dicts) == 3
            assert dicts[0] == {"role": "system", "content": "You are a helpful assistant."}
            assert dicts[1] == {"role": "user", "content": "Hi!"}
            assert dicts[2] == {"role": "assistant", "content": "Hello! How can I help?"}
        finally:
            t.close()


# -----------------------------------------------------------------------
# Auto-generated commit message tests
# -----------------------------------------------------------------------


class TestAutoMessage:
    """Tests for auto-generated commit messages."""

    def test_instruction_auto_message(self):
        """commit(InstructionContent) without message= generates auto-message."""
        from tract import Tract
        from tract.models.content import InstructionContent

        t = Tract.open()
        try:
            info = t.commit(InstructionContent(text="Be helpful"))
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert commit.message == "Be helpful"
        finally:
            t.close()

    def test_dialogue_auto_message(self):
        """commit(DialogueContent) without message= generates auto-message."""
        from tract import Tract
        from tract.models.content import DialogueContent

        t = Tract.open()
        try:
            info = t.commit(DialogueContent(role="user", text="Hello world"))
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert commit.message == "Hello world"
        finally:
            t.close()

    def test_shorthand_inherits_auto_message(self):
        """Shorthand methods also generate auto-messages."""
        from tract import Tract

        t = Tract.open()
        try:
            info = t.system("Be helpful")
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert commit.message == "Be helpful"
        finally:
            t.close()

    def test_long_text_truncated_at_500(self):
        """Long text is truncated with '...' to stay within 500 chars."""
        from tract import Tract

        t = Tract.open()
        try:
            long_text = "A" * 600
            info = t.system(long_text)
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert len(commit.message) <= 500
            assert commit.message.endswith("...")
        finally:
            t.close()

    def test_text_under_500_stored_in_full(self):
        """Text under 500 chars is stored without truncation."""
        from tract import Tract

        t = Tract.open()
        try:
            text = "A" * 200
            info = t.system(text)
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert commit.message == text
        finally:
            t.close()

    def test_empty_string_message_not_auto_generated(self):
        """message='' stores empty string, does NOT trigger auto-generation."""
        from tract import Tract
        from tract.models.content import InstructionContent

        t = Tract.open()
        try:
            info = t.commit(InstructionContent(text="Be helpful"), message="")
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert commit.message == ""
        finally:
            t.close()

    def test_custom_message_preserved(self):
        """message='custom' stores 'custom', not auto-generated."""
        from tract import Tract
        from tract.models.content import InstructionContent

        t = Tract.open()
        try:
            info = t.commit(InstructionContent(text="Be helpful"), message="custom")
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert commit.message == "custom"
        finally:
            t.close()

    def test_dict_content_auto_message(self):
        """Dict content also gets auto-message after validation."""
        from tract import Tract

        t = Tract.open()
        try:
            info = t.commit({"content_type": "instruction", "text": "Be helpful"})
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert commit.message == "Be helpful"
        finally:
            t.close()

    def test_multiline_text_flattened(self):
        """Multi-line text is flattened to a single line."""
        from tract import Tract

        t = Tract.open()
        try:
            info = t.user("Hello\nworld\nfoo")
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            assert "\n" not in commit.message
            assert "Hello world foo" in commit.message
        finally:
            t.close()

    def test_freeform_dict_payload_auto_message(self):
        """FreeformContent with dict payload generates reasonable message."""
        from tract import Tract
        from tract.models.content import FreeformContent

        t = Tract.open()
        try:
            info = t.commit(FreeformContent(payload={"key": "value"}))
            commit = t.get_commit(info.commit_hash)
            assert commit is not None
            # FreeformContent extract_text returns JSON string of payload
            assert commit.message.startswith("{")  # JSON preview, no type prefix
        finally:
            t.close()
