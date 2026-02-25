"""Tests for reasoning trace handling.

Validates:
- ReasoningContent model with format field
- BUILTIN_TYPE_HINTS / DEFAULT_TYPE_PRIORITIES skip for reasoning
- ChatResponse reasoning fields
- extract_reasoning() on OpenAIClient (4 provider formats)
- t.reasoning() shorthand
- _generate_once() reasoning extraction + commit flow
- compile(include_reasoning=True) override
- commit_reasoning=False on Tract.open()
- reasoning=False per-call opt-out
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from tract import (
    ChatResponse,
    CommitInfo,
    CompiledContext,
    LLMConfig,
    Priority,
    ReasoningContent,
    Tract,
)
from tract.models.annotations import DEFAULT_TYPE_PRIORITIES
from tract.models.commit import CommitOperation
from tract.models.content import BUILTIN_TYPE_HINTS
from tract.protocols import TokenUsage


# ---------------------------------------------------------------------------
# MockLLMClient with reasoning support
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM client that can return reasoning traces."""

    def __init__(self, responses=None, model="mock-model"):
        self.responses = responses or [{"content": "Mock response"}]
        self._call_count = 0
        self.last_messages = None
        self.last_kwargs: dict = {}
        self._model = model

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        resp_data = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1

        # Support both simple string and dict responses
        if isinstance(resp_data, str):
            resp_data = {"content": resp_data}

        message: dict = {"content": resp_data.get("content", "")}
        if "reasoning" in resp_data:
            message["reasoning"] = resp_data["reasoning"]
        if "reasoning_content" in resp_data:
            message["reasoning_content"] = resp_data["reasoning_content"]

        return {
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model": kwargs.get("model", self._model),
        }

    @staticmethod
    def extract_content(response: dict) -> str:
        return response["choices"][0]["message"].get("content") or ""

    @staticmethod
    def extract_usage(response: dict) -> dict | None:
        return response.get("usage")

    @staticmethod
    def extract_reasoning(response: dict) -> tuple[str, str] | None:
        """Delegate to OpenAIClient's extract_reasoning for testing."""
        from tract.llm.client import OpenAIClient
        return OpenAIClient.extract_reasoning(response)

    def close(self):
        pass


class MockLLMClientNoReasoning:
    """Mock LLM client WITHOUT extract_reasoning (duck-typed optional)."""

    def __init__(self, responses=None, model="mock-model"):
        self.responses = responses or ["Mock response"]
        self._call_count = 0

    def chat(self, messages, **kwargs):
        text = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": kwargs.get("model", self._model if hasattr(self, '_model') else "mock"),
        }


# ---------------------------------------------------------------------------
# ReasoningContent model tests
# ---------------------------------------------------------------------------


class TestReasoningContentModel:
    """Tests for the ReasoningContent Pydantic model."""

    def test_default_format(self):
        rc = ReasoningContent(text="thinking...")
        assert rc.content_type == "reasoning"
        assert rc.text == "thinking..."
        assert rc.format == "parsed"

    def test_explicit_format(self):
        for fmt in ("parsed", "raw", "think_tags", "anthropic"):
            rc = ReasoningContent(text="thinking", format=fmt)
            assert rc.format == fmt

    def test_invalid_format_rejected(self):
        with pytest.raises(Exception):
            ReasoningContent(text="x", format="invalid")

    def test_serialization_roundtrip(self):
        rc = ReasoningContent(text="step by step", format="think_tags")
        data = rc.model_dump()
        assert data["format"] == "think_tags"
        assert data["content_type"] == "reasoning"
        rc2 = ReasoningContent.model_validate(data)
        assert rc2 == rc


# ---------------------------------------------------------------------------
# Type hints / priority defaults
# ---------------------------------------------------------------------------


class TestReasoningDefaults:
    """Tests for reasoning default priority being SKIP."""

    def test_builtin_type_hints_reasoning_skip(self):
        hints = BUILTIN_TYPE_HINTS["reasoning"]
        assert hints.default_priority == "skip"
        assert hints.default_role == "assistant"
        assert hints.compression_priority == 40

    def test_default_type_priorities_reasoning_skip(self):
        assert DEFAULT_TYPE_PRIORITIES["reasoning"] == Priority.SKIP


# ---------------------------------------------------------------------------
# ChatResponse reasoning fields
# ---------------------------------------------------------------------------


class TestChatResponseReasoning:
    """Tests for reasoning fields on ChatResponse."""

    def test_reasoning_defaults_none(self):
        info = CommitInfo(
            commit_hash="abc", tract_id="t1", parent_hash=None,
            content_hash="b1", content_type="dialogue",
            operation=CommitOperation.APPEND, message="test",
            token_count=10, created_at=datetime.now(timezone.utc),
        )
        resp = ChatResponse(
            text="Hello",
            usage=None,
            commit_info=info,
            generation_config=LLMConfig(),
        )
        assert resp.reasoning is None
        assert resp.reasoning_commit is None

    def test_reasoning_fields_set(self):
        info = CommitInfo(
            commit_hash="abc", tract_id="t1", parent_hash=None,
            content_hash="b1", content_type="dialogue",
            operation=CommitOperation.APPEND, message="test",
            token_count=10, created_at=datetime.now(timezone.utc),
        )
        reasoning_info = CommitInfo(
            commit_hash="def", tract_id="t1", parent_hash=None,
            content_hash="b2", content_type="reasoning",
            operation=CommitOperation.APPEND, message="reasoning",
            token_count=5, created_at=datetime.now(timezone.utc),
        )
        resp = ChatResponse(
            text="Hello",
            usage=None,
            commit_info=info,
            generation_config=LLMConfig(),
            reasoning="Let me think...",
            reasoning_commit=reasoning_info,
        )
        assert resp.reasoning == "Let me think..."
        assert resp.reasoning_commit.commit_hash == "def"


# ---------------------------------------------------------------------------
# extract_reasoning() tests
# ---------------------------------------------------------------------------


class TestExtractReasoning:
    """Tests for OpenAIClient.extract_reasoning() auto-detect."""

    def test_cerebras_parsed_field(self):
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {
                "content": "The answer is 42.",
                "reasoning": "I need to think about this...",
            }}],
        }
        result = OpenAIClient.extract_reasoning(response)
        assert result is not None
        text, fmt = result
        assert text == "I need to think about this..."
        assert fmt == "parsed"

    def test_openai_reasoning_content(self):
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {
                "content": "The answer is 42.",
                "reasoning_content": "Step 1: Consider...",
            }}],
        }
        result = OpenAIClient.extract_reasoning(response)
        assert result is not None
        text, fmt = result
        assert text == "Step 1: Consider..."
        assert fmt == "reasoning_content"

    def test_anthropic_thinking_blocks(self):
        from tract.llm.client import OpenAIClient
        response = {
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "The answer is 42."},
            ],
        }
        result = OpenAIClient.extract_reasoning(response)
        assert result is not None
        text, fmt = result
        assert text == "Let me reason about this..."
        assert fmt == "anthropic"

    def test_anthropic_multiple_thinking_blocks(self):
        from tract.llm.client import OpenAIClient
        response = {
            "content": [
                {"type": "thinking", "thinking": "Part 1..."},
                {"type": "text", "text": "middle"},
                {"type": "thinking", "thinking": "Part 2..."},
            ],
        }
        result = OpenAIClient.extract_reasoning(response)
        assert result is not None
        text, fmt = result
        assert "Part 1..." in text
        assert "Part 2..." in text
        assert fmt == "anthropic"

    def test_think_tags_in_content(self):
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {
                "content": "<think>Let me think step by step...</think>The answer is 42.",
            }}],
        }
        result = OpenAIClient.extract_reasoning(response)
        assert result is not None
        text, fmt = result
        assert text == "Let me think step by step..."
        assert fmt == "think_tags"

    def test_no_reasoning_returns_none(self):
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {
                "content": "Just a normal response.",
            }}],
        }
        result = OpenAIClient.extract_reasoning(response)
        assert result is None

    def test_priority_order_parsed_over_reasoning_content(self):
        """Parsed field takes priority over reasoning_content."""
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {
                "content": "answer",
                "reasoning": "parsed reasoning",
                "reasoning_content": "should not be used",
            }}],
        }
        result = OpenAIClient.extract_reasoning(response)
        assert result is not None
        assert result[0] == "parsed reasoning"
        assert result[1] == "parsed"

    def test_empty_response_returns_none(self):
        from tract.llm.client import OpenAIClient
        result = OpenAIClient.extract_reasoning({})
        assert result is None

    def test_malformed_response_returns_none(self):
        from tract.llm.client import OpenAIClient
        result = OpenAIClient.extract_reasoning({"choices": []})
        assert result is None


# ---------------------------------------------------------------------------
# t.reasoning() shorthand
# ---------------------------------------------------------------------------


class TestReasoningShorthand:
    """Tests for Tract.reasoning() shorthand method."""

    def test_reasoning_basic(self):
        t = Tract.open()
        info = t.reasoning("Let me think step by step...")

        assert isinstance(info, CommitInfo)
        assert info.content_type == "reasoning"
        assert info.operation == CommitOperation.APPEND
        t.close()

    def test_reasoning_with_format(self):
        t = Tract.open()
        info = t.reasoning("thinking...", format="think_tags")

        # Verify content was stored correctly
        content = t.get_content(info.commit_hash)
        assert content["format"] == "think_tags"
        t.close()

    def test_reasoning_auto_message(self):
        t = Tract.open()
        info = t.reasoning("Let me think about this problem carefully")

        # Auto-message should be a preview of the text
        assert "Let me think" in info.message
        t.close()

    def test_reasoning_explicit_message(self):
        t = Tract.open()
        info = t.reasoning("thinking...", message="custom message")
        assert info.message == "custom message"
        t.close()

    def test_reasoning_with_metadata(self):
        t = Tract.open()
        info = t.reasoning("thinking...", metadata={"source": "o1"})
        meta = t.get_metadata(info.commit_hash)
        assert meta["source"] == "o1"
        t.close()

    def test_reasoning_excluded_from_compile_by_default(self):
        """Reasoning commits have SKIP priority and should not appear in compile."""
        t = Tract.open()
        t.system("Be helpful")
        t.user("Hello")
        t.reasoning("Let me think...")
        t.assistant("Hi there!")

        compiled = t.compile()
        # Should have 3 messages: system, user, assistant (reasoning excluded)
        assert compiled.commit_count == 3
        texts = [m.content for m in compiled.messages]
        assert "Let me think..." not in texts
        t.close()


# ---------------------------------------------------------------------------
# compile(include_reasoning=True)
# ---------------------------------------------------------------------------


class TestCompileIncludeReasoning:
    """Tests for compile(include_reasoning=True) override."""

    def test_include_reasoning_shows_reasoning_commits(self):
        t = Tract.open()
        t.system("Be helpful")
        t.user("Hello")
        t.reasoning("Let me think about this...")
        t.assistant("Hi there!")

        compiled = t.compile(include_reasoning=True)
        # Should have 4 messages: system, user, reasoning, assistant
        assert compiled.commit_count == 4
        texts = [m.content for m in compiled.messages]
        assert "Let me think about this..." in texts
        t.close()

    def test_include_reasoning_preserves_order(self):
        t = Tract.open()
        t.user("Hello")
        t.reasoning("Thinking...")
        t.assistant("Response")

        compiled = t.compile(include_reasoning=True)
        roles = [m.role for m in compiled.messages]
        # user -> assistant(reasoning) -> assistant
        assert roles[0] == "user"
        assert compiled.messages[1].content == "Thinking..."
        assert compiled.messages[2].content == "Response"
        t.close()

    def test_explicit_annotation_overrides_include_reasoning(self):
        """If user manually annotates reasoning as PINNED, it always shows."""
        t = Tract.open()
        t.user("Hello")
        info = t.reasoning("Important thinking")
        t.annotate(info.commit_hash, Priority.PINNED, reason="keep this")
        t.assistant("Response")

        # Even without include_reasoning, PINNED reasoning should appear
        compiled = t.compile(include_reasoning=False)
        texts = [m.content for m in compiled.messages]
        assert "Important thinking" in texts
        t.close()

    def test_explicit_skip_annotation_overrides_include_reasoning(self):
        """If user manually annotates reasoning as SKIP, include_reasoning doesn't override."""
        t = Tract.open()
        t.user("Hello")
        info = t.reasoning("Thinking")
        t.annotate(info.commit_hash, Priority.SKIP, reason="exclude this")
        t.assistant("Response")

        # Even with include_reasoning, explicit SKIP should be respected
        compiled = t.compile(include_reasoning=True)
        texts = [m.content for m in compiled.messages]
        assert "Thinking" not in texts
        t.close()


# ---------------------------------------------------------------------------
# generate() reasoning extraction + commit
# ---------------------------------------------------------------------------


class TestGenerateReasoning:
    """Tests for reasoning trace handling in generate()."""

    def test_generate_extracts_reasoning(self):
        """generate() should extract reasoning and put it on ChatResponse."""
        t = Tract.open()
        mock = MockLLMClient(responses=[{
            "content": "The answer is 42.",
            "reasoning": "I need to calculate...",
        }])
        t.configure_llm(mock)
        t.system("System")
        t.user("Question")

        resp = t.generate()

        assert resp.text == "The answer is 42."
        assert resp.reasoning == "I need to calculate..."
        assert resp.reasoning_commit is not None
        assert resp.reasoning_commit.content_type == "reasoning"
        t.close()

    def test_generate_commits_reasoning_before_assistant(self):
        """Reasoning commit should be parent of assistant commit."""
        t = Tract.open()
        mock = MockLLMClient(responses=[{
            "content": "Answer",
            "reasoning": "Thinking...",
        }])
        t.configure_llm(mock)
        t.user("Question")

        resp = t.generate()

        # Chain should be: user -> reasoning -> assistant
        log = t.log(limit=10)
        # log is newest-first, so: [assistant, reasoning, user]
        assert log[0].content_type == "dialogue"  # assistant
        assert log[1].content_type == "reasoning"
        assert log[2].content_type == "dialogue"  # user

        # Assistant's parent should be the reasoning commit
        assert resp.commit_info.parent_hash == resp.reasoning_commit.commit_hash
        t.close()

    def test_generate_no_reasoning_in_response(self):
        """When LLM returns no reasoning, ChatResponse.reasoning is None."""
        t = Tract.open()
        mock = MockLLMClient(responses=[{
            "content": "Just a response",
        }])
        t.configure_llm(mock)
        t.user("Question")

        resp = t.generate()

        assert resp.text == "Just a response"
        assert resp.reasoning is None
        assert resp.reasoning_commit is None
        t.close()

    def test_generate_reasoning_false_skips_commit(self):
        """reasoning=False should skip the commit but still extract."""
        t = Tract.open()
        mock = MockLLMClient(responses=[{
            "content": "Answer",
            "reasoning": "Thinking...",
        }])
        t.configure_llm(mock)
        t.user("Question")

        resp = t.generate(reasoning=False)

        # Reasoning text should still be available
        assert resp.reasoning == "Thinking..."
        # But no commit was made
        assert resp.reasoning_commit is None

        # Log should only have user + assistant (no reasoning)
        log = t.log(limit=10)
        assert len(log) == 2
        content_types = [e.content_type for e in log]
        assert "reasoning" not in content_types
        t.close()

    def test_generate_client_without_extract_reasoning(self):
        """Client without extract_reasoning should skip reasoning silently."""
        t = Tract.open()
        mock = MockLLMClientNoReasoning(responses=["Response"])
        t.configure_llm(mock)
        t.user("Question")

        resp = t.generate()

        assert resp.text == "Response"
        assert resp.reasoning is None
        assert resp.reasoning_commit is None
        t.close()

    def test_generate_think_tags_strips_from_content(self):
        """When <think> tags detected, they should be stripped from text."""
        t = Tract.open()

        class ThinkTagClient:
            def chat(self, messages, **kwargs):
                return {
                    "choices": [{"message": {
                        "content": "<think>Deep thoughts...</think>The answer is 42.",
                    }}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "model": "test",
                }

            @staticmethod
            def extract_reasoning(response):
                from tract.llm.client import OpenAIClient
                return OpenAIClient.extract_reasoning(response)

            @staticmethod
            def extract_content(response):
                return response["choices"][0]["message"].get("content") or ""

        t.configure_llm(ThinkTagClient())
        t.user("Question")
        resp = t.generate()

        assert resp.reasoning == "Deep thoughts..."
        assert "<think>" not in resp.text
        assert resp.text == "The answer is 42."
        t.close()


# ---------------------------------------------------------------------------
# chat() reasoning
# ---------------------------------------------------------------------------


class TestChatReasoning:
    """Tests for reasoning trace handling in chat()."""

    def test_chat_extracts_reasoning(self):
        t = Tract.open()
        mock = MockLLMClient(responses=[{
            "content": "The answer is 42.",
            "reasoning": "Let me calculate...",
        }])
        t.configure_llm(mock)

        resp = t.chat("What is 6*7?")

        assert resp.reasoning == "Let me calculate..."
        assert resp.reasoning_commit is not None
        assert resp.prompt == "What is 6*7?"
        t.close()

    def test_chat_reasoning_false(self):
        t = Tract.open()
        mock = MockLLMClient(responses=[{
            "content": "Answer",
            "reasoning": "Thinking...",
        }])
        t.configure_llm(mock)

        resp = t.chat("Question", reasoning=False)

        assert resp.reasoning == "Thinking..."
        assert resp.reasoning_commit is None
        t.close()


# ---------------------------------------------------------------------------
# Tract.open(commit_reasoning=False)
# ---------------------------------------------------------------------------


class TestCommitReasoningConfig:
    """Tests for commit_reasoning config on Tract.open()."""

    def test_commit_reasoning_false_disables_globally(self):
        t = Tract.open(commit_reasoning=False)
        mock = MockLLMClient(responses=[{
            "content": "Answer",
            "reasoning": "Thinking...",
        }])
        t.configure_llm(mock)
        t.user("Question")

        resp = t.generate()

        # Reasoning extracted but not committed
        assert resp.reasoning == "Thinking..."
        assert resp.reasoning_commit is None

        # No reasoning commit in log
        log = t.log(limit=10)
        content_types = [e.content_type for e in log]
        assert "reasoning" not in content_types
        t.close()

    def test_commit_reasoning_true_default(self):
        t = Tract.open()
        assert t._commit_reasoning is True
        t.close()

    def test_commit_reasoning_false_still_allows_manual(self):
        """t.reasoning() shorthand should still work even when commit_reasoning=False."""
        t = Tract.open(commit_reasoning=False)
        info = t.reasoning("Manual reasoning")

        assert info.content_type == "reasoning"
        t.close()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestReasoningFormatting:
    """Tests for reasoning visual distinction in formatting."""

    def test_role_colors_has_reasoning(self):
        from tract.formatting import _ROLE_COLORS
        assert "reasoning" in _ROLE_COLORS
        assert "dim cyan" in _ROLE_COLORS["reasoning"]

    def test_role_styles_has_reasoning(self):
        from tract.formatting import _ROLE_STYLES
        assert "reasoning" in _ROLE_STYLES
        title, color = _ROLE_STYLES["reasoning"]
        assert title == "Reasoning"
        assert "dim cyan" in color
