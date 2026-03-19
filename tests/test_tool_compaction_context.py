"""Tests for context-aware tool compaction.

Verifies that compress_tool_calls(context=...) compiles the
current context and passes it to the LLM for relevance-aware compaction,
and that the default (context=None) path is unchanged.
"""

from __future__ import annotations

import json

import pytest

from tract import Tract
from tract.models.compression import ToolCompactResult
from tract.prompts.summarize import (
    TOOL_COMPACT_CONTEXT_SYSTEM,
    TOOL_COMPACT_SYSTEM,
    build_tool_compact_prompt,
)


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLM client that records calls and returns canned responses."""

    def __init__(self, responses=None):
        self.responses = responses or [json.dumps(["Summary."])]
        self._call_count = 0
        self.last_messages = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        text = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tool_call_tract() -> tuple[Tract, list, str]:
    """Build a tract with a realistic tool-call sequence.

    Returns (tract, tool_result_commit_infos, answer_hash).
    """
    t = Tract.open()
    t.system("You are a code search agent.")
    t.user("Find the hidden comment in helpers.py.")

    # Turn 1: assistant calls a tool
    t.assistant(
        "",
        metadata={
            "tool_calls": [
                {"id": "call_1", "name": "grep", "arguments": {"q": "TODO"}},
            ]
        },
    )

    # Turn 1: tool result
    tr1 = t.tool_result(
        "call_1",
        "grep",
        "helpers.py:42: # TODO: refactor this ugly loop\n"
        "helpers.py:99: # TODO: add error handling\n"
        "helpers.py:150: # TODO: optimise query\n"
        "Full file listing follows...\n" * 20,
    )

    # Turn 2: final answer
    answer = t.assistant(
        "Found 3 TODO comments in helpers.py at lines 42, 99, and 150."
    )

    return t, [tr1], answer.commit_hash


# ===========================================================================
# build_tool_compact_prompt tests
# ===========================================================================


class TestBuildToolCompactPrompt:
    """Tests for the context_text parameter on build_tool_compact_prompt."""

    def test_without_context_text(self):
        """Default call (no context_text) produces original prompt format."""
        prompt = build_tool_compact_prompt("tool sequence", result_count=2)
        assert prompt.startswith("Compact the tool results")
        assert "conversation so far" not in prompt

    def test_with_context_text(self):
        """When context_text is provided, it is prepended to the prompt."""
        ctx = "user: Find bugs\nassistant: Searching..."
        prompt = build_tool_compact_prompt(
            "tool sequence", result_count=1, context_text=ctx
        )
        assert "Here is the conversation so far:" in prompt
        assert ctx in prompt
        assert "relevant to the conversation above" in prompt
        assert "tool sequence" in prompt

    def test_context_text_none_is_no_op(self):
        """Explicitly passing context_text=None behaves like omitting it."""
        prompt = build_tool_compact_prompt(
            "seq", result_count=1, context_text=None
        )
        assert "conversation so far" not in prompt

    def test_context_text_with_target_tokens(self):
        """context_text and target_tokens coexist correctly."""
        prompt = build_tool_compact_prompt(
            "seq",
            result_count=1,
            context_text="some context",
            target_tokens=80,
        )
        assert "conversation so far" in prompt
        assert "Target length: 80 tokens" in prompt

    def test_context_text_with_instructions(self):
        """context_text and instructions coexist correctly."""
        prompt = build_tool_compact_prompt(
            "seq",
            result_count=1,
            context_text="some context",
            instructions="Be brief.",
        )
        assert "conversation so far" in prompt
        assert "Be brief." in prompt


# ===========================================================================
# compress_tool_calls(context=...) tests
# ===========================================================================


class TestCompressToolCallsContext:
    """Tests for the context parameter on compress_tool_calls."""

    def test_context_truthy_uses_context_system_prompt(self):
        """context=True selects TOOL_COMPACT_CONTEXT_SYSTEM."""
        t, _, _ = _build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted."])])
        t.config.configure_llm(mock)

        t._compression_mgr.compress_tool_calls(context=True)

        assert mock.last_messages is not None
        system_msg = mock.last_messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == TOOL_COMPACT_CONTEXT_SYSTEM

    def test_context_truthy_includes_context_in_prompt(self):
        """context=True embeds compiled context in the user prompt."""
        t, _, _ = _build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted."])])
        t.config.configure_llm(mock)

        t._compression_mgr.compress_tool_calls(context=True)

        user_msg = mock.last_messages[1]
        assert user_msg["role"] == "user"
        # The compiled context includes the system message we committed
        assert "code search agent" in user_msg["content"]
        assert "Here is the conversation so far:" in user_msg["content"]

    def test_context_string_uses_literal_text(self):
        """context='some text' uses the string directly as context_text."""
        t, _, _ = _build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted."])])
        t.config.configure_llm(mock)

        t._compression_mgr.compress_tool_calls(context="custom context for compaction")

        user_msg = mock.last_messages[1]
        assert "custom context for compaction" in user_msg["content"]
        system_msg = mock.last_messages[0]
        assert system_msg["content"] == TOOL_COMPACT_CONTEXT_SYSTEM

    def test_context_none_uses_default_system_prompt(self):
        """Default (context=None) uses TOOL_COMPACT_SYSTEM."""
        t, _, _ = _build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted."])])
        t.config.configure_llm(mock)

        t._compression_mgr.compress_tool_calls()

        system_msg = mock.last_messages[0]
        assert system_msg["content"] == TOOL_COMPACT_SYSTEM

    def test_context_none_no_context_in_prompt(self):
        """Default path does not embed context in the user prompt."""
        t, _, _ = _build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted."])])
        t.config.configure_llm(mock)

        t._compression_mgr.compress_tool_calls(context=None)

        user_msg = mock.last_messages[1]
        assert "conversation so far" not in user_msg["content"]

    def test_context_truthy_result_is_tool_compact_result(self):
        """context=True returns a valid ToolCompactResult."""
        t, _, _ = _build_tool_call_tract()

        mock = MockLLMClient(responses=[json.dumps(["Compacted grep output."])])
        t.config.configure_llm(mock)

        result = t._compression_mgr.compress_tool_calls(context=True)

        assert isinstance(result, ToolCompactResult)
        assert result.turn_count == 1
        assert len(result.edit_commits) == 1
        assert "grep" in result.tool_names

    def test_system_prompt_override_takes_precedence_over_context(self):
        """Explicit system_prompt= overrides even when context=True."""
        t, _, _ = _build_tool_call_tract()

        custom = "You are a custom compactor."
        mock = MockLLMClient(responses=[json.dumps(["Custom."])])
        t.config.configure_llm(mock)

        t._compression_mgr.compress_tool_calls(context=True, system_prompt=custom)

        system_msg = mock.last_messages[0]
        assert system_msg["content"] == custom

    def test_no_regression_basic_compaction(self):
        """Basic compaction without context works as before."""
        t, tool_results, _ = _build_tool_call_tract()

        mock = MockLLMClient(
            responses=[json.dumps(["Found TODO comments at lines 42, 99, 150."])]
        )
        t.config.configure_llm(mock)

        result = t._compression_mgr.compress_tool_calls(target_tokens=50)

        assert isinstance(result, ToolCompactResult)
        assert len(result.edit_commits) == 1
        assert result.turn_count == 1

        # Compiled context still has the assistant answer
        ctx = t.compile()
        messages = ctx.to_dicts()
        assert any("Found 3 TODO" in m.get("content", "") for m in messages)


# ===========================================================================
# TOOL_COMPACT_CONTEXT_SYSTEM existence test
# ===========================================================================


class TestToolCompactContextSystem:
    """Verify the new system prompt constant exists and is well-formed."""

    def test_constant_exists(self):
        assert isinstance(TOOL_COMPACT_CONTEXT_SYSTEM, str)
        assert len(TOOL_COMPACT_CONTEXT_SYSTEM) > 50

    def test_mentions_json_array(self):
        assert "JSON array" in TOOL_COMPACT_CONTEXT_SYSTEM

    def test_mentions_relevance(self):
        assert "relevant" in TOOL_COMPACT_CONTEXT_SYSTEM.lower()
